import builtins
import datetime
import os
import time
from argparse import Namespace

import accelerate
import einops
import ml_collections
import torch
from datasets import get_dataset
from loguru import logger
from torch import multiprocessing as mp
from torch.utils._pytree import tree_map
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import taming.models.vqgan
import wandb
import torch.utils.data
from libs.muse import MUSE
from tools.fid_score import calculate_fid_given_paths

import utils

logging = logger

torch.multiprocessing.set_sharing_strategy('file_system')  # todo


def convert_model_dtype(models, dtype):
    logging.info(f'Converting model to {dtype}')
    if not isinstance(models, (list, tuple)):
        models = [models]
    for model in models:
        if model is None:
            continue
        if dtype == torch.float16:
            model.half()
        elif dtype == torch.bfloat16:
            model.bfloat16()


def LSimple(x0, nnet, schedule, **kwargs):
    labels, masked_ids = schedule.sample(x0)
    logits = nnet(masked_ids, **kwargs)
    # b (h w) c, b (h w)
    loss = schedule.loss(logits, labels)
    return loss


def train(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.ConfigDict(config)

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logging.info(config)
        wandb.init(dir=os.path.abspath(config.workdir), project=f'cc3m', config=config.to_dict(),
                   job_type='train', mode='online', settings=wandb.Settings(start_method='fork'))
    else:
        logging.remove()
        logger.add(sys.stderr, level='ERROR')
        builtins.print = lambda *args: None
    logging.info(f'Run on {accelerator.num_processes} devices')

    ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(config.ckpt_root)))
    if not ckpts:
        resume_step = 0
    else:
        steps = map(lambda x: int(x.split(".")[0]), ckpts)
        resume_step = max(steps)

    logger.info(f'world size is {accelerator.num_processes}')
    dist_eval = config.wds.dist_eval

    webdataset_args = Namespace(
        train_data=config.wds.train_data,
        val_data=config.wds.val_data,
        dist_eval=dist_eval,
        ctx_path=config.wds.ctx_path,
        seed=config.seed,
        batch_size=mini_batch_size,
        val_batch_size=config.sample.mini_batch_size,
        workers=config.train.num_workers,
        world_size=accelerator.num_processes,
        train_num_samples=getattr(config, 'wds.train_num_samples', 6091948),
        val_num_samples=getattr(config, 'wds.val_num_samples', 13818),
        dataset_type='webdataset',
    )

    dataset = get_dataset(**config.dataset,
                          args=webdataset_args,
                          step=resume_step)
    # assert os.path.exists(dataset.fid_stat)
    train_dataset = dataset.get_split(split='train', labeled=True)
    test_dataset = dataset.get_split(split='test', labeled=True)  # for sampling
    train_dataset_loader = train_dataset.dataloader
    test_dataset_loader = test_dataset.dataloader

    autoencoder = taming.models.vqgan.get_model(**config.autoencoder)
    autoencoder.to(device)

    train_state = utils.initialize_train_state(config, device)
    nnet, nnet_ema, optimizer = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer)
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.ckpt_root)

    @torch.cuda.amp.autocast()
    def encode(_batch):
        res = autoencoder.encode(_batch)[-1][-1].reshape(len(_batch), -1)
        return res

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode_code(_batch)

    def get_data_generator():
        while True:
            for data in tqdm(train_dataset_loader, disable=not accelerator.is_main_process, desc='epoch'):
                yield data

    data_generator = get_data_generator()

    def get_context_generator():
        while True:
            for data in test_dataset_loader:
                _, _context = data
                yield _context

    context_generator = get_context_generator()

    muse = MUSE(codebook_size=autoencoder.n_embed, device=device, **config.muse)

    def cfg_nnet(x, context, scale=None):
        _cond = nnet_ema(x, context=context)
        _empty_context = torch.tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = nnet_ema(x, context=_empty_context)
        res = _cond + scale * (_cond - _uncond)
        return res

    def train_step(_batch):
        _metrics = dict()
        optimizer.zero_grad()
        _z, context = proc_batch_feat(_batch)
        loss = LSimple(_z, nnet, muse, context=context)  # currently only support the extracted feature version
        metric_logger.update(loss=accelerator.gather(loss.detach()).mean())
        accelerator.backward(loss.mean())
        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        loss_scale, grad_norm = accelerator.scaler.get_scale(), utils.get_grad_norm_(nnet.parameters())
        metric_logger.update(loss_scale=loss_scale)
        metric_logger.update(grad_norm=grad_norm)
        return dict(lr=train_state.optimizer.param_groups[0]['lr'],
                    **{k: v.value for k, v in metric_logger.meters.items()})

    def proc_batch_feat(_batch):
        _z = _batch[0].reshape(-1, 256)
        context = _batch[1].reshape(_z.shape[0], 77, -1)
        assert context.shape[-1] == config.nnet.clip_dim
        return _z, context

    def eval_step(n_samples, sample_steps):
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}'
                     f'mini_batch_size={config.sample.mini_batch_size}')

        def sample_fn(_n_samples):
            _context = next(context_generator)
            _context = _context.to(device).reshape(-1, 77, config.nnet.clip_dim)
            kwargs = dict(context=_context)
            return muse.generate(config, _n_samples, cfg_nnet, decode, **kwargs)

        if accelerator.is_main_process:
            path = f'{config.workdir}/eval_samples/{train_state.step}_{datetime.datetime.now().strftime("%m%d_%H%M%S")}'
            logging.info(f'Path for FID images: {path}')
        else:
            path = None

        utils.sample2dir(accelerator, path, n_samples, config.sample.mini_batch_size, sample_fn,
                         dataset.unpreprocess, dist=dist_eval)

        _fid = 0
        if accelerator.is_main_process:
            _fid = calculate_fid_given_paths((dataset.fid_stat, path))
            logging.info(f'step={train_state.step} fid{n_samples}={_fid}')
            with open(os.path.join(config.workdir, 'eval.log'), 'a') as f:
                print(f'step={train_state.step} fid{n_samples}={_fid}', file=f)
            wandb.log({f'fid{n_samples}': _fid}, step=train_state.step)
        _fid = torch.tensor(_fid, device=device)
        _fid = accelerator.reduce(_fid, reduction='sum')

        return _fid.item()

    if eval_ckpt_path := os.getenv('EVAL_CKPT', ''):
        nnet.eval()
        train_state.resume(eval_ckpt_path)
        logging.info(f'Eval {train_state.step}...')
        eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps)
        return

    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')
    step_fid = []
    metric_logger = utils.MetricLogger()
    while train_state.step < config.train.n_steps:
        nnet.train()
        data_time_start = time.time()
        batch = tree_map(lambda x: x.to(device), next(data_generator))
        metric_logger.update(data_time=time.time() - data_time_start)
        metrics = train_step(batch)

        nnet.eval()

        if train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:
            torch.cuda.empty_cache()
            logging.info(f'Save checkpoint {train_state.step}...')
            if accelerator.local_process_index == 0:
                train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
        accelerator.wait_for_everyone()

        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            logger.info(f'step: {train_state.step} {metric_logger}')
            wandb.log(metrics, step=train_state.step)

        if train_state.step % config.train.eval_interval == 0:
            torch.cuda.empty_cache()
            logging.info('Save a grid of images...')
            contexts = torch.tensor(dataset.contexts, device=device)[: 2 * 5]
            samples = muse.generate(config, 2 * 5, cfg_nnet, decode, context=contexts)
            samples = make_grid(dataset.unpreprocess(samples), 5)
            save_image(samples, os.path.join(config.sample_dir, f'{train_state.step}_{accelerator.process_index}.png'))
            if accelerator.is_main_process:
                wandb.log({'samples': wandb.Image(samples)}, step=train_state.step)
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        if train_state.step % config.train.fid_interval == 0 or train_state.step == config.train.n_steps:
            torch.cuda.empty_cache()
            logging.info(f'Eval {train_state.step}...')
            fid = eval_step(n_samples=config.eval.n_samples,
                            sample_steps=config.eval.sample_steps)  # calculate fid of the saved checkpoint
            step_fid.append((train_state.step, fid))
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    logging.info(f'Finish fitting, step={train_state.step}')
    logging.info(f'step_fid: {step_fid}')
    step_best = sorted(step_fid, key=lambda x: x[1])[0][0]
    logging.info(f'step_best: {step_best}')
    train_state.load(os.path.join(config.ckpt_root, f'{step_best}.ckpt'))
    del metrics
    accelerator.wait_for_everyone()
    eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps)


from absl import flags
from absl import app
from ml_collections import config_flags
import sys
from pathlib import Path

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_bool("disable_val", False, 'help')


def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--config='):
            return Path(argv[i].split('=')[-1]).stem


def main(argv):
    config = FLAGS.config
    config.config_name = get_config_name()
    config.workdir = os.getenv('OUTPUT_DIR',
                               Path.home() / 'exp/default' / datetime.datetime.now().strftime("%m%d_%H%M%S"))
    config.disable_val = FLAGS.disable_val
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    train(config)


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    app.run(main)
