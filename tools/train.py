import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '..'))

import argparse

from utils import Config
from version import __version__
import env
from utils import log, parse_losses
from models import build_detector
from datasets import get_dataset, build_dataloader
from modules import DistributedDataParallel
from run import Runner



def batch_processor(model, data, train_mode):
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs


def train_detector(model,
                   datasets,
                   cfg,
                   logger=None):
    if logger is None:
        logger = log.get_root_logger(cfg.log_level)

    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            dist=True) for dataset in datasets
    ]


    # put model on gpus
    model = DistributedDataParallel(model.cuda())
    # build runner
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir,
                    cfg.log_level)
    # register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
    
    


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args



def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # set training environment, e.g. distribution, cudnn_benchmark, random_seed for re-prodution
    env.set_env(cfg.env_config)

    # init logger before other steps
    logger = log.get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(True))

    if cfg.checkpoint_config is not None:
        # save satdet version in checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            satdet_version=__version__, config=cfg.text)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    train_dataset = get_dataset(cfg.data.train)
    val_dataset = get_dataset(cfg.data.val)
    
    train_detector(
        model,
        [train_dataset, val_dataset],
        cfg,
        logger=logger)
    



if __name__ == '__main__':
    main()
