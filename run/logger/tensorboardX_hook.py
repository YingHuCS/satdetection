import os.path as osp
import cv2

import torch
import torchvision

from utils import master_only
from ..hooks import Hook

class TensorboardXHook(Hook):

    def __init__(self, log_dir=None):
        self.log_dir = log_dir


    def before_run(self, runner):
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            raise ImportError('Please install tensorboardX '
                              'to use TensorboardXHook.')
        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'tbX_logs')
        self.writer = SummaryWriter(self.log_dir)



    def after_run(self, runner):
        self.writer.close()


    def after_val_epoch(self, runner):
        if runner.iter % 1 == 0 and runner.epoch < runner.max_epochs:
            loss_rpn_cls = runner.tensorboardX_buffer.output['loss_rpn_cls']
            loss_rpn_reg = runner.tensorboardX_buffer.output['loss_rpn_reg']
            loss_cls = runner.tensorboardX_buffer.output['loss_cls']
            loss_reg = runner.tensorboardX_buffer.output['loss_reg']
            loss = runner.tensorboardX_buffer.output['loss']
            acc = runner.tensorboardX_buffer.output['acc']

            self.writer.add_scalar('loss_rpn_cls', loss_rpn_cls, runner.epoch)
            self.writer.add_scalar('loss_rpn_reg', loss_rpn_reg, runner.epoch)
            self.writer.add_scalar('loss_cls', loss_cls, runner.epoch)
            self.writer.add_scalar('loss_reg', loss_reg, runner.epoch)
            self.writer.add_scalar('loss', loss, runner.epoch)
            self.writer.add_scalar('acc', acc, runner.epoch)
                       
            imgs_with_boxes = runner.tensorboardX_buffer.output['imgs_with_boxes']   
            imgs_with_boxes = torch.from_numpy(imgs_with_boxes)
            img_grid = torchvision.utils.make_grid(imgs_with_boxes)
            self.writer.add_image('images_with_boxes', img_grid, runner.epoch)
            
            
