import abc
import glob
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import enum

from typing import Tuple, Dict, Any
from tensorboardX import SummaryWriter
from collections import defaultdict
from pytorch_sound.settings import SAMPLE_RATE
from pytorch_sound.utils.commons import get_loadable_checkpoint, log
from pytorch_sound.utils.plots import imshow_to_buf, plot_to_buf
from pytorch_sound.utils.tensor import to_device, to_numpy


# switch matplotlib backend
plt.switch_backend('Agg')


class LogType(enum.Enum):
    SCALAR: int = 1
    IMAGE: int = 2
    ENG: int = 3
    AUDIO: int = 4
    PLOT: int = 5


class Trainer:
    """
    Generalized training helper class.

    This class focuses remove repetitive sources in general training pipeline.
    And almost things has similar patterns to train some models, but, in major,
    forwarding process is mainly different in most cases.
    So, if engineer extends this class as their own cases, he/she just override forward function.

    Args:
        model: a main model to be saved and to be forwarded
        optimizer: optimizer module
        train_dataset: dataset on train phase
        valid_dataset: dataset on validation phase
        max_step: maximum iteration step
        valid_max_step: maximum iteration steps on each validation time.
        save_interval: save and validate interval (in iteration)
        log_interval: log interval (in iteration)
        save_dir: base directory to save checkpoints and logs
        save_prefix: a prefix to categorize each experiment
        grad_clip: scalars to clamp gradients
        grad_norm: maximum norm of gradients to be clipped
        pretrained_path: specific file path of checkpoint
        sr: sampling rate
        scheduler: learning rate scheduler

    Examples::
        class MyTrainer(Trainer):

            def forward(self, input: torch.tensor, target: torch.tensor, is_logging: bool):
                # forward model
                out = self.model(input)

                # calc your own loss
                loss = calc_loss(out, target)

                # build meta for logging
                meta = {
                    'loss': (loss.item(), LogType.SCALAR),
                    'out': (out[0], LogType.PLOT)
                }
                return loss, meta
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 train_dataset, valid_dataset,
                 max_step: int, valid_max_step: int, save_interval: int, log_interval: int,
                 save_dir: str, save_prefix: str = 'save',
                 grad_clip: float = 0.0, grad_norm: float = 0.0,
                 pretrained_path: str = None, sr: int = None, scheduler: torch.optim.lr_scheduler._LRScheduler = None):

        # save project info
        self.pretrained_trained = pretrained_path

        # model
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        # log how many parameters in the model
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log('Model {} was loaded. Total {} params.'.format(self.model.__class__.__name__, n_params))

        # adopt repeating function on datasets
        self.train_dataset = self.repeat(train_dataset)
        self.valid_dataset = self.repeat(valid_dataset)

        # save parameters
        self.step = 0
        if sr:
            self.sr = sr
        else:
            self.sr = SAMPLE_RATE
        self.max_step = max_step
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.save_dir = save_dir
        self.save_prefix = save_prefix
        self.grad_clip = grad_clip
        self.grad_norm = grad_norm
        self.valid_max_step = valid_max_step

        # make dirs
        self.log_dir = os.path.join(save_dir, 'logs', self.save_prefix)
        self.model_dir = os.path.join(save_dir, 'models')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        # load previous checkpoint
        # set seed
        self.seed = None
        self.load()

        if not self.seed:
            self.seed = np.random.randint(np.iinfo(np.int32).max)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)

        # load pretrained model
        if self.step == 0 and pretrained_path:
            self.load_pretrained_model()

        # valid loss
        self.best_valid_loss = np.finfo(np.float32).max
        self.cur_best_valid_loss = self.best_valid_loss
        self.save_valid_loss = np.finfo(np.float32).max

    @abc.abstractmethod
    def forward(self, *inputs, is_logging: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        :param inputs: Loaded Data Points from Speech Loader
        :param is_logging: log or not
        :return: Loss Tensor, Log Dictionary
        """
        raise NotImplemented

    def run(self) -> float:
        try:
            # training loop
            for i in range(self.step + 1, self.max_step + 1):

                # update step
                self.step = i

                # logging
                if i % self.save_interval == 1:
                    log('------------- TRAIN step : %d -------------' % i)

                # do training step
                if self.scheduler is not None:
                    self.scheduler.step(i)
                self.model.train()
                self.train(i)

                # save model
                if i % self.save_interval == 0:
                    log('------------- VALID step : %d -------------' % i)
                    # valid
                    self.model.eval()
                    self.validate(i)
                    # save model checkpoint file
                    self.save(i)

        except KeyboardInterrupt:
            log('Train is canceled !!')

        return self.best_valid_loss

    def clip_grad(self):
        if self.grad_clip:
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad = p.grad.clamp(-self.grad_clip, self.grad_clip)
        if self.grad_norm:
            torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],
                                           self.grad_norm)

    def train(self, step: int) -> torch.Tensor:

        # update model
        self.optimizer.zero_grad()

        # flag for logging
        log_flag = step % self.log_interval == 0

        # forward model
        loss, meta = self.forward(*to_device(next(self.train_dataset)), is_logging=log_flag)

        # check loss nan
        if loss != loss:
            log('{} cur step NAN is occured'.format(step))
            return

        loss.backward()
        self.clip_grad()
        self.optimizer.step()

        # logging
        if log_flag:
            # console logging
            self.console_log('train', meta, step)
            # tensorboard logging
            self.tensorboard_log('train', meta, step)

    def validate(self, step: int):

        loss = 0.
        stat = defaultdict(float)

        for i in range(self.valid_max_step):

            # forward model
            with torch.no_grad():
                batch_loss, meta = self.forward(*to_device(next(self.valid_dataset)), is_logging=True)
                loss += batch_loss

            # update stat
            for key, (value, log_type) in meta.items():
                if log_type == LogType.SCALAR:
                    stat[key] += value

            # console logging of this step
            if (i + 1) % self.log_interval == 0:
                self.console_log('valid', meta, i + 1)

        meta_non_scalar = {
            key: (value, log_type) for key, (value, log_type) in meta.items()
            if not log_type == LogType.SCALAR
        }
        self.tensorboard_log('valid', meta_non_scalar, step)

        # averaging stat
        loss /= self.valid_max_step
        for key in stat.keys():
            stat[key] = stat[key] / self.valid_max_step

        # update best valid loss
        if loss < self.best_valid_loss:
            self.best_valid_loss = loss

        # console logging of total stat
        msg = 'step {} / total stat'.format(step)
        for key, value in sorted(stat.items()):
            msg += '\t{}: {:.6f}'.format(key, value)
        log(msg)

        # tensor board logging of scalar stat
        for key, value in stat.items():
            self.writer.add_scalar('valid/{}'.format(key), value, global_step=step)

    @property
    def save_name(self):
        if isinstance(self.model, nn.parallel.DataParallel):
            module = self.model.module
        else:
            module = self.model
        return self.save_prefix + '/' + module.__class__.__name__

    def load(self, load_optim: bool = True):
        # make name
        save_name = self.save_name

        # save path
        save_path = os.path.join(self.model_dir, save_name)

        # get latest file
        check_files = glob.glob(os.path.join(save_path, '*'))
        if check_files:
            # load latest state dict
            latest_file = max(check_files, key=os.path.getctime)
            state_dict = torch.load(latest_file)
            if 'seed' in state_dict:
                self.seed = state_dict['seed']
            # load model
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(get_loadable_checkpoint(state_dict['model']))
            else:
                self.model.load_state_dict(get_loadable_checkpoint(state_dict['model']))
            if load_optim:
                self.optimizer.load_state_dict(state_dict['optim'])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(state_dict['scheduler'])
            self.step = state_dict['step']
            log('checkpoint \'{}\' is loaded. previous step={}'.format(latest_file, self.step))
        else:
            log('No any checkpoint in {}. Loading network skipped.'.format(save_path))

    def save(self, step: int):

        # state dict
        state_dict = get_loadable_checkpoint(self.model.state_dict())

        # train
        state_dict = {
            'step': step,
            'model': state_dict,
            'optim': self.optimizer.state_dict(),
            'pretrained_step': step,
            'seed': self.seed
        }
        if self.scheduler is not None:
            state_dict.update({
                'scheduler': self.scheduler.state_dict()
            })

        # save for training
        save_name = self.save_name

        save_path = os.path.join(self.model_dir, save_name)
        os.makedirs(save_path, exist_ok=True)
        torch.save(state_dict, os.path.join(save_path, 'step_{:06d}.chkpt'.format(step)))

        # save best
        if self.best_valid_loss != self.cur_best_valid_loss:
            save_path = os.path.join(self.model_dir, save_name + '.best.chkpt')
            torch.save(state_dict, save_path)
            self.cur_best_valid_loss = self.best_valid_loss

        # logging
        log('step %d / saved model.' % step)

    def load_pretrained_model(self):
        assert os.path.exists(self.pretrained_trained), 'You must define pretrained path!'
        self.model.load_state_dict(get_loadable_checkpoint(torch.load(self.pretrained_trained)['model']))

    def console_log(self, tag: str, meta: Dict[str, Any], step: int):
        # console logging
        msg = '{}\t{:06d} it'.format(tag, step)
        for key, (value, log_type) in sorted(meta.items()):
            if log_type == LogType.SCALAR:
                msg += '\t{}: {:.6f}'.format(key, value)
        log(msg)

    def tensorboard_log(self, tag: str, meta: Dict[str, Any], step: int):
        for key, (value, log_type) in meta.items():
            if log_type != LogType.SCALAR and type(value) == torch.Tensor:
                value = to_numpy(value)
            if log_type == LogType.IMAGE:
                self.writer.add_image('{}/{}'.format(tag, key), imshow_to_buf(value), global_step=step)
            elif log_type == LogType.AUDIO:
                self.writer.add_audio('{}/{}'.format(tag, key), value, global_step=step, sample_rate=self.sr)
            elif log_type == LogType.SCALAR:
                self.writer.add_scalar('{}/{}'.format(tag, key), value, global_step=step)
            elif log_type == LogType.PLOT:
                self.writer.add_image('{}/{}'.format(tag, key), plot_to_buf(value), global_step=step)

    @staticmethod
    def repeat(iterable):
        while True:
            for x in iterable:
                yield x
