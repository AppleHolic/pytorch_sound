import abc
import glob
import os
import pathlib
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import enum
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from pytorch_sound.data.dataset import SpeechDataLoader
from pytorch_sound.utils.commons import get_loadable_checkpoint, tprint
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

    def __init__(self, model: nn.Module, train_dataset, valid_dataset,
                 sr: int, lr: float, betas: float, weight_decay: float,
                 batch_size: int, num_workers: int,
                 max_step: int, valid_max_step: int, save_interval: int,
                 save_dir: str,
                 grad_clip: float = 0.0, grad_norm: float = 0.0,
                 pretrained_path: str = None):

        # save project info
        self.pretrained_trained = pretrained_path

        # model
        self.model = model

        # logging
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        tprint('Model {} was loaded. Total {} params.'.format(self.model.__class__.__name__, n_params))

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=lr, betas=betas,
                                          weight_decay=weight_decay, eps=1e-6)

        self.dataset = dict()
        self.dataset['train'] = self.repeat(SpeechDataLoader(train_dataset, batch_size, num_workers))
        self.dataset['valid'] = self.repeat(SpeechDataLoader(valid_dataset, batch_size, num_workers))

        # save parameters
        self.step = 0
        self.sr = sr
        self.max_step = max_step
        self.save_interval = save_interval
        self.grad_clip = grad_clip
        self.grad_norm = grad_norm
        self.valid_max_step = valid_max_step

        # load previous checkpoint
        # set seed
        self.seed = None
        self.load()

        if not self.seed:
            self.seed = np.random.randint(np.iinfo(np.int32).max)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)

        if torch.cuda.device_count() > 1:
            self.model = nn.parallel.DataParallel(self.model)

        # load pretrained model
        if self.step == 0 and pretrained_path is not None:
            self.load_pretrained_model()

        # tensorboard logging path
        self.save_dir = save_dir
        # make dirs
        self.log_dir = os.path.join(save_dir, 'logs')
        self.model_dir = os.path.join(save_dir, 'models')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        # valid loss
        self.best_valid_loss = float(1e+5)
        self.save_valid_loss = float(1e+5)

    @abc.abstractmethod
    def forward(self, *inputs):
        raise NotImplemented

    def run(self):
        try:
            # training loop
            for i in range(self.step + 1, self.max_step + 1):

                # update step
                self.step = i

                # logging
                if i % self.save_interval:
                    tprint('------------- TRAIN step : %d -------------' % i)

                # do training step
                self.train_step(i)

                # save model
                if i % self.save_interval == 0:
                    # do validation first
                    self.valid_step(i)
                    # save model checkpoint file
                    self.save(i)

        except KeyboardInterrupt:
            tprint('Train is canceled !!')

    def get_data(self, set_name):
        return to_device(next(self.dataset[set_name]))

    def clip_grad(self):
        if self.grad_clip:
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad = p.grad.clamp(-self.grad_clip, self.grad_clip)
        if self.grad_norm:
            torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],
                                           self.grad_norm)

    def train_step(self, step):

        # flag for logging
        log_flag = step % self.log_interval == 0

        # forward model
        loss, meta = self.forward(*self.get_data('train'), log_flag)

        # update model
        self.optimizer.zero_grad()
        loss.backward()
        self.clip_grad()
        self.optimizer.step()

        # logging
        if log_flag:
            # console logging
            self.console_log('train', meta, step)
            # tensorboard logging
            self.tensorboard_log('train', meta, step)
        return loss

    def valid_step(self, step):
        # switch to evaluation mode
        self.model.eval()

        loss, stat = 0., defaultdict(float)
        for i in range(self.valid_max_step):

            # forward model
            with torch.no_grad():
                loss_this, meta = self.forward(*self.get_data('valid'), True)
                loss += loss_this

            # update stat
            for key, (value, log_type) in meta.items():
                if log_type == LogType.SCALAR:
                    stat[key] += value

            # console logging of this step
            if (i + 1) % self.log_interval == 0:
                self.console_log('valid', meta, i + 1)

            if i == self.valid_max_step - 1:
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
        tprint(msg)

        # tensor board logging of scalar stat
        for key, value in stat.items():
            self.writer.add_scalar('valid/{}'.format(key), value, global_step=step)

        # switch to training mode
        self.model.train()

    def load(self, load_optim=True):
        # make name
        # TODO : save name
        save_name = None

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
            self.model.load_state_dict(get_loadable_checkpoint(state_dict['model']))
            if load_optim:
                self.optimizer.load_state_dict(state_dict['optim'])
            self.step = state_dict['step']
            tprint('checkpoint \'{}\' is loaded. previous step={}'.format(latest_file, self.step))
        else:
            tprint('No any checkpoint in {}. Loading network skipped.'.format(save_path))

    def save(self, step):

        # save latest
        state_dict_inf = {
            'step': step,
            'model': self.model.state_dict(),
            'seed': self.seed,
        }

        # train
        state_dict_train = {
            'step': step,
            'model': self.model.state_dict(),
            'optim': self.optimizer.state_dict(),
            'pretrained_step': step,
            'seed': self.seed
        }

        # save for training
        save_name = None

        save_path = os.path.join(self.model_dir, save_name)
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        torch.save(state_dict_train, os.path.join(save_path, 'step_{:06d}.chkpt'.format(step)))

        # save for inference
        save_path = os.path.join(self.project_path, 'model', save_name + '.latest.chkpt')
        torch.save(state_dict_inf, save_path)

        # logging
        tprint('step %d / saved model.' % step)

    def load_pretrained_model(self):
        assert os.path.exists(self.pretrained_trained), 'You must define pretrained path!'
        self.model.load_state_dict(torch.load(self.pretrained_trained)['model'])

    def console_log(self, tag, meta, step):
        # console logging
        msg = '{}\t{:06d} it'.format(tag, step)
        for key, (value, log_type) in sorted(meta.items()):
            if log_type == LogType.SCALAR:
                msg += '\t{}: {:.6f}'.format(key, value)
        tprint(msg)

    def tensorboard_log(self, tag, meta, step):
        for key, (value, log_type) in meta.items():
            if log_type == LogType.IMAGE:
                self.writer.add_image('{}/{}'.format(tag, key), self.imshow_to_buf(to_numpy(value)), global_step=step)
            elif log_type == LogType.AUDIO:
                self.writer.add_audio('{}/{}'.format(tag, key), to_numpy(value), global_step=step,
                                        sample_rate=self.sr)
            elif log_type == LogType.SCALAR:
                self.writer.add_scalar('{}/{}'.format(tag, key), value, global_step=step)
            elif log_type == LogType.PLOT:
                self.writer.add_image('{}/{}'.format(tag, key), self.plot_to_buf(to_numpy(value)), global_step=step)

    @staticmethod
    def repeat(iterable):
        while True:
            for x in iterable:
                yield x

    @staticmethod
    def plot_to_buf(x, align=True):
        fig, ax = plt.subplots()
        ax.plot(x)
        if align:
            ax.set_ylim([-1, 1])
        fig.canvas.draw()
        im = np.array(fig.canvas.renderer._renderer)
        plt.clf()
        plt.close('all')
        return np.rollaxis(im[..., :3], 2)

    @staticmethod
    def imshow_to_buf(x):
        if len(x.shape) == 3:
            x = x[0]
        fig, ax = plt.subplots()
        ax.imshow(x, cmap='magma', aspect='auto')
        fig.canvas.draw()
        im = np.array(fig.canvas.renderer._renderer)
        plt.clf()
        plt.close('all')
        return np.rollaxis(im[..., :3], 2)
