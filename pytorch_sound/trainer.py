import abc
import glob
import os
import pathlib
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import enum
from collections import defaultdict
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from pytorch_sound.data.dataset import SpeechDataLoader, SpeechDataset
from pytorch_sound.utils.settings import CFG, get_save_name
from pytorch_sound.utils.tensor import to_device, to_numpy
from pytorch_sound.utils.text import kor_i2t
from pytorch_sound.utils.settings import get_hparam, get_model


# switch matplotlib backend
plt.switch_backend('Agg')


LogType = enum.Enum('LogType', 'SCALAR IMAGE KOR ENG AUDIO PLOT')


class Trainer:

    def __init__(self, rank: int, project_path: str, task_type: str, setting_name: str = '', pretrained_path: str = None):

        # save project info
        self.rank = rank
        self.project_path = project_path
        self.task_type = task_type
        self.pretrained_trained = pretrained_path
        self.setting_name = setting_name

        # set seed
        seed = np.random.randint(2**16)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # load hyper params
        hparam = get_hparam(self.task_type, setting_name=setting_name)
        self.hparam = hparam

        # model
        self.model = get_model(self.task_type, setting_name=setting_name, pretrained_path=pretrained_path)

        # logging
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.tprint('Model [{}] was loaded. Total {} params.'.format(task_type, n_params))

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.hparam['lr'], betas=self.hparam['betas'],
                                          weight_decay=self.hparam['weight_decay'], eps=1e-6)

        # dataset
        def data_loader_func(is_train):
            dataset = SpeechDataset(rank, project_path, task_type, setting_name, is_train=is_train)
            return SpeechDataLoader(dataset, hparam['batch_size'],
                                    hparam['num_workers'], is_bucket=hparam['is_bucket'])

        self.dataset = dict()
        self.dataset['train'] = self.repeat(data_loader_func(True))
        self.dataset['valid'] = self.repeat(data_loader_func(False))

        # current step
        self.step = 0

        # load previous checkpoint
        self.load()

        # load pretrained model
        if self.step == 0 and pretrained_path is not None:
            self.load_pretrained_model()

        # tensorboard logging path
        log_path = os.path.join(self.project_path, 'log', self.task_type,
                                '{}-run-{}'.format(setting_name, time.strftime('%Y%m%d-%H%M%S')))
        self.writer = SummaryWriter(log_dir=log_path, flush_secs=10)

        # valid loss
        self.best_valid_loss = float(1e+5)
        self.save_valid_loss = float(1e+5)

    @abc.abstractmethod
    def forward(self, *inputs):
        raise NotImplemented

    def run(self):

        try:
            # training loop
            for i in range(self.step + 1, self.hparam['max_step'] + 1):

                # update step
                self.step = i

                # logging
                if i % CFG.SAVE_INTERVAL == 1:
                    self.tprint('------------- TRAIN step : %d -------------' % i)

                # do training step
                self.train_step(i)

                # save model ( main controller only )
                if self.rank <= 0 and i % CFG.SAVE_INTERVAL == 0:
                    # do validation first
                    self.valid_step(i)
                    # save model checkpoint file
                    self.save(i)

        except KeyboardInterrupt:
            self.tprint('Train is canceled !!')

    def get_data(self, set_name):
        return to_device(next(self.dataset[set_name]))

    def train_step(self, step):

        def clip_grad():
            if 'grad_clip' in self.hparam:
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad = p.grad.clamp(-self.hparam['grad_clip'], self.hparam['grad_clip'])
            if 'grad_norm' in self.hparam:
                torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],
                                               self.hparam['grad_norm'])

        def lr_schedule():
            if 'lrs_start' in self.hparam and \
               'lrs_gamma' in self.hparam and \
               'lr_target' in self.hparam:
                lrs_start = self.hparam['lrs_start']
                lrs_gamma = 1 - self.hparam['lrs_gamma']
                lr_target = self.hparam['lr_target']
                lr = self.hparam['lr'] * lrs_gamma ** max(0, self.step - lrs_start)
                lr = max(lr_target, lr)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

        # flag for logging
        log_flag = self.rank <= 0 and step % CFG.LOG_INTERVAL == 0

        # forward model
        loss, meta = self.forward(*self.get_data('train'), log_flag)

        # learning rate schedule
        lr_schedule()

        # update model
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad()
        self.optimizer.step()

        # logging
        if log_flag:
            # console logging
            self.console_log('train', meta, step)
            # tensorboard logging
            self.tensorboard_log('train', meta, step)

        return loss

    def valid_step(self, step):

        self.tprint('------------- Valid step : %d -------------' % step)

        # switch to evaluation mode
        self.model.eval()

        loss, stat = 0, defaultdict(float)
        for i in range(CFG.VALID_MAX_STEP):

            # forward model
            with torch.no_grad():
                loss_this, meta = self.forward(*self.get_data('valid'), True)
                loss += loss_this

            # update stat
            for key, (value, log_type) in meta.items():
                if log_type == LogType.SCALAR:
                    stat[key] += value

            # console logging of this step
            if (i + 1) % CFG.LOG_INTERVAL == 0:
                self.console_log('valid', meta, i + 1)

            if i == CFG.VALID_MAX_STEP - 1:
                meta_non_scalar = {
                    key: (value, log_type) for key, (value, log_type) in meta.items()
                    if not log_type == LogType.SCALAR
                }
                self.tensorboard_log('valid', meta_non_scalar, step)

        # averaging stat
        loss /= CFG.VALID_MAX_STEP
        for key in stat.keys():
            stat[key] = stat[key] / CFG.VALID_MAX_STEP

        # update best valid loss
        if loss < self.best_valid_loss:
            self.best_valid_loss = loss

        # console logging of total stat
        msg = 'step {} / total stat'.format(step)
        for key, value in sorted(stat.items()):
            msg += '\t{}: {:.6f}'.format(key, value)
        self.tprint(msg)

        # tensor board logging of scalar stat
        for key, value in stat.items():
            self.writer.add_scalar('valid/{}'.format(key), value, global_step=step)

        # switch to training mode
        self.model.train()

    def load(self, load_optim=True):
        # make name
        save_name = get_save_name(self.task_type, self.hparam['model']['name'], self.setting_name)

        # save path
        save_path = os.path.join(self.project_path, 'model', save_name)

        # get latest file
        check_files = glob.glob(os.path.join(save_path, '*'))
        if check_files:
            # load latest state dict
            latest_file = max(check_files, key=os.path.getctime)
            state_dict = torch.load(latest_file)
            # check model version
            assert self.hparam['version'] == state_dict['version'], \
                'Expected model ver = {}, but saved model ver = {}'.format(self.hparam['version'],
                                                                           state_dict['version'])
            # load model
            self.model.load_state_dict(state_dict['model'])
            if load_optim:
                self.optimizer.load_state_dict(state_dict['optim'])
            self.step = state_dict['step']
            self.tprint('checkpoint \'{}\' is loaded. previous step={}'.format(latest_file, self.step))
        else:
            self.tprint('No any checkpoint in {}. Loading network skipped.'.format(save_path))

    def save(self, step):

        # save latest
        state_dict_inf = {
            'version': self.hparam['version'],
            'step': step,
            'model': self.model.state_dict()
        }

        # train
        state_dict_train = {
            'step': step,
            'model': self.model.state_dict(),
            'optim': self.optimizer.state_dict(),
            'pretrained_step': step,
        }

        # save for training
        save_name = get_save_name(self.task_type, self.hparam['model']['name'], self.setting_name)

        save_path = os.path.join(self.project_path, 'model', save_name)
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        torch.save(state_dict_train, os.path.join(save_path, 'step_{:06d}.chkpt'.format(step)))

        # save for inference
        save_path = os.path.join(self.project_path, 'model', save_name + '.latest.chkpt')
        torch.save(state_dict_inf, save_path)

        # logging
        self.tprint('step %d / saved model.' % step)

    def load_pretrained_model(self):
        assert os.path.exists(self.pretrained_trained), 'You must define pretrained path!'
        self.model.load_state_dict(torch.load(self.pretrained_trained)['model'])

    def console_log(self, tag, meta, step):
        # console logging
        msg = '{}\t{:06d} it'.format(tag, step)
        for key, (value, log_type) in sorted(meta.items()):
            if log_type == LogType.SCALAR:
                msg += '\t{}: {:.6f}'.format(key, value)
        self.tprint(msg)

    def tensorboard_log(self, tag, meta, step):
        for key, (value, log_type) in meta.items():
            if log_type == LogType.IMAGE:
                self.writer.add_image('{}/{}'.format(tag, key), self.imshow_to_buf(to_numpy(value)), global_step=step)
            elif log_type == LogType.KOR:
                self.writer.add_text('{}/{}'.format(tag, key), kor_i2t(to_numpy(value)), global_step=step)
            elif log_type == LogType.AUDIO:
                self.writer.add_audio('{}/{}'.format(tag, key), to_numpy(value), global_step=step,
                                        sample_rate=CFG.SAMPLE_RATE)
            elif log_type == LogType.SCALAR:
                self.writer.add_scalar('{}/{}'.format(tag, key), value, global_step=step)
            elif log_type == LogType.PLOT:
                self.writer.add_image('{}/{}'.format(tag, key), self.plot_to_buf(to_numpy(value)), global_step=step)

    def tprint(self, msg):
        if self.rank <= 0:
            print('[{}] {}'.format(time.strftime('%Y%m%d %H:%M:%S'), msg))

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

    @staticmethod
    def apply_gradient_allreduce(module):
        # based on :
        # https://github.com/NVIDIA/waveglow/blob/master/distributed.py

        #
        # inner functions
        #
        def flatten_tensors(tensors):
            return torch.cat([t.contiguous().view(-1) for t in tensors], dim=0)

        def unflatten_tensors(flat, tensors):
            offset, res = 0, list()
            for tensor in tensors:
                numel = tensor.numel()
                res.append(flat.narrow(0, offset, numel).view_as(tensor))
                offset += numel
            return tuple(res)

        def bucketing(tensors):
            buckets = {}
            for tensor in tensors:
                tp = tensor.dtype  # bucket by type
                if tp not in buckets:
                    buckets[tp] = []
                buckets[tp].append(tensor)
            return buckets

        def sync_buckets(buckets):
            for tp in buckets:
                buffers = [buf for buf in buckets[tp]]
                coalesced = flatten_tensors(buffers)
                torch.distributed.all_reduce(coalesced)
                coalesced /= torch.distributed.get_world_size()
                for buf, synced in zip(buffers, unflatten_tensors(coalesced, buffers)):
                    buf.copy_(synced)

        def all_reduce_grads():
            if module.needs_reduction:
                module.needs_reduction = False
                # bucketing for efficiency
                buckets = bucketing([param.grad.data for param in module.parameters()
                                     if param.requires_grad and param.grad is not None])
                # sync gradients
                sync_buckets(buckets)

        def all_reduce_buffers():
            # bucketing for efficiency
            buckets = bucketing([buf.data for buf in module.buffers()])
            # gradients sync
            sync_buckets(buckets)

        def grad_hook(*unused):
            Variable._execution_engine.queue_callback(all_reduce_grads)

        def forward_hook(self, input, output):
            # set flag
            self.needs_reduction = True
            # sync buffers(for BN like modules)
            all_reduce_buffers()

        #
        # sync initial parameters
        #
        for p in module.state_dict().values():
            if torch.is_tensor(p):
                torch.distributed.broadcast(p, 0)

        #
        # register grad hook
        #
        for param in list(module.parameters()):
            if param.requires_grad:
                param.register_hook(grad_hook)

        #
        # register forward hook
        #
        module.register_forward_hook(forward_hook)

        return module
