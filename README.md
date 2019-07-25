# Pytorch Sound

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FAppleholic%2Fpytorch_sound)](https://hits.seeyoufarm.com)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

---


## Introduction

 *Pytorch Sound* is a modeling toolkit that allows engineers to train custom models for sound related tasks.
 It focuses on removing repetitive patterns that builds deep learning pipelines.


- Register models and call it other side.
  - It is inspired https://github.com/pytorch/fairseq

```python
import torch.nn as nn
from pytorch_sound.models import register_model, register_model_architecture


@register_model('my_model')
class Model(nn.Module):
...


@register_model_architecture('my_model', 'my_model_base')
def my_model_base():
    return {'hidden_dim': 256}
```

```python
from pytorch_sound.models import build_model


# build model
model_name = 'my_model_base'
model = build_model(model_name)
```


- Several dataset sources (preprocess, meta, general sound dataset)

> LibriTTS, Maestro, VCTK and Voice Bank are prepared at now.
>
> Freely suggest me a dataset !


- Abstract Training Process
  - You just build forward function (data to loss)
  - Provide various logging type
    - Tensorboard, Console
    - scalar, plot, image, audio

```python
import torch
from pytorch_sound.trainer import Trainer, LogType


class MyTrainer(Trainer):

    def forward(input: torch.tensor, target: torch.tensor, is_logging: bool):
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
```


- General sound settings and sources


## Environment

- Python > 3.6
- pytorch 1.0
- ffmpeg

```bash
$ sudo add-apt-repository ppa:jonathonf/ffmpeg-4
$ sudo apt updated
$ sudo apt install ffmpeg
$ ffmpeg -version
```

## Components

1. Data and its meta file
2. Data Preprocess
3. General functions and modules in sound tasks
4. Abstract training process


## To be updated soon

- *Add test codes and CI*
- *Examples (external repositories)*


## LICENSE

- This repository is under BSD-2 clause license. Check out the LICENSE file.