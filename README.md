# Pytorch Sound

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FAppleholic%2Fpytorch_sound)](https://hits.seeyoufarm.com)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

---


## Introduction

 *Pytorch Sound* is a modeling toolkit that allows engineers to train custom models for sound related tasks.
 It focuses on removing repetitive patterns that builds deep learning pipelines to boost speed of related experiments.


- Register models and call it other side.
  - It is inspired by https://github.com/pytorch/fairseq


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

> LibriTTS, Maestro, VCTK and VoiceBank are prepared at now.
>
> Freely suggest me a dataset or PR is welcome!


- Abstract Training Process
  - Build forward function (from data to loss, meta)
  - Provide various logging type
    - Tensorboard, Console
    - scalar, plot, image, audio

```python
import torch
from pytorch_sound.trainer import Trainer, LogType


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
```


- English handler sources are brought from https://github.com/keithito/tacotron
  - Add types

- General sound settings and sources


## Usage

### Install

- ffmpeg v4

```bash
$ sudo add-apt-repository ppa:jonathonf/ffmpeg-4
$ sudo apt update
$ sudo apt install ffmpeg
$ ffmpeg -version
```

- install package

```bash
$ pip install -e .
```


### Preprocess / Handling Meta

1. Download data files
  - In the LibriTTS case, checkout [READMD](https://github.com/AppleHolic/pytorch_sound/blob/master/pytorch_sound/scripts/libri_tts/README.md)

2. Run commands (If you want to change sound settings, Change settings.py)

```bash
$ python pytorch_sound/scripts/preprocess.py [libri_tts / vctk / voice_bank] in_dir out_dir
```

3. Checkout preprocessed data, meta files.
  - Maestro dataset is not required running preprocess code at now.


### Examples

- Source (Speech) Separation with audioset : https://github.com/AppleHolic/source_separation


## Environment

- Python > 3.6
- pytorch 1.0
- ubuntu 16.04


## Components

1. Data and its meta file
2. Data Preprocess
3. General functions and modules in sound tasks
4. Abstract training process


## To be updated soon

- Preprocess docs in README.md
- *Add test codes and CI*
- Document website.


## LICENSE

- This repository is under BSD-2 clause license. Check out the LICENSE file.
