import os
import torch


def load_weights(chkpt_path: str, version: float):
    # check path
    assert os.path.exists(chkpt_path), '{} does not exist.'.format(chkpt_path)
    state_dict = torch.load(chkpt_path)
    # check model version
    assert state_dict['version'] == version, \
        'Expected model ver = {}, but saved model ver = {}'.format(version, state_dict['version'])
    return state_dict['model']


def get_latest_weights(project_path: str, task_type: str, version: float):
    saved_path = os.path.join(project_path, 'model', task_type.name.lower() + '.latest.chkpt')
    return load_weights(saved_path, version)
