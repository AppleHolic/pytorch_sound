from pytorch_sound import settings as CFG


def get_save_name(task_type: str, model_name: str, setting_name: str = ''):
    result = task_type + '__' + model_name
    if setting_name:
        result += '__' + setting_name
    return result


def get_loadable_checkpoint(checkpoint):
    """
    If model is saved with DataParallel, checkpoint keys is started with 'module.' remove it and return new state dict
    :param checkpoint:
    :return: new checkpoint
    """
    new_checkpoint = {}
    for key, val in checkpoint.items():
        new_key = key.replace('module.', '')
        new_checkpoint[new_key] = val
    return new_checkpoint


def get_model():
    # TOOD: rebuild get model function
    raise NotImplementedError


def get_hparam(task_type: str, setting_name: str = ''):
    setup_task_settings(task_type)
    setting_name = setting_name if setting_name else CFG.DEFAULT_MODEL_SETTING
    return getattr(CFG, setting_name, {})()


def setup_task_settings(task_type: str):
    # TODO: rebuild setup process
    raise NotImplementedError


__all__ = ['get_model', 'get_hparam']
