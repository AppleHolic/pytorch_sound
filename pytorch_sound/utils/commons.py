import multiprocessing
import time


def go_multiprocess(worker_func, inputs):

    # declare pool
    cpu_count = multiprocessing.cpu_count() // 2
    pool = multiprocessing.Pool(cpu_count)

    res = []
    try:
        for i in range(0, len(inputs), cpu_count):
            start_idx, end_idx = i, i + cpu_count
            res += pool.map(worker_func, inputs[start_idx:end_idx])
            print('{}/{}\t{}() processed.'.format(i + 1, len(inputs), worker_func.__name__))
    finally:
        pool.close()

    return res


def tprint(msg):
    print('[{}] {}'.format(time.strftime('%Y%m%d %H:%M:%S'), msg))


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
