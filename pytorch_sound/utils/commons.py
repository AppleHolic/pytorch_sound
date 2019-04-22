import multiprocessing


def go_multiprocess(worker_func, inputs):

    # declare pool
    cpu_count = multiprocessing.cpu_count() // 2
    pool = multiprocessing.Pool(cpu_count)

    res = []
    try:
        for i in range(0, len(inputs), cpu_count):
            # chunk 로 나누기.
            start_idx, end_idx = i, i + cpu_count
            # multi-processing 으로 worker 실행.
            res += pool.map(worker_func, inputs[start_idx:end_idx])
            print('{}/{}\t{}() processed.'.format(i + 1, len(inputs), worker_func.__name__))
    finally:
        pool.close()

    return res
