import os
import sys
from multiprocessing import Pool

def run(item):
    gpu_id = item[0]
    start_index = item[1]
    end_index = item[2]
    os.system('python pose.py {} {} {}'.format(gpu_id, start_index, end_index))
    sys.stdout.flush()

def main():
    pool = Pool(4)
    path_given = "/home/saa2/work/temporal-segment-networks/data/ucf101_frames"
    gpu_ids = [0,1,3,4]

    dirs = os.listdir(path_given)
    length = len(dirs)
    offset = length // 5 + 1

    start_indices = []
    end_indices = []

    for i in range(6):
        start_index = offset*i
        end_index = start_index + offset
        if end_index > length:
            end_index = length

        start_indices.append(start_index)
        end_indices.append(end_index)

    pool.map(run, zip(gpu_ids, start_indices, end_indices))

if __name__ == "__main__":
    main()