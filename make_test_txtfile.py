import os
import glob
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="./data/tables", help="data directory path")

args = parser.parse_args()
data_dir = args.data_dir


if __name__ == '__main__':

    set_dir = os.path.join(data_dir, 'ImageSets/Main')
    if not os.path.exists(set_dir):
        os.makedirs(set_dir)
    
    #import ipdb
    #ipdb.set_trace()
    
    names = [os.path.splitext(os.path.basename(obj))[0] for obj in glob.glob(os.path.join(data_dir, 'JPEGImages/*.*'))]

    with open(os.path.join(set_dir, 'test.txt'), 'w') as f:
        for name in names:
            f.write(name + '\n')
