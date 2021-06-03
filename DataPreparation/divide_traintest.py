import glob
import argparse
from random import shuffle
parser = argparse.ArgumentParser()
parser.add_argument('--rate', type=float, default=0.9, help='training data portion')
args = parser.parse_args()

if __name__ == "__main__":
    with open('../data/VOC2007/ImageSets/Main/whole.txt', 'r') as f:
        lines = f.readlines()

    lines = sorted(lines)
    shuffle(lines)

    # train 90 : test 10

    train = []
    test = []
    trainNum = round(len(lines)*args.rate)
    trainlines = lines[0:trainNum]
    trainlines = sorted(trainlines)
    testlines = lines[trainNum:]
    testlines = sorted(testlines)
    for line in trainlines:
        train.append(line)
    for line in testlines:
        test.append(line)

    print('whole %d, train %d, test %d'%(len(lines), len(train), len(test)))
    with open('../data/VOC2007/ImageSets/Main/trainval.txt', 'w') as f:
        for line in train:
            f.write(line)

    with open('../data/VOC2007/ImageSets/Main/test.txt', 'w') as f:
        for line in test:
            f.write(line)
