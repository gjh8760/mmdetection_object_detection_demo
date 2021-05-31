import glob

if __name__ == "__main__":
      with open('../data/VOC/ImageSets/Main/whole.txt', 'r') as f:
            lines = f.readlines()

      lines = sorted(lines)

      # train 90 : test 10

      train = []
      test = []
      for i in range(len(lines)):
            line = lines[i]
            if i%10 == 0:
                  test.append(line)
            else:
                  train.append(line)

      print('whole %d, train %d, test %d'%(len(lines), len(train), len(test)))
      with open('../data/VOC/ImageSets/Main/trainval.txt', 'w') as f:
            for line in train:
                  f.write(line)

      with open('../data/VOC/ImageSets/Main/test.txt', 'w') as f:
            for line in test:
                  f.write(line)
