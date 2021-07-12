import os, sys, glob
import numpy as np
import argparse
import re
import xml.etree.ElementTree as ET
import time
import matplotlib
import matplotlib.pylab as plt
import mmcv
from mmcv.runner import load_checkpoint
import mmcv.visualization.image as mmcv_image
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result, init_detector
import torch
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/VOC2007', help='data directory path')
parser.add_argument('--config_dir', default='./Config', help='config directory path')
parser.add_argument('--epoch', type=int, default=12, help='maximum training epoch')
parser.add_argument('--gpu', type=int, default=2)
parser.add_argument('--output_dir', help='output image directory, e.g., ./Result/test')
parser.add_argument('--focal', type=int, default=0, help='0 for CELoss, 1 for FocalLoss')
parser.add_argument('--save_pickle', default=1, help='1 for saving pickle')
parser.add_argument('--pickle_dir', default='./prediction/VOC2007', help='bbox prediction pickle directory path')
args = parser.parse_args()

if args.focal:
    MODELS_CONFIG = {'cascade_mask_rcnn_hrnet': {'config_file': 'cascade_mask_rcnn_hrnetv2p_w32_20e_focal.py'}}
else:
    MODELS_CONFIG = {'cascade_mask_rcnn_hrnet': {'config_file': 'cascade_mask_rcnn_hrnetv2p_w32_20e_ce.py'}}

DATA_DIR = args.data_dir
CONFIG_DIR = args.config_dir
MAX_EPOCH = args.epoch
OUT_DIR = args.output_dir
PICKLE_DIR = args.pickle_dir
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PICKLE_DIR, exist_ok=True)

if __name__ == "__main__":
    # parse data classes
    anno_path = os.path.join(DATA_DIR, "Annotations")
#    classes_names = []
#    xml_list = []
#    for xml_file in glob.glob(os.path.join(anno_path, "*.xml")):
#        tree = ET.parse(xml_file)
#        root = tree.getroot()
#        for member in root.findall("object"):
#            classes_names.append(member[0].text)
#    classes_names = list(set(classes_names))
#    classes_names.sort()
    classes_names = ['bordered', 'borderless']
    print(classes_names)

    # modify config file - 아마 이미 다 되있는 것 같음
    selected_model = 'cascade_mask_rcnn_hrnet'
    config_file = MODELS_CONFIG[selected_model]['config_file']

    config_fname = os.path.join(CONFIG_DIR, config_file)
    assert os.path.isfile(config_fname), '`{}` not exist'.format(config_fname)

    with open(config_fname) as f:
        s = f.read()
        work_dir = re.findall(r"work_dir = \'(.*?)\'", s)[0]
#        s = re.sub('num_classes=.*?,',
#               'num_classes={},'.format(len(classes_names) + 1), s)
#        s = re.sub('ann_file=.*?\],',
#                   "ann_file=data_root + 'VOC2007/ImageSets/Main/trainval.txt',", s, flags=re.S)
#        s = re.sub('total_epochs = \d+',
#                   'total_epochs = {} #'.format(MAX_EPOCH), s)
#        if "CocoDataset" in s:
#            s = re.sub("data_root = 'data/coco/'",
#                       "data_root = 'data/'", s)
#            s = re.sub("annotations/instances_train2017.json",
#                       "coco/trainval.json", s)
#            s = re.sub("annotations/instances_val2017.json",
#                       "coco/test.json", s)
#            s = re.sub("annotations/instances_val2017.json",
#                       "coco/test.json", s)
#            s = re.sub("train2017", "VOC2007/JPEGImages", s)
#            s = re.sub("val2017", "VOC2007/JPEGImages", s)
#        else:
#            s = re.sub('img_prefix=.*?\],',
#                       "img_prefix=data_root + 'VOC2007/JPEGImages',".format(MAX_EPOCH), s)
    with open(config_fname, 'w') as f:
        f.write(s)


    # Test
    score_thr = 0.7
    checkpoint_file = os.path.join(work_dir, "epoch_%d.pth"%(MAX_EPOCH))
    assert os.path.isfile(checkpoint_file), '`{}` not exist'.format(checkpoint_file)
    checkpoint_file = os.path.abspath(checkpoint_file)

    device = torch.device("cuda:%d"%(args.gpu) if torch.cuda.is_available() else 'cpu')
    model = init_detector(config_fname, checkpoint_file, device=device)
    with open(os.path.join(DATA_DIR, 'ImageSets', 'Main', 'test.txt'), 'r') as f:
        test_filenames = f.readlines()
        
    prediction_dict = {}
    
    for i in range(len(test_filenames)):
        start_time = time.time()
        test_filename = test_filenames[i]
        # test img
        img = os.path.join(DATA_DIR, 'JPEGImages', test_filename.replace('\n', '.png'))
        if os.path.basename(img) not in os.listdir(os.path.join(DATA_DIR, 'JPEGImages')):
            img = os.path.join(DATA_DIR, 'JPEGImages', test_filename.replace('\n', '.jpg'))
                
        result = inference_detector(model, img)
        
        prediction_dict[os.path.basename(img)] = result[0][:2]  # result[2]는 cell정보? 인것 같으므로 TD에선 제외
        
        show_result(img, result, classes_names, score_thr=score_thr, out_file=os.path.join(OUT_DIR, test_filename.replace('\n', '.jpg')))
        run_time = time.time() - start_time
        print('%s\t\t\t\t%.2fs' % (test_filename.replace('\n', ''), run_time))
    
    if args.save_pickle:
        with open(os.path.join(PICKLE_DIR, 'pred.pickle'), 'wb') as f:
            pickle.dump(prediction_dict, f, pickle.HIGHEST_PROTOCOL)
