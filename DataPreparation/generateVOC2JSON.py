## Script for Converting Pascal VOC annotations to Coco Json format
## This script shows and example conversion of our table dataset pascal voc
## annotations conversion to coco annotations

## Usage :
# You need to first create a txt file containing names of all pascal voc files
# You can use following linux command
# ls -1 | sed -e 's/\.xml$//' | sort -n > "/path/to/folder/coco.txt"
# And then read the comments in this script to understand its working

import os
import xml.etree.ElementTree as ET
import xmltodict
import json
from xml.dom import minidom
from collections import OrderedDict
import shutil #ADD
import re
import xml.etree.ElementTree as ET

def generateVOC2Json(rootDir,xmlFiles,cocoName,dstDir,startId):
  attrDict = dict()
  # Add categories according to you Pascal VOC annotations
  attrDict["categories"]=[{"supercategory":"none","id":1,"name":"bordered"}, #Table"},
                          {"supercategory":"none","id":2,"name":"cell"},
                          {"supercategory":"none","id":3,"name":"borderless"}
        # {"supercategory":"none","id":4,"name":"item_name"},
        # {"supercategory":"none","id":5,"name":"item_desc"},
        # {"supercategory":"none","id":6,"name":"price"},
        # {"supercategory":"none","id":7,"name":"total_price_text"},
        # {"supercategory":"none","id":8,"name":"total_price"},
        # {"supercategory":"none","id":9,"name":"footer"}
            ]
  images = list()
  annotations = list()
  id1 = 1

  int2filename = dict()

  # Some random variables
  cnt_bor = 0
  cnt_cell = 0
  cnt_bless = 0

  # Main execution loop
  for root, dirs, files in os.walk(rootDir):
    image_id = startId
    for file in xmlFiles:
      image_id = image_id + 1
      if file in files:
        annotation_path = os.path.abspath(os.path.join(root, file))
        file_annotations= []
        file_annotations_bor = []
        file_cnt_bor = 0
        file_cnt_cell = 0
        file_cnt_bless = 0
        file_check = 0
        image = dict()
        doc = xmltodict.parse(open(annotation_path).read())
        int2filename[image_id] = str(doc['annotation']['filename']) #ADD
        # image['file_name'] = str(doc['annotation']['filename'])
        image['file_name'] = '%d.png'%(image_id)
        image['height'] = int(doc['annotation']['size']['height'])
        image['width'] = int(doc['annotation']['size']['width'])
        image['id'] = image_id
        print("File Name: {} and image_id {}".format(file, image_id))
        images.append(image)
        if 'object' in doc['annotation']:
          file_annotation = []
          for key,vals in doc['annotation'].items():
            if(key=='object'):
              for value in attrDict["categories"]:
                if(not isinstance(vals, list)):
                  vals = [vals]
                for val in vals:
                  if str(val['name']) == value["name"]:
                    annotation = dict()
                    annotation["iscrowd"] = 0
                    annotation["image_id"] = image_id
                    x1 = int(val["bndbox"]["xmin"])  - 1
                    y1 = int(val["bndbox"]["ymin"]) - 1
                    x2 = int(val["bndbox"]["xmax"]) - x1
                    y2 = int(val["bndbox"]["ymax"]) - y1
                    annotation["bbox"] = [x1, y1, x2, y2]
                    annotation["area"] = float(x2 * y2)
                    annotation["category_id"] = value["id"]

                    # Tracking the count
                    if(value["id"] == 1):
                      file_cnt_bor += 1
                      file_check = 1
                      print("BORDERED!!!" , file)
                    if(value["id"] == 2):
                      file_cnt_cell += 1
                    if(value["id"] == 3):
                      file_cnt_bless += 1

                    annotation["ignore"] = 0
                    annotation["id"] = id1
                    annotation["segmentation"] = [[x1,y1,x1,(y1 + y2), (x1 + x2), (y1 + y2), (x1 + x2), y1]]
                    id1 +=1
                    file_annotation.append(annotation)
                    if value["name"] != "cell":
                      file_annotations.append(file_annotation)
                      file_annotation = []
                      # if (value["id"] == 1):
                      #   file_annotations.append(file_annotation)
                      #   #file_annotations_bor.append(file_annotation)
                      #   file_annotation = []
                      # elif (value["id"] == 3):
                      #   file_annotations.append(file_annotation)
                      #   file_annotation = []
                  if value["name"] == "cell":
                    if val['name'] == 'bordered':
                      # file_annotations.append([annotation])
                      # file_annotations_bor.append([annotation])
                      file_annotation = []
                      file_cnt_cell = 0
                    elif val['name'] == 'borderless':
                      file_annotations.append(file_annotation)
                      file_annotation = []

          # if not file_check:
          for file_annotation in file_annotations:
            annotations += file_annotation
          cnt_bor += file_cnt_bor
          cnt_cell += file_cnt_cell
          cnt_bless += file_cnt_bless
          # else:
          #   cnt_bor += file_cnt_bor
          #   for file_annotation_bor in file_annotations_bor:
          #     annotations += file_annotation_bor

          with open(annotation_path, 'rt') as f:
            target_tree = ET.parse(f)
          target_root = target_tree.getroot()
          target_tag = target_root.find("filename")
          target_tag.text = "%d.png"%(image_id)
          target_tree.write(os.path.join(dstDir, "Annotations", "%d.xml"%(image_id)))
          # shutil.copy2(annotation_path, os.path.join(dstDir, "Annotations", "%d.xml"%(image_id)))

          image_path = annotation_path.replace("Annotations", "JPEGImages").replace(".xml",".png")
          shutil.copy2(image_path, os.path.join(dstDir, "JPEGImages", "%d.png"%(image_id)))

        else:
          print("File: {} doesn't have any object".format(file))
      else:
        print("File: {} not found".format(file))

  attrDict["images"] = images
  attrDict["annotations"] = annotations
  attrDict["type"] = "instances"

  # Printing out some statistics
  print(len(images))
  print("Bordered : ",cnt_bor," Cell : ",cnt_cell," Bless : ",cnt_bless)
  print(len(annotations))

  # Save the final JSON file
  # jsonString = json.dumps(attrDict)
  jsonString = json.dumps(attrDict, indent = 4, sort_keys=True)
  with open(cocoName, "w") as f: #"/content/drive/My Drive/ICDAR 13 dataset/coco.json", "w") as f:
    f.write(jsonString)

  with open(os.path.join(dstDir, os.path.basename(cocoName).replace('.json', '_dict.json')), 'w') as f:
    json.dump(int2filename, f, indent=4)

  return image_id

# Path to the txt file (see at the top of this script)
trainFile = "../data/VOC2007/ImageSets/Main/trainval.txt"#"/content/drive/My Drive/ICDAR 13 dataset/coco.txt"
trainXMLFiles = list()
with open(trainFile, "r") as f:
	for line in f:
		fileName = line.strip()
		print(fileName)
		trainXMLFiles.append(fileName + ".xml")

# Path to the pascal voc xml files
rootDir = "../data/VOC2007/Annotations"#"/content/drive/My Drive/ICDAR 13 dataset/2Be Fine Tuned"
destDir = "../data/VOC2008"
if not os.path.exists(destDir):
  os.makedirs(os.path.join(destDir, "Annotations"))
  os.makedirs(os.path.join(destDir, "JPEGImages"))
  os.makedirs(os.path.join(destDir, "ImageSets", "Main"))
# Start execution
startId = 0
endId = generateVOC2Json(rootDir, trainXMLFiles, "../data/VOC2008/train.json", destDir, startId)
with open(os.path.join(destDir, "ImageSets/Main/trainval.txt"), "w") as f:
  for i in range(startId+1, endId+1):
    f.write("%d\n"%(i))

# Path to the txt file (see at the top of this script)
trainFile = "../data/VOC2007/ImageSets/Main/test.txt"#"/content/drive/My Drive/ICDAR 13 dataset/coco.txt"
trainXMLFiles = list()
with open(trainFile, "r") as f:
	for line in f:
		fileName = line.strip()
		print(fileName)
		trainXMLFiles.append(fileName + ".xml")

# Path to the pascal voc xml files
rootDir = "../data/VOC2007/Annotations"#"/content/drive/My Drive/ICDAR 13 dataset/2Be Fine Tuned"
destDir = "../data/VOC2008"
# Start execution
startId = endId
endId = generateVOC2Json(rootDir, trainXMLFiles, "../data/VOC2008/test.json", destDir, startId)
with open(os.path.join(destDir, "ImageSets/Main/test.txt"), "w") as f:
  for i in range(startId+1, endId+1):
    f.write("%d\n"%(i))

