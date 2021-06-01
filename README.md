# [How to train an object detection model with mmdetection](https://www.dlology.com/blog/how-to-train-an-object-detection-model-with-mmdetection/) | DLology blog

## Quick start
Train an object detection with Google Colab and free GPU.

Train with custom Pascal VOC dataset.
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tony607/mmdetection_object_detection_demo/blob/master/mmdetection_train_custom_data.ipynb)

Train with custom COCO dataset.
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tony607/mmdetection_object_detection_demo/blob/master/mmdetection_train_custom_coco_data.ipynb)

*The `data/VOC2007` folder provides a reference structure of custom dataset ready for training. Fork my repository and replace them with your custom annotated dataset as necessary.*


Further instruction on how to create your own datasets, read the tutorials
- [How to train an object detection model with mmdetection](https://www.dlology.com/blog/how-to-train-an-object-detection-model-with-mmdetection/) - Custom Pascal VOC dataset.
- [How to create custom COCO data set for object detection](https://www.dlology.com/blog/how-to-create-custom-coco-data-set-for-object-detection/) - Custom COCO dataset.

### Note: most of the codes in DataPreparation are from [CascadeTabNet](https://github.com/DevashishPrasad/CascadeTabNet)

## conda env setting
conda create -n envName python=3.7 -y
conda activate envName

cd mmdetection
pip install -r requirements/build.txt
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"

conda config --add channels conda-forge
conda install cudatoolkit-dev

python setup.py install
python setup.py develop

pip install -r requirements.txt

pip install pillow==6.2.1
pip install mmcv==0.4.3

