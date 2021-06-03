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

#### Note: Dilation,Smudge,generateVOC2JSON.py in DataPreparation are from [CascadeTabNet](https://github.com/DevashishPrasad/CascadeTabNet)

### conda env setting
가상환경 생성  
```
    conda create -n envName python=3.7 -y   
    conda activate envName  
```
이 github 프로젝트를 다운로드(sub module도 같이 다운)  
```
    git clone --recurse-submodules $[Git-address]  
```
mmdetection dependency 설치  
```
    cd mmdetection  
    pip install -q mmcv terminaltables  
    pip install -r requirements/build.txt  
```
conda env에 nvcc 설치를 위한 cudatoolkit 설치  
```
    pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"  
    conda config --add channels conda-forge  
    conda install cudatoolkit-dev  
```
다시 mmdetection dependency 설치  
```
    python setup.py install  
    python setup.py develop  
    pip install -r requirements.txt  
    pip install pillow==6.2.1  
    pip install mmcv==0.4.3  
```
### Folder setting
> data  
> (VOC data to COCO data, by running DataPreparation/generateVOC2JSON.py)  
> > Annotations  
> > > *.xml  

> > JPEGImages  
> > > *.jpg or *.png  
>
> > ImageSets/Main  
> > > trainval.json   
> > > test.json  
>
> Model  
> (set TableBank Latex (TD) pretrained model as pretraining model)  
> > cascade_mask_rcnn_hrnetv2p_w32_20e_eunji  
> > > epoch_14.pth (TableBank Latex pretrained model, from CascadeTabNet)  

