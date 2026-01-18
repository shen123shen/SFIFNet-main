# SFIF-Net-main

Accurate lesion segmentation in medical image analysis is critical for diagnosis and treatment planning. However, traditional U-shaped architectures often struggle with large lesion-scale variations and blurred boundaries. To address these challenges, a model called SFIF-Net has been proposed in this study for robust feature extraction. In particular, SFIF-Net strengthens feature interaction through four key components: a Hierarchical Feature Aggregation (HFA) module to capture global context; a Layer-wise Feature Aggregation (LWFA) module in skip connections for dynamic multi-scale fusion; an Interactive Feature Fusion (IFF) module with a Spectral Feature Migration (SFM) component in the decoder to restore fine boundaries via spatial-frequency fusion; and a Multi-scale Feature Enhancement (MFE) module applied across stages to improve multi-level feature learning. Experiments on four public datasets—ISIC 2018, BUSI, Glas, and CVC-ClinicDB—using Dice, mIoU, HD95, and Specificity show that SFIF-Net outperforms state-of-the-art methods. It achieves the highest average Dice score on each dataset, outperforming the second-best method by 1.02\%, 1.22\%, 0.06\%, and 0.06\%, respectively. The source code is available at https://github.com/shen123shen/SFIF-Net-main.

# Experiment
In the experimental section, four publicly available and widely utilized datasets are employed for testing purposes. These datasets are:\
ISIC-2018 (dermoscopy, 2,594 images fortraining, 100 images for validation, and 1,000 images for testing)\
Glas (colorectal adenocarcinoma, 85 images for training, 80 images for validation)\
BUSI (breast ultrasound, 399 images for training.113 images for validation, and 118 images for testing)\
CVC-ClinicDB (colorectal cancer, 367 images for training, 123images for validation, and 122 images for testing)\
The dataset path may look like:
```
/The Dataset Path/
├── ISIC-2018/
    ├── Train_Folder/
    │   ├── img
    │   ├── labelcol
    │
    ├── Val_Folder/
    │   ├── img
    │   ├── labelcol
    │
    ├── Test_Folder/
        ├── img
        ├── labelcol
```
 # Usage
 Installation
 ```
 git clone git@github.com:shen123shen/SFIFNet-main
.git
 conda create -n shen python=3.8
 conda activate shen
 conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```
Training
 ```
python train_cuda.py
 ```
Evaluation
 ```
python Test.py
 ```
# Citation

 ```
@ARTICLE{40030292,
  author  = {Haozhou Shen, Shiren Li, Guangguang Yang},
  journal = {Expert Systems with Applications}
  title   = {SFIF-Net: Spatial–Frequency Interactive Feature Learning for Medical Image Segmentation},
  year    = {2025}
}
 ```
