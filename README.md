# GANCDM

This repo contains information and code of the weakly-supervised cloud detection method for remote sensing images presented in the work

><b>A hybrid generative adversarial network for weakly-supervised cloud detection in multispectral images</b>
Jun Li, Zhaocong Wu,*, Qinghong Sheng,*, Bo Wang, Zhongwen Hu, Shaobo Zheng, Gustau Camps-Valls, Matthieu Molinier<br>
Submitted, 2022

## Abstract

Cloud detection is a crucial step in the optical satellite image processing pipeline for Earth observation. Clouds in optical remote sensing images seriously affect the visibility of background and greatly reduce the usability of images for land applications. Traditional methods based on thresholding, multi-temporal or multi-spectral information are often specific to a particular satellite sensor. Convolutional Neural Networks for cloud detection often require labelled cloud masks for training that are very time-consuming and expensive to obtain. In order to overcome these challenges, this paper presents a hybrid cloud detection method based on the synergistic combination of generative adversarial networks (GAN) and a physics-based cloud distortion model (CDM). The proposed weakly-supervised GAN-CDM method only requires patch-level labels for training, and can produce cloud masks at pixel-level in both training and testing stages. GAN-CDM is trained on a new globally distributed Landsat 8 dataset (WHUL8-CDb, available online https://github.com/Neooolee/WHU-CDb) including image blocks and corresponding block-level labels. Experimental results show that the proposed GAN-CDM method trained on Landsat 8 image blocks achieves much higher cloud detection accuracy than baseline deep learning-based methods, not only in Landsat 8 images (L8 Biome dataset, 90.20% versus 70.112.09%) but also in Sentinel-2 images (“S2 Cloud Mask Catalogue” dataset, 92.54% versus 82.5177.00%). This suggests that the proposed method provides accurate cloud detection in Landsat images, has good transferability to Sentinel-2 images, and can quickly be adapted for different optical satellite sensors.

## Keywords: Cloud detection, generative adversarial networks (GAN), cloud distortion model, deep learning, remote sensing

# Code

Code snippets and demos can be found here: 

# Data

The training dataset is on: https://github.com/Neooolee/WHUL8-CDb

# How to cite our work

If you find this useful, consider citing our work:

><b>A hybrid generative adversarial network for weakly-supervised cloud detection in multispectral images</b>
Jun Lia, Zhaocong Wu,*, Qinghong Sheng,*, Bo Wang, Zhongwen Hu, Shaobo Zheng, Gustau Camps-Valls, Matthieu Molinier
Submitted, 2022

```
@article {Li22gancdm,
  author = {Jun Li, Zhaocong Wu,*, Qinghong Sheng,*, Bo Wang, Zhongwen Hu, Shaobo Zheng, Gustau Camps-Valls, Matthieu Molinier},
  title = {A hybrid generative adversarial network for weakly-supervised cloud detection in multispectral images},
  volume = {},
  number = {},
  elocation-id = {},
  year = {2022},
  doi = {},
  publisher = {},
  URL = {},
  eprint = {},
  journal = {Under review}
}
```


