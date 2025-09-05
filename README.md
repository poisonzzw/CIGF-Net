# CIGF-Net
After the publication of the paper, we will upload the source code and relevant data of CIGF-Net.

# Abstract
RGB-T semantic segmentation aims to enhance the robustness of segmentation methods in complex environments by utilizing thermal information. To facilitate the effective interaction and fusion of multimodal information, we propose a novel Cross-modality Interaction and Global-feature Fusion Network, namely CIGF-Net. In each feature extraction stage, we propose a Cross-modality Interaction Module (CIM) to enable effective interaction between the RGB and thermal modalities. CIM utilizes channel and spatial attention mechanisms to process the feature information from both modalities. By encouraging cross-modal information exchange, the CIM facilitates the integration of complementary information and improves the overall segmentation performance. Subsequently, the Global-feature Fusion Module (GFM) is proposed to focus on fusing the information provided by the CIM. GFM assigns different weights to the multimodal features to achieve cross-modality fusion. Experimental results show that CIGF-Net achieves state-of-the-art performance on RGB-T image semantic segmentation datasets, with a remarkable 60.8 mIoU on the MFNet dataset and 86.93 mIoU on the PST900 dataset.

# Requirements
CUDA 11.2，torchvision 0.13.1，Tensorboard 2.9.0，Python 3.9，PyTorch 1.12.1。

# Dataset
The MFNet datesets for RGB-T semantic segmentation could be found in [here](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/).  

The PST900 datesets for RGB-T semantic segmentation could be found in [here](https://drive.google.com/file/d/1hZeM-MvdUC_Btyok7mdF00RV-InbAadm/view?pli=1).  

# Pretrain weight
Download the pretrained ConvNext V2 - tiny here [pretrained ConvNext V2](https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_384_ema.pt).

# test 
运行test.py文件，导入模型权重即可。
Model weights download：[百度网盘](https://pan.baidu.com/s/1wqXyt5-c43Qfz-JsnR4pHA).
提取码s2t4。



# Citation
@article{zhang2023ms,
  title={MS-IRTNet: Multistage information interaction network for RGB-T semantic segmentation},
  author={Zhang, Zhiwei and Liu, Yisha and Xue, Weimin},
  journal={Information Sciences},
  volume={9},
  pages={2440 - 2451},
  year={2023},
  publisher={Elsevier}
}
@article{zhang2024cigf,
  title={CIGF-net: Cross-modality interaction and global-feature fusion for RGB-t semantic segmentation},
  author={Zhang, Zhiwei and Liu, Yisha and Xue, Weimin and Zhuang, Yan},
  journal={IEEE Transactions on Emerging Topics in Computational Intelligence},
  volume={9},
  pages={2440--2451},
  year={2024},
  publisher={IEEE}
}

# Contact
Please drop me an email for further problems or discussion: 1519968317@qq.com
