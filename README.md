# City-seg 

We will also release the all the state-of-the-art models and code trained on Cityscape dataset including Deeplabv3, Deeplabv3+, PSPnet, DAnet. All the codes are still in refactoring process.


# GALD-Net (BMVC 2019)
We propose Global Aggregation then Local Distribution (GALD) scheme to distribute global information to each position adaptively according to the local information around the position. GALD net achieves **top performance** on Cityscapes dataset. Both source code and models will be available soon. The work was done at [DeepMotion AI Research](https://deepmotion.ai/) 

# GFF-Net ([arxiv](https://arxiv.org/abs/1904.01803))
We proposed Gated Fully Fusion (GFF) to fuse features from multiple levels through gates in a fully connected way. Specifically, features at each level are enhanced by higher-level features with stronger semantics and lower-level features with more details, and gates are used to control the pass of useful information which significantly reducing noise propagation during fusion. (Joint work: Key Laboratory of Machine Perception, School of EECS @Peking University and DeepMotion AI Research )


##  Comparisons with state-of-the-art models on Cityscapes dataset 
Method | Conference | Backbone | mIoU(\%) 
---- | --- | --- | --- 
RefineNet |  CVPR2017  | ResNet-101  |  73.6 
SAC  |  ICCV2017  | ResNet-101  |  78.1 
PSPNet |  CVPR2017  | ResNet-101  |  78.4
DUC-HDC | WACV2018 | ResNet-101 | 77.6 
AAF |   ECCV2018  | ResNet-101  |  77.1 
BiSeNet |   ECCV2018  | ResNet-101  |  78.9 
PSANet |  ECCV2018  | ResNet-101  |  80.1 
DFN  |  CVPR2018  | ResNet-101  |  79.3 
DSSPN | CVPR2018  | ResNet-101  | 77.8 
DenseASPP  |  CVPR2018  | DenseNet-161  |  80.6
OCNet| - |  ResNet-101 | 81.7
CCNet| ICCV2019 | ResNet-101 | 81.4
GALD-Net | BMVC2019 | ResNet50 |**80.8**
GALD-Net | BMVC2019| ResNet101 |**81.8**
GFF-Net | - | ResNet101 | **82.3**
Dual-GCN-Net | BMVC2019 | ResNet101 | **82.0**
GALD-Net(use coarse data) |BMVC2019 | ResNet101 |**82.9**
GALD-Net(use Mapillary)|BMVC2019 |ResNet101| **83.3**

## Detailed Results are shown 
GALD-Net:
[here](https://www.cityscapes-dataset.com/anonymous-results/?id=5ee0f5098e160aa56db6e9ed01c5fbc73d4ac736b6b61751b50ad31067b0d5bd)   
GFF-Net:[here](https://www.cityscapes-dataset.com/method-details/?submissionID=3719)  
Both are (**Single Model Result**)  

