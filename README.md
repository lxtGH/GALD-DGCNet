# GALD_net
We propose Global Aggregation then Local Distribution (GALD) scheme to distribute global information to each position adaptively according to the local information surrounding the position. GALD net achieves **top performance** on Cityscapes dataset. Both source code and models will be available soon.


##  Comparison with the art models on Cityscapes dataset 
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
CCNet| - | ResNet-101 | 81.4
GALD-net | - | ResNet50 |**80.8**
GALD-net | -| ResNet101 |**81.8**
GALD-net(use coarse) |- | ResNet101 |**82.9**
GALD-net(use map)|- |ResNet101| **83.3**

Detailed Results are shown [here](https://www.cityscapes-dataset.com/anonymous-results/?id=5ee0f5098e160aa56db6e9ed01c5fbc73d4ac736b6b61751b50ad31067b0d5bd)
