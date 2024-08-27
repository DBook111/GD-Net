# GD-Net
This is the official source code for the paper "GCN-Enhanced Spatial-Spectral Dual-Encoder Network for Simultaneous Segmentation of Retinal Layers and Fluid in OCT Images"  
Coded by [Zhilin Zhou](https://github.com/DBook111)
# Citation
If you use this code for any academic purpose, please cite:
[link to the paper](https://www.sciencedirect.com/science/article/abs/pii/S1746809424007602)
```
title={GCN-Enhanced Spatial-Spectral Dual-Encoder Network for Simultaneous Segmentation of Retinal Layers and Fluid in OCT Images}
author={Guogang Cao, Zhilin Zhou, Yan Wu, Zeyu Peng, Rugang Yan, Yunqing Zhang, Bin Jiang}
journal={Biomedical Signal Processing and Control}
year={2024}
organization={Elsevier}
```
# Environment
The requirements.txt file includes the required libraries for this project.
```
conda create --name GDNet python=3.8.17
conda activate GDNet
pip install -r requirements.txt
```
# Dataset
The dataset folder hierarchy should look like this:  
```
dataset
--DUKE
----train
------img
------mask
----test
------img
------mask
----eval
------img
------mask
```
1.DUKE DME: https://people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm  
2.RETOUCH: https://retouch.grand-challenge.org/  
3.Peripapillary OCT dataset: http://www.yuyeling.com/project/mgu-net/  
# Train and test  
Run the following script to train and test our model  
```
python train&test.py --name GDNet --batch-size 1 --epoch 50 --lr 0.001
```
# Acknowledgements
The codes are built on [Li](https://github.com/Jiaxuan-Li/MGU-Net). We sincerely appreciate the authors for sharing their codes.
# Contact
If you have any questions,please contact 226142168@mail.sit.edu.cn



