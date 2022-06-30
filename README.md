[![GitHub license](https://img.shields.io/github/license/Carrotsniper/Multiclass-SGCN)](https://github.com/Carrotsniper/Multiclass-SGCN/blob/main/LICENSE)
## Multiclass-SGCN: Sparse Graph-based Trajectory Prediction with Agent Class Embedding

This is the official implementation of our paper **Multiclass-SGCN**

![](images/struct.png)

## Abstract
Trajectory prediction of road users in real-world scenarios is challenging because their movement patterns are stochastic and complex. Previous pedestrian-oriented works have been successful in modelling the complex interactions among pedestrians, but fail in predicting trajectories when other types of road users are involved (*e.g., cars, cyclists, etc.*), because they ignore user types. Although a few recent works construct densely connected graphs with user label information, they suffer from superfluous spatial interactions and temporal dependencies. To address these issues, we propose **Multiclass-SGCN**, a sparse graph convolution network based approach for multi-class trajectory prediction that takes into consideration velocity and agent label information and uses a novel interaction mask to adaptively decide the spatial and temporal connections of agents based on their interaction scores. The proposed approach significantly outperformed state-of-the-art approaches on the Stanford Drone Dataset ([SDD](https://cvgl.stanford.edu/projects/uav_data/))
, providing more realistic and plausible trajectory predictions.

## Models

All the experiments are done with Stanford Drone Dataset (SDD), we compare our proposed work with 8 models.
Method | mADE | mFDE
:---:|:---:|:---:
Linear            | 37.11 | 63.51 
SF                | 36.48 | 58.14 
Social-LSTM       | 31.19 | 56.97 
Social-GAN        | 27.25 | 41.44
CAR-Net           | 25.72 | 51.80
DESIRE            | 19.25 | 34.05
Social-STGCNN     | 26.46 | 42.71
Semantics-STGCNN  | 18.12 | 29.70
:---:|:---:|:---:
**Multiclass-SGCN (ours)** | **14.36** | **25.99**

 

### To train the model run 
> python train.py

### To test the model run  
> python test_pred.py
