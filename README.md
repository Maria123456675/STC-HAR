STC-HAR: Skeleton-Based Human Action Recognition with Spatio-Temporal-Channel Attention

This repository contains the official implementation of the paper:  
"Enhancing Human Action Recognition with a Hybrid 3D-CNN-LSTM-GCN Architecture and Spatio-Temporal-Channel Attention",  
an international peer-reviewed journal (Springer).

This repository provides reproducible research code accompanying the manuscript.

## Key Contributions
- A unified hybrid 3D-CNN–LSTM–GCN framework for skeleton-based HAR
- Joint modeling of spatial, temporal, and channel-wise attention
- Skeleton-only approach without RGB or depth inputs

## Requirements
- Python 3.8+
- See [`requirements.txt`](requirements.txt)

## Dataset Preparation
Download datasets:
- [NTU RGB+D](https://github.com/shahroudy/NTURGB-D)  
- [Kinetics-Skeleton](https://github.com/yysijie/st-gcn/blob/master/OLD_README.md#kinetics-skeleton)  
- [Penn Action](http://dreamdragon.github.io/PennAction/)  
- [Human3.6M](http://vision.imar.ro/human3.6m/description.php)  


Place skeleton JSON files in `data/` folder:

