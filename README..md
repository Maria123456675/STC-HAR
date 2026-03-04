# STC-HAR: Hybrid 3D-CNN-LSTM-GCN with Spatio-Temporal-Channel Attention

**Official implementation of:**

> *"Enhancing Human Action Recognition with a Hybrid 3D-CNN-LSTM-GCN Architecture and Spatio-Temporal-Channel Attention"*
> Hafiza Maria Rafique, Jin Qi, Anees Khalil
> School of Information and Communication Engineering, UESTC, Chengdu, China

---

## Architecture

```
Input (B, C, T, V)
  -> 3D-CNN           local spatio-temporal feature extraction
  -> STC Attention    joint / frame / channel recalibration  [NOVEL]
  -> Adaptive GCN     dynamic spatial dependency modelling
  -> Bi-LSTM          long-range temporal modelling
  -> Weighted Fusion  of three streams
  -> Classifier       action class probabilities
```

---

## Results

| Dataset           | Metric   | Our Model |
|-------------------|----------|-----------|
| NTU RGB+D         | CS Top-1 | 92.10%    |
| NTU RGB+D         | CV Top-1 | 98.10%    |
| Kinetics-Skeleton | Top-1    | 40.20%    |
| Kinetics-Skeleton | Top-5    | 63.50%    |
| Penn Action       | Top-1    | 92.34%    |
| Human3.6M         | Top-1    | 89.80%    |

---

## Repository Structure

```
STC-HAR/
├── architecture.py      Main model (3D-CNN + GCN + BiLSTM + STC)
├── preprocess.py        Dataset loading and augmentation
├── train.py             Training script
├── evaluate.py          Evaluation and attention visualisation
├── requirements.txt     Python dependencies
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Maria123456675/STC-HAR.git
cd STC-HAR
pip install -r requirements.txt
```

Requirements: Python 3.8+, PyTorch 2.0+, CUDA 12.x

---

## Dataset Preparation

| Dataset | Link |
|---------|------|
| NTU RGB+D | https://rose1.ntu.edu.sg/dataset/actionRecognition/ |
| Kinetics-Skeleton | https://github.com/open-mmlab/mmskeleton |
| Penn Action | https://dreamdragon.github.io/PennAction/ |
| Human3.6M | http://vision.imar.ro/human3.6m/ |

Place skeleton JSON files in `data/` folder.

---

## Training

```bash
# NTU RGB+D Cross-Subject
python train.py \
  --train_data   data/ntu_train_skeletons.json \
  --train_labels data/ntu_train_labels.json \
  --test_data    data/ntu_test_skeletons.json \
  --test_labels  data/ntu_test_labels.json \
  --num_classes  60 --num_joints 25 --num_frames 150 \
  --in_channels  3  --batch_size  32 --epochs 50 \
  --lr 0.01 --seed 42

# Kinetics-Skeleton
python train.py --num_classes 400 --num_joints 18 --in_channels 2 --epochs 65
```

---

## Evaluation

```bash
python evaluate.py \
  --checkpoint  checkpoints/best_model.pth \
  --test_data   data/ntu_test_skeletons.json \
  --test_labels data/ntu_test_labels.json \
  --dataset_key NTU_RGBD_CS

# Noise robustness test
python evaluate.py --noise_test
```

---

## Reproducibility

```python
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
```

Results: mean ± std over 5 seeds (42, 123, 256, 512, 1024). Variance < ±0.35%.

---

## Hardware

- GPU: NVIDIA RTX 4090 (24 GB) | CPU: Intel Xeon | RAM: 128 GB
- CUDA 12.x | PyTorch 2.x

---

## Citation

```bibtex
@article{rafique2026stchar,
  title   = {Enhancing Human Action Recognition with a Hybrid 3D-CNN-LSTM-GCN
             Architecture and Spatio-Temporal-Channel Attention},
  author  = {Rafique, Hafiza Maria and Qi, Jin and Khalil, Anees},
  journal = {Computer Vision and Image Understanding},
  year    = {2026}
}
```

---

## Contact

Hafiza Maria Rafique — mariarafique@std.uestc.edu.cn
UESTC, Chengdu, China
