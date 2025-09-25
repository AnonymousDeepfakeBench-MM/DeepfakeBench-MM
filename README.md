# DeepfakeBench-MM and Mega-MMDF

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-brightgreen.svg)](https://creativecommons.org/licenses/by-nc/4.0/) ![Release .10](https://img.shields.io/badge/Release-0.4-brightgreen) ![PyTorch](https://img.shields.io/badge/PyTorch-1.12-brightgreen) ![Python](https://img.shields.io/badge/Python-3.7.2-brightgreen)

## 🧠 Overview

Welcome to **DeepfakeBench-MM**, your one-stop solution for multimodal deepfake detection! This work is currently under double-blind review period of ICLR2026. Key contributions include:

[//]: # (- 💽 **[Mega-MMDF Dataset]&#40;https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/J4DVAA&#41;**  )
- 💽 **Mega-MMDF Dataset**  
  One of the largest **multimodal deepfake detection dataset**. To mitigate potential social impact caused by Deepfake data, we require request before accessing this dataset.

- 🧪 **DeepfakeBench-MM Benchmark**  
  A modular and extensible **benchmark codebase** for training and evaluating multimodal deepfake detection methods. Supported Datasets are models are listed below.

|              | Paper                                                                                                                                                            |
|--------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
 | MDS          | [Not made for each other- Audio-Visual Dissonance-based Deepfake Detection and Localization](https://arxiv.org/abs/2005.14405) MM2024                            |
| AVTS         | [Hearing and Seeing Abnormality: Self-Supervised Audio-Visual Mutual Learning for Deepfake Detection](https://ieeexplore.ieee.org/document/10095247) ICASSP 2023 |
| AVAD         | [Self-Supervised Video Forensics by Audio-Visual Anomaly Detection](https://cfeng16.github.io/audio-visual-forensics/) CVPR2023                                  |
| MRDF         | [Cross-Modality and Within-Modality Regularization for Audio-Visual Deepfake Detection](https://ieeexplore.ieee.org/document/10447248) ICASSP2024                |
| AVFF         | [AVFF: Audio-Visual Feature Fusion for Video Deepfake Detection](https://arxiv.org/abs/2406.02951) CVPR2024                                                                                                                                                    |
| FRADE        | [Forgery-aware Audio-distilled Multimodal Learning for Deepfake Detection](https://dl.acm.org/doi/10.1145/3664647.3681672) MM2024                                |
| AVH          | [Circumventing shortcuts in audio-visual deepfake detection datasets with unsupervised learning](https://arxiv.org/abs/2412.00175) CVPR2025                      |
| Baseline     | -                                                                                                                                                                |
| Ensemble     | -                                                                                                                                                                |
| Qwen2.5-Omni | [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215) arxiv 2025                                                                                     |
| Video-Llama2 | [VideoLLaMA 2: Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLM](https://arxiv.org/abs/2406.07476) arxiv 2024                            |

| Dataset           | Real Videos | Fake Videos | Total Videos | Forgery Methods | Original Repository                                                   |
|-------------------|-------------|-------------|--------------|-----------------|-----------------------------------------------------------------------|
| FakeAVCeleb_v1.2  | 500         | 21,044      | 21,544       | 4               | [Hyper-link](https://github.com/DASH-Lab/FakeAVCeleb)                 |
| LAV-DF            | 36,431      | 99,873      | 136,304      | 2               | [Hyper-link](https://github.com/ControlNet/LAV-DF)                    |
| IDForge_v1        | 80,000      | 170000      | 25,0000      | 6               | [Hyper-link](https://github.com/xyyandxyy/IDForge?tab=readme-ov-file) |
| AVDeepfake1M      | 286,721     | 860,039     | 1,146,760    | 3               | [Hyper-link](https://github.com/ControlNet/AV-Deepfake1M)             |
| Mega-MMDF         | 100,000     | 1,100,000   | 1,200,000    | 28              | Coming Soon                                                           |



---


## ⏳ Quick Start
### 1️⃣  Installation
<a href="#top">[Back to top]</a>
```
conda create -n DeepfakeBench python=3.7.2
conda activate DeepfakeBench
bash install.sh
```
### 2️⃣  Data Preprocessing
<a href="#top">[Back to top]</a>

All datasets must be preprocessed to a unified format, including:

- Audio-video stream separation
- Audio resampling and video frame rate adjustment
- Face alignment and cropping

After preprocessing, a JSON file is generated to organize audio/video clips with their corresponding labels and metadata.

Preprocessed version and corresponding JSON files are in preparation and will be released soon. 🛡️ **Copyright of the above datasets belongs to their original providers.**


Example command:
```
python preprocess/fakeavceleb_preprocesor.py
```
Thanks to our modular design, additional datasets can be integrated with ease. More details can be found in `preprocess/README.md`.

### 3️⃣  Training
<a href="#top">[Back to top]</a>

Our benchmark provides flexible training scripts with support for various configurations, including model architecture, optimizer, batch size, number of epochs, etc.

To train a custom model:

1. **Define your model**:  
   Inherit from `detectors/abstract_detectors.py` and implement required methods. We decouple `forward()` into `features()` and `classifier()` to encourage backbone reuse.

2. **Define a customized loss**:  
   Inherit from `losses/abstract_loss.py`.

3. **Register your components**:  
   Add them into `utils/registry.py`.

4. **Configure your experiment**:  
   - `configs/path.yaml`: paths for logs, datasets, JSON files  
   - `configs/detectors/${YourModel}.yaml`: model, training, validation settings

5. **Run training**:
```
# With out DDP:
python train.py --detector_path configs/detectors/${YourModel}.yaml [other_args]

# With DDP:
bash train.sh ${num_GPUs} --detector_path configs/detectors/${YourModel}.yaml [other_args]
```
Optional arguments (overriding config settings):

| Argument           | Description                               |
| ------------------ | ----------------------------------------- |
| `--train_datasets` | `[list]` Training datasets to concatenate |
| `--val_datsets`    | `[list]` Validation datasets              |
| `--save_ckpt`      | `[bool]` Save checkpoint after each epoch |
| `--log-dir`        | `[str]` Custom log directory path         |



### 4️⃣  Evaluation
<a href="#top">[Back to top]</a>

To evaluate a trained model on both in-domain and cross-domain datasets:
```
python test.py --detector_path configs/detectors/${YourModel}.yaml --weights_path ${YourWeight}.yaml
```
This will report performance metrics including accuracy, AUC, and more, depending on the configuration.
