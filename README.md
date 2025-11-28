# 📢 CleanUNet2 — A Hybrid Speech Denoising Model on Waveform and Spectrogram

CleanUNet2 is a deep-learning architecture for **speech enhancement**, combining:

* **CleanUNet (waveform UNet)**
* **CleanSpecNet (frequency-domain transformer network)**
* **Multi-resolution STFT losses**
* **Phase-aware losses**
* **A hybrid spectrogram-to-waveform conditioning block**

This repository includes full training, validation, inference workflows using **PyTorch Lightning**.

---

## 🚀 Features

* **Hybrid enhancement**: spectrogram refinement + waveform denoising
* **Multi-Resolution STFT Loss (MR-STFT)**
* **Anti-Wrapping Phase Loss** for improved phase reconstruction
* **Fully configurable training (YAML-based)**
* **Trainer, callbacks, and logging (TensorBoard)**
* **Modular dataset pipeline (VoiceBank-DEMAND compatible)**
* **Clean code with English documentation and comments**

---

## 📁 Repository Structure

```
.
│── lightning_modules/
│    ├── cleanunet_module.py     # Lightning training module
│    ├── data_module.py          # DataModule for dataset handling
│
│── cleanunet/
│    ├── cleanunet.py            # CleanUNet (waveform model)
│    ├── cleanspecnet.py         # CleanSpecNet (spectrogram model)
│    ├── cleanunet2.py           # Full hybrid CleanUNet2 system
│
│── filelists/
│── configs/
│    ├── config.yaml             # Training configuration
│    ├── inference.yaml          # Inference configuration
│
│── train.py                     # Training script
│── inference.py                 # Inference script
│── metrics.py                   # PESQ/STOI/SI-SDR prediction
│── losses.py                    # Loss functions (MR-STFT, Phase Loss)
│── spec_dataset.py              # Dataset loader
│── README.md
```

---

## 🔧 Installation

```bash
git clone https://github.com/your-repo/CleanUNet2.git
cd CleanUNet2
pip install -r requirements.txt
```

---

## 🎚️ Training

Edit `configs/train.yaml` as needed, then run:

```bash
python train.py --config=configs/train.yaml
```

TensorBoard logs will appear under:

```
logs/cleanunet2/
```

To view them:

```bash
tensorboard --logdir logs
```

---

## 🎤 Inference (Denoising Audio)

Configure `configs/inference.yaml`, then run:

```bash
python inference.py --config=configs/inference.yaml
```

Denoised WAV files are saved to:

```
denoised_results/
```

---

## 📦 Datasets

The default setup assumes the **VoiceBank-DEMAND (16 kHz)** dataset.

Expected filelist format (`train.csv`, `test.csv`):

```
clean_file.wav|noisy_file.wav
```

Example directory:

```
VoiceBank-DEMAND-16k/
│── clean/
│── noisy/
│── filelists/
│    ├── train.csv
│    └── test.csv
```

---

## 🧠 Model Overview

### CleanUNet (Waveform Domain)

* Multi-scale encoder-decoder UNet
* Convolutional downsampling & upsampling
* Transformer bottleneck
* No skip-connection misalignment thanks to input padding logic

### CleanSpecNet (Spectrogram Domain)

* Convolutional feature extractor
* Multiple transformer layers
* GLU gating
* Causal or non-causal mask support

### Hybrid Combination

The denoising workflow:

1. **CleanSpecNet** refines the noisy spectrogram
2. **Upsampler** expands spectrogram into a waveform-length feature
3. **WaveformConditioner** fuses noisy + upsampled features
4. **CleanUNet** produces the enhanced waveform

---

## 🎧 Losses

| Loss Type                          | Purpose                              |
| ---------------------------------- | ------------------------------------ |
| **L1/L2 waveform loss**            | Basic reconstruction                 |
| **MR-STFT Loss**                   | Spectral convergence + log magnitude |
| **Anti-Wrapping Phase Loss**       | Phase consistency                    |
| **Spectrogram Log-Magnitude Loss** | Auxiliary stabilization              |

---

## 📊 Validation Metrics

During validation:

* **PESQ**
* **STOI**
* **SI-SDR**

Metrics are computed per sample and averaged.

---

## ⚙️ Configuration (YAML)

Everything is configured through YAML:

### Training Config

`configs/config.yaml` includes:

* Trainer settings
* Loss weights
* MR-STFT FFT sizes
* Batch size
* Data paths

### Inference Config

`configs/inference.yaml` includes:

* Input directory
* Output directory
* Checkpoint path
* CPU/GPU override

---

## 📦 Checkpoints

To resume training:

```yaml
resume_from_checkpoint: "logs/checkpoints/last.ckpt"
```

To load pretrained weights in inference mode:

```yaml
checkpoint_path: "logs_cleanunet/checkpoints/last.ckpt"
```

---

## 🧪 Example Output

```python
model = CleanUNet2().cuda()
noisy = torch.randn(1, 1, 64000).cuda()
noisy_spec = torch.randn(1, 513, 256).cuda()

enhanced, enhanced_spec = model(noisy, noisy_spec)
print(enhanced.shape)   # (1, 1, 64000)
```

---

## 🤝 Contributing

Pull requests are welcome!
Please open an issue before major feature changes.

---

## 📄 License

This project is licensed under the **MIT License**.

---

If you want, I can now generate:

✅ A **PDF version** using reportlab
✅ A more elaborate README including diagrams
✅ A minimal version for PyPI
✅ A citation section (BibTeX)

Would you like any of these?

