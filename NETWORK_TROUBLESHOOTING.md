# Network Troubleshooting: X-Vector Model Download

## Problem

You're seeing an error like:

```
socket.gaierror: [Errno -3] Temporary failure in name resolution
Failed to resolve 'cas-bridge.xethub.hf.co'
```

This means the training machine cannot access HuggingFace Hub to download the X-Vector model.

---

## Quick Solutions

### ✅ Solution 1: Download on Machine with Internet (Recommended)

If your training machine doesn't have internet, download the model on a different machine and transfer it.

**Step 1: On a machine WITH internet, run:**

```bash
python download_xvector_model.py
```

This will download the model to `pretrained_models/spkrec-xvect-voxceleb/`

**Step 2: Copy the directory to your training machine**

Using SCP, rsync, USB drive, or any file transfer method:

```bash
# Example with SCP
scp -r pretrained_models/spkrec-xvect-voxceleb/ user@training-machine:/path/to/CleanUNet2-Vanilla_xvectors/pretrained_models/
```

**Step 3: Update your config**

Edit `configs/train_xvector_vanilla_stage1.yaml`:

```yaml
model:
  xvector_dim: 512
  conditioning_type: "addition"
  xvector_local_path: "pretrained_models/spkrec-xvect-voxceleb"  # Add this line
```

**Step 4: Start training**

```bash
python train_xvector.py --config configs/train_xvector_vanilla_stage1.yaml --stage stage1
```

---

### ✅ Solution 2: Fix Network Connection

If you SHOULD have internet access but it's not working:

**Check internet connection:**

```bash
ping -c 3 huggingface.co
```

**Check DNS resolution:**

```bash
nslookup cas-bridge.xethub.hf.co
nslookup huggingface.co
```

**Try alternative DNS servers:**

```bash
# Temporarily use Google DNS
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf.temp
sudo mv /etc/resolv.conf.temp /etc/resolv.conf
```

**Check proxy settings:**

```bash
echo $http_proxy
echo $https_proxy
```

If you're behind a corporate proxy, set these:

```bash
export http_proxy=http://proxy.company.com:8080
export https_proxy=http://proxy.company.com:8080
```

Then retry:

```bash
python download_xvector_model.py
```

---

### ✅ Solution 3: Use Pre-Trained Stage-1 Checkpoint (Skip X-Vector Download)

If you have access to a pre-trained Stage-1 checkpoint, you can skip Stage-1 entirely:

**Step 1: Obtain Stage-1 checkpoint**

From a colleague, shared drive, or previous training run.

**Step 2: Update Stage-2 config**

Edit `configs/train_xvector_vanilla_stage2.yaml`:

```yaml
stage1_checkpoint: "/path/to/stage1-best.ckpt"
```

**Step 3: Run Stage-2 directly**

```bash
python train_xvector.py --config configs/train_xvector_vanilla_stage2.yaml --stage stage2
```

**Note:** You still need the latents from Stage-1 validation. Copy the `stored_latents_stage1/` directory as well.

---

### ✅ Solution 4: Manual Download (Advanced)

If the automated script fails, download manually:

**Step 1: Visit HuggingFace Hub**

https://huggingface.co/speechbrain/spkrec-xvect-voxceleb

**Step 2: Download these files:**

- `hyperparams.yaml`
- `embedding_model.ckpt`
- `label_encoder.txt`
- `mean_var_norm.ckpt`

**Step 3: Create directory structure**

```bash
mkdir -p pretrained_models/spkrec-xvect-voxceleb
cd pretrained_models/spkrec-xvect-voxceleb
```

**Step 4: Place downloaded files**

```
pretrained_models/spkrec-xvect-voxceleb/
├── hyperparams.yaml
├── embedding_model.ckpt
├── label_encoder.txt
└── mean_var_norm.ckpt
```

**Step 5: Update config and train**

Add to config:

```yaml
model:
  xvector_local_path: "pretrained_models/spkrec-xvect-voxceleb"
```

---

## Verification

After downloading, verify the model works:

```python
from speechbrain.pretrained import EncoderClassifier

classifier = EncoderClassifier.from_hparams(
    source="pretrained_models/spkrec-xvect-voxceleb",
    savedir="pretrained_models/spkrec-xvect-voxceleb",
    run_opts={"device": "cpu"}
)

import torch
dummy_audio = torch.randn(1, 16000)  # 1 second of audio at 16kHz
embeddings = classifier.encode_batch(dummy_audio)
print(f"Embeddings shape: {embeddings.shape}")  # Should be (1, 1, 512)
print("✅ Model loaded successfully!")
```

---

## Common Issues

### Issue: "Model files exist but loading fails"

**Solution:** Delete the directory and re-download:

```bash
rm -rf pretrained_models/spkrec-xvect-voxceleb
python download_xvector_model.py
```

### Issue: "Partial download due to interrupted connection"

**Solution:** The download script will detect existing files. Delete incomplete files:

```bash
rm -rf pretrained_models/spkrec-xvect-voxceleb/*.incomplete
python download_xvector_model.py
```

### Issue: "Permission denied when creating directory"

**Solution:** Create the directory with proper permissions:

```bash
mkdir -p pretrained_models
chmod 755 pretrained_models
python download_xvector_model.py
```

### Issue: "Model downloads but training still fails"

**Solution:** Check config file syntax:

```bash
# Verify YAML is valid
python -c "import yaml; yaml.safe_load(open('configs/train_xvector_vanilla_stage1.yaml'))"
```

Make sure `xvector_local_path` is under the `model:` section:

```yaml
model:
  xvector_dim: 512
  xvector_local_path: "pretrained_models/spkrec-xvect-voxceleb"  # Correct indentation
```

NOT:

```yaml
model:
  xvector_dim: 512
xvector_local_path: "pretrained_models/spkrec-xvect-voxceleb"  # Wrong - not under model
```

---

## File Sizes (for verification)

After successful download, you should have:

```
pretrained_models/spkrec-xvect-voxceleb/
├── hyperparams.yaml           (~2 KB)
├── embedding_model.ckpt       (~450-500 MB)
├── label_encoder.txt          (~200 KB)
├── mean_var_norm.ckpt         (~1 KB)
└── custom_dummy.py            (~1 KB, auto-generated)
```

**Total size:** ~500 MB

---

## Alternative: Use Different X-Vector Model

If `speechbrain/spkrec-xvect-voxceleb` is unavailable, you can use alternatives:

1. `speechbrain/spkrec-ecapa-voxceleb` (ECAPA-TDNN, 192-dim embeddings)
2. Train your own X-Vector model on your dataset

**Note:** This requires modifying `xvector_dim` in the config to match the embedding dimension.

---

## Still Having Issues?

Check these resources:

1. **SpeechBrain Documentation**: https://speechbrain.github.io/
2. **HuggingFace Status**: https://status.huggingface.co/
3. **Model Page**: https://huggingface.co/speechbrain/spkrec-xvect-voxceleb

Or contact your system administrator to:
- Allow access to `huggingface.co` and `*.hf.co` domains
- Configure proxy settings correctly
- Enable external network access for the training machine

---

## Quick Reference Commands

```bash
# Download model (with internet)
python download_xvector_model.py

# Verify download
ls -lh pretrained_models/spkrec-xvect-voxceleb/

# Test model loading
python -c "from speechbrain.pretrained import EncoderClassifier; \
           c = EncoderClassifier.from_hparams(source='pretrained_models/spkrec-xvect-voxceleb', \
           savedir='pretrained_models/spkrec-xvect-voxceleb'); \
           print('Model OK!')"

# Start training with local model
python train_xvector.py --config configs/train_xvector_vanilla_stage1.yaml --stage stage1
```

---

**Remember:** Once the model is downloaded once, it's cached and won't be downloaded again!
