"""Model package.

- ``cleanunet.CleanUNet``: waveform U-Net denoiser with a Transformer, Mamba, Mamba2 or LSTM bottleneck.
- ``cleanspecnet.CleanSpecNet``: spectrogram denoiser (plus the ``tf_mamba`` and ``subband`` variants).
- ``cleanunet2.CleanUNet2``: the hybrid model that wires the two branches together.
- ``blend``: training-free blend of a waveform estimate with a CleanSpecNet magnitude.
- ``streaming``: frame-by-frame inference of the Mamba CleanUNet.
- ``fusion``, ``postfilter``: learned ways of feeding spectral information into CleanUNet.
- ``metric_disc``: metric discriminator (MetricGAN-style PESQ predictor) used as a training loss.
- ``sequence``, ``mamba_torch``: Mamba / LSTM stacks and a pure-PyTorch Mamba.
- ``spectral``: the single STFT used for inputs, losses and resynthesis.
- ``checkpoint_io``: loaders for Lightning, CleanUNet2 and CleanUMamba checkpoints.
- ``util``, ``logger``: utilities kept from the original CleanUNet code (mostly legacy).
"""
