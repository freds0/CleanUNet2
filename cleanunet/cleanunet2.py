"""
CleanUNet2: Hybrid speech denoising model on waveform and spectrogram.
Based on "CleanUNet 2: A Hybrid Speech Denoising Model on Waveform and Spectrogram".

Pipeline:
  1. The noisy waveform is normalized once by its standard deviation (both branches are scale invariant).
  2. A single STFT front-end (cleanunet.spectral.Spectrogram) computes the noisy spectrogram inside
     the model, so training, inference and the spectral loss always use the same transform.
  3. CleanSpecNet denoises the (compressed) spectrogram.
  4. SpecUpsampler turns it into a waveform-rate feature; frames are aligned with the waveform and
     the result is cropped (never stretched) to the input length.
  5. The Conditioner combines it with the noisy waveform (addition, concatenation or FiLM).
  6. CleanUNet denoises the conditioned waveform. Its bottleneck can be a Transformer, Mamba, Mamba2 or LSTM.
  Optional: "bottleneck fusion" injects CleanSpecNet's hidden features directly into the CleanUNet
  bottleneck (FiLM), exploiting that the STFT hop equals (a multiple of) the U-Net total stride.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cleanspecnet import CleanSpecNet
from .cleanunet import CleanUNet
from .spectral import Spectrogram


class SpecUpsampler(nn.Module):
    """
    Upsamples the spectrogram in time (by hop = up_strides[0] * up_strides[1]) and collapses the
    frequency axis to produce a time-domain feature.

    Input: spec (B, F, T)
    Output: time_feat (B, 1, T * hop)
    """
    def __init__(self, in_channels=1, hidden_channels=32, freq_kernel=3, up_strides=(16, 16), leaky_slope=0.4):
        super().__init__()
        s1, s2 = up_strides
        self.up1 = nn.ConvTranspose2d(in_channels, hidden_channels, (freq_kernel, s1), stride=(1, s1),
                                      padding=(freq_kernel // 2, 0))
        self.act1 = nn.LeakyReLU(leaky_slope)
        self.up2 = nn.ConvTranspose2d(hidden_channels, in_channels, (freq_kernel, s2), stride=(1, s2),
                                      padding=(freq_kernel // 2, 0))
        self.act2 = nn.LeakyReLU(leaky_slope)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """(B, F, T) spectrogram -> (B, 1, T * hop) waveform-rate condition."""
        x = spec.unsqueeze(1) if spec.dim() == 3 else spec      # (B, 1, F, T)
        x = self.act2(self.up2(self.act1(self.up1(x))))
        return x.mean(dim=2)                                     # (B, 1, T * hop)


class Conditioner(nn.Module):
    """
    Combines the noisy waveform and the upsampled spectrogram feature.

    Methods:
     1. 'addition': waveform + alpha * condition (alpha = 1 fixed, or learnable and 0 at init when zero_init).
     2. 'concatenation': concatenation along channels + projection
        (with zero_init it becomes residual: waveform + proj(cat), last conv initialized to zero).
     3. 'film': (1 + gamma) * waveform + beta, with gamma/beta predicted from the condition
        (with zero_init the generator starts at zero, i.e. identity).

    With zero_init=True the conditioner starts as the identity on the waveform, so a pretrained
    waveform branch behaves exactly as before at the start of joint training.
    """
    def __init__(self, method: str = "addition", input_channels: int = 1, cond_channels: int = 1,
                 zero_init: bool = False):
        super().__init__()
        self.method = method.lower()
        self.zero_init = zero_init

        if self.method == "concatenation":
            total_channels = input_channels + cond_channels
            self.concat_proj = nn.Sequential(
                nn.Conv1d(total_channels, 16, kernel_size=7, padding=3),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(0.2),
                nn.Conv1d(16, input_channels, kernel_size=1)
            )
            if zero_init:
                nn.init.zeros_(self.concat_proj[-1].weight)
                nn.init.zeros_(self.concat_proj[-1].bias)
        elif self.method == "film":
            self.film_gen = nn.Conv1d(cond_channels, input_channels * 2, kernel_size=3, padding=1)
            if zero_init:
                nn.init.zeros_(self.film_gen.weight)
                nn.init.zeros_(self.film_gen.bias)
        elif self.method == "addition":
            if zero_init:
                self.alpha = nn.Parameter(torch.zeros(1))
        else:
            raise ValueError(f"Unknown conditioning method '{method}'. Use: addition, concatenation, film")

    def forward(self, waveform: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Combine the waveform (B, C, L) with the upsampled condition (B, C', L)."""
        if self.method == "addition":
            return waveform + (self.alpha * condition if self.zero_init else condition)
        if self.method == "concatenation":
            out = self.concat_proj(torch.cat([waveform, condition], dim=1))
            return waveform + out if self.zero_init else out
        gamma, beta = torch.chunk(self.film_gen(condition), 2, dim=1)
        return (1.0 + gamma) * waveform + beta


class BottleneckFusion(nn.Module):
    """FiLM of the CleanUNet bottleneck features with CleanSpecNet hidden features.

    The STFT hop must be a multiple of the U-Net total stride (hop = 256 = 2^8 for 8 layers,
    4 bottleneck tokens per frame for 6 layers). Initialized to the identity.
    """
    def __init__(self, spec_hidden_dim: int, d_model: int, repeat: int):
        super().__init__()
        self.repeat = repeat
        self.proj = nn.Linear(spec_hidden_dim, 2 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, spec_hidden: torch.Tensor) -> torch.Tensor:
        """x: (B, d_model, T_b); spec_hidden: (B, T_s, H)."""
        params = self.proj(spec_hidden).transpose(1, 2)            # (B, 2*d_model, T_s)
        if self.repeat > 1:
            params = params.repeat_interleave(self.repeat, dim=-1)
        T_b = x.shape[-1]
        if params.shape[-1] < T_b:                                  # replicate the last frame
            params = F.pad(params, (0, T_b - params.shape[-1]), mode="replicate")
        gamma, beta = params[..., :T_b].chunk(2, dim=1)
        return x * (1.0 + gamma.to(x.dtype)) + beta.to(x.dtype)


class CleanUNet2(nn.Module):
    """Hybrid denoiser: CleanSpecNet on the spectrogram feeds CleanUNet on the waveform.

    The input is normalized once by its standard deviation, and the model computes its own STFT
    (``cleanunet.spectral.Spectrogram``), so training, inference and the spectral loss share one transform. The
    spectral branch can condition the waveform branch in three ways, alone or together: the original upsampled
    waveform condition (``conditioning_type``), FiLM in the CleanUNet bottleneck (``bottleneck_fusion``), and the
    generic fusion hub (``fusion``: named injection points, post-filter, input estimate, Mamba exchange).
    ``use_spec_branch: false`` gives a waveform-only CleanUNet with the same checkpoint layout.
    """
    def __init__(self,
                 conditioning_type: str = "addition",
                 cleanunet_params: dict = None,
                 cleanspecnet_params: dict = None,
                 spectrogram: dict = None,
                 use_spec_branch: bool = True,
                 conditioning_zero_init: bool = False,
                 bottleneck_fusion: bool = False,
                 normalize_input: bool = True,
                 fusion: dict = None):
        super().__init__()

        cleanunet_params = dict(cleanunet_params or {})
        cleanspecnet_params = dict(cleanspecnet_params or {})
        spectrogram = dict(spectrogram or {})
        self.use_spec_branch = use_spec_branch
        self.normalize_input = normalize_input
        self.bottleneck_fusion = bottleneck_fusion

        # 1. CleanUNet (waveform denoiser). Normalization is done once here, not inside CleanUNet.
        if normalize_input:
            cleanunet_params["normalize_input"] = False
        self.clean_unet = CleanUNet(**cleanunet_params)

        # Spectrogram branch (skipped entirely for a waveform-only model, e.g. stage-A training)
        self.spectrogram = Spectrogram(**spectrogram)
        if use_spec_branch:
            # 2. Spectrogram front-end + CleanSpecNet (spectrogram denoiser)
            cleanspecnet_params.setdefault("input_channels", self.spectrogram.n_freq)
            if cleanspecnet_params["input_channels"] != self.spectrogram.n_freq:
                raise ValueError(f"cleanspecnet input_channels={cleanspecnet_params['input_channels']} does not match "
                                 f"n_fft//2+1={self.spectrogram.n_freq}")
            # Spectral-branch architecture (IMPROVEMENT_PLAN.md, track N): conv1d (original CleanSpecNet),
            # tf_mamba (dual-path time-frequency Mamba, N2) or subband (full-band + sub-band, N3).
            # input_features: mag (compressed magnitude) or mag_phase (+ cos/sin of the noisy phase, N5).
            spec_arch = cleanspecnet_params.pop("architecture", "conv1d")
            if spec_arch != "conv1d" and cleanspecnet_params.get("backbone", "") is None:
                cleanspecnet_params.pop("backbone")      # presets inheriting a conv1d config set it to null
            self.spec_input_features = cleanspecnet_params.pop("input_features", "mag")
            if self.spec_input_features not in ("mag", "mag_phase"):
                raise ValueError(f"input_features must be mag | mag_phase, got '{self.spec_input_features}'")
            cleanspecnet_params["feature_groups"] = 3 if self.spec_input_features == "mag_phase" else 1
            if spec_arch == "conv1d":
                spec_cls = CleanSpecNet
            elif spec_arch == "tf_mamba":
                from .tf_mamba import TFMambaSpecNet as spec_cls
            elif spec_arch == "subband":
                from .subband import SubbandSpecNet as spec_cls
            else:
                raise ValueError(f"Unknown CleanSpecNet architecture '{spec_arch}' (conv1d | tf_mamba | subband)")
            self.clean_spec_net = spec_cls(**cleanspecnet_params)

            # 3-4. Original waveform conditioning: upsampler (hop = product of the two strides) + conditioner.
            #      conditioning_type 'none' disables both (fusion-only models).
            hop = self.spectrogram.hop_length
            if conditioning_type != "none":
                s1 = 16 if hop % 16 == 0 else hop
                self.spec_upsampler = SpecUpsampler(up_strides=(s1, hop // s1))
                self.conditioner = Conditioner(method=conditioning_type, input_channels=1, cond_channels=1,
                                               zero_init=conditioning_zero_init)

            # 5. Optional bottleneck fusion
            if bottleneck_fusion:
                total_stride = self.clean_unet.total_stride
                if hop % total_stride != 0:
                    raise ValueError(f"bottleneck_fusion needs hop ({hop}) to be a multiple of the "
                                     f"CleanUNet total stride ({total_stride})")
                self.fusion = BottleneckFusion(self.clean_spec_net.hidden_dim, self.clean_unet.tsfm_d_model,
                                               repeat=hop // total_stride)
        elif bottleneck_fusion:
            raise ValueError("bottleneck_fusion requires use_spec_branch=True")

        # 6. Generic fusion (IMPROVEMENT_PLAN.md): frame-rate condition -> hooks at CleanUNet points,
        #    optional spectral post-filter, input estimate and Mamba-stack exchange.
        self.fusion_cfg = dict(fusion or {})
        self.oracle_prob = 0.0          # oracle-condition curriculum (set by the training loop)
        if self.fusion_cfg:
            if not use_spec_branch:
                raise ValueError("fusion requires use_spec_branch=True (it uses the model STFT / CleanSpecNet)")
            self._build_fusion(self.fusion_cfg)

        specnet = self.clean_spec_net.backbone_type if use_spec_branch else None
        self.conditioning_type = conditioning_type
        fusion_desc = {k: v for k, v in self.fusion_cfg.items() if k in ("signal", "points", "exchange",
                                                                          "input_estimate")}
        print(f"[INFO] CleanUNet2: conditioning={conditioning_type}, bottleneck={self.clean_unet.bottleneck_type}, "
              f"specnet={specnet}, spec_branch={use_spec_branch}, "
              f"bottleneck_fusion={bottleneck_fusion}, fusion={fusion_desc or None}")

    # ------------------------------------------------------------------
    SIGNALS = ("spec_hidden", "spec", "mask", "noisy_spec", "oracle_spec", "oracle_mask")

    def _build_fusion(self, cfg: dict):
        from .fusion import BranchExchange, FusionHub, InputEstimateFusion
        from .postfilter import SpectralPostFilter

        known = {"signal", "trunk", "resample", "shift", "zero_init", "points", "input_estimate", "exchange",
                 "detach", "postfilter"}
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f"Unknown fusion options: {sorted(unknown)}")
        signals = cfg.get("signal", "spec_hidden")
        self.fusion_signals = [signals] if isinstance(signals, str) else list(signals)
        for sig in self.fusion_signals:
            if sig not in self.SIGNALS:
                raise ValueError(f"Unknown fusion signal '{sig}' (use {self.SIGNALS})")
        n_freq, hidden = self.spectrogram.n_freq, self.clean_spec_net.hidden_dim
        cond_dim = sum(hidden if sig == "spec_hidden" else n_freq for sig in self.fusion_signals)
        hop = self.spectrogram.hop_length
        zero_init = cfg.get("zero_init", "last")

        points = dict(cfg.get("points") or {})
        out_opts = points.pop("out", None)
        if points:
            self.fusion_hub = FusionHub(self.clean_unet.fusion_points(), cond_dim, hop, points,
                                        trunk=cfg.get("trunk"), resample=cfg.get("resample", "repeat"),
                                        shift=int(cfg.get("shift", 0)), zero_init=zero_init)
        if out_opts is not None:
            opts = dict(out_opts or {})
            opts.pop("op", None)
            self.postfilter = SpectralPostFilter(n_freq, n_freq, zero_init=zero_init, **opts)
        if cfg.get("input_estimate", False):
            ch, stride = self.clean_unet.fusion_points()["enc_s2"]
            self.input_estimate = InputEstimateFusion(ch, self.clean_unet.kernel_size, self.clean_unet.stride,
                                                      zero_init=zero_init)
        if cfg.get("exchange", False):
            cu, cs = self.clean_unet, self.clean_spec_net
            if cu.bottleneck_type != "mamba" or cs.backbone_type != "mamba":
                raise ValueError("exchange requires Mamba in both the CleanUNet bottleneck and CleanSpecNet")
            if hop != cu.total_stride:
                raise ValueError("exchange requires hop == CleanUNet total stride (frame j <-> token j)")
            if points:
                raise ValueError("exchange cannot be combined with fusion points")
            self.exchange = BranchExchange(cu.tsfm_d_model, cs.hidden_dim, len(cu.tsfm_Mamba_layers),
                                           len(cs.backbone.layers), zero_init=zero_init)

    @property
    def needs_clean(self) -> bool:
        """True when the forward pass uses the clean target (oracle signals / oracle curriculum)."""
        sig = getattr(self, "fusion_signals", [])
        return any(s.startswith("oracle") for s in sig) or self.oracle_prob > 0

    # ------------------------------------------------------------------
    def input_scale(self, noisy: torch.Tensor) -> torch.Tensor:
        """Per-example normalization factor (B, 1, 1)."""
        if not self.normalize_input:
            return torch.ones(noisy.shape[0], 1, 1, device=noisy.device, dtype=noisy.dtype)
        return noisy.std(dim=-1, keepdim=True) + 1e-3

    def target_spectrogram(self, clean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Compressed spectrogram of the clean target, using the same normalization as the noisy input."""
        if clean.dim() == 2:
            clean = clean.unsqueeze(1)
        return self.spectrogram(clean / scale)

    def spec_features(self, wav_norm: torch.Tensor):
        """Input of the spectral branch for a normalized waveform.

        Returns (features (B, G*F, T), compressed magnitude (B, F, T), complex STFT (B, F, T) or None).
        """
        if getattr(self, "spec_input_features", "mag") == "mag":
            mag = self.spectrogram(wav_norm)
            return mag, mag, None
        with torch.autocast(device_type=wav_norm.device.type, enabled=False):
            X = self.spectrogram.complex_stft(wav_norm.float())
            mag = self.spectrogram.compress(X.abs() * self.spectrogram.scale)
            phase = torch.angle(X)
            return torch.cat([mag, torch.cos(phase), torch.sin(phase)], dim=1), mag, X

    def spec_branch(self, wav_norm: torch.Tensor) -> dict:
        """Run only the spectral branch (analysis / diagnostics): CleanSpecNet outputs plus the noisy
        compressed magnitude and the complex noisy STFT."""
        feats, mag, X = self.spec_features(wav_norm)
        out = dict(self.clean_spec_net.run(feats))
        out["noisy_spec"] = mag
        out["noisy_stft"] = X if X is not None else self.spectrogram.complex_stft(wav_norm.float())
        return out

    def target_stft(self, clean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Complex STFT of the clean target with the input normalization (for phase-aware losses)."""
        return self.spectrogram.complex_stft((clean.view(clean.shape[0], 1, -1) / scale).float())

    def spec_to_wave(self, spec: torch.Tensor, wav_norm: torch.Tensor, length: int, phase: torch.Tensor = None):
        """Resynthesize a compressed magnitude (B, F, T) with the predicted phase, or with the phase of the
        normalized waveform `wav_norm` when `phase` is None. Returns (B, 1, length) in the normalized domain."""
        with torch.autocast(device_type=spec.device.type, enabled=False):
            mag = self.spectrogram.decompress(spec.float()) / self.spectrogram.scale
            if phase is None:
                X = self.spectrogram.complex_stft(wav_norm.float())
                unit = X / X.abs().clamp(min=1e-8)
            else:
                unit = torch.polar(torch.ones_like(phase, dtype=torch.float32), phase.float())
            T = min(mag.shape[-1], unit.shape[-1])
            return self.spectrogram.istft(mag[..., :T] * unit[..., :T], length)

    def forward(self, noisy_waveform: torch.Tensor, spec_only: bool = False, clean: torch.Tensor = None,
                cond_noisy: torch.Tensor = None, cond_clean: torch.Tensor = None,
                fusion_enabled: bool = True) -> dict:
        """
        Args:
            noisy_waveform: (B, 1, L) or (B, L)
            spec_only: only run the spectrogram branch (stage-B training)
            clean: clean target (B, 1, L); only used by oracle conditions (diagnostics) and the oracle curriculum
            cond_noisy / cond_clean: compute the spectral condition from these signals instead of the input
                (same length L); used to measure how much the model depends on its condition
            fusion_enabled: False bypasses every conditioning path (waveform FiLM, fusion, post-filter)
        Returns:
            dict with "wav" (B, 1, L) [None if spec_only], "spec" (B, F, T) [None if CleanSpecNet did not run],
            "phase" (B, F, T) [None without a phase head] and "scale" (B, 1, 1).
        """
        if noisy_waveform.dim() == 2:
            noisy_waveform = noisy_waveform.unsqueeze(1)
        L = noisy_waveform.shape[-1]
        scale = self.input_scale(noisy_waveform)
        x = noisy_waveform / scale

        if spec_only and not self.use_spec_branch:
            raise RuntimeError("spec_only=True requires use_spec_branch=True")
        if not self.use_spec_branch:
            return {"wav": self.clean_unet(x) * scale, "spec": None, "phase": None, "scale": scale}

        # Source of the spectral condition (normally the input itself)
        if cond_noisy is not None:
            if cond_noisy.dim() == 2:
                cond_noisy = cond_noisy.unsqueeze(1)
            c_scale = self.input_scale(cond_noisy)
            xc = cond_noisy / c_scale
            cc = None if cond_clean is None else cond_clean.view_as(cond_noisy) / c_scale
        else:
            xc, cc = x, (None if clean is None else clean.view_as(noisy_waveform) / scale)

        noisy_feats, noisy_spec, _ = self.spec_features(xc)
        clean_feats = clean_spec = None
        if cc is not None and self.needs_clean:
            clean_feats, clean_spec, _ = self.spec_features(cc)
        spec_in = noisy_feats
        if self.training and self.oracle_prob > 0 and clean_feats is not None:
            pick = torch.rand(noisy_feats.shape[0], 1, 1, device=noisy_feats.device) < self.oracle_prob
            spec_in = torch.where(pick, clean_feats, noisy_feats)

        signals = getattr(self, "fusion_signals", [])
        use_exchange = hasattr(self, "exchange") and fusion_enabled
        run_specnet = (spec_only or hasattr(self, "conditioner") or hasattr(self, "fusion")
                       or hasattr(self, "input_estimate") or hasattr(self, "exchange")
                       or any(sg in ("spec_hidden", "spec", "mask") for sg in signals))

        denoised_spec = spec_hidden = phase = None
        runner = None
        if run_specnet and not use_exchange:
            res = self.clean_spec_net.run(spec_in)
            denoised_spec, spec_hidden, phase = res["spec"], res["hidden"], res["phase"]
            if spec_only:
                return {"wav": None, "spec": denoised_spec, "phase": phase, "scale": scale}
        elif use_exchange:
            store = {}
            spec_front = self.clean_spec_net.encode(spec_in)
            backbone = self.clean_spec_net.backbone

            def runner(tokens):
                out_w, out_s = self.exchange.run(self.clean_unet.tsfm_Mamba_layers, self.clean_unet.norm_f,
                                                 backbone.layers, backbone.norm_f, tokens, spec_front)
                store["hidden"] = out_s
                return out_w

        # Original waveform conditioning (CleanUNet2 paper) and legacy bottleneck fusion
        x_in, hooks = x, {}
        if fusion_enabled and hasattr(self, "conditioner"):
            cond = self.spec_upsampler(denoised_spec.float())[..., :L].to(x.dtype)
            x_in = self.conditioner(x, cond)
        if fusion_enabled and self.bottleneck_fusion:
            hooks["bott_pre"] = lambda h: self.fusion(h, spec_hidden)  # noqa: E731

        # Generic fusion
        cond = spec_like = None
        if fusion_enabled and (hasattr(self, "fusion_hub") or hasattr(self, "postfilter")):
            cond, spec_like = self._fusion_condition(signals, denoised_spec, spec_hidden, noisy_spec, clean_spec)
            if hasattr(self, "fusion_hub"):
                hooks.update(self.fusion_hub.hooks(cond))
        if fusion_enabled and hasattr(self, "input_estimate"):
            detach = self.fusion_cfg.get("detach")
            est_spec = denoised_spec.detach() if detach else denoised_spec
            est_phase = phase.detach() if (detach and phase is not None) else phase
            est = self.spec_to_wave(est_spec, xc, L, phase=est_phase)
            prev = hooks.get("enc_s2")
            hooks["enc_s2"] = lambda h, prev=prev: self.input_estimate(prev(h) if prev else h, est,  # noqa: E731
                                                                        self.clean_unet)

        wav = self.clean_unet(x_in, hooks=hooks, bottleneck_runner=runner)
        if use_exchange:
            denoised_spec = self.clean_spec_net.head(store["hidden"], spec_in)
            phase = self.clean_spec_net.phase_head(store["hidden"])
        if fusion_enabled and hasattr(self, "postfilter"):
            if spec_like is None:
                spec_like = denoised_spec
            wav = self.postfilter(wav, spec_like, noisy_spec, self.spectrogram)
        return {"wav": wav * scale, "spec": denoised_spec, "phase": phase, "scale": scale}

    def _fusion_condition(self, signals, denoised_spec, spec_hidden, noisy_spec, clean_spec):
        """Frame-rate condition (B, D, T) from the configured signals, and a spectrogram-like (B, F, T) one."""
        detach = self.fusion_cfg.get("detach", False)
        dn = (lambda t: t.detach()) if detach else (lambda t: t)  # noqa: E731
        parts, spec_like = [], None
        for sig in signals:
            if sig == "spec_hidden":
                parts.append(dn(spec_hidden).transpose(1, 2))
                continue
            if sig.startswith("oracle") and clean_spec is None:
                raise ValueError(f"fusion signal '{sig}' needs the clean target (forward(..., clean=...))")
            t = {"spec": lambda: dn(denoised_spec),
                 "mask": lambda: dn(denoised_spec) - noisy_spec,
                 "noisy_spec": lambda: noisy_spec,
                 "oracle_spec": lambda: clean_spec,
                 "oracle_mask": lambda: clean_spec - noisy_spec}[sig]()
            parts.append(t)
            if spec_like is None and sig in ("spec", "noisy_spec", "oracle_spec"):
                spec_like = t
        return torch.cat([p.float() for p in parts], dim=1), spec_like

    @torch.no_grad()
    def enhance(self, noisy_waveform: torch.Tensor) -> torch.Tensor:
        """Convenience inference helper: (B, L) / (B, 1, L) -> (B, 1, L)."""
        return self.forward(noisy_waveform)["wav"]

    # ------------------------------------------------------------------
    # Checkpoint Loading Helpers
    # ------------------------------------------------------------------
    def load_cleanunet_weights(self, checkpoint_path, prefix=None):
        """Loads pre-trained weights for the CleanUNet submodule (Lightning, CleanUNet2 or CleanUMamba files)."""
        from .checkpoint_io import load_submodule_weights
        load_submodule_weights(self.clean_unet, checkpoint_path, "clean_unet", prefix)

    def load_cleanspecnet_weights(self, checkpoint_path, prefix=None):
        """Loads pre-trained weights for the CleanSpecNet submodule."""
        from .checkpoint_io import load_submodule_weights
        load_submodule_weights(self.clean_spec_net, checkpoint_path, "clean_spec_net", prefix)

    def load_cleanumamba_weights(self, checkpoint_path):
        """Initializes the waveform branch from an official CleanUMamba checkpoint (.pkl)."""
        from .checkpoint_io import load_cleanumamba_into_cleanunet
        load_cleanumamba_into_cleanunet(self.clean_unet, checkpoint_path)

    def load_cleanunet2_weights(self, checkpoint_path, strict=True):
        """Loads pre-trained weights for the full CleanUNet2 model."""
        from .checkpoint_io import extract_state_dict
        state_dict = extract_state_dict(checkpoint_path)
        if any(k.startswith("model.") for k in state_dict):
            # Lightning checkpoint: keep the model weights only (loss modules also store buffers)
            state_dict = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
        self.load_state_dict(state_dict, strict=strict)
        print("[INFO] Weights loaded successfully into the full CleanUNet2 model.")
