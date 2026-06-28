"""
SSL extractor factory.

Single dispatch point that maps an `ssl_type` string to the matching extractor
class. Every extractor exposes the identical constructor signature, so the only
per-family differences (backbone class + feature path) stay encapsulated inside
the extractor modules; the rest of the pipeline is family-agnostic.

The `+` vs `++` distinction is NOT a different class -- it is the single
`selected_layers` argument:
    +   -> selected_layers = [first, middle, last]   (3 selected hidden states)
    ++  -> selected_layers = 'all'                    (every hidden state)
Both use a learnable softmax over the chosen layers.
"""

# ssl_type (lowercased) -> (module, class name)
_REGISTRY = {
    'wav2vec2': ('.wav2vec2_extractor', 'Wav2Vec2Extractor'),
    'wav2vec':  ('.wav2vec2_extractor', 'Wav2Vec2Extractor'),
    'hubert':   ('.hubert_extractor',   'HubertExtractor'),
    'wavlm':    ('.wavlm_extractor',    'WavLMExtractor'),
    'w2v-bert': ('.w2vbert_extractor',  'W2VBertExtractor'),
    'w2vbert':  ('.w2vbert_extractor',  'W2VBertExtractor'),
    'w2v_bert': ('.w2vbert_extractor',  'W2VBertExtractor'),
    'whisper':  ('.whisper_extractor',  'WhisperExtractor'),
}


def build_ssl_extractor(ssl_type, model_name, device='cpu', layer=12,
                        pooling_method='mean', num_attention_heads=8,
                        selected_layers=None, num_selected_layers=3,
                        use_weighted_layers=True):
    """
    Instantiate the SSL extractor for `ssl_type`.

    Args:
        ssl_type (str): one of wav2vec2 | hubert | wavlm | w2v-bert | whisper.
        model_name (str): HuggingFace model id for that family.
        selected_layers: list of indices for the '+' (3-layer) strategy, or the
                         string 'all' for the '++' (all-layer) strategy. None
                         falls back to `num_selected_layers` evenly-spaced layers.
        (remaining args mirror the extractor constructors.)

    Returns:
        nn.Module: the family-specific extractor.
    """
    key = ssl_type.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown ssl_type '{ssl_type}'. Supported: "
            f"{sorted(set(_REGISTRY))}"
        )

    import importlib
    module_name, class_name = _REGISTRY[key]
    module = importlib.import_module(module_name, package=__package__)
    extractor_cls = getattr(module, class_name)

    return extractor_cls(
        model_name=model_name,
        device=device,
        layer=layer,
        pooling_method=pooling_method,
        num_attention_heads=num_attention_heads,
        num_selected_layers=num_selected_layers,
        selected_layers=selected_layers,
        use_weighted_layers=use_weighted_layers,
    )


def ssl_args_from_config(model_config):
    """
    Translate a `model` config block into the `ssl_*` kwargs accepted by
    CleanUNet2WithSSLEmbeddings. Reads the generic `model.ssl.*` schema and
    falls back to the legacy nested `model.wav2vec2.*` block for old configs.

    The single '+' vs '++' switch is `ssl.selected_layers`:
        [i, j, k]  -> '+'  (3 selected hidden states)
        'all'      -> '++' (every hidden state)
    """
    ssl = model_config.get('ssl', {})
    legacy = model_config.get('wav2vec2', {})  # backward compatibility

    return {
        'ssl_type': ssl.get('type', 'wav2vec2'),
        'ssl_model': ssl.get('model_name', legacy.get('model_name', 'facebook/wav2vec2-xls-r-2b')),
        'ssl_layer': ssl.get('layer', legacy.get('wav2vec2_layer', 24)),
        'ssl_cache_dir': ssl.get('cache_dir', legacy.get('wav2vec2_cache_dir')),
        'use_preextracted_embeddings': ssl.get('use_preextracted', legacy.get('use_preextracted', False)),
        'ssl_pooling_method': ssl.get('pooling_method', 'mean'),
        'ssl_attention_heads': ssl.get('attention_heads', 8),
        'ssl_use_weighted_layers': ssl.get('use_weighted_layers', legacy.get('use_weighted_layers', True)),
        'ssl_selected_layers': ssl.get('selected_layers'),   # list ('+'), 'all' ('++'), or None
        'ssl_num_selected_layers': ssl.get('num_selected_layers', 3),
        'ssl_embedding_dim': ssl.get('embedding_dim'),       # required by Stage 2 (no extractor)
    }


def fusion_args_from_config(model_config):
    """
    Translate the `model.fusion` block into the `fusion_*` kwargs accepted by
    CleanUNet2WithSSLEmbeddings. Defaults to the hierarchical multi-scale fusion.

    Schema (model.fusion):
        type: 'hierarchical_multiscale' (default) | 'legacy_pooling'
        acoustic_layers: [lo, hi] | null   (inclusive indices into the selected SSL stack)
        semantic_layers: [lo, hi] | null   (null -> lower/upper halves, auto per SSL model)
    """
    fusion = model_config.get('fusion', {})
    return {
        'fusion_type': fusion.get('type', 'hierarchical_multiscale'),
        'acoustic_layers': fusion.get('acoustic_layers'),   # None -> lower half (auto)
        'semantic_layers': fusion.get('semantic_layers'),   # None -> upper half (auto)
    }


def latent_predictor_args_from_config(model_config):
    """
    Translate the `model.latent_predictor` block into the kwargs accepted by
    CleanUNet2WithSSLEmbeddings for the Stage-2 latent predictor. Defaults to the
    original 'baseline' predictor (2-layer 1x1 conv) for back-compat.

    Schema (model.latent_predictor):
        type: 'baseline' | 'tcn'
        params: { ... } | null   (forwarded to the predictor constructor)
    """
    lp = model_config.get('latent_predictor', {})
    return {
        'latent_predictor_type': lp.get('type', 'baseline'),
        'latent_predictor_params': lp.get('params'),
    }
