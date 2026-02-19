"""
Script to validate that all WavLM configs are properly updated
Checks for correct model names, dimensions, and consistency
"""

import yaml
from pathlib import Path
from collections import defaultdict

def validate_config(config_path):
    """Validate a single config file"""
    errors = []
    warnings = []
    info = []

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        errors.append(f"Failed to load config: {e}")
        return errors, warnings, info

    # Check if this is a wavlm config
    if 'wavlm' not in config_path.name and 'wav2vec2' not in config_path.name:
        info.append("Not a WavLM config, skipping")
        return errors, warnings, info

    # Check model section
    if 'model' not in config:
        errors.append("No 'model' section found")
        return errors, warnings, info

    model = config['model']

    # Check wav2vec2_model
    if 'wav2vec2_model' in model:
        expected_model = "microsoft/wavlm-base"
        actual_model = model['wav2vec2_model']

        if actual_model != expected_model:
            errors.append(f"Wrong model: '{actual_model}' (expected '{expected_model}')")
        else:
            info.append(f"✓ Model: {actual_model}")
    else:
        warnings.append("No 'wav2vec2_model' specified")

    # Check for old model references
    config_str = str(config)
    if 'facebook/wav2vec2' in config_str or 'xls-r-300m' in config_str:
        errors.append("Found reference to old Wav2Vec2 model (facebook/wav2vec2-xls-r-300m)")

    # Check embedding dimension comments
    if '1024' in config_str and 'embedding' in config_str.lower():
        # Check if it's in STFT context (which is OK)
        if 'stft' not in config_str.lower():
            warnings.append("Found '1024' in embedding context (should be 768 for WavLM)")

    # Check use_wav2vec2 flag
    if 'use_wav2vec2' in model:
        if model['use_wav2vec2']:
            info.append("✓ use_wav2vec2: true")
        else:
            warnings.append("use_wav2vec2 is false")

    # Check use_xvector flag
    if 'use_xvector' in model:
        if not model['use_xvector']:
            info.append("✓ use_xvector: false (correct for WavLM)")
        else:
            errors.append("use_xvector is true (should be false for WavLM)")

    # Check for Stage-1 specific settings
    if 'stage1' in config_path.name:
        if 'use_preextracted_embeddings' in model:
            if model['use_preextracted_embeddings']:
                info.append("✓ Using pre-extracted embeddings (recommended)")
            else:
                warnings.append("Not using pre-extracted embeddings (slower training)")

        if 'wav2vec2_cache_dir' in model:
            cache_dir = model['wav2vec2_cache_dir']
            info.append(f"✓ Cache dir: {cache_dir}")

    # Check for Stage-2 specific settings
    if 'stage2' in config_path.name:
        if 'stage1_checkpoint' not in config:
            errors.append("Stage-2 config missing 'stage1_checkpoint'")
        else:
            info.append(f"✓ Stage-1 checkpoint: {config['stage1_checkpoint']}")

        if 'latents_dir' not in config:
            errors.append("Stage-2 config missing 'latents_dir'")
        else:
            info.append(f"✓ Latents dir: {config['latents_dir']}")

    return errors, warnings, info


def main():
    configs_dir = Path("configs")

    print("=" * 80)
    print("WavLM Config Validation")
    print("=" * 80)

    # Find all wavlm configs (updated from wav2vec2)
    wavlm_configs = sorted(list(configs_dir.glob("*wavlm*.yaml")) + list(configs_dir.glob("*wav2vec2*.yaml")))

    if not wavlm_configs:
        print("\n❌ No WavLM configs found!")
        return

    print(f"\nFound {len(wavlm_configs)} WavLM config(s):\n")

    results = defaultdict(lambda: {'errors': [], 'warnings': [], 'info': []})

    for config_path in wavlm_configs:
        print(f"\n{'─' * 80}")
        print(f"📄 {config_path.name}")
        print('─' * 80)

        errors, warnings, info = validate_config(config_path)

        results[config_path.name]['errors'] = errors
        results[config_path.name]['warnings'] = warnings
        results[config_path.name]['info'] = info

        # Print results
        if errors:
            print("\n❌ ERRORS:")
            for err in errors:
                print(f"   • {err}")

        if warnings:
            print("\n⚠️  WARNINGS:")
            for warn in warnings:
                print(f"   • {warn}")

        if info:
            print("\n✓ INFO:")
            for i in info:
                print(f"   {i}")

        if not errors and not warnings:
            print("\n✅ Config is valid!")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_errors = sum(len(r['errors']) for r in results.values())
    total_warnings = sum(len(r['warnings']) for r in results.values())

    print(f"\nTotal configs checked: {len(wavlm_configs)}")
    print(f"Total errors: {total_errors}")
    print(f"Total warnings: {total_warnings}")

    if total_errors == 0 and total_warnings == 0:
        print("\n🎉 All configs are valid and properly configured for WavLM!")
    elif total_errors == 0:
        print("\n✅ All configs are valid (some warnings present)")
    else:
        print("\n❌ Some configs have errors that need to be fixed")

    print("\n" + "=" * 80)

    # List Stage-1 and Stage-2 configs
    stage1_configs = [c.name for c in wavlm_configs if 'stage1' in c.name]
    stage2_configs = [c.name for c in wavlm_configs if 'stage2' in c.name]

    print("\nStage-1 Configs:")
    for cfg in stage1_configs:
        print(f"  ✓ {cfg}")

    print("\nStage-2 Configs:")
    for cfg in stage2_configs:
        print(f"  ✓ {cfg}")

    if len(stage1_configs) != len(stage2_configs):
        print(f"\n⚠️  Warning: {len(stage1_configs)} Stage-1 configs but {len(stage2_configs)} Stage-2 configs")
        print("   Each Stage-1 config should have a corresponding Stage-2 config")
    else:
        print(f"\n✓ Matching Stage-1 and Stage-2 configs ({len(stage1_configs)} each)")


if __name__ == '__main__':
    main()
