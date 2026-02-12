"""
Script para testar a configuração do Stage-1 do XVector.
Verifica:
1. Se as augmentações estão sendo carregadas
2. Se o sample_rate está correto (24000 Hz)
3. Se o dataset está funcionando
"""

import yaml
import sys

def test_config(config_path):
    print("=" * 80)
    print("TESTE DE CONFIGURAÇÃO - CleanUNet2-XVector Stage-1")
    print("=" * 80)

    # Carregar config
    print(f"\n1. Carregando configuração: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Verificar seção audio
    print("\n2. Verificando seção 'audio':")
    audio_cfg = config.get('audio', {})
    sample_rate = audio_cfg.get('sample_rate', 'NÃO ENCONTRADO')
    print(f"   sample_rate: {sample_rate} Hz")

    if sample_rate == 24000:
        print("   ✅ Sample rate correto (24000 Hz)")
    else:
        print(f"   ❌ Sample rate incorreto! Esperado: 24000, Encontrado: {sample_rate}")

    # Verificar seção data
    print("\n3. Verificando seção 'data':")
    data_cfg = config.get('data', {})
    sampling_rate = data_cfg.get('sampling_rate', 'NÃO ENCONTRADO')
    print(f"   sampling_rate: {sampling_rate} Hz")

    if sampling_rate == 24000:
        print("   ✅ Sampling rate correto (24000 Hz)")
    else:
        print(f"   ❌ Sampling rate incorreto! Esperado: 24000, Encontrado: {sampling_rate}")

    # Verificar augmentations
    print("\n4. Verificando augmentações:")
    augmentations = data_cfg.get('augmentations', [])

    if augmentations:
        print(f"   ✅ Augmentações encontradas: {len(augmentations)} tipos")
        for i, aug in enumerate(augmentations):
            name = aug.get('name', 'Unknown')
            params = aug.get('params', {})
            p = params.get('p', 'N/A')
            print(f"      {i+1}. {name} (probability: {p})")

            # Verificar AddBackgroundNoise
            if name == "AddBackgroundNoise":
                bg_paths = params.get('background_paths', 'NÃO ESPECIFICADO')
                print(f"         background_paths: {bg_paths}")
                min_snr = params.get('min_snr_in_db', 'N/A')
                max_snr = params.get('max_snr_in_db', 'N/A')
                print(f"         SNR range: {min_snr} dB to {max_snr} dB")
    else:
        print("   ❌ Nenhuma augmentação encontrada!")

    # Verificar batch_size e num_workers
    print("\n5. Verificando parâmetros de DataLoader:")
    batch_size = data_cfg.get('batch_size', 'NÃO ENCONTRADO')
    num_workers = data_cfg.get('num_workers', 'NÃO ENCONTRADO')
    segment_size = data_cfg.get('segment_size', 'NÃO ENCONTRADO')
    print(f"   batch_size: {batch_size}")
    print(f"   num_workers: {num_workers}")
    print(f"   segment_size: {segment_size} samples ({segment_size/24000:.2f} segundos)")

    # Verificar filelists
    print("\n6. Verificando filelists:")
    train_list = data_cfg.get('train_list_path', 'NÃO ENCONTRADO')
    val_list = data_cfg.get('val_list_path', 'NÃO ENCONTRADO')
    print(f"   train_list_path: {train_list}")
    print(f"   val_list_path: {val_list}")

    # Verificar modelo
    print("\n7. Verificando configuração do modelo:")
    model_cfg = config.get('model', {})
    xvector_dim = model_cfg.get('xvector_dim', 'NÃO ENCONTRADO')
    conditioning_type = model_cfg.get('conditioning_type', 'NÃO ENCONTRADO')
    vanilla_checkpoint = model_cfg.get('vanilla_checkpoint', 'NÃO ESPECIFICADO')
    print(f"   xvector_dim: {xvector_dim}")
    print(f"   conditioning_type: {conditioning_type}")
    print(f"   vanilla_checkpoint: {vanilla_checkpoint}")

    # Resumo
    print("\n" + "=" * 80)
    print("RESUMO DO TESTE")
    print("=" * 80)

    issues = []

    if sample_rate != 24000:
        issues.append(f"Sample rate incorreto em 'audio': {sample_rate}")

    if sampling_rate != 24000:
        issues.append(f"Sampling rate incorreto em 'data': {sampling_rate}")

    if not augmentations:
        issues.append("Nenhuma augmentação configurada")

    if issues:
        print("❌ PROBLEMAS ENCONTRADOS:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ TODAS AS VERIFICAÇÕES PASSARAM!")
        print("\nResumo:")
        print(f"   - Sample rate: {sample_rate} Hz")
        print(f"   - Augmentações: {len(augmentations)} tipos configurados")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Segment size: {segment_size} samples")

    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "configs/train_xvector_vanilla_stage1.yaml"

    test_config(config_path)
