#!/usr/bin/env python3
"""
Script de diagnóstico para verificar problemas no dataset.
"""

import os
import yaml
from pathlib import Path
import torch
import torchaudio

def get_dataset_filelist(filename):
    """Carrega lista de arquivos do dataset."""
    with open(filename, encoding='utf-8') as f:
        files = [line.strip().split('|') for line in f]
    return files


def check_audio_file(file_path):
    """Verifica se um arquivo de áudio pode ser carregado."""
    try:
        if not os.path.exists(file_path):
            return False, "File not found"

        # Tentar carregar
        waveform, sr = torchaudio.load(file_path)

        if waveform.numel() == 0:
            return False, "Empty waveform"

        return True, f"OK (sr={sr}, samples={waveform.shape[-1]})"

    except Exception as e:
        return False, str(e)


def main():
    print("=" * 80)
    print("DIAGNÓSTICO DO DATASET")
    print("=" * 80)

    # Carregar config
    config_path = "configs/train_xvector_optimized_stage2.yaml"
    print(f"\nCarregando config: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Verificar data module config
    data_config = config.get('data', {})
    data_dir = data_config.get('data_dir', 'data')
    train_files = data_config.get('training_files', 'train_files.txt')
    val_files = data_config.get('validation_files', 'val_files.txt')

    print(f"\nDataset config:")
    print(f"  Data dir: {data_dir}")
    print(f"  Train files: {train_files}")
    print(f"  Val files: {val_files}")

    # Verificar se diretório existe
    if not os.path.exists(data_dir):
        print(f"\n❌ ERRO: Data directory não existe: {data_dir}")
        return

    print(f"\n✅ Data directory existe: {data_dir}")

    # Verificar validation files
    print(f"\n{'='*80}")
    print("VERIFICANDO VALIDATION FILES")
    print(f"{'='*80}")

    if not os.path.exists(val_files):
        print(f"❌ ERRO: Validation files list não existe: {val_files}")
        return

    print(f"✅ Validation files list existe: {val_files}")

    # Carregar lista de arquivos
    try:
        file_list = get_dataset_filelist(val_files)
        print(f"✅ Carregados {len(file_list)} pares de arquivos")
    except Exception as e:
        print(f"❌ ERRO ao carregar file list: {e}")
        return

    # Verificar primeiros arquivos
    num_to_check = min(10, len(file_list))
    print(f"\n📋 Verificando primeiros {num_to_check} arquivos...")

    errors = []
    for i in range(num_to_check):
        clean_rel, noisy_rel = file_list[i]
        clean_path = os.path.join(data_dir, clean_rel)
        noisy_path = os.path.join(data_dir, noisy_rel)

        print(f"\n--- Arquivo {i+1}/{num_to_check} ---")
        print(f"Clean: {clean_rel}")
        print(f"Noisy: {noisy_rel}")

        # Verificar clean
        clean_ok, clean_msg = check_audio_file(clean_path)
        if clean_ok:
            print(f"  ✅ Clean: {clean_msg}")
        else:
            print(f"  ❌ Clean: {clean_msg}")
            errors.append((clean_path, clean_msg))

        # Verificar noisy
        noisy_ok, noisy_msg = check_audio_file(noisy_path)
        if noisy_ok:
            print(f"  ✅ Noisy: {noisy_msg}")
        else:
            print(f"  ❌ Noisy: {noisy_msg}")
            errors.append((noisy_path, noisy_msg))

    # Resumo
    print(f"\n{'='*80}")
    print("RESUMO")
    print(f"{'='*80}")

    if errors:
        print(f"\n❌ Encontrados {len(errors)} erros:")
        for file_path, error_msg in errors[:10]:
            print(f"  - {file_path}")
            print(f"    {error_msg}")
        if len(errors) > 10:
            print(f"  ... e mais {len(errors) - 10} erros")
    else:
        print(f"\n✅ Todos os arquivos verificados estão OK!")

    # Verificar se algum arquivo existe no data_dir
    print(f"\n{'='*80}")
    print("VERIFICANDO CONTEÚDO DO DATA_DIR")
    print(f"{'='*80}")

    data_path = Path(data_dir)
    if data_path.exists():
        # Listar subdiretórios
        subdirs = [d for d in data_path.iterdir() if d.is_dir()]
        print(f"\n📁 Subdiretórios encontrados: {len(subdirs)}")
        for subdir in subdirs[:5]:
            # Contar arquivos de áudio
            audio_files = list(subdir.glob('*.wav')) + list(subdir.glob('*.flac'))
            print(f"  - {subdir.name}: {len(audio_files)} arquivos de áudio")

        # Contar total de arquivos de áudio
        total_wav = len(list(data_path.rglob('*.wav')))
        total_flac = len(list(data_path.rglob('*.flac')))
        print(f"\n📊 Total de arquivos de áudio:")
        print(f"  - WAV: {total_wav}")
        print(f"  - FLAC: {total_flac}")
        print(f"  - Total: {total_wav + total_flac}")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
