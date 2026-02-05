#!/usr/bin/env python3
"""
Script para criar filelists de treino e validação para o dataset LJSpeech.

O LJSpeech contém ~13100 arquivos de áudio. Este script cria um split
treino/validação (por padrão 95%/5% = ~12400 treino, ~650 validação).

Uso:
    python scripts/prepare_ljspeech_filelists.py \
        --ljspeech_dir /path/to/LJSpeech-1.1 \
        --output_dir filelists \
        --val_split 0.05
"""

import os
import argparse
from pathlib import Path
import random


def create_ljspeech_filelists(ljspeech_dir, output_dir, val_split=0.05, seed=1234):
    """
    Cria filelists de treino e validação para LJSpeech.

    Args:
        ljspeech_dir: Caminho para o diretório LJSpeech-1.1
        output_dir: Diretório onde salvar os arquivos CSV
        val_split: Fração dos dados para validação (default: 0.05 = 5%)
        seed: Seed para reproducibilidade
    """
    # Configura seed
    random.seed(seed)

    # Diretório com os arquivos WAV
    wavs_dir = os.path.join(ljspeech_dir, "wavs")

    if not os.path.exists(wavs_dir):
        raise FileNotFoundError(f"Diretório de áudios não encontrado: {wavs_dir}")

    # Lista todos os arquivos .wav
    wav_files = sorted([f for f in os.listdir(wavs_dir) if f.endswith('.wav')])

    if not wav_files:
        raise ValueError(f"Nenhum arquivo .wav encontrado em {wavs_dir}")

    print(f"Encontrados {len(wav_files)} arquivos de áudio em {wavs_dir}")

    # Embaralha para criar split aleatório
    random.shuffle(wav_files)

    # Calcula tamanho do split
    val_size = int(len(wav_files) * val_split)
    train_size = len(wav_files) - val_size

    train_files = wav_files[:train_size]
    val_files = wav_files[train_size:]

    print(f"\nSplit:")
    print(f"  Treino: {len(train_files)} arquivos ({100*(1-val_split):.1f}%)")
    print(f"  Validação: {len(val_files)} arquivos ({100*val_split:.1f}%)")

    # Cria diretório de saída
    os.makedirs(output_dir, exist_ok=True)

    # Escreve arquivo de treino
    train_path = os.path.join(output_dir, "ljspeech_train.csv")
    with open(train_path, 'w', encoding='utf-8') as f:
        for wav_file in train_files:
            # Formato: clean_path|noisy_path
            # Como só temos áudio limpo, repetimos o mesmo arquivo
            # O áudio ruidoso será gerado via augmentation
            rel_path = f"wavs/{wav_file}"
            f.write(f"{rel_path}|{rel_path}\n")

    print(f"\n✓ Arquivo de treino criado: {train_path}")

    # Escreve arquivo de validação
    val_path = os.path.join(output_dir, "ljspeech_val.csv")
    with open(val_path, 'w', encoding='utf-8') as f:
        for wav_file in val_files:
            rel_path = f"wavs/{wav_file}"
            f.write(f"{rel_path}|{rel_path}\n")

    print(f"✓ Arquivo de validação criado: {val_path}")

    # Mostra exemplos
    print(f"\nPrimeiras 5 linhas do arquivo de treino:")
    with open(train_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(f"  {line.strip()}")

    print("\n" + "=" * 80)
    print("Próximos passos:")
    print("=" * 80)
    print("\n1. Edite o arquivo de configuração train_ljspeech_augmentation.yaml:")
    print(f"   data_dir: \"{ljspeech_dir}\"")
    print(f"   train_list_path: \"{train_path}\"")
    print(f"   val_list_path: \"{val_path}\"")
    print("\n2. Configure o caminho para arquivos de ruído em augmentation.augmentations")
    print("\n3. Execute o treinamento:")
    print("   python train.py --config configs/train_ljspeech_augmentation.yaml")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Cria filelists de treino/validação para LJSpeech"
    )
    parser.add_argument(
        "--ljspeech_dir",
        type=str,
        required=True,
        help="Caminho para o diretório LJSpeech-1.1"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="filelists",
        help="Diretório onde salvar os arquivos CSV (default: filelists)"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.05,
        help="Fração dos dados para validação (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed para reproducibilidade (default: 1234)"
    )

    args = parser.parse_args()

    # Valida argumentos
    if not 0 < args.val_split < 1:
        parser.error("--val_split deve estar entre 0 e 1")

    try:
        create_ljspeech_filelists(
            args.ljspeech_dir,
            args.output_dir,
            args.val_split,
            args.seed
        )
    except Exception as e:
        print(f"\n✗ Erro: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
