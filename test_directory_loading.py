#!/usr/bin/env python3
"""
Script para testar a funcionalidade de carregamento de áudio a partir de diretórios.

Uso:
    python test_directory_loading.py --path /path/to/audio/directory
    python test_directory_loading.py --path filelists/train.csv --data_dir /path/to/dataset
"""

import argparse
import sys
from spec_dataset import get_dataset_filelist


def main():
    parser = argparse.ArgumentParser(description="Testa o carregamento de arquivos de áudio")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Caminho para arquivo CSV ou diretório com arquivos .wav"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Diretório base (usado para caminhos relativos)"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Teste de Carregamento de Dataset")
    print("=" * 80)
    print(f"\nCaminho especificado: {args.path}")
    print(f"Diretório base: {args.data_dir if args.data_dir else '(não especificado)'}")
    print("\n" + "-" * 80)

    try:
        # Tenta carregar a lista de arquivos
        file_pairs = get_dataset_filelist(args.path, args.data_dir)

        print(f"\n✓ Sucesso! Encontrados {len(file_pairs)} pares de arquivos.")
        print("\nPrimeiros 10 pares:")
        print("-" * 80)

        for i, (clean, noisy) in enumerate(file_pairs[:10], 1):
            print(f"{i:3d}. Clean: {clean}")
            print(f"     Noisy: {noisy}")
            print()

        if len(file_pairs) > 10:
            print(f"... e mais {len(file_pairs) - 10} pares.")

        # Estatísticas
        print("\n" + "=" * 80)
        print("Estatísticas:")
        print(f"  Total de pares: {len(file_pairs)}")

        # Verifica se há pares onde clean == noisy (modo clean_only)
        same_files = sum(1 for c, n in file_pairs if c == n)
        if same_files > 0:
            print(f"  Pares idênticos (clean==noisy): {same_files}")
            print(f"  → Detectado modo 'clean_only' - augmentation recomendada")
        else:
            print(f"  Pares diferentes: {len(file_pairs)}")
            print(f"  → Detectado modo 'two_folders' - dados já pareados")

        print("=" * 80)
        print("\n✓ Teste concluído com sucesso!")

    except Exception as e:
        print(f"\n✗ Erro ao carregar dataset:")
        print(f"  {type(e).__name__}: {str(e)}")
        print("\nDicas:")
        print("  - Verifique se o caminho existe")
        print("  - Se for CSV, certifique-se que o formato é 'clean|noisy'")
        print("  - Se for diretório, certifique-se que contém arquivos .wav")
        print("  - Para modo pareado, use subpastas 'clean/' e 'noisy/'")
        sys.exit(1)


if __name__ == "__main__":
    main()
