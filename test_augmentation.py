#!/usr/bin/env python3
"""
Script para testar a funcionalidade de busca recursiva de arquivos de áudio
e a configuração de data augmentation.

Uso:
    python test_augmentation.py --noise_dir /path/to/noise/directory
    python test_augmentation.py --config configs/train.yaml
"""

import argparse
import yaml
import os
from augmentation import get_audio_files_recursively, AudioAugmenter


def test_recursive_search(directory):
    """Testa a busca recursiva de arquivos em um diretório."""
    print(f"\n{'='*60}")
    print(f"Testando busca recursiva em: {directory}")
    print(f"{'='*60}\n")

    if not os.path.exists(directory):
        print(f"❌ ERRO: Diretório não existe: {directory}")
        return

    audio_files = get_audio_files_recursively(directory)

    if not audio_files:
        print(f"⚠️  Nenhum arquivo de áudio encontrado em {directory}")
        print("\nVerifique se o diretório contém arquivos .wav, .flac, .mp3 ou .ogg")
        return

    print(f"✅ Encontrados {len(audio_files)} arquivos de áudio\n")

    # Agrupar por extensão
    extensions = {}
    for f in audio_files:
        ext = os.path.splitext(f)[1]
        extensions[ext] = extensions.get(ext, 0) + 1

    print("Arquivos por extensão:")
    for ext, count in sorted(extensions.items()):
        print(f"  {ext}: {count} arquivo(s)")

    print("\nPrimeiros 10 arquivos:")
    for i, f in enumerate(audio_files[:10], 1):
        rel_path = os.path.relpath(f, directory)
        print(f"  {i}. {rel_path}")

    if len(audio_files) > 10:
        print(f"  ... e mais {len(audio_files) - 10} arquivo(s)")

    return audio_files


def test_augmentation_config(config_path):
    """Testa a configuração de augmentation de um arquivo YAML."""
    print(f"\n{'='*60}")
    print(f"Testando configuração de augmentation: {config_path}")
    print(f"{'='*60}\n")

    if not os.path.exists(config_path):
        print(f"❌ ERRO: Arquivo de configuração não existe: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extrair configuração de augmentations
    augmentations = config.get('data', {}).get('augmentations', [])
    sample_rate = config.get('model', {}).get('sample_rate', 16000)

    if not augmentations or augmentations is None:
        print("⚠️  Nenhuma augmentation configurada")
        print("Para habilitar augmentations, adicione uma seção 'augmentations' em 'data'")
        return

    print(f"Taxa de amostragem alvo: {sample_rate} Hz")
    print(f"Número de augmentations configuradas: {len(augmentations)}\n")

    # Testar cada augmentation
    for i, aug in enumerate(augmentations, 1):
        name = aug.get('name', 'Unknown')
        params = aug.get('params', {})

        print(f"{i}. {name}")
        print(f"   Parâmetros:")
        for key, value in params.items():
            if key in ['background_paths', 'ir_paths']:
                print(f"     {key}: {value}")
                # Testar se é um diretório válido
                if isinstance(value, str):
                    if os.path.isdir(value):
                        files = get_audio_files_recursively(value)
                        print(f"       ✅ Diretório válido com {len(files)} arquivo(s)")
                    elif os.path.isfile(value):
                        print(f"       ✅ Arquivo válido")
                    else:
                        print(f"       ❌ Caminho inválido ou não existe")
                elif isinstance(value, list):
                    total_files = 0
                    for path in value:
                        if os.path.isdir(path):
                            files = get_audio_files_recursively(path)
                            total_files += len(files)
                        elif os.path.isfile(path):
                            total_files += 1
                    print(f"       ✅ {total_files} arquivo(s) encontrados no total")
            else:
                print(f"     {key}: {value}")
        print()

    # Tentar instanciar o AudioAugmenter
    print("Tentando instanciar AudioAugmenter...")
    try:
        augmenter = AudioAugmenter(augmentations, device='cpu', seed=42)
        print("✅ AudioAugmenter criado com sucesso!")
        print(f"   Compose contém {len(augmenter.compose.transforms)} transform(s)")
    except Exception as e:
        print(f"❌ Erro ao criar AudioAugmenter: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Testa funcionalidades de data augmentation"
    )
    parser.add_argument(
        '--noise_dir',
        type=str,
        help='Diretório para testar busca recursiva de arquivos de ruído'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Arquivo de configuração YAML para testar'
    )
    args = parser.parse_args()

    if not args.noise_dir and not args.config:
        parser.print_help()
        print("\n❌ Erro: Você deve especificar --noise_dir ou --config")
        return

    if args.noise_dir:
        test_recursive_search(args.noise_dir)

    if args.config:
        test_augmentation_config(args.config)

    print("\n" + "="*60)
    print("Teste concluído!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
