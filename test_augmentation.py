#!/usr/bin/env python3
"""
Script para testar a funcionalidade de busca recursiva de arquivos de áudio
e a configuração de data augmentation.

Uso:
    # Testar busca recursiva
    python test_augmentation.py --noise_dir /path/to/noise/directory

    # Testar configuração
    python test_augmentation.py --config configs/train.yaml

    # Aplicar augmentation e salvar os 10 primeiros arquivos
    python test_augmentation.py --config configs/train.yaml --input_dir /path/to/clean/audio --output_dir output_test
"""

import argparse
import yaml
import os
from pathlib import Path
from augmentation import get_audio_files_recursively, AudioAugmenter
import torchaudio
import soundfile as sf
from tqdm import tqdm


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
    # Suporta duas estruturas:
    # 1. data.augmentations: [...] (estrutura antiga/simplificada)
    # 2. data.augmentation.augmentations: [...] (estrutura nova com enabled/mode)
    data_cfg = config.get('data', {})

    # Tentar estrutura nova primeiro (augmentation como dict)
    augmentation_cfg = data_cfg.get('augmentation', {})
    if isinstance(augmentation_cfg, dict):
        enabled = augmentation_cfg.get('enabled', False)
        mode = augmentation_cfg.get('mode', 'two_folders')
        augmentations = augmentation_cfg.get('augmentations', [])
    else:
        # Estrutura antiga (augmentations direto)
        augmentations = data_cfg.get('augmentations', [])
        enabled = bool(augmentations)
        mode = 'unknown'

    sample_rate = config.get('model', {}).get('sample_rate', 16000)

    if not augmentations or augmentations is None:
        if augmentation_cfg and not enabled:
            print("⚠️  Augmentation está configurada mas desabilitada (enabled: false)")
        else:
            print("⚠️  Nenhuma augmentation configurada")
            print("Para habilitar augmentations, adicione uma seção 'augmentation' em 'data'")
        return

    if not enabled:
        print("⚠️  Augmentation está configurada mas desabilitada (enabled: false)")
        print(f"   Para habilitar, defina 'enabled: true' na seção 'augmentation'")
        return

    print(f"✅ Augmentation habilitada!")
    print(f"   Taxa de amostragem alvo: {sample_rate} Hz")
    print(f"   Modo: {mode}")
    print(f"   Número de augmentations: {len(augmentations)}\n")

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
        return augmenter, sample_rate
    except Exception as e:
        print(f"❌ Erro ao criar AudioAugmenter: {e}")
        return None, sample_rate


def apply_augmentation_to_files(config_path, input_dir, output_dir, num_files=10):
    """
    Aplica augmentation aos primeiros N arquivos de áudio e salva em disco.

    Args:
        config_path: Caminho para o arquivo de configuração YAML
        input_dir: Diretório com arquivos de áudio limpos
        output_dir: Diretório onde salvar os arquivos augmentados
        num_files: Número de arquivos a processar (padrão: 10)
    """
    print(f"\n{'='*60}")
    print(f"Aplicando Augmentation e Salvando Arquivos")
    print(f"{'='*60}\n")

    # Carregar configuração
    if not os.path.exists(config_path):
        print(f"❌ ERRO: Arquivo de configuração não existe: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extrair configuração de augmentations
    data_cfg = config.get('data', {})
    augmentation_cfg = data_cfg.get('augmentation', {})

    if isinstance(augmentation_cfg, dict):
        enabled = augmentation_cfg.get('enabled', False)
        mode = augmentation_cfg.get('mode', 'two_folders')
        augmentations = augmentation_cfg.get('augmentations', [])
    else:
        augmentations = data_cfg.get('augmentations', [])
        enabled = bool(augmentations)
        mode = 'unknown'

    sample_rate = config.get('model', {}).get('sample_rate', 16000)

    if not enabled or not augmentations:
        print("❌ Augmentation não está habilitada ou não está configurada")
        return

    print(f"Configuração:")
    print(f"  - Config: {config_path}")
    print(f"  - Input dir: {input_dir}")
    print(f"  - Output dir: {output_dir}")
    print(f"  - Sample rate: {sample_rate} Hz")
    print(f"  - Modo: {mode}")
    print(f"  - Número de augmentations: {len(augmentations)}\n")

    # Buscar arquivos de áudio no input_dir
    if not os.path.exists(input_dir):
        print(f"❌ ERRO: Diretório de entrada não existe: {input_dir}")
        return

    print(f"Buscando arquivos de áudio em: {input_dir}")
    audio_files = get_audio_files_recursively(input_dir)

    if not audio_files:
        print(f"❌ Nenhum arquivo de áudio encontrado em {input_dir}")
        return

    print(f"✅ Encontrados {len(audio_files)} arquivos de áudio")

    # Limitar ao número especificado
    files_to_process = audio_files[:num_files]
    print(f"Processando os primeiros {len(files_to_process)} arquivos...\n")

    # Criar o diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ Diretório de saída criado/verificado: {output_dir}\n")

    # Criar o augmenter
    try:
        augmenter = AudioAugmenter(augmentations, device='cpu', seed=42)
        print("✅ AudioAugmenter criado com sucesso!\n")
    except Exception as e:
        print(f"❌ Erro ao criar AudioAugmenter: {e}")
        return

    # Processar cada arquivo
    print("Processando arquivos:")
    print("-" * 60)

    for i, audio_path in enumerate(tqdm(files_to_process, desc="Augmentando"), 1):
        try:
            # Carregar áudio
            waveform, sr = torchaudio.load(audio_path)

            # Resamplear se necessário
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
                waveform = resampler(waveform)
                sr = sample_rate

            # Aplicar augmentation
            # AudioAugmenter espera (samples,) e retorna (1, samples)
            waveform_squeezed = waveform.squeeze(0)  # Remove channel dim se houver
            augmented_waveform = augmenter.apply(waveform_squeezed, sr)

            # Preparar para salvar
            augmented_waveform = augmented_waveform.squeeze()  # (samples,)

            # Nome do arquivo de saída
            input_filename = os.path.basename(audio_path)
            name_without_ext, ext = os.path.splitext(input_filename)
            output_filename = f"{name_without_ext}_augmented{ext}"
            output_path = os.path.join(output_dir, output_filename)

            # Salvar usando soundfile (suporta float32)
            augmented_waveform_np = augmented_waveform.cpu().numpy()
            sf.write(output_path, augmented_waveform_np, sr)

            print(f"  {i:2d}. {input_filename:40s} → {output_filename}")

        except Exception as e:
            print(f"  ❌ Erro ao processar {os.path.basename(audio_path)}: {e}")

    print("\n" + "="*60)
    print(f"✅ Processamento concluído!")
    print(f"   Arquivos salvos em: {output_dir}")
    print(f"   Total processado: {len(files_to_process)} arquivos")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Testa funcionalidades de data augmentation e aplica augmentation a arquivos"
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
    parser.add_argument(
        '--input_dir',
        type=str,
        help='Diretório com arquivos de áudio limpo para aplicar augmentation (requer --config e --output_dir)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Diretório onde salvar os arquivos augmentados (requer --config e --input_dir)'
    )
    parser.add_argument(
        '--num_files',
        type=int,
        default=10,
        help='Número de arquivos a processar (padrão: 10)'
    )
    args = parser.parse_args()

    # Validar argumentos
    if not args.noise_dir and not args.config:
        parser.print_help()
        print("\n❌ Erro: Você deve especificar --noise_dir ou --config")
        return

    # Se especificar input_dir ou output_dir, precisa de ambos + config
    if args.input_dir or args.output_dir:
        if not args.config:
            print("\n❌ Erro: --input_dir e --output_dir requerem --config")
            return
        if not args.input_dir or not args.output_dir:
            print("\n❌ Erro: Você deve especificar tanto --input_dir quanto --output_dir")
            return

    # Executar testes
    if args.noise_dir:
        test_recursive_search(args.noise_dir)

    if args.config and not args.input_dir:
        # Apenas testar configuração (sem aplicar augmentation)
        test_augmentation_config(args.config)

    if args.config and args.input_dir and args.output_dir:
        # Aplicar augmentation e salvar arquivos
        apply_augmentation_to_files(args.config, args.input_dir, args.output_dir, args.num_files)

    print("\n" + "="*60)
    print("Teste concluído!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
