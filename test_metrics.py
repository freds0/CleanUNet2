"""
Script para testar métricas de áudio (PESQ, STOI, SI-SDR)

Este script:
1. Carrega um áudio limpo
2. Aplica data augmentation (ruído)
3. Calcula métricas entre áudio limpo e áudio com ruído
4. Verifica se as métricas estão funcionando corretamente

Uso:
    python test_metrics.py
    python test_metrics.py --audio path/to/audio.wav
"""

import torch
import torchaudio
import argparse
from pathlib import Path
import sys

# TorchMetrics
from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
    ScaleInvariantSignalNoiseRatio
)

# Augmentation
try:
    from torch_audiomentations import Compose, AddColoredNoise, Gain
    AUGMENTATION_AVAILABLE = True
except ImportError:
    print("[WARNING] torch-audiomentations not installed. Install with: pip install torch-audiomentations")
    AUGMENTATION_AVAILABLE = False


def load_audio(audio_path, target_sr=16000):
    """
    Carrega áudio e reamostra se necessário.

    Args:
        audio_path (str): Caminho para o arquivo de áudio
        target_sr (int): Taxa de amostragem desejada

    Returns:
        tuple: (waveform, sample_rate)
    """
    print(f"\n[Load] Carregando áudio: {audio_path}")
    waveform, sample_rate = torchaudio.load(audio_path)

    print(f"  - Shape original: {waveform.shape}")
    print(f"  - Sample rate: {sample_rate} Hz")

    # Converter para mono se necessário
    if waveform.shape[0] > 1:
        print(f"  - Convertendo para mono...")
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resampling se necessário
    if sample_rate != target_sr:
        print(f"  - Reamostrando de {sample_rate} Hz para {target_sr} Hz...")
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
        sample_rate = target_sr

    print(f"  - Shape final: {waveform.shape}")
    print(f"  - Duração: {waveform.shape[1] / sample_rate:.2f} segundos")

    return waveform, sample_rate


def apply_augmentation(waveform, sample_rate):
    """
    Aplica augmentações ao áudio limpo.

    Args:
        waveform (torch.Tensor): Áudio limpo (1, samples)
        sample_rate (int): Taxa de amostragem

    Returns:
        torch.Tensor: Áudio com augmentação aplicada
    """
    if not AUGMENTATION_AVAILABLE:
        print("\n[Augmentation] torch-audiomentations não disponível")
        print("[Augmentation] Criando ruído manualmente...")

        # Criar ruído gaussiano manualmente
        noise_level = 0.05  # SNR ~26 dB
        noise = torch.randn_like(waveform) * noise_level
        augmented = waveform + noise

        return augmented

    print("\n[Augmentation] Aplicando augmentações...")

    # Criar pipeline de augmentação
    augment = Compose([
        AddColoredNoise(
            min_snr_in_db=10.0,
            max_snr_in_db=15.0,
            min_f_decay=-1.0,
            max_f_decay=1.0,
            p=1.0,
            sample_rate=sample_rate
        ),
        Gain(
            min_gain_in_db=-3.0,
            max_gain_in_db=3.0,
            p=1.0
        )
    ])

    # Aplicar augmentação
    # torch-audiomentations espera (batch, channels, samples)
    waveform_batch = waveform.unsqueeze(0)  # (1, 1, samples)
    augmented_batch = augment(waveform_batch, sample_rate=sample_rate)
    augmented = augmented_batch.squeeze(0)  # (1, samples)

    print(f"  - Augmentação aplicada com sucesso")
    print(f"  - Shape: {augmented.shape}")

    return augmented


def calculate_metrics(clean, degraded, sample_rate):
    """
    Calcula métricas entre áudio limpo e degradado.

    Args:
        clean (torch.Tensor): Áudio limpo (referência)
        degraded (torch.Tensor): Áudio degradado (predição)
        sample_rate (int): Taxa de amostragem

    Returns:
        dict: Dicionário com as métricas calculadas
    """
    print("\n" + "=" * 80)
    print("CALCULANDO MÉTRICAS")
    print("=" * 80)

    metrics = {}

    # Garantir que tensores tenham o mesmo tamanho
    min_len = min(clean.shape[-1], degraded.shape[-1])
    clean = clean[..., :min_len]
    degraded = degraded[..., :min_len]

    # Remover dimensão de canal se presente (1, samples) -> (samples,)
    if clean.dim() == 2 and clean.shape[0] == 1:
        clean = clean.squeeze(0)
    if degraded.dim() == 2 and degraded.shape[0] == 1:
        degraded = degraded.squeeze(0)

    # PESQ precisa de batch dimension
    clean_batch = clean.unsqueeze(0)  # (1, samples)
    degraded_batch = degraded.unsqueeze(0)  # (1, samples)

    # -------------------------
    # 1. PESQ
    # -------------------------
    print("\n1. Calculando PESQ...")

    # PESQ suporta apenas 8kHz ou 16kHz
    if sample_rate not in [8000, 16000]:
        print(f"   ⚠️  Sample rate {sample_rate} Hz não suportado pelo PESQ")
        print(f"   Reamostrando para 16 kHz...")

        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        clean_pesq = resampler(clean_batch)
        degraded_pesq = resampler(degraded_batch)
        pesq_sr = 16000
    else:
        clean_pesq = clean_batch
        degraded_pesq = degraded_batch
        pesq_sr = sample_rate

    try:
        pesq_metric = PerceptualEvaluationSpeechQuality(fs=pesq_sr, mode='wb')

        # IMPORTANTE: PESQ(reference, degraded) = PESQ(clean, degraded)
        pesq_score = pesq_metric(degraded_pesq, clean_pesq)
        metrics['pesq'] = pesq_score.item()

        print(f"   ✓ PESQ: {metrics['pesq']:.3f}")
        print(f"     Intervalo esperado: -0.5 a 4.5 (melhor = 4.5)")

        # Validação do resultado
        if metrics['pesq'] < -0.5 or metrics['pesq'] > 4.5:
            print(f"   ⚠️  WARNING: PESQ fora do intervalo esperado!")
        elif metrics['pesq'] > 4.0:
            print(f"   ⚠️  WARNING: PESQ muito alto (> 4.0) - pode indicar erro!")

    except Exception as e:
        print(f"   ✗ Erro ao calcular PESQ: {e}")
        metrics['pesq'] = None

    # -------------------------
    # 2. STOI
    # -------------------------
    print("\n2. Calculando STOI...")

    try:
        stoi_metric = ShortTimeObjectiveIntelligibility(fs=sample_rate, extended=False)

        # STOI também usa (reference, degraded)
        stoi_score = stoi_metric(degraded_batch, clean_batch)
        metrics['stoi'] = stoi_score.item()

        print(f"   ✓ STOI: {metrics['stoi']:.3f}")
        print(f"     Intervalo: 0.0 a 1.0 (melhor = 1.0)")

        # Validação
        if metrics['stoi'] < 0.0 or metrics['stoi'] > 1.0:
            print(f"   ⚠️  WARNING: STOI fora do intervalo esperado!")

    except Exception as e:
        print(f"   ✗ Erro ao calcular STOI: {e}")
        metrics['stoi'] = None

    # -------------------------
    # 3. SI-SDR
    # -------------------------
    print("\n3. Calculando SI-SDR...")

    try:
        sisdr_metric = ScaleInvariantSignalNoiseRatio()

        # SI-SDR também usa (reference, degraded)
        sisdr_score = sisdr_metric(degraded_batch, clean_batch)
        metrics['si_sdr'] = sisdr_score.item()

        print(f"   ✓ SI-SDR: {metrics['si_sdr']:.2f} dB")
        print(f"     Valores típicos: -10 dB a 30 dB (melhor = maior)")

    except Exception as e:
        print(f"   ✗ Erro ao calcular SI-SDR: {e}")
        metrics['si_sdr'] = None

    return metrics


def print_summary(metrics_clean_vs_noisy, metrics_clean_vs_clean=None):
    """
    Imprime resumo das métricas.

    Args:
        metrics_clean_vs_noisy (dict): Métricas entre clean e noisy
        metrics_clean_vs_clean (dict): Métricas entre clean e clean (sanidade)
    """
    print("\n" + "=" * 80)
    print("RESUMO DAS MÉTRICAS")
    print("=" * 80)

    if metrics_clean_vs_clean:
        print("\n📊 Teste de Sanidade (Clean vs Clean):")
        print("-" * 80)
        print(f"  PESQ:   {metrics_clean_vs_clean.get('pesq', 'N/A')}")
        print(f"  STOI:   {metrics_clean_vs_clean.get('stoi', 'N/A')}")
        print(f"  SI-SDR: {metrics_clean_vs_clean.get('si_sdr', 'N/A')} dB")
        print("\n  Valores esperados:")
        print("    - PESQ: ~4.5 (máximo)")
        print("    - STOI: ~1.0 (máximo)")
        print("    - SI-SDR: muito alto (> 40 dB)")

    print("\n📊 Teste Real (Clean vs Noisy):")
    print("-" * 80)
    print(f"  PESQ:   {metrics_clean_vs_noisy.get('pesq', 'N/A')}")
    print(f"  STOI:   {metrics_clean_vs_noisy.get('stoi', 'N/A')}")
    print(f"  SI-SDR: {metrics_clean_vs_noisy.get('si_sdr', 'N/A')} dB")
    print("\n  Valores esperados:")
    print("    - PESQ: 2.0 - 3.5 (depende do nível de ruído)")
    print("    - STOI: 0.6 - 0.9")
    print("    - SI-SDR: 5 - 20 dB")

    # Validação
    print("\n" + "=" * 80)
    print("VALIDAÇÃO")
    print("=" * 80)

    all_ok = True

    # Validar PESQ
    pesq = metrics_clean_vs_noisy.get('pesq')
    if pesq is not None:
        if pesq < -0.5 or pesq > 4.5:
            print("❌ PESQ fora do intervalo válido (-0.5 a 4.5)")
            all_ok = False
        elif pesq > 4.0:
            print("⚠️  PESQ muito alto (> 4.0) - pode indicar que argumentos estão invertidos!")
            all_ok = False
        elif pesq < 1.5:
            print("⚠️  PESQ muito baixo (< 1.5) - ruído muito alto ou problema no cálculo")
        else:
            print("✅ PESQ dentro do intervalo esperado")

    # Validar STOI
    stoi = metrics_clean_vs_noisy.get('stoi')
    if stoi is not None:
        if stoi < 0.0 or stoi > 1.0:
            print("❌ STOI fora do intervalo válido (0.0 a 1.0)")
            all_ok = False
        elif stoi > 0.95:
            print("⚠️  STOI muito alto (> 0.95) - pode indicar problema")
        elif stoi < 0.5:
            print("⚠️  STOI muito baixo (< 0.5) - ruído muito alto")
        else:
            print("✅ STOI dentro do intervalo esperado")

    # Validar SI-SDR
    sisdr = metrics_clean_vs_noisy.get('si_sdr')
    if sisdr is not None:
        if sisdr > 30:
            print("⚠️  SI-SDR muito alto (> 30 dB) - pode indicar problema")
        elif sisdr < 0:
            print("⚠️  SI-SDR negativo - ruído muito alto ou problema no cálculo")
        else:
            print("✅ SI-SDR dentro do intervalo esperado")

    if all_ok:
        print("\n✅ TODAS AS MÉTRICAS PARECEM ESTAR FUNCIONANDO CORRETAMENTE!")
    else:
        print("\n⚠️  ALGUMAS MÉTRICAS PODEM ESTAR COM PROBLEMAS!")

    print("=" * 80)


def save_audio_samples(clean, noisy, output_dir="test_metrics_output"):
    """
    Salva amostras de áudio para inspeção manual.

    Args:
        clean (torch.Tensor): Áudio limpo
        noisy (torch.Tensor): Áudio com ruído
        output_dir (str): Diretório de saída
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\n[Save] Salvando amostras em: {output_path}")

    # Salvar clean
    torchaudio.save(
        str(output_path / "clean.wav"),
        clean,
        sample_rate=16000
    )
    print(f"  ✓ clean.wav")

    # Salvar noisy
    torchaudio.save(
        str(output_path / "noisy.wav"),
        noisy,
        sample_rate=16000
    )
    print(f"  ✓ noisy.wav")

    print(f"\n  Você pode ouvir os arquivos para verificar a qualidade:")
    print(f"    - {output_path / 'clean.wav'}")
    print(f"    - {output_path / 'noisy.wav'}")


def main():
    parser = argparse.ArgumentParser(description='Testar métricas de áudio')
    parser.add_argument('--audio', type=str, default=None,
                        help='Caminho para arquivo de áudio de teste')
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='Taxa de amostragem alvo (default: 16000)')
    parser.add_argument('--save-samples', action='store_true',
                        help='Salvar amostras de áudio')
    parser.add_argument('--sanity-check', action='store_true',
                        help='Fazer teste de sanidade (clean vs clean)')

    args = parser.parse_args()

    print("=" * 80)
    print("TESTE DE MÉTRICAS DE ÁUDIO")
    print("=" * 80)
    print(f"Sample rate alvo: {args.sample_rate} Hz")
    print(f"Augmentação disponível: {'Sim' if AUGMENTATION_AVAILABLE else 'Não (manual)'}")

    # Encontrar arquivo de áudio
    if args.audio:
        audio_path = args.audio
        if not Path(audio_path).exists():
            print(f"\n❌ Erro: Arquivo não encontrado: {audio_path}")
            sys.exit(1)
    else:
        # Procurar arquivo de teste
        print("\n[Search] Procurando arquivo de áudio de teste...")

        possible_paths = [
            "/home/fred/Projetos/DATASETS/VoiceBank-DEMAND-16k/test/clean/p232_001.wav",
            "/home/fred/Projetos/DATASETS/LJSpeech-1.1/wavs/LJ001-0001.wav",
            "./test_audio.wav"
        ]

        audio_path = None
        for path in possible_paths:
            if Path(path).exists():
                audio_path = path
                print(f"  ✓ Encontrado: {path}")
                break

        if not audio_path:
            print("\n❌ Erro: Nenhum arquivo de áudio encontrado!")
            print("\nEspecifique um arquivo com: python test_metrics.py --audio path/to/audio.wav")
            print("\nOu crie um arquivo de teste com:")
            print("  python -c \"import torch, torchaudio; torchaudio.save('test_audio.wav', torch.randn(1, 16000), 16000)\"")
            sys.exit(1)

    # Carregar áudio
    clean, sample_rate = load_audio(audio_path, target_sr=args.sample_rate)

    # Aplicar augmentação
    noisy = apply_augmentation(clean, sample_rate)

    # Salvar amostras se solicitado
    if args.save_samples:
        save_audio_samples(clean, noisy)

    # Calcular métricas: Clean vs Noisy
    metrics_clean_vs_noisy = calculate_metrics(clean, noisy, sample_rate)

    # Teste de sanidade: Clean vs Clean (opcional)
    metrics_clean_vs_clean = None
    if args.sanity_check:
        print("\n" + "=" * 80)
        print("TESTE DE SANIDADE (Clean vs Clean)")
        print("=" * 80)
        print("Calculando métricas entre áudio limpo e ele mesmo...")
        print("Valores esperados: PESQ ~4.5, STOI ~1.0, SI-SDR muito alto")

        metrics_clean_vs_clean = calculate_metrics(clean, clean, sample_rate)

    # Imprimir resumo
    print_summary(metrics_clean_vs_noisy, metrics_clean_vs_clean)

    print("\n✅ Teste concluído!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Teste interrompido pelo usuário.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Erro durante o teste: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
