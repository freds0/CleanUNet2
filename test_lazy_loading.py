#!/usr/bin/env python3
"""
Script para testar o lazy loading de latents no Stage-2.
Verifica que a inicialização é rápida e não carrega tudo na RAM.
"""

import time
import os
import yaml
from pathlib import Path

from lightning_modules.cleanunet_xvector_stage2_module import CleanUNet2Stage2Module


def get_memory_usage_mb():
    """Retorna o uso de memória do processo atual em MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        # Fallback se psutil não estiver disponível
        return 0.0


def test_lazy_loading():
    """Testa o lazy loading de latents."""
    print("=" * 80)
    print("TESTE: Lazy Loading de Latents no Stage-2")
    print("=" * 80)

    # Configuração mínima
    config = {
        'model': {
            'xvector_dim': 512,
            'conditioning_type': 'addition',
            'cleanunet_params': {},
            'cleanspecnet_params': {}
        },
        'losses': {
            'weight_waveform': 10.0,
            'weight_spec': 5.0,
            'weight_phase': 5.0,
            'gamma_latent': 0.05,
            'stft_config': {
                'fft_sizes': [512, 1024, 2048],
                'hop_sizes': [128, 256, 512],
                'win_lengths': [512, 1024, 2048]
            }
        },
        'audio': {
            'sample_rate': 16000
        },
        'optimizer': {
            'lr': 5e-5,
            'betas': [0.9, 0.999]
        },
        'stage1_checkpoint': 'experiments/exp_xvector_optimized_stage1_lr_decay/checkpoints/last.ckpt',
        'latents_dir': 'experiments/exp_xvector_optimized_stage1_lr_decay/stored_latents_stage1'
    }

    print("\n📊 Medindo uso de memória ANTES da inicialização...")
    mem_before = get_memory_usage_mb()
    print(f"   Memória inicial: {mem_before:.1f} MB")

    print("\n⏱️  Iniciando módulo Stage-2 (com lazy loading)...")
    start_time = time.time()

    try:
        module = CleanUNet2Stage2Module(config)
        init_time = time.time() - start_time

        mem_after = get_memory_usage_mb()
        mem_increase = mem_after - mem_before

        print(f"\n✅ Inicialização concluída em {init_time:.2f} segundos")
        print(f"\n📊 Uso de memória:")
        print(f"   Antes: {mem_before:.1f} MB")
        print(f"   Depois: {mem_after:.1f} MB")
        print(f"   Aumento: {mem_increase:.1f} MB")

        # Verificar que não carregou tudo
        if mem_increase < 5000:  # Menos de 5GB
            print(f"\n✅ SUCESSO: Lazy loading funcionando!")
            print(f"   Aumento de memória: {mem_increase:.1f} MB (esperado < 5000 MB)")
        else:
            print(f"\n⚠️  AVISO: Aumento de memória muito grande: {mem_increase:.1f} MB")
            print(f"   Pode não estar usando lazy loading corretamente")

        # Testar acesso a latents individuais
        print("\n🔍 Testando acesso a latents individuais...")

        test_indices = [0, 1, 100, 500, 1000]
        for idx in test_indices:
            latent = module._get_latent(idx)
            if latent is not None:
                print(f"   ✅ Batch {idx}: Carregado ({latent['fused_latent'].shape})")
            else:
                print(f"   ⚠️  Batch {idx}: Não encontrado")

        # Verificar tamanho do cache
        print(f"\n📦 Cache de latents:")
        print(f"   Tamanho atual: {len(module.latent_cache)}")
        print(f"   Tamanho máximo: {module.latent_cache_size}")

        return True

    except Exception as e:
        print(f"\n❌ ERRO durante inicialização: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Função principal."""
    print("\nVerificando dependências...")

    # Verificar se o diretório de latents existe
    latents_dir = Path('experiments/exp_xvector_optimized_stage1_lr_decay/stored_latents_stage1')
    if not latents_dir.exists():
        print(f"❌ Diretório de latents não encontrado: {latents_dir}")
        print("   Execute o Stage-1 primeiro.")
        return

    # Contar arquivos
    num_files = len(list(latents_dir.glob('val_batch_*.pt')))
    print(f"✅ Encontrado diretório de latents com {num_files} arquivos")

    # Executar teste
    success = test_lazy_loading()

    print("\n" + "=" * 80)
    if success:
        print("🎉 TESTE BEM-SUCEDIDO!")
    else:
        print("❌ TESTE FALHOU")
    print("=" * 80)


if __name__ == "__main__":
    main()
