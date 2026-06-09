#!/usr/bin/env python3
"""
Script simplificado para verificar descongelamento de parâmetros.
Não depende de checkpoints existentes.
"""

import yaml
import torch
from pathlib import Path

from lightning_modules.cleanunet_xvector_stage1_module import CleanUNet2Stage1Module


def create_minimal_config():
    """Cria uma configuração mínima para teste."""
    return {
        'model': {
            'xvector_dim': 512,
            'conditioning_type': 'film',
            'xvector_cache_enabled': False,
            'cleanunet_params': {
                'channels_H': 64,
                'max_H': 768,
                'encoder_n_layers': 8
            },
            'cleanspecnet_params': {}
        },
        'losses': {
            'weight_waveform': 10.0,
            'weight_spec': 5.0,
            'weight_phase': 5.0,
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
        'latents_dir': 'stored_latents_stage1'
    }


def main():
    """Função principal."""
    print("=" * 80)
    print("VERIFICAÇÃO SIMPLIFICADA: Descongelamento de Parâmetros")
    print("=" * 80)

    # Criar configuração mínima
    config = create_minimal_config()

    print("\nCriando módulo Stage-1...")
    module = CleanUNet2Stage1Module(config)

    print("\nConfigurando otimizador (aciona descongelamento)...")
    optimizer = module.configure_optimizers()

    print("\n" + "=" * 80)
    print("ANÁLISE DE PARÂMETROS")
    print("=" * 80)

    # Categorizar parâmetros
    frozen_xvector = []
    frozen_other = []
    trainable_xvector = []
    trainable_other = []

    for name, param in module.named_parameters():
        is_xvector = 'xvector_extractor' in name
        is_trainable = param.requires_grad

        if is_xvector and is_trainable:
            trainable_xvector.append(name)
        elif is_xvector and not is_trainable:
            frozen_xvector.append(name)
        elif not is_xvector and is_trainable:
            trainable_other.append(name)
        elif not is_xvector and not is_trainable:
            frozen_other.append(name)

    # Contadores
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n📊 Estatísticas:")
    print(f"   Total de parâmetros: {total_params:,}")
    print(f"   Parâmetros treináveis: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"   Parâmetros congelados: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")

    print(f"\n✅ Módulos Treináveis (exceto X-Vector): {len(trainable_other)}")
    if trainable_other:
        print("   Exemplos:")
        for name in trainable_other[:5]:
            print(f"     - {name}")
        if len(trainable_other) > 5:
            print(f"     ... e mais {len(trainable_other) - 5}")

    print(f"\n❄️  X-Vector Extractor (Congelado): {len(frozen_xvector)}")
    if frozen_xvector:
        print("   Exemplos:")
        for name in frozen_xvector[:5]:
            print(f"     - {name}")
        if len(frozen_xvector) > 5:
            print(f"     ... e mais {len(frozen_xvector) - 5}")

    # Verificar problemas
    print("\n" + "=" * 80)
    print("VERIFICAÇÃO DE CONFORMIDADE")
    print("=" * 80)

    problems = []

    if trainable_xvector:
        problems.append(f"❌ X-Vector extractor tem {len(trainable_xvector)} parâmetros treináveis (deveria estar congelado)")
        print(f"\n{problems[-1]}:")
        for name in trainable_xvector:
            print(f"     - {name}")

    if frozen_other:
        problems.append(f"❌ {len(frozen_other)} parâmetros fora do X-Vector estão congelados (deveriam estar treináveis)")
        print(f"\n{problems[-1]}:")
        for name in frozen_other:
            print(f"     - {name}")

    if not problems:
        print("\n✅ PERFEITO! Configuração está correta:")
        print("   ✅ Todos os módulos principais: TREINÁVEIS")
        print("   ✅ X-Vector extractor: CONGELADO")
        print("   ✅ CleanUNet: TREINÁVEL")
        print("   ✅ CleanSpecNet: TREINÁVEL")
        print("   ✅ Integration Block: TREINÁVEL")
        print("   ✅ Conditioner: TREINÁVEL")
        print("   ✅ SpecUpsampler: TREINÁVEL")

    print("\n" + "=" * 80)

    # Verificar módulos específicos
    print("\nVERIFICAÇÃO DETALHADA POR MÓDULO:")
    print("=" * 80)

    modules_to_check = [
        ('CleanUNet', 'model.clean_unet'),
        ('CleanSpecNet', 'model.clean_spec_net'),
        ('Integration Block', 'model.integration_block'),
        ('Conditioner', 'model.conditioner'),
        ('SpecUpsampler', 'model.spec_upsampler'),
        ('X-Vector Extractor', 'model.xvector_extractor')
    ]

    for module_name, prefix in modules_to_check:
        module_params = [name for name, _ in module.named_parameters() if name.startswith(prefix)]
        trainable = sum(1 for name, param in module.named_parameters()
                       if name.startswith(prefix) and param.requires_grad)
        frozen = len(module_params) - trainable

        if trainable > 0:
            status = "✅ TREINÁVEL"
        else:
            status = "❄️  CONGELADO"

        print(f"\n{module_name}: {status}")
        print(f"   Parâmetros: {len(module_params)} | Treináveis: {trainable} | Congelados: {frozen}")

    print("\n" + "=" * 80)

    if not problems:
        print("\n🎉 SUCESSO COMPLETO! Todos os módulos estão configurados corretamente.")
        return True
    else:
        print(f"\n⚠️  {len(problems)} problema(s) encontrado(s).")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
