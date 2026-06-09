#!/usr/bin/env python3
"""
Script para testar se todos os módulos estão sendo descongelados corretamente
(exceto o X-Vector extractor que deve permanecer congelado).
"""

import yaml
import torch
from pathlib import Path

# Importar módulos
from lightning_modules.cleanunet_xvector_stage1_module import CleanUNet2Stage1Module
from lightning_modules.cleanunet_xvector_stage2_module import CleanUNet2Stage2Module


def load_config(config_path):
    """Carrega configuração YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def test_stage1_unfreezing(config_path):
    """Testa descongelamento de parâmetros no Stage-1."""
    print("=" * 80)
    print("TESTANDO STAGE-1: Descongelamento de Parâmetros")
    print("=" * 80)

    config = load_config(config_path)
    module = CleanUNet2Stage1Module(config)

    # Configurar otimizador (isso aciona o descongelamento)
    optimizer = module.configure_optimizers()

    print("\n" + "=" * 80)
    print("VERIFICAÇÃO DE PARÂMETROS STAGE-1")
    print("=" * 80)

    frozen_params = []
    trainable_params = []

    for name, param in module.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)

    print(f"\n✅ Parâmetros Treináveis: {len(trainable_params)}")
    if trainable_params:
        print("\nExemplos (primeiros 10):")
        for name in trainable_params[:10]:
            print(f"  - {name}")
        if len(trainable_params) > 10:
            print(f"  ... e mais {len(trainable_params) - 10} parâmetros")

    print(f"\n❄️  Parâmetros Congelados: {len(frozen_params)}")
    if frozen_params:
        print("\nLista completa:")
        for name in frozen_params:
            print(f"  - {name}")

    # Verificar que APENAS X-Vector extractor está congelado
    print("\n" + "=" * 80)
    print("VERIFICAÇÃO DE CONFORMIDADE")
    print("=" * 80)

    incorrect_frozen = [p for p in frozen_params if 'xvector_extractor' not in p]
    incorrect_trainable = [p for p in trainable_params if 'xvector_extractor' in p]

    if incorrect_frozen:
        print(f"\n⚠️  ERRO: {len(incorrect_frozen)} parâmetros NÃO deveriam estar congelados:")
        for name in incorrect_frozen:
            print(f"  - {name}")

    if incorrect_trainable:
        print(f"\n⚠️  ERRO: {len(incorrect_trainable)} parâmetros do X-Vector NÃO deveriam estar treináveis:")
        for name in incorrect_trainable:
            print(f"  - {name}")

    if not incorrect_frozen and not incorrect_trainable:
        print("\n✅ SUCESSO: Todos os parâmetros estão configurados corretamente!")
        print("   - X-Vector extractor: CONGELADO ❄️")
        print("   - Todos os outros módulos: TREINÁVEIS ✅")

    print("\n" + "=" * 80)
    return len(incorrect_frozen) == 0 and len(incorrect_trainable) == 0


def test_stage2_unfreezing(config_path):
    """Testa descongelamento de parâmetros no Stage-2."""
    print("\n" + "=" * 80)
    print("TESTANDO STAGE-2: Descongelamento de Parâmetros")
    print("=" * 80)

    config = load_config(config_path)
    module = CleanUNet2Stage2Module(config)

    # Configurar otimizador (isso aciona o descongelamento)
    optimizer = module.configure_optimizers()

    print("\n" + "=" * 80)
    print("VERIFICAÇÃO DE PARÂMETROS STAGE-2")
    print("=" * 80)

    frozen_params = []
    trainable_params = []

    for name, param in module.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)

    print(f"\n✅ Parâmetros Treináveis: {len(trainable_params)}")
    if trainable_params:
        print("\nExemplos (primeiros 10):")
        for name in trainable_params[:10]:
            print(f"  - {name}")
        if len(trainable_params) > 10:
            print(f"  ... e mais {len(trainable_params) - 10} parâmetros")

    print(f"\n❄️  Parâmetros Congelados: {len(frozen_params)}")
    if frozen_params:
        print("\nLista completa:")
        for name in frozen_params:
            print(f"  - {name}")

    # No Stage-2, TODOS os parâmetros devem estar treináveis
    print("\n" + "=" * 80)
    print("VERIFICAÇÃO DE CONFORMIDADE")
    print("=" * 80)

    if frozen_params:
        print(f"\n⚠️  ERRO: {len(frozen_params)} parâmetros NÃO deveriam estar congelados no Stage-2:")
        for name in frozen_params:
            print(f"  - {name}")
        success = False
    else:
        print("\n✅ SUCESSO: TODOS os parâmetros estão treináveis no Stage-2!")
        success = True

    print("\n" + "=" * 80)
    return success


def main():
    """Função principal."""
    # Encontrar arquivos de configuração
    config_dir = Path("configs")

    # Tentar encontrar configs de stage1 e stage2
    stage1_configs = list(config_dir.glob("*stage1*.yaml"))
    stage2_configs = list(config_dir.glob("*stage2*.yaml"))

    if not stage1_configs:
        print("⚠️  Nenhuma configuração Stage-1 encontrada em configs/")
        print("   Usando configuração de exemplo...")
        stage1_config = config_dir / "train_xvector_stage1.yaml"
    else:
        stage1_config = stage1_configs[0]

    if not stage2_configs:
        print("⚠️  Nenhuma configuração Stage-2 encontrada em configs/")
        stage2_config = None
    else:
        stage2_config = stage2_configs[0]

    # Testar Stage-1
    print(f"\nUsando config Stage-1: {stage1_config}")
    try:
        stage1_success = test_stage1_unfreezing(stage1_config)
    except Exception as e:
        print(f"\n❌ ERRO ao testar Stage-1: {e}")
        import traceback
        traceback.print_exc()
        stage1_success = False

    # Testar Stage-2
    if stage2_config:
        print(f"\nUsando config Stage-2: {stage2_config}")
        try:
            stage2_success = test_stage2_unfreezing(stage2_config)
        except Exception as e:
            print(f"\n❌ ERRO ao testar Stage-2: {e}")
            import traceback
            traceback.print_exc()
            stage2_success = False
    else:
        print("\n⚠️  Pulando teste Stage-2 (nenhuma configuração encontrada)")
        stage2_success = True

    # Resumo final
    print("\n" + "=" * 80)
    print("RESUMO FINAL")
    print("=" * 80)

    if stage1_success:
        print("✅ Stage-1: OK")
    else:
        print("❌ Stage-1: FALHOU")

    if stage2_config:
        if stage2_success:
            print("✅ Stage-2: OK")
        else:
            print("❌ Stage-2: FALHOU")

    if stage1_success and stage2_success:
        print("\n🎉 TODOS OS TESTES PASSARAM!")
    else:
        print("\n⚠️  ALGUNS TESTES FALHARAM")

    print("=" * 80)


if __name__ == "__main__":
    main()
