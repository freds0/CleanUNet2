#!/usr/bin/env python3
"""
Script para testar as configurações de validação.

Testa:
1. Validação com arquivo separado (val_list_path)
2. Validação com split automático (val_split)

Uso:
    # Testar configuração de arquivo YAML
    python test_validation_config.py --config configs/train.yaml

    # Testar split automático manualmente
    python test_validation_config.py --split 0.1 --train_list filelists/train.csv --data_dir /path/to/data
"""

import argparse
import yaml
import sys
from lightning_modules.data_module import CleanUNetDataModule


def test_config_file(config_path):
    """Testa a configuração de validação de um arquivo YAML."""
    print(f"\n{'='*70}")
    print(f"Testando configuração: {config_path}")
    print(f"{'='*70}\n")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Erro ao carregar configuração: {e}")
        return False

    data_cfg = config.get('data', {})

    # Verificar qual modo está configurado
    has_val_list = data_cfg.get('val_list_path') is not None
    has_val_split = data_cfg.get('val_split') is not None

    print("Configuração detectada:")
    print(f"  - train_list_path: {data_cfg.get('train_list_path')}")
    print(f"  - val_list_path: {data_cfg.get('val_list_path')}")
    print(f"  - val_split: {data_cfg.get('val_split')}")
    print()

    if has_val_list and has_val_split:
        print("⚠️  AVISO: Ambos val_list_path e val_split estão configurados!")
        print("   val_list_path terá prioridade (val_split será ignorado)")
        print()

    if not has_val_list and not has_val_split:
        print("❌ ERRO: Nenhuma configuração de validação encontrada!")
        print("   Configure val_list_path OU val_split")
        return False

    # Tentar instanciar o DataModule
    print("Tentando instanciar CleanUNetDataModule...")
    try:
        data_module = CleanUNetDataModule(**data_cfg)
        print("✅ DataModule instanciado com sucesso!")
        print()

        # Tentar fazer setup
        print("Tentando fazer setup dos datasets...")
        data_module.setup()
        print()

        # Verificar tamanhos
        train_size = len(data_module.train_dataset)
        val_size = len(data_module.val_dataset)

        print(f"✅ Setup concluído com sucesso!")
        print(f"   - Dataset de treino: {train_size} amostras")
        print(f"   - Dataset de validação: {val_size} amostras")
        print(f"   - Proporção val/total: {val_size/(train_size+val_size)*100:.2f}%")

        return True

    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_manual_split(split_ratio, train_list, data_dir):
    """Testa split automático com parâmetros manuais."""
    print(f"\n{'='*70}")
    print(f"Testando split automático: {split_ratio*100:.1f}% para validação")
    print(f"{'='*70}\n")

    print(f"Configuração:")
    print(f"  - data_dir: {data_dir}")
    print(f"  - train_list_path: {train_list}")
    print(f"  - val_split: {split_ratio}")
    print()

    try:
        data_module = CleanUNetDataModule(
            data_dir=data_dir,
            train_list_path=train_list,
            val_split=split_ratio,
            batch_size=8,
            num_workers=0
        )

        print("✅ DataModule instanciado com sucesso!")
        print()

        print("Fazendo setup dos datasets...")
        data_module.setup()
        print()

        train_size = len(data_module.train_dataset)
        val_size = len(data_module.val_dataset)

        print(f"✅ Setup concluído com sucesso!")
        print(f"   - Dataset de treino: {train_size} amostras")
        print(f"   - Dataset de validação: {val_size} amostras")
        print(f"   - Proporção val/total: {val_size/(train_size+val_size)*100:.2f}%")
        print(f"   - Split esperado: {split_ratio*100:.1f}%")

        return True

    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Testa configurações de validação do DataModule"
    )

    # Opção 1: Testar arquivo de configuração
    parser.add_argument(
        '--config',
        type=str,
        help='Caminho para arquivo de configuração YAML'
    )

    # Opção 2: Testar split manual
    parser.add_argument(
        '--split',
        type=float,
        help='Fração para validação (ex: 0.1 = 10%%)'
    )
    parser.add_argument(
        '--train_list',
        type=str,
        help='Caminho para arquivo de lista de treino'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        help='Diretório raiz dos dados'
    )

    args = parser.parse_args()

    success = False

    # Modo 1: Teste de arquivo de configuração
    if args.config:
        success = test_config_file(args.config)

    # Modo 2: Teste de split manual
    elif args.split and args.train_list and args.data_dir:
        if not (0.0 < args.split < 1.0):
            print(f"❌ Erro: split deve estar entre 0.0 e 1.0, recebido: {args.split}")
            sys.exit(1)
        success = test_manual_split(args.split, args.train_list, args.data_dir)

    else:
        parser.print_help()
        print("\n❌ Erro: Especifique --config OU (--split, --train_list, --data_dir)")
        sys.exit(1)

    print(f"\n{'='*70}")
    if success:
        print("✅ TESTE CONCLUÍDO COM SUCESSO!")
    else:
        print("❌ TESTE FALHOU")
    print(f"{'='*70}\n")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
