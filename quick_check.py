#!/usr/bin/env python3
"""
Script de verificação rápida para testar todas as funcionalidades implementadas.

Executa testes básicos para:
1. Busca recursiva de arquivos de áudio
2. Configuração de validação (split automático)
3. Data augmentation

Uso:
    python quick_check.py
"""

import sys
import os


def print_header(title):
    """Imprime um cabeçalho formatado."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def test_imports():
    """Testa se todos os imports necessários estão disponíveis."""
    print_header("1. Testando Imports")

    required_modules = [
        ('pytorch_lightning', 'PyTorch Lightning'),
        ('torch', 'PyTorch'),
        ('torchaudio', 'TorchAudio'),
        ('torch_audiomentations', 'torch-audiomentations'),
        ('yaml', 'PyYAML'),
    ]

    all_ok = True
    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            print(f"✅ {display_name:25s} OK")
        except ImportError as e:
            print(f"❌ {display_name:25s} FALTANDO")
            print(f"   Instale com: pip install {module_name}")
            all_ok = False

    return all_ok


def test_project_files():
    """Verifica se os arquivos do projeto foram criados/modificados."""
    print_header("2. Verificando Arquivos do Projeto")

    required_files = [
        'augmentation.py',
        'spec_dataset.py',
        'lightning_modules/data_module.py',
        'configs/train.yaml',
        'configs/train_with_augmentation.yaml',
        'configs/train_with_auto_split.yaml',
        'test_augmentation.py',
        'test_validation_config.py',
        'AUGMENTATION_README.md',
        'CHANGELOG_VALIDATION.md',
    ]

    all_ok = True
    for filepath in required_files:
        if os.path.exists(filepath):
            print(f"✅ {filepath}")
        else:
            print(f"❌ {filepath:40s} FALTANDO")
            all_ok = False

    return all_ok


def test_augmentation_code():
    """Testa se o código de augmentation está funcional."""
    print_header("3. Testando Código de Augmentation")

    try:
        from augmentation import get_audio_files_recursively, AudioAugmenter
        print("✅ Imports de augmentation OK")

        # Testar busca recursiva em diretório fictício
        files = get_audio_files_recursively('/tmp/test_dir_nao_existe')
        print(f"✅ Função get_audio_files_recursively() OK (retornou {len(files)} arquivos)")

        # Testar criação de augmenter vazio
        augmenter = AudioAugmenter([], device='cpu')
        print("✅ AudioAugmenter pode ser instanciado")

        return True

    except Exception as e:
        print(f"❌ Erro: {e}")
        return False


def test_data_module():
    """Testa se o DataModule suporta as novas opções."""
    print_header("4. Testando DataModule")

    try:
        from lightning_modules.data_module import CleanUNetDataModule
        print("✅ Import de CleanUNetDataModule OK")

        # Testar se aceita val_split
        try:
            dm = CleanUNetDataModule(
                data_dir="/tmp",
                train_list_path="/tmp/dummy.csv",
                val_split=0.1,
                batch_size=8
            )
            print("✅ DataModule aceita parâmetro val_split")
        except TypeError:
            print("❌ DataModule não aceita parâmetro val_split")
            return False

        # Testar validação de parâmetros
        try:
            dm = CleanUNetDataModule(
                data_dir="/tmp",
                train_list_path="/tmp/dummy.csv",
                batch_size=8
                # Sem val_list_path nem val_split - deve dar erro
            )
            print("❌ DataModule não está validando parâmetros obrigatórios")
            return False
        except ValueError as e:
            if "val_list_path" in str(e) or "val_split" in str(e):
                print("✅ DataModule valida parâmetros corretamente")
            else:
                print(f"⚠️  DataModule deu erro, mas mensagem inesperada: {e}")

        return True

    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_configs():
    """Verifica se as configurações estão corretas."""
    print_header("5. Verificando Arquivos de Configuração")

    import yaml

    configs_to_check = {
        'configs/train.yaml': {
            'should_have_val_comment': True,
            'should_have_augmentation_section': True,
        },
        'configs/train_with_auto_split.yaml': {
            'should_have_val_split': True,
        },
    }

    all_ok = True
    for config_file, checks in configs_to_check.items():
        try:
            with open(config_file, 'r') as f:
                content = f.read()
                config = yaml.safe_load(content)

            print(f"\n📄 {config_file}")

            # Verificar se tem seção data
            if 'data' in config:
                print("  ✅ Seção 'data' encontrada")

                data_cfg = config['data']

                # Verificar val_split ou val_list_path
                has_val_list = 'val_list_path' in data_cfg and data_cfg['val_list_path'] is not None
                has_val_split = 'val_split' in data_cfg and data_cfg['val_split'] is not None

                if has_val_list:
                    print(f"  ✅ Configurado com val_list_path: {data_cfg['val_list_path']}")
                if has_val_split:
                    print(f"  ✅ Configurado com val_split: {data_cfg['val_split']}")

                if not has_val_list and not has_val_split:
                    print("  ⚠️  Nenhuma opção de validação configurada (pode estar em comentário)")

                # Verificar augmentations
                if 'augmentations' in data_cfg:
                    aug = data_cfg['augmentations']
                    if aug is None:
                        print("  ℹ️  Augmentations desabilitadas (null)")
                    elif isinstance(aug, list) and len(aug) == 0:
                        print("  ℹ️  Augmentations desabilitadas (lista vazia)")
                    elif isinstance(aug, list) and len(aug) > 0:
                        print(f"  ✅ Augmentations configuradas ({len(aug)} augmentation(s))")
            else:
                print("  ❌ Seção 'data' não encontrada")
                all_ok = False

        except Exception as e:
            print(f"  ❌ Erro ao ler {config_file}: {e}")
            all_ok = False

    return all_ok


def main():
    """Função principal."""
    print("\n" + "="*70)
    print("  VERIFICAÇÃO RÁPIDA - CleanUNet2 Augmentation & Validation")
    print("="*70)

    results = []

    # Executar testes
    results.append(("Imports", test_imports()))
    results.append(("Arquivos", test_project_files()))
    results.append(("Código Augmentation", test_augmentation_code()))
    results.append(("DataModule", test_data_module()))
    results.append(("Configurações", check_configs()))

    # Resumo
    print_header("RESUMO")

    all_passed = True
    for name, passed in results:
        status = "✅ PASSOU" if passed else "❌ FALHOU"
        print(f"{name:25s} {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("🎉 TODAS AS VERIFICAÇÕES PASSARAM!")
        print("\nPróximos passos:")
        print("1. Configure o diretório de ruído em configs/train.yaml")
        print("2. Escolha o modo de validação (val_list_path ou val_split)")
        print("3. Execute: python test_validation_config.py --config configs/train.yaml")
        print("4. Execute: python train.py --config configs/train.yaml")
    else:
        print("⚠️  ALGUMAS VERIFICAÇÕES FALHARAM")
        print("\nVerifique os erros acima e:")
        print("1. Instale dependências faltantes")
        print("2. Certifique-se que todos os arquivos foram criados")
        print("3. Execute novamente este script")
    print("="*70 + "\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
