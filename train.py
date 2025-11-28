# train.py (Corrigido)

import yaml
import argparse
from argparse import Namespace
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Verifique se os caminhos de importação estão corretos
from lightning_modules.cleanunet_module import CleanUNetLightningModule
from lightning_modules.data_module import CleanUNetDataModule

def train(config):
    """
    Função principal de treinamento que usa um dicionário de configuração.
    """
    # Combina os parâmetros para o LightningModule
    hparams_dict = {**config.get('model', {}), **config.get('data', {})}
    hparams = Namespace(**hparams_dict)

    # Instancia os módulos
    data_module = CleanUNetDataModule(**config['data'])
    model = CleanUNetLightningModule(hparams)

    # --- CORREÇÃO AQUI: Instanciação segura dos Callbacks ---
    callbacks = []
    for cb_name, cb_config in config.get('callbacks', {}).items():
        # Pega o nome da classe do config (ex: "pytorch_lightning.callbacks.ModelCheckpoint")
        cb_class_str = cb_config.pop('_target_')
        
        # Usa um if/elif para instanciar a classe correta de forma segura
        if "ModelCheckpoint" in cb_class_str:
            callbacks.append(ModelCheckpoint(**cb_config))
        elif "EarlyStopping" in cb_class_str:
            callbacks.append(EarlyStopping(**cb_config))
        else:
            print(f"Aviso: Callback '{cb_class_str}' não reconhecido e será ignorado.")
    # --- FIM DA CORREÇÃO ---

    # --- CORREÇÃO AQUI: Instanciação segura do Logger ---
    logger_config = config['logger']
    logger_class_str = logger_config.pop('_target_')
    
    logger = None
    if "TensorBoardLogger" in logger_class_str:
        logger = TensorBoardLogger(**logger_config)
    else:
        raise ValueError(f"Logger '{logger_class_str}' não suportado.")
    # --- FIM DA CORREÇÃO ---

    # Instancia o Trainer
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        **config['trainer']
    )

    # Inicia o treinamento (com ou sem retomada de checkpoint)
    ckpt_path = config.get('resume_from_checkpoint', None)
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Caminho para o arquivo de configuração YAML."
    )
    args = parser.parse_args()

    # Carregar o arquivo de configuração YAML
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Adiciona a dica de precisão para Tensor Cores
    torch.set_float32_matmul_precision('medium')

    train(config)