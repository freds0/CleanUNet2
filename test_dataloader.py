import torch
from lightning_modules.data_module import CleanUNetDataModule  # ajuste o caminho conforme necessário

def main():
    # Parâmetros de teste
    data_dir = "/home/fred/Projetos/DATASETS/VoiceBank-DEMAND-16k/"
    train_list = "filelists/train.csv"
    val_list = "filelists/test.csv"
    batch_size = 2
    num_workers = 0

    # Inicializa o DataModule
    dm = CleanUNetDataModule(
        data_dir=data_dir,
        train_list_path=train_list,
        val_list_path=val_list,
        batch_size=batch_size,
        num_workers=num_workers
    )

    dm.setup()

    # Obtém o DataLoader de treino
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # Itera sobre o primeiro batch de treino
    print("\n▶️ Testando train_dataloader...")
    for i, batch in enumerate(train_loader):
        noisy, noisy_spec, clean, clean_spec, features = batch
        print(f"Batch {i + 1}")
        print("  Noisy audio shape      :", noisy.shape)
        print("  Noisy spectrogram shape:", noisy_spec.shape)
        print("  Clean audio shape      :", clean.shape)
        print("  Clean spectrogram shape:", clean_spec.shape)
        print("  Feature sample type    :", type(features[0]))
        print("  Feature sample shape   :", features.shape)
        break

    # Itera sobre o primeiro batch de validação
    print("\n▶️ Testando val_dataloader...")
    for i, batch in enumerate(val_loader):
        noisy, noisy_spec, clean, clean_spec, features = batch
        print(f"Batch {i + 1}")
        print("  Noisy audio shape      :", noisy.shape)
        print("  Noisy spectrogram shape:", noisy_spec.shape)
        print("  Clean audio shape      :", clean.shape)
        print("  Clean spectrogram shape:", clean_spec.shape)
        print("  Feature sample type    :", type(features[0]))
        print("  Feature sample shape   :", features.shape)
        break

if __name__ == "__main__":
    main()
