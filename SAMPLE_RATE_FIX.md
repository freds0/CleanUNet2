# Fix: Audio Sample Rate Mismatch in TensorBoard/WandB Logs

## 🐛 Problema Identificado

Os áudios salvos nos logs do TensorBoard e WandB estavam com taxa de amostragem incorreta, causando reprodução em velocidade errada.

### Exemplo:
- **LJSpeech**: Taxa real é 22050 Hz
- **Áudio carregado**: 16000 Hz (default do dataset)
- **Taxa informada no log**: 22050 Hz (do model config)
- **Resultado**: Áudio tocava 1.38x mais rápido que deveria

---

## 🔍 Análise da Causa Raiz

### Fluxo Incorreto (Antes do Fix):

1. **Arquivo de Configuração** (`train_ljspeech_augmentation.yaml`):
   ```yaml
   model:
     sample_rate: 22050  # LJSpeech usa 22050 Hz

   data:
     # Sem parâmetro sampling_rate!
   ```

2. **Data Module** (`data_module.py`):
   - Não aceitava parâmetro `sampling_rate`
   - Não passava para `MelDataset`

3. **MelDataset** (`spec_dataset.py:305`):
   ```python
   def __init__(self, ..., sampling_rate: int = 16000, ...):
       # Default: 16000 Hz
   ```
   - Áudio carregado em **16000 Hz** (ERRADO para LJSpeech!)

4. **Lightning Module** (`cleanunet_module.py:309`):
   ```python
   sample_rate = int(getattr(self.hparams, "sample_rate", 16000))
   # Lê 22050 do model config

   self.logger.experiment.add_audio(..., sample_rate=sample_rate)
   # Loga áudio de 16000 Hz como se fosse 22050 Hz!
   ```

---

## ✅ Solução Implementada

### 1. Atualizar `CleanUNetDataModule`

**Arquivo**: `lightning_modules/data_module.py`

#### Mudanças:
- Adicionar parâmetro `sampling_rate` no `__init__`
- Passar `sampling_rate` para `MelDataset`
- Atualizar docstring

```python
class CleanUNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        train_list_path: str,
        val_list_path: str = None,
        val_split: float = None,
        batch_size: int = 8,
        num_workers: int = 4,
        persistent_workers: bool = False,
        segment_size: int = None,
        sampling_rate: int = 16000,  # ✅ NOVO PARÂMETRO
        augmentation: dict = None
    ):
        super().__init__()
        self.sampling_rate = sampling_rate  # ✅ ARMAZENA
        # ...

    def setup(self, stage=None):
        dataset_kwargs = {}
        if self.segment_size is not None:
            dataset_kwargs["segment_size"] = self.segment_size
        if self.sampling_rate is not None:
            dataset_kwargs["sampling_rate"] = self.sampling_rate  # ✅ PASSA PARA DATASET
        # ...
```

### 2. Atualizar Arquivos de Configuração

**Arquivos**:
- `configs/train_ljspeech_augmentation.yaml`
- `configs/train_xvector_ljspeech_augmented.yaml`
- `configs/train.yaml`

#### Mudança:
Adicionar `sampling_rate` na seção `data:`:

```yaml
data:
  data_dir: "/path/to/dataset"
  train_list_path: "filelists/train.csv"
  val_list_path: "filelists/val.csv"

  batch_size: 16
  num_workers: 8
  segment_size: 24576
  sampling_rate: 22050  # ✅ DEVE CORRESPONDER AO model.sample_rate
```

---

## 📊 Fluxo Correto (Depois do Fix)

```
Config (YAML)
├── model.sample_rate: 22050    → usado para logging
└── data.sampling_rate: 22050   → usado para carregar áudio
         ↓
    DataModule
         ↓
    MelDataset(sampling_rate=22050)
         ↓
    load_wav(target_sr=22050)     → áudio reamostrado para 22050
         ↓
    Lightning Module
         ↓
    logger.add_audio(..., sample_rate=22050)  ✅ CORRETO!
```

---

## 🎯 Projetos Atualizados

| Projeto | Data Module | Config Files |
|---------|-------------|--------------|
| CleanUNet2-Vanilla | ✅ | ✅ train.yaml |
| CleanUNet2-Data_Augmentation | ✅ | ✅ train_ljspeech_augmentation.yaml |
| CleanUNet2-GAN | ✅ (já usava **kwargs) | ✅ (já tinha sampling_rate) |
| CleanUNet2-Vanilla_xvectors | ✅ | ✅ train_xvector_ljspeech_augmented.yaml |

---

## 🧪 Como Verificar o Fix

### Teste 1: Verificar parâmetro no dataset

```python
from lightning_modules.data_module import CleanUNetDataModule
import yaml

# Carregar config
with open("configs/train_ljspeech_augmentation.yaml") as f:
    config = yaml.safe_load(f)

# Criar data module
data_cfg = config["data"]
dm = CleanUNetDataModule(**data_cfg)

# Verificar
print(f"DataModule sampling_rate: {dm.sampling_rate}")  # Deve ser 22050

# Setup
dm.setup()

# Verificar dataset
print(f"Train dataset sampling_rate: {dm.train_dataset.sampling_rate}")  # Deve ser 22050
```

### Teste 2: Verificar áudio durante treinamento

```python
# Durante on_validation_epoch_end:
sample_rate = int(getattr(self.hparams, "sample_rate", 16000))
print(f"Logging sample_rate: {sample_rate}")

# Verificar se corresponde ao dataset
print(f"Dataset sampling_rate: {self.trainer.datamodule.sampling_rate}")
```

### Teste 3: Reprodução Manual

1. Treinar modelo por algumas épocas
2. Abrir TensorBoard:
   ```bash
   tensorboard --logdir logs/
   ```
3. Ir na aba "Audio"
4. Tocar um sample de áudio
5. Comparar com arquivo original usando `ffplay`:
   ```bash
   ffplay -ar 22050 original.wav
   ```
6. Velocidades devem ser iguais!

---

## ⚠️ IMPORTANTE: Configuração Consistente

**SEMPRE** mantenha os valores de `sample_rate` consistentes:

```yaml
# ✅ CORRETO
model:
  sample_rate: 22050

data:
  sampling_rate: 22050  # Igual ao model!
```

```yaml
# ❌ ERRADO - Taxas diferentes!
model:
  sample_rate: 22050

data:
  sampling_rate: 16000  # Diferente do model!
```

---

## 📝 Recomendações por Dataset

### LJSpeech
```yaml
model:
  sample_rate: 22050

data:
  sampling_rate: 22050
  segment_size: 24576  # ~1.1 seg a 22050 Hz
```

### VoiceBank-DEMAND
```yaml
model:
  sample_rate: 16000

data:
  sampling_rate: 16000
  segment_size: 8192  # ~0.5 seg a 16000 Hz
```

### LibriSpeech
```yaml
model:
  sample_rate: 16000

data:
  sampling_rate: 16000
  segment_size: 16384  # ~1 seg a 16000 Hz
```

---

## 🔧 Retrocompatibilidade

O fix é **retrocompatível**:
- Se `sampling_rate` não for especificado no config, usa default 16000 Hz
- Projetos existentes continuam funcionando
- Novos projetos devem incluir o parâmetro explicitamente

---

## 📚 Arquivos Modificados

### CleanUNet2-Data_Augmentation
- `lightning_modules/data_module.py` (linhas 31-53, 81-85)
- `configs/train_ljspeech_augmentation.yaml` (linha 90)

### CleanUNet2-Vanilla
- `lightning_modules/data_module.py` (linhas 30-52, 80-83)
- `configs/train.yaml` (linha 92)

### CleanUNet2-Vanilla_xvectors
- `lightning_modules/data_module.py` (linhas 31-53, 81-85)
- `configs/train_xvector_ljspeech_augmented.yaml` (linha 117)

### CleanUNet2-GAN
- ✅ Já estava correto (usava **kwargs)

---

## 🎉 Resultado

Após o fix:
- ✅ Áudio carregado na taxa correta
- ✅ Logs mostram taxa correta
- ✅ Reprodução na velocidade correta
- ✅ Consistência entre model e data
- ✅ Funcionamento em todos os projetos
