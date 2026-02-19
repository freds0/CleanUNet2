# 🚀 Início Rápido - WavLM

Guia rápido para treinar CleanUNet2 com WavLM embeddings.

## ⚡ Setup Rápido (5 minutos)

### 1. Teste a Instalação

```bash
python test_wavlm_model.py
```

**Saída esperada**:
```
✓ Model loaded successfully!
✓ Embedding dimension: 768
✓ All tests passed!
```

### 2. Escolha um Dataset/Config

| Config | Dataset | Recomendação |
|--------|---------|--------------|
| `train_wavlm_stage1.yaml` | Alcateia | ⭐ Uso geral |
| `train_wavlm_stage1_self_attention.yaml` | Alcateia | Melhor qualidade |
| `train_ljspeech_wavlm_stage1.yaml` | LJSpeech | Dataset específico |

## 📋 Treino em 3 Passos

### Passo 1: Pré-extrair Embeddings

```bash
python extract_wavlm_embeddings.py --config configs/train_wavlm_stage1.yaml --device cuda
```

**Tempo**: ~15 min para 10k arquivos (GPU) | ~2-3h (CPU)

### Passo 2: Treinar Stage-1

```bash
python train_xvector.py --config configs/train_wavlm_stage1.yaml --stage stage1
```

**Duração**: Depende do dataset e GPU

### Passo 3: Treinar Stage-2

```bash
python train_xvector.py --config configs/train_wavlm_stage2.yaml --stage stage2
```

**Duração**: Geralmente mais rápido que Stage-1

## 🎯 Exemplos Completos

### Exemplo 1: Setup Básico (Mean Pooling)

```bash
# Pré-extração
python extract_wavlm_embeddings.py \
    --config configs/train_wavlm_stage1.yaml \
    --device cuda

# Stage 1
python train_xvector.py \
    --config configs/train_wavlm_stage1.yaml \
    --stage stage1

# Stage 2 (após Stage-1 completo)
python train_xvector.py \
    --config configs/train_wavlm_stage2.yaml \
    --stage stage2
```

### Exemplo 2: Melhor Qualidade (Self-Attention)

```bash
# Pré-extração
python extract_wavlm_embeddings.py \
    --config configs/train_wavlm_stage1_self_attention.yaml \
    --device cuda

# Stage 1
python train_xvector.py \
    --config configs/train_wavlm_stage1_self_attention.yaml \
    --stage stage1

# Stage 2
python train_xvector.py \
    --config configs/train_wavlm_stage2_self_attention.yaml \
    --stage stage2
```

### Exemplo 3: LJSpeech Dataset

```bash
# Pré-extração
python extract_wavlm_embeddings.py \
    --config configs/train_ljspeech_wavlm_stage1.yaml \
    --device cuda

# Stage 1
python train_xvector.py \
    --config configs/train_ljspeech_wavlm_stage1.yaml \
    --stage stage1

# Stage 2
python train_xvector.py \
    --config configs/train_ljspeech_wavlm_stage2.yaml \
    --stage stage2
```

## 📊 O que Esperar

### Durante Pré-extração

```
[Extract] Loading config from: configs/train_wavlm_stage1.yaml
[WavLMExtractor] Loading model: microsoft/wavlm-base
[WavLMExtractor] Embedding dimension: 768
Extracting embeddings: 100%|██████████| 10000/10000 [15:23<00:00, 10.84it/s]
✓ 10000 embeddings extracted
```

### Durante Stage-1

```
Epoch 0: 100%|██████████| 8455/8455 [35:01<00:00,  4.02it/s]
val/pesq: 2.45 | val/stoi: 0.89 | val/si-sdr: 15.2
✓ Best checkpoint saved
```

### Durante Stage-2

```
Epoch 0: 100%|██████████| 8455/8455 [30:45<00:00,  4.58it/s]
val/pesq: 2.48 | val/stoi: 0.90 | val/si-sdr: 15.8
loss_latent: 0.023
✓ Best checkpoint saved
```

## 🔍 Validação

### Verificar Configs

```bash
python validate_configs.py
```

**Esperado**: 6 configs válidos, 0 erros

### Monitorar Treino

```bash
# TensorBoard
tensorboard --logdir logs/wavlm_stage1

# WandB
# Acesse o link fornecido no terminal
```

## 📁 Estrutura Gerada

Após extração e treino:

```
CleanUNet2-WavLM/
├── wavlm_embeddings/                 # Embeddings pré-extraídos (~30MB para 10k)
├── pretrained_models/wavlm/          # Modelo WavLM baixado (~500MB)
├── logs/
│   ├── wavlm_stage1_checkpoints/     # Checkpoints Stage-1
│   └── wavlm_stage2_checkpoints/     # Checkpoints Stage-2
└── stored_latents_wavlm_stage1/      # Latentes para Stage-2 (~5MB)
```

## ⚙️ Opções Comuns

### Forçar Re-extração

```bash
python extract_wavlm_embeddings.py \
    --config configs/train_wavlm_stage1.yaml \
    --force
```

### Usar CPU (sem GPU)

```bash
python extract_wavlm_embeddings.py \
    --config configs/train_wavlm_stage1.yaml \
    --device cpu
```

### Retomar Treino

Descomente no config:
```yaml
# resume_from_checkpoint: "logs/wavlm_stage1_checkpoints/last.ckpt"
```

## 🐛 Problemas Comuns

### "Model not found"

**Solução**: Primeira vez demora (download do modelo)
```bash
# Aguarde download (pode levar 5-10 min)
# Modelo será salvo em: pretrained_models/wavlm/
```

### "CUDA out of memory"

**Solução**: Reduza batch_size no config
```yaml
data:
  batch_size: 32  # Era 64, reduza pela metade
```

### "No files found"

**Solução**: Verifique paths no config
```yaml
data:
  data_dir: "./"  # Verifique este path
  train_list_path: "filelists/alcateia_train.csv"  # E este
```

## 📖 Mais Informações

Para detalhes completos:

- `RENOMEACAO_WAVLM.md` - Mudanças de nomenclatura
- `CONFIG_UPDATES_WAVLM.md` - Detalhes dos configs
- `ATUALIZACAO_COMPLETA.md` - Resumo técnico completo

## ✅ Checklist Pré-Treino

Antes de começar:

- [ ] GPU disponível (recomendado) ou CPU (mais lento)
- [ ] Dataset preparado (arquivos de áudio + filelists)
- [ ] Espaço em disco (~1GB para modelo + embeddings + checkpoints)
- [ ] Config revisado (especialmente paths)

Durante treino:

- [ ] Monitorar métricas (PESQ, STOI, SI-SDR)
- [ ] Verificar uso de GPU/memória
- [ ] Aguardar Stage-1 completar antes de Stage-2

## 🎯 Meta de Qualidade

Valores típicos esperados:

| Métrica | Baseline | Stage-1 | Stage-2 |
|---------|----------|---------|---------|
| PESQ | 1.97 | 2.45 | 2.48 |
| STOI | 0.92 | 0.89 | 0.90 |
| SI-SDR | 8.5 | 15.2 | 15.8 |

*(Valores variam conforme dataset)*

## 🚀 Próximos Passos

Após treino completo:

1. **Inferência**: Use o checkpoint Stage-2 para denoising
2. **Avaliação**: Teste em dados não vistos
3. **Fine-tuning**: Ajuste para seu domínio específico

## 💡 Dicas

- ✓ **Sempre pré-extraia embeddings** (muito mais rápido)
- ✓ **Use GPU para extração** (10x mais rápido que CPU)
- ✓ **Monitore métricas** durante treino
- ✓ **Aguarde Stage-1 completar** antes de iniciar Stage-2
- ✓ **Salve checkpoints regularmente**

---

**Versão**: 2.0
**Status**: ✅ Pronto para uso
**Última atualização**: 2026-02-19
