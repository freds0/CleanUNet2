# Guia Rápido - WavLM no CleanUNet2

## 🚀 Início Rápido

### 1. Teste a Integração (Opcional)

```bash
python test_wavlm_model.py
```

**Resultado esperado**: ✅ Modelo carrega, dimensão 768

### 2. Escolha um Config

**3 opções disponíveis:**

| Config | Dataset | Pooling | Recomendado para |
|--------|---------|---------|------------------|
| `train_wav2vec2_stage1.yaml` | Alcateia | Mean | Treino rápido |
| `train_wav2vec2_stage1_self_attention.yaml` | Alcateia | Self-Attention | Melhor qualidade |
| `train_ljspeech_wav2vec2_stage1.yaml` | LJSpeech | Mean | Dataset LJSpeech |

### 3. Pré-extraia Embeddings

```bash
# Escolha um config
python extract_wav2vec2_embeddings.py --config configs/train_wav2vec2_stage1.yaml

# Tempo estimado: ~15 min para 10k arquivos (GPU)
```

### 4. Treine Stage-1

```bash
python train_xvector.py --config configs/train_wav2vec2_stage1.yaml --stage stage1
```

### 5. Treine Stage-2

```bash
python train_xvector.py --config configs/train_wav2vec2_stage2.yaml --stage stage2
```

## 📋 Comandos Completos

### Opção A: Mean Pooling (Mais Rápido)

```bash
# Pré-extração
python extract_wav2vec2_embeddings.py \
    --config configs/train_wav2vec2_stage1.yaml \
    --device cuda

# Stage 1
python train_xvector.py \
    --config configs/train_wav2vec2_stage1.yaml \
    --stage stage1

# Stage 2
python train_xvector.py \
    --config configs/train_wav2vec2_stage2.yaml \
    --stage stage2
```

### Opção B: Self-Attention (Melhor Qualidade)

```bash
# Pré-extração
python extract_wav2vec2_embeddings.py \
    --config configs/train_wav2vec2_stage1_self_attention.yaml \
    --device cuda

# Stage 1
python train_xvector.py \
    --config configs/train_wav2vec2_stage1_self_attention.yaml \
    --stage stage1

# Stage 2
python train_xvector.py \
    --config configs/train_wav2vec2_stage2_self_attention.yaml \
    --stage stage2
```

## 🔍 Validação

### Verificar Configs

```bash
python validate_configs.py
```

**Resultado esperado**:
```
Total configs checked: 6
Total errors: 0
Total warnings: 0
🎉 All configs are valid!
```

### Verificar Modelo

```bash
python test_wavlm_model.py
```

**Resultado esperado**:
```
✓ Model loaded successfully!
✓ Embedding dimension: 768
✓ Shape matches expected!
```

## 📊 Estrutura de Diretórios

Após extração e treino:

```
CleanUNet2-WavLM/
├── wav2vec2_embeddings/          # Embeddings pré-extraídos
│   ├── *.pt                       # ~3KB cada
│   └── metadata.yaml
│
├── pretrained_models/
│   └── wavlm/                     # Modelo WavLM baixado
│
├── logs/
│   ├── wav2vec2_stage1_checkpoints/     # Checkpoints Stage-1
│   └── wav2vec2_stage2_checkpoints/     # Checkpoints Stage-2
│
└── stored_latents_wav2vec2_stage1/      # Latentes para Stage-2
    └── val_batch_*.pt
```

## 🐛 Troubleshooting

### Problema: "Model not found"

```bash
# Solução: Baixar modelo manualmente
python -c "
from transformers import WavLMModel
model = WavLMModel.from_pretrained(
    'microsoft/wavlm-base',
    cache_dir='pretrained_models/wavlm',
    use_safetensors=True
)
print('Model downloaded!')
"
```

### Problema: "Embedding dimension mismatch"

**Causa**: Usando checkpoint antigo (Wav2Vec2 1024)

**Solução**: Re-treinar do zero com WavLM
```bash
# Remover checkpoints antigos
rm -rf logs/wav2vec2_stage1_checkpoints/*
rm -rf logs/wav2vec2_stage2_checkpoints/*

# Re-treinar
python extract_wav2vec2_embeddings.py --config configs/train_wav2vec2_stage1.yaml
python train_xvector.py --config configs/train_wav2vec2_stage1.yaml --stage stage1
```

### Problema: "Latents directory not found"

**Causa**: Stage-2 não encontra latentes do Stage-1

**Solução**: Verificar diretório
```bash
# Checar se latentes existem
ls -lh stored_latents_wav2vec2_stage1/

# Re-executar Stage-1 validação se necessário
python train_xvector.py --config configs/train_wav2vec2_stage1.yaml --stage stage1
```

### Problema: "Batch size mismatch"

**Status**: ✅ JÁ CORRIGIDO

Se ainda ocorrer:
```bash
# Verificar que tem a correção
grep -A5 "Handle batch size mismatch" lightning_modules/cleanunet_xvector_stage2_module.py
```

## 💡 Dicas

### Performance

1. **GPU recomendada**: CUDA para extração de embeddings (10x mais rápido)
2. **Pre-extração**: Sempre pré-extraia embeddings (não use on-the-fly)
3. **Batch size**: Ajuste conforme GPU RAM disponível

### Qualidade

1. **Self-Attention**: Melhor qualidade, mas +8M parâmetros
2. **Mean Pooling**: Mais rápido, boa qualidade
3. **Stage-2**: Essencial para inferência rápida

### Storage

- Embeddings: ~3KB por arquivo
- 10k arquivos: ~30 MB
- 100k arquivos: ~300 MB

## 📖 Documentação Completa

Para mais detalhes, consulte:

- `ATUALIZACAO_COMPLETA.md` - Resumo completo
- `CONFIG_UPDATES_WAVLM.md` - Detalhes de configs
- `WAVLM_MIGRATION.md` - Migração técnica
- `BUGFIX_BATCH_SIZE_MISMATCH.md` - Correção de bug

## ✅ Checklist

Antes de treinar:

- [ ] Modelo WavLM testado (`test_wavlm_model.py`)
- [ ] Config validado (`validate_configs.py`)
- [ ] Embeddings pré-extraídos
- [ ] Dataset preparado
- [ ] Paths corretos no config

Durante treino:

- [ ] Stage-1 completo
- [ ] Latentes salvos
- [ ] Melhor checkpoint salvo
- [ ] Métricas registradas

Antes de Stage-2:

- [ ] Latentes de validação existem
- [ ] Stage-1 checkpoint especificado
- [ ] Paths corretos no config Stage-2

## 🎯 Configs Disponíveis

### Stage-1

| Arquivo | Dataset | Pooling | Sample Rate |
|---------|---------|---------|-------------|
| `train_wav2vec2_stage1.yaml` | Alcateia | Mean | 24kHz |
| `train_wav2vec2_stage1_self_attention.yaml` | Alcateia | Self-Attention | 24kHz |
| `train_ljspeech_wav2vec2_stage1.yaml` | LJSpeech | Mean | 22.05kHz |

### Stage-2

| Arquivo | Corresponde a | Latents Dir |
|---------|---------------|-------------|
| `train_wav2vec2_stage2.yaml` | stage1.yaml | `stored_latents_wav2vec2_stage1/` |
| `train_wav2vec2_stage2_self_attention.yaml` | stage1_self_attention.yaml | `stored_latents_wav2vec2_stage1_self_attn/` |
| `train_ljspeech_wav2vec2_stage2.yaml` | ljspeech_stage1.yaml | `stored_latents_ljspeech_wav2vec2_stage1/` |

## 🆘 Ajuda

Se tiver problemas:

1. Execute validações:
   ```bash
   python test_wavlm_model.py
   python validate_configs.py
   ```

2. Verifique logs:
   ```bash
   tail -f logs/wav2vec2_stage1/wav2vec2_stage1/version_0/events.out.tfevents.*
   ```

3. Consulte documentação detalhada nos arquivos `.md`

## 📝 Referência Rápida

### Modelo
```
Nome: microsoft/wavlm-base
Dimensão: 768
Cache: pretrained_models/wavlm/
```

### Embeddings
```
Dir: wav2vec2_embeddings/
Tamanho: ~3KB/arquivo
Formato: PyTorch (.pt)
```

### Checkpoints
```
Stage-1: logs/wav2vec2_stage1_checkpoints/
Stage-2: logs/wav2vec2_stage2_checkpoints/
```

### Latentes
```
Dir: stored_latents_wav2vec2_stage1/
Formato: val_batch_XXXXXX.pt
Uso: Carregados no Stage-2
```

---

**Versão**: 1.0
**Status**: ✅ Pronto para uso
**Última atualização**: 2026-02-19
