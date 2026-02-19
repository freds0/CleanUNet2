# Atualização dos Configs para WavLM

## Resumo

Todos os arquivos de configuração foram atualizados para usar `microsoft/wavlm-base` no lugar de `facebook/wav2vec2-xls-r-300m`.

## Mudanças Principais

### 1. Modelo
- **Antes**: `wav2vec2_model: "facebook/wav2vec2-xls-r-300m"`
- **Depois**: `wav2vec2_model: "microsoft/wavlm-base"`

### 2. Dimensão de Embedding
- **Antes**: 1024 dimensões (wav2vec2-xls-r-300m)
- **Depois**: 768 dimensões (microsoft/wavlm-base)
- **Nota**: A dimensão é detectada automaticamente pelo código

### 3. Cache Directory
- **Mantido**: `wav2vec2_embeddings/` (mantido para compatibilidade)
- **Modelo salvo**: `pretrained_models/wavlm/`

## Arquivos de Configuração

### Stage 1 Configs (Atualizados)

#### ✅ `train_wav2vec2_stage1.yaml`
- Modelo: `microsoft/wavlm-base`
- Embedding dim: 768 (auto-detectado)
- Sample rate: 24000 Hz
- Dataset: Alcateia
- Cache: `wav2vec2_embeddings/`

#### ✅ `train_wav2vec2_stage1_self_attention.yaml`
- Modelo: `microsoft/wavlm-base`
- Embedding dim: 768 (auto-detectado)
- Pooling: Self-Attention (8 heads)
- Sample rate: 24000 Hz
- Dataset: Alcateia

#### ✅ `train_ljspeech_wav2vec2_stage1.yaml`
- Modelo: `microsoft/wavlm-base`
- Embedding dim: 768 (auto-detectado)
- Sample rate: 22050 Hz
- Dataset: LJSpeech
- Augmentação adaptada para LJSpeech

### Stage 2 Configs (Novos - Criados)

#### ✅ `train_wav2vec2_stage2.yaml` (NOVO)
- Corresponde ao: `train_wav2vec2_stage1.yaml`
- Stage-1 checkpoint: `logs/wav2vec2_stage1_checkpoints/`
- Latents dir: `stored_latents_wav2vec2_stage1/`
- Não usa extractor (Stage-2 replica latentes)
- Gamma latent: 0.05

#### ✅ `train_wav2vec2_stage2_self_attention.yaml` (NOVO)
- Corresponde ao: `train_wav2vec2_stage1_self_attention.yaml`
- Stage-1 checkpoint: `logs/wav2vec2_stage1_self_attn_checkpoints/`
- Latents dir: `stored_latents_wav2vec2_stage1_self_attn/`
- Pooling: Self-Attention (mesmo do Stage-1)

#### ✅ `train_ljspeech_wav2vec2_stage2.yaml` (NOVO)
- Corresponde ao: `train_ljspeech_wav2vec2_stage1.yaml`
- Stage-1 checkpoint: `logs/ljspeech_wav2vec2_stage1_checkpoints/`
- Latents dir: `stored_latents_ljspeech_wav2vec2_stage1/`
- Dataset: LJSpeech

### Configs Não Modificados

Estes configs não foram alterados pois não usam WavLM/Wav2Vec2:

- ❌ `inference.yaml` - Inference genérico
- ❌ `inference_xvector.yaml` - Inference com X-Vectors
- ❌ `meu_config.yaml` - Config customizado
- ❌ `train.yaml` - Treino básico
- ❌ `train_xvector_*.yaml` - Configs de X-Vector

**Nota**: Esses configs mantêm referências a `1024`, mas são para `n_fft` (STFT), não embedding dimension.

## Estrutura de Diretórios

### Stage 1
```
logs/
  wav2vec2_stage1_checkpoints/          # Checkpoints Stage-1
  wav2vec2_stage1_self_attn_checkpoints/  # Checkpoints Stage-1 Self-Attention
  ljspeech_wav2vec2_stage1_checkpoints/   # Checkpoints Stage-1 LJSpeech

stored_latents_wav2vec2_stage1/          # Latentes salvos Stage-1
stored_latents_wav2vec2_stage1_self_attn/  # Latentes Stage-1 Self-Attention
stored_latents_ljspeech_wav2vec2_stage1/   # Latentes Stage-1 LJSpeech

wav2vec2_embeddings/                      # Embeddings pré-extraídos
pretrained_models/wavlm/                  # Modelo WavLM baixado
```

### Stage 2
```
logs/
  wav2vec2_stage2_checkpoints/          # Checkpoints Stage-2
  wav2vec2_stage2_self_attn_checkpoints/  # Checkpoints Stage-2 Self-Attention
  ljspeech_wav2vec2_stage2_checkpoints/   # Checkpoints Stage-2 LJSpeech
```

## Uso dos Configs

### Treino Completo (2 Stages)

#### Opção 1: Dataset Alcateia (Mean Pooling)

```bash
# Stage 1: Treinar com WavLM embeddings
# 1. Pré-extrair embeddings
python extract_wav2vec2_embeddings.py --config configs/train_wav2vec2_stage1.yaml

# 2. Treinar Stage-1
python train_xvector.py --config configs/train_wav2vec2_stage1.yaml --stage stage1

# Stage 2: Replicar latentes sem embeddings
python train_xvector.py --config configs/train_wav2vec2_stage2.yaml --stage stage2
```

#### Opção 2: Dataset Alcateia (Self-Attention Pooling)

```bash
# Stage 1: Treinar com WavLM + Self-Attention
# 1. Pré-extrair embeddings
python extract_wav2vec2_embeddings.py --config configs/train_wav2vec2_stage1_self_attention.yaml

# 2. Treinar Stage-1
python train_xvector.py --config configs/train_wav2vec2_stage1_self_attention.yaml --stage stage1

# Stage 2: Replicar latentes sem embeddings
python train_xvector.py --config configs/train_wav2vec2_stage2_self_attention.yaml --stage stage2
```

#### Opção 3: Dataset LJSpeech

```bash
# Stage 1: Treinar com WavLM (LJSpeech)
# 1. Pré-extrair embeddings
python extract_wav2vec2_embeddings.py --config configs/train_ljspeech_wav2vec2_stage1.yaml

# 2. Treinar Stage-1
python train_xvector.py --config configs/train_ljspeech_wav2vec2_stage1.yaml --stage stage1

# Stage 2: Replicar latentes sem embeddings
python train_xvector.py --config configs/train_ljspeech_wav2vec2_stage2.yaml --stage stage2
```

## Parâmetros Importantes

### Stage 1
- `use_wav2vec2: true` - Usa WavLM embeddings
- `use_xvector: false` - Não usa X-Vectors
- `use_preextracted_embeddings: true` - Usa embeddings pré-extraídos (RECOMENDADO)
- `wav2vec2_cache_dir: "wav2vec2_embeddings"` - Diretório com embeddings
- `latents_dir: "stored_latents_*"` - Onde salvar latentes para Stage-2

### Stage 2
- `use_wav2vec2: true` - Arquitetura inclui integração WavLM
- `use_preextracted_embeddings: false` - Stage-2 não usa embeddings
- `stage1_checkpoint: "logs/.../best.ckpt"` - Checkpoint do Stage-1
- `latents_dir: "stored_latents_*"` - Onde carregar latentes do Stage-1
- `gamma_latent: 0.05` - Peso da loss de replicação de latentes

## Validação

Para verificar que os configs estão corretos:

```bash
# Testar modelo WavLM
python test_wavlm_model.py

# Verificar dimensões nos configs
grep -r "768\|microsoft/wavlm" configs/train_*wav2vec2*.yaml
```

## Troubleshooting

### Erro: "Model dimension mismatch"
- **Causa**: Config tem dimensão antiga (1024)
- **Solução**: Verificar que o config usa `microsoft/wavlm-base` e não menciona 1024 para embeddings

### Erro: "Latents directory not found"
- **Causa**: Stage-2 não encontra latentes do Stage-1
- **Solução**: Verificar que `latents_dir` no Stage-2 corresponde ao usado no Stage-1

### Erro: "Checkpoint incompatible"
- **Causa**: Tentando usar checkpoint Wav2Vec2 (1024) com WavLM (768)
- **Solução**: Re-treinar Stage-1 com WavLM

## Resumo de Mudanças

| Config | Status | Modelo | Embedding Dim |
|--------|--------|--------|---------------|
| train_wav2vec2_stage1.yaml | ✅ Atualizado | microsoft/wavlm-base | 768 |
| train_wav2vec2_stage1_self_attention.yaml | ✅ Atualizado | microsoft/wavlm-base | 768 |
| train_ljspeech_wav2vec2_stage1.yaml | ✅ Atualizado | microsoft/wavlm-base | 768 |
| train_wav2vec2_stage2.yaml | ✅ Criado | microsoft/wavlm-base | 768 |
| train_wav2vec2_stage2_self_attention.yaml | ✅ Criado | microsoft/wavlm-base | 768 |
| train_ljspeech_wav2vec2_stage2.yaml | ✅ Criado | microsoft/wavlm-base | 768 |

## Status

✅ **TODOS OS CONFIGS ATUALIZADOS PARA WAVLM**

Todos os arquivos de configuração estão prontos para uso com o modelo WavLM.
