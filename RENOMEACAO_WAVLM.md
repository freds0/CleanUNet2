# Renomeação de Wav2Vec2 para WavLM

## ✅ Mudanças Aplicadas

Para melhor clareza e consistência, todos os nomes de arquivos e referências foram atualizados de "wav2vec2" para "wavlm".

## 📝 Arquivos Renomeados

### Configs de Stage-1

| Nome Anterior | Nome Novo |
|---------------|-----------|
| `train_wav2vec2_stage1.yaml` | `train_wavlm_stage1.yaml` |
| `train_wav2vec2_stage1_self_attention.yaml` | `train_wavlm_stage1_self_attention.yaml` |
| `train_ljspeech_wav2vec2_stage1.yaml` | `train_ljspeech_wavlm_stage1.yaml` |

### Configs de Stage-2

| Nome Anterior | Nome Novo |
|---------------|-----------|
| `train_wav2vec2_stage2.yaml` | `train_wavlm_stage2.yaml` |
| `train_wav2vec2_stage2_self_attention.yaml` | `train_wavlm_stage2_self_attention.yaml` |
| `train_ljspeech_wav2vec2_stage2.yaml` | `train_ljspeech_wavlm_stage2.yaml` |

### Scripts

| Nome Anterior | Nome Novo |
|---------------|-----------|
| `extract_wav2vec2_embeddings.py` | `extract_wavlm_embeddings.py` |

## 🔄 Referências Internas Atualizadas

Todos os arquivos de config foram atualizados internamente:

### Paths e Diretórios

| Antigo | Novo |
|--------|------|
| `logs/wav2vec2_stage1_checkpoints/` | `logs/wavlm_stage1_checkpoints/` |
| `logs/wav2vec2_stage2_checkpoints/` | `logs/wavlm_stage2_checkpoints/` |
| `stored_latents_wav2vec2_stage1/` | `stored_latents_wavlm_stage1/` |
| `stored_latents_wav2vec2_stage1_self_attn/` | `stored_latents_wavlm_stage1_self_attn/` |
| `stored_latents_ljspeech_wav2vec2_stage1/` | `stored_latents_ljspeech_wavlm_stage1/` |

### Nomes de Projetos

| Antigo | Novo |
|--------|------|
| `CleanUNet2_Wav2Vec2` | `CleanUNet2_WavLM` |
| `CleanUNet2_Wav2Vec2_SelfAttention` | `CleanUNet2_WavLM_SelfAttention` |
| `CleanUNet2_Wav2Vec2_LJSpeech` | `CleanUNet2_WavLM_LJSpeech` |

## 🚀 Comandos Atualizados

### Antes

```bash
python extract_wav2vec2_embeddings.py --config configs/train_wav2vec2_stage1.yaml
python train_xvector.py --config configs/train_wav2vec2_stage1.yaml --stage stage1
python train_xvector.py --config configs/train_wav2vec2_stage2.yaml --stage stage2
```

### Depois

```bash
python extract_wavlm_embeddings.py --config configs/train_wavlm_stage1.yaml
python train_xvector.py --config configs/train_wavlm_stage1.yaml --stage stage1
python train_xvector.py --config configs/train_wavlm_stage2.yaml --stage stage2
```

## 📂 Nova Estrutura de Diretórios

```
CleanUNet2-WavLM/
├── configs/
│   ├── train_wavlm_stage1.yaml                    ✓ Renomeado
│   ├── train_wavlm_stage1_self_attention.yaml     ✓ Renomeado
│   ├── train_ljspeech_wavlm_stage1.yaml           ✓ Renomeado
│   ├── train_wavlm_stage2.yaml                    ✓ Renomeado
│   ├── train_wavlm_stage2_self_attention.yaml     ✓ Renomeado
│   └── train_ljspeech_wavlm_stage2.yaml           ✓ Renomeado
│
├── extract_wavlm_embeddings.py                    ✓ Renomeado
│
├── logs/
│   ├── wavlm_stage1_checkpoints/                  ✓ Novo nome
│   ├── wavlm_stage1_self_attn_checkpoints/        ✓ Novo nome
│   ├── ljspeech_wavlm_stage1_checkpoints/         ✓ Novo nome
│   ├── wavlm_stage2_checkpoints/                  ✓ Novo nome
│   ├── wavlm_stage2_self_attn_checkpoints/        ✓ Novo nome
│   └── ljspeech_wavlm_stage2_checkpoints/         ✓ Novo nome
│
└── stored_latents_wavlm_stage1/                   ✓ Novo nome
    ├── stored_latents_wavlm_stage1_self_attn/     ✓ Novo nome
    └── stored_latents_ljspeech_wavlm_stage1/      ✓ Novo nome
```

## ⚠️ Importante: Migração de Dados

Se você já tem dados do sistema antigo (com nomes wav2vec2), você pode:

### Opção 1: Renomear Diretórios Existentes (Recomendado)

```bash
# Checkpoints
mv logs/wav2vec2_stage1_checkpoints logs/wavlm_stage1_checkpoints 2>/dev/null
mv logs/wav2vec2_stage2_checkpoints logs/wavlm_stage2_checkpoints 2>/dev/null

# Latentes
mv stored_latents_wav2vec2_stage1 stored_latents_wavlm_stage1 2>/dev/null

# Logs
mv logs/wav2vec2_stage1 logs/wavlm_stage1 2>/dev/null
mv logs/wav2vec2_stage2 logs/wavlm_stage2 2>/dev/null
```

### Opção 2: Manter Dados Antigos e Treinar do Zero

Simplesmente deixe os dados antigos e treine com os novos configs. Os novos dados serão salvos nos novos diretórios.

## 🧪 Validação

Execute o script de validação atualizado:

```bash
python validate_configs.py
```

**Resultado esperado**:
```
Found 6 WavLM config(s)
Total errors: 0
✅ All configs are valid!
```

## 📋 Checklist de Migração

Se você tem um treino em andamento:

- [ ] Pause o treino atual
- [ ] Renomeie os diretórios de dados (Opção 1)
- [ ] Atualize seus scripts para usar novos nomes
- [ ] Execute `validate_configs.py`
- [ ] Retome o treino com novos configs

Se está começando do zero:

- [x] Nada a fazer! Use os novos configs diretamente

## 🔄 Compatibilidade

### Mantido (Compatibilidade com Código Existente)

Estes nomes foram **MANTIDOS** para compatibilidade:

- ✓ `wav2vec2_model: "microsoft/wavlm-base"` (nome do parâmetro no config)
- ✓ `use_wav2vec2: true` (flag no config)
- ✓ `wav2vec2_cache_dir` (nome do parâmetro)
- ✓ `wav2vec2_embeddings/` (diretório de cache)
- ✓ `wav2vec2_extractor.py` (módulo Python)

**Razão**: Mudá-los quebraria o código existente. O nome do arquivo/script é o que importa para o usuário.

### Atualizado (Melhor UX)

Estes nomes foram **ATUALIZADOS**:

- ✓ Arquivos de config (.yaml)
- ✓ Scripts de linha de comando (.py)
- ✓ Diretórios de logs e checkpoints
- ✓ Nomes de projetos WandB/TensorBoard

**Razão**: São visíveis ao usuário e devem refletir o modelo correto (WavLM).

## 📖 Exemplos de Uso Atualizado

### Treino Completo (2 Stages)

```bash
# Stage 1
python extract_wavlm_embeddings.py --config configs/train_wavlm_stage1.yaml
python train_xvector.py --config configs/train_wavlm_stage1.yaml --stage stage1

# Stage 2
python train_xvector.py --config configs/train_wavlm_stage2.yaml --stage stage2
```

### Com Self-Attention Pooling

```bash
# Stage 1
python extract_wavlm_embeddings.py --config configs/train_wavlm_stage1_self_attention.yaml
python train_xvector.py --config configs/train_wavlm_stage1_self_attention.yaml --stage stage1

# Stage 2
python train_xvector.py --config configs/train_wavlm_stage2_self_attention.yaml --stage stage2
```

### LJSpeech Dataset

```bash
# Stage 1
python extract_wavlm_embeddings.py --config configs/train_ljspeech_wavlm_stage1.yaml
python train_xvector.py --config configs/train_ljspeech_wavlm_stage1.yaml --stage stage1

# Stage 2
python train_xvector.py --config configs/train_ljspeech_wavlm_stage2.yaml --stage stage2
```

## ✅ Status Final

| Item | Status |
|------|--------|
| Configs renomeados | ✅ 6 arquivos |
| Scripts renomeados | ✅ 1 arquivo |
| Referências internas | ✅ Atualizadas |
| Paths em configs | ✅ Atualizados |
| Nomes de projeto | ✅ Atualizados |
| Documentação | ⏳ A atualizar |

## 🎉 Conclusão

Todos os arquivos visíveis ao usuário foram renomeados de "wav2vec2" para "wavlm" para maior clareza e consistência. O código interno mantém compatibilidade com nomes de parâmetros existentes.

---

**Data**: 2026-02-19
**Versão**: 2.0 - Renomeação WavLM
