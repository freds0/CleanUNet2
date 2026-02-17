# Self-Attention Pooling para Wav2Vec2

## Visão Geral

Esta implementação substitui o **mean pooling** original por **Self-Attention Pooling** na agregação temporal dos embeddings wav2vec2.

### Motivação

O mean pooling original simplesmente calcula a média sobre todos os timesteps:
```python
embeddings = hidden_states.mean(dim=1)  # (batch, time, 1024) → (batch, 1024)
```

**Problema:** Todas as partes temporais têm o mesmo peso, perdendo informação sobre quais partes do áudio são mais importantes.

**Solução:** Self-Attention Pooling aprende a dar **pesos diferentes** para cada timestep, focando nas partes mais relevantes do áudio.

---

## Arquitetura do Self-Attention Pooling

```
Input: hidden_states (batch, time_steps, 1024)
         ↓
    [Learnable Query Vector] (1, 1, 1024)
         ↓
    Multi-Head Attention (8 heads)
    Query attends to hidden_states
         ↓
    Attention Output (batch, 1, 1024)
         ↓
    Squeeze + LayerNorm
         ↓
Output: pooled (batch, 1024)
```

### Componentes

1. **Learnable Query Vector**: Vetor aprendível que representa "o que procurar" nos embeddings
2. **Multi-Head Attention**: 8 attention heads para capturar diferentes aspectos
3. **Layer Normalization**: Estabilização da saída

---

## Uso

### 1. Wav2Vec2Extractor Standalone

```python
from cleanunet.wav2vec2_extractor import Wav2Vec2Extractor

# Com Self-Attention Pooling (padrão agora)
extractor = Wav2Vec2Extractor(
    model_name="facebook/wav2vec2-xls-r-300m",
    device='cuda',
    layer=-1,
    pooling_method='self_attention',  # Novo padrão
    num_attention_heads=8
)

# Extrair embeddings
waveform = torch.randn(4, 16000 * 2)  # 2 segundos, 16kHz
embeddings = extractor.extract_embeddings(
    waveform,
    sample_rate=16000,
    return_mean=True
)
# Output: (4, 1024) - pooled via self-attention

# Para usar mean pooling (antigo comportamento)
extractor_mean = Wav2Vec2Extractor(
    model_name="facebook/wav2vec2-xls-r-300m",
    device='cuda',
    pooling_method='mean'  # Backward compatibility
)
```

### 2. CleanUNet2WithXVector (Treinamento)

#### Configuração YAML

```yaml
# configs/train_wav2vec2_stage1.yaml

model:
  stage: 'stage1'
  use_xvector: false
  use_wav2vec2: true
  wav2vec2_model: 'facebook/wav2vec2-xls-r-300m'
  use_preextracted_embeddings: false

  # Configurações de Pooling (NOVO)
  wav2vec2_pooling_method: 'self_attention'  # ou 'mean'
  wav2vec2_attention_heads: 8  # número de attention heads

  cleanunet_params:
    channels_H: 64
    # ... outros parâmetros
```

#### Código Python

```python
from cleanunet.cleanunet2_with_xvector import CleanUNet2WithXVector

model = CleanUNet2WithXVector(
    stage='stage1',
    use_wav2vec2=True,
    wav2vec2_model='facebook/wav2vec2-xls-r-300m',
    use_preextracted_embeddings=False,

    # Configuração de pooling
    wav2vec2_pooling_method='self_attention',
    wav2vec2_attention_heads=8,

    cleanunet_params={...}
)
```

---

## Treinabilidade

### Parâmetros Congelados vs Treináveis

```python
# Wav2Vec2 model: CONGELADO (não treina)
for param in model.embedding_extractor.model.parameters():
    param.requires_grad = False

# Self-Attention Pooling: TREINÁVEL (treina!)
for param in model.embedding_extractor.attention_pooling.parameters():
    param.requires_grad = True
```

**Implicação:** Durante Stage 1, apenas o **Self-Attention Pooling** será treinado junto com o resto do modelo CleanUNet2. O Wav2Vec2 permanece congelado.

### Número de Parâmetros

- **Wav2Vec2 (congelado)**: ~300M parâmetros
- **Self-Attention Pooling (treinável)**: ~8M parâmetros (para 8 heads, 1024 dim)

**Overhead:** Mínimo (~2.5% dos parâmetros do wav2vec2)

---

## Comparação: Mean vs Self-Attention Pooling

| Aspecto | Mean Pooling | Self-Attention Pooling |
|---------|--------------|------------------------|
| **Parâmetros** | 0 (sem parâmetros) | ~8M treináveis |
| **Flexibilidade** | Pesos fixos (1/T para todos) | Pesos aprendíveis |
| **Foco temporal** | Uniforme | Adaptativo |
| **Custo computacional** | Mínimo | Baixo (linear em T) |
| **Adequado para** | Baseline simples | Melhor performance |

---

## Testes

Execute o script de teste:

```bash
cd CleanUNet2-wav2vec2
python test_self_attention_pooling.py
```

**Testes realizados:**
1. ✓ SelfAttentionPooling module funciona corretamente
2. ✓ Wav2Vec2Extractor com self-attention pooling
3. ✓ Wav2Vec2Extractor com mean pooling (backward compatibility)
4. ✓ Verificação de treinabilidade (wav2vec2 frozen, attention trainable)

---

## Visualização dos Pesos de Atenção

Para visualizar quais partes do áudio recebem mais atenção:

```python
# Modificar SelfAttentionPooling.forward() temporariamente
attn_output, attn_weights = self.self_attention(
    query=query,
    key=hidden_states,
    value=hidden_states,
    need_weights=True,  # Retornar pesos
    average_attn_weights=True  # Média sobre heads
)

# attn_weights shape: (batch, 1, time_steps)
# Mostra quanto de atenção cada timestep recebe
```

---

## Resultados Esperados

Com Self-Attention Pooling, esperamos:

1. **Melhor preservação de qualidade vocal**: Foco nas partes mais informativas
2. **Maior robustez a ruído**: Downweight de regiões ruidosas
3. **Melhor inteligibilidade**: Atenção a segmentos fonéticos importantes

---

## Backward Compatibility

O código é **totalmente compatível** com versões anteriores:

```python
# Comportamento antigo (mean pooling)
extractor = Wav2Vec2Extractor(
    model_name="facebook/wav2vec2-xls-r-300m",
    pooling_method='mean'  # Explicitamente usar mean pooling
)

# Novo comportamento padrão (self-attention)
extractor = Wav2Vec2Extractor(
    model_name="facebook/wav2vec2-xls-r-300m"
    # pooling_method='self_attention' é o padrão
)
```

---

## Próximos Passos

Para melhorar ainda mais:

1. **Layer Weighting**: Combinar múltiplas layers do wav2vec2
2. **Bidirectional Attention**: Atenção bidirecional sobre a sequência
3. **Hierarchical Pooling**: Pooling hierárquico (chunk-level → utterance-level)
4. **Statistics Pooling**: Adicionar std aos embeddings (como x-vectors)

---

## Referências

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [wav2vec 2.0 (Baevski et al., 2020)](https://arxiv.org/abs/2006.11477)
- [X-vectors (Snyder et al., 2018)](https://www.danielpovey.com/files/2018_odyssey_xvector.pdf) - statistics pooling
