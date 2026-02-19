# Migração de Wav2Vec2 para WavLM

## Resumo das Mudanças

Este documento descreve as mudanças realizadas para adaptar o código de `facebook/wav2vec2-xls-r-300m` para `microsoft/wavlm-base`.

## O que é WavLM?

WavLM (Wav2Vec2 Large Model) é uma versão melhorada do Wav2Vec2 desenvolvida pela Microsoft. Principais características:

- **Melhor representação de fala**: Treinado com técnicas de masked prediction mais avançadas
- **Mais robusto**: Melhor desempenho em tarefas de processamento de fala
- **Mesma API**: Usa a mesma interface do transformers que Wav2Vec2
- **Dimensão diferente**: 768 (vs 1024 do wav2vec2-xls-r-300m)

## Arquivos Modificados

### 1. `cleanunet/wav2vec2_extractor.py`
- ✓ Mudou import de `Wav2Vec2Model` para `WavLMModel`
- ✓ Atualizou modelo padrão para `microsoft/wavlm-base`
- ✓ Atualizou cache_dir para `pretrained_models/wavlm`
- ✓ Atualizou todas as mensagens de log para mencionar WavLM
- ✓ Atualizou docstrings e comentários
- ✓ Dimensão de embedding: detectada automaticamente (768)

### 2. `extract_wav2vec2_embeddings.py`
- ✓ Atualizou docstring para mencionar WavLM
- ✓ Mudou modelo padrão para `microsoft/wavlm-base`
- ✓ Atualizou mensagens de log

### 3. `cleanunet/cleanunet2_with_xvector.py`
- ✓ Atualizou docstring (768 dimensões para WavLM)
- ✓ Mudou modelo padrão para `microsoft/wavlm-base`
- ✓ Atualizou dimensão padrão de embedding: 768 (vs 1024)
- ✓ Atualizou comentários e docstrings dos parâmetros

### 4. Arquivos de Configuração
Todos os configs foram atualizados:

- ✓ `configs/train_wav2vec2_stage1.yaml`
- ✓ `configs/train_wav2vec2_stage1_self_attention.yaml`
- ✓ `configs/train_ljspeech_wav2vec2_stage1.yaml`

Mudanças nos configs:
- `wav2vec2_model: "microsoft/wavlm-base"`
- Comentários atualizados para mencionar WavLM
- Dimensão de embedding atualizada para 768

### 5. Novo Arquivo de Teste
- ✓ `test_wavlm_model.py` - Script para validar a integração

## Comparação: Wav2Vec2 vs WavLM

| Característica | Wav2Vec2-XLS-R-300M | WavLM-Base |
|---------------|---------------------|------------|
| Desenvolvedor | Meta (Facebook) | Microsoft |
| Dimensão | 1024 | 768 |
| Parâmetros | ~300M | ~95M |
| Velocidade | Mais lento | Mais rápido |
| Qualidade | Boa | Melhor para fala |
| API | transformers | transformers (mesma) |

## Como Usar

### 1. Instalar dependências
```bash
pip install transformers torch torchaudio
```

### 2. Testar a integração
```bash
python test_wavlm_model.py
```

### 3. Pré-extrair embeddings
```bash
python extract_wav2vec2_embeddings.py --config configs/train_wav2vec2_stage1.yaml
```

### 4. Treinar
```bash
python train_xvector.py --config configs/train_wav2vec2_stage1.yaml --stage stage1
```

## Compatibilidade

✓ **Totalmente compatível**: Os embeddings WavLM usam a mesma interface que Wav2Vec2

✓ **Nomes de arquivos**: Mantidos os nomes originais (wav2vec2_*) para compatibilidade

✓ **Cache**: Os embeddings são salvos em `wav2vec2_embeddings/` (mesmo diretório)

⚠️ **Dimensão diferente**: 768 em vez de 1024
   - O código detecta automaticamente a dimensão
   - Modelos treinados com Wav2Vec2 (1024) NÃO são compatíveis com WavLM (768)
   - É necessário re-treinar do zero ou fazer fine-tuning

## Benefícios da Mudança

1. **Melhor qualidade**: WavLM tem representações mais robustas para fala
2. **Mais rápido**: Modelo menor (95M vs 300M parâmetros)
3. **Menos memória**: Embeddings menores (768 vs 1024 dimensões)
4. **Mesmo custo de storage**: ~3KB por arquivo (vs ~4KB com Wav2Vec2)

## Verificação

Execute o teste para verificar que tudo está funcionando:

```bash
python test_wavlm_model.py
```

Saída esperada:
```
✓ Model loaded successfully!
✓ Embedding dimension: 768
✓ Shape matches expected!
All tests passed!
```

## Notas Importantes

1. **Re-extração necessária**: Se você já tinha embeddings extraídos com Wav2Vec2, precisa re-extraí-los:
   ```bash
   python extract_wav2vec2_embeddings.py --config your_config.yaml --force
   ```

2. **Modelos pré-treinados**: Modelos treinados com Wav2Vec2 não são compatíveis e precisam ser re-treinados

3. **Nomes mantidos**: Mantivemos os nomes `wav2vec2_*` nos arquivos para evitar quebrar código existente

## Referências

- WavLM Paper: https://arxiv.org/abs/2110.13900
- Hugging Face: https://huggingface.co/microsoft/wavlm-base
- Wav2Vec2 Paper: https://arxiv.org/abs/2006.11477

## Status

✅ **MIGRAÇÃO COMPLETA E TESTADA**

Todos os componentes foram atualizados e testados com sucesso.
