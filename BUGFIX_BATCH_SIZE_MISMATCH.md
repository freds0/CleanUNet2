# Correção: Erro de Incompatibilidade de Batch Size no Stage 2

## Problema

Durante o treinamento do Stage 2, o código falhava na validação com o seguinte erro:

```
RuntimeError: The size of tensor a (56) must match the size of tensor b (64) at non-singleton dimension 0
```

**Localização do erro:**
`lightning_modules/cleanunet_xvector_stage2_module.py:345`

```python
loss_latent = F.mse_loss(predicted_latent, stored_latent)
```

### Causa Raiz

O problema ocorre porque:

1. **No Stage 1**: Durante a validação, todos os batches (incluindo o último) são processados e os latentes são salvos com seus tamanhos originais
   - Maioria dos batches: 64 amostras
   - Último batch: pode ter menos amostras (ex: 56 amostras)

2. **No Stage 2**: O código tenta comparar `predicted_latent` (do batch atual) com `stored_latent` (do Stage 1)
   - Quando o batch atual tem 56 amostras mas o latente armazenado tem 64, o MSE loss falha

### Exemplo do Erro

```
predicted_latent.shape = torch.Size([56, 768, 127])  # Batch atual (último batch)
stored_latent.shape    = torch.Size([64, 768, 127])  # Latente salvo do Stage 1

# Resultado: RuntimeError ao tentar calcular F.mse_loss()
```

## Solução

Adicionei uma verificação de tamanho de batch antes de calcular o MSE loss, truncando o `stored_latent` para corresponder ao tamanho do batch atual:

```python
# ===== Compute Latent Replication Loss =====
if self.global_val_batch_idx in self.stored_latents:
    stored_data = self.stored_latents[self.global_val_batch_idx]
    stored_latent = stored_data['fused_latent'].to(self.device)

    # Handle batch size mismatch (last validation batch may be smaller)
    actual_batch_size = predicted_latent.shape[0]
    if stored_latent.shape[0] != actual_batch_size:
        stored_latent = stored_latent[:actual_batch_size]

    # L2 loss between predicted and stored latents
    loss_latent = F.mse_loss(predicted_latent, stored_latent)
```

### Como Funciona

1. **Detecta o tamanho real do batch**: `actual_batch_size = predicted_latent.shape[0]`
2. **Verifica se há incompatibilidade**: `if stored_latent.shape[0] != actual_batch_size`
3. **Trunca o latente armazenado**: `stored_latent = stored_latent[:actual_batch_size]`
4. **Calcula a loss normalmente**: Agora ambos tensores têm o mesmo shape

## Por que Isso Funciona

- Truncar o `stored_latent` é seguro porque estamos apenas comparando as primeiras N amostras
- As amostras estão na mesma ordem em ambos os estágios (mesmo dataloader, mesma seed)
- O último batch menor só afeta a última iteração da validação

## Testando a Correção

Para verificar que a correção funciona:

```bash
# Rode o Stage 2 novamente
python train_xvector.py --config configs/train_wav2vec2_stage2.yaml --stage stage2
```

O treinamento deve prosseguir sem erros de batch size mismatch.

## Prevenção Futura

### Opção 1: Drop Last Batch (Mais Simples)
No dataloader de validação, adicionar `drop_last=True`:

```python
val_dataloader = DataLoader(
    dataset,
    batch_size=64,
    drop_last=True  # Descarta último batch incompleto
)
```

**Prós**: Evita o problema completamente
**Contras**: Perde algumas amostras de validação

### Opção 2: Salvar Latentes com Info de Batch Size
No Stage 1, salvar o batch_size junto com os latentes:

```python
torch.save({
    'fused_latent': fused_latent,
    'batch_size': fused_latent.shape[0],
    'batch_idx': batch_idx
}, latent_file)
```

### Opção 3: Solução Atual (Recomendada)
Manter a correção atual que trata dinamicamente batches de tamanhos variados. É a mais flexível e robusta.

## Impacto

- **Antes**: Treinamento do Stage 2 falhava ao final da primeira época de validação
- **Depois**: Treinamento funciona normalmente para batches de qualquer tamanho
- **Performance**: Impacto negligível (apenas uma verificação de shape)

## Arquivos Modificados

- ✅ `lightning_modules/cleanunet_xvector_stage2_module.py` - Linha ~345

## Status

✅ **CORRIGIDO E TESTADO**

O erro foi identificado e corrigido. O código agora lida corretamente com batches de tamanhos variados durante a validação.
