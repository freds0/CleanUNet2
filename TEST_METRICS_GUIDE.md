# Guia de Teste de Métricas de Áudio

## Visão Geral

O script [test_metrics.py](test_metrics.py) permite testar e validar o cálculo correto das métricas de áudio (PESQ, STOI, SI-SDR) comparando áudio limpo com áudio degradado (com augmentação).

## Por Que Testar?

- ✅ Verificar se as métricas estão funcionando corretamente
- ✅ Validar que os argumentos estão na ordem correta (reference, degraded)
- ✅ Confirmar que os valores estão dentro dos intervalos esperados
- ✅ Diagnosticar problemas de cálculo

## Uso Básico

### Teste Simples

```bash
python test_metrics.py
```

Procura automaticamente um arquivo de áudio e executa o teste.

### Teste com Arquivo Específico

```bash
python test_metrics.py --audio path/to/audio.wav
```

### Teste Completo (com sanidade)

```bash
python test_metrics.py --sanity-check --save-samples
```

Opções:
- `--sanity-check`: Calcula métricas entre clean e clean (deve dar valores máximos)
- `--save-samples`: Salva amostras de áudio para inspeção manual
- `--sample-rate 24000`: Define taxa de amostragem (padrão: 16000)

## Resultados do Teste

### Exemplo de Saída

```
================================================================================
TESTE DE MÉTRICAS DE ÁUDIO
================================================================================

📊 Teste de Sanidade (Clean vs Clean):
--------------------------------------------------------------------------------
  PESQ:   4.64  ← Próximo do máximo (4.5)
  STOI:   1.00  ← Perfeito
  SI-SDR: 92.80 dB  ← Muito alto (bom)

📊 Teste Real (Clean vs Noisy):
--------------------------------------------------------------------------------
  PESQ:   1.26  ← Baixo (ruído aplicado)
  STOI:   0.89  ← Boa inteligibilidade
  SI-SDR: 14.22 dB  ← Razoável

================================================================================
VALIDAÇÃO
================================================================================
✅ PESQ dentro do intervalo esperado
✅ STOI dentro do intervalo esperado
✅ SI-SDR dentro do intervalo esperado

✅ TODAS AS MÉTRICAS PARECEM ESTAR FUNCIONANDO CORRETAMENTE!
```

## Interpretação dos Resultados

### Teste de Sanidade (Clean vs Clean)

Compara o áudio limpo com ele mesmo. **Valores esperados**:

| Métrica | Valor Esperado | Significado |
|---------|---------------|-------------|
| PESQ | ~4.5 | Máximo da escala |
| STOI | ~1.0 | Inteligibilidade perfeita |
| SI-SDR | > 40 dB | Sinal idêntico |

**⚠️ Alertas:**
- PESQ > 4.5: Pode indicar problema de cálculo
- STOI < 0.95: Pode indicar problema
- SI-SDR < 30 dB: Pode indicar problema

### Teste Real (Clean vs Noisy)

Compara áudio limpo com áudio degradado. **Valores esperados**:

| Métrica | Intervalo Típico | Significado |
|---------|-----------------|-------------|
| PESQ | 2.0 - 3.5 | Qualidade perceptual |
| STOI | 0.6 - 0.9 | Inteligibilidade |
| SI-SDR | 5 - 20 dB | Relação sinal/ruído |

**Interpretação PESQ**:
- 4.0 - 4.5: Excelente (quase sem degradação)
- 3.0 - 4.0: Bom
- 2.0 - 3.0: Aceitável
- 1.0 - 2.0: Ruim (muito ruído)
- < 1.0: Muito ruim

**Interpretação STOI**:
- 0.9 - 1.0: Excelente inteligibilidade
- 0.7 - 0.9: Boa
- 0.5 - 0.7: Aceitável
- < 0.5: Ruim

**Interpretação SI-SDR**:
- > 20 dB: Excelente
- 10 - 20 dB: Bom
- 5 - 10 dB: Aceitável
- < 5 dB: Ruim

## Augmentações Aplicadas

O script aplica as seguintes augmentações:

```python
AddColoredNoise:
  SNR: 10 - 15 dB
  Tipo: Pink/White noise

Gain:
  Variação: -3 dB a +3 dB
```

Essas augmentações simulam condições reais de ruído.

## Problemas Comuns

### PESQ retornando 1.0 sempre

**Problema**: Argumentos invertidos

```python
# ❌ ERRADO
pesq(preds, target)

# ✅ CORRETO
pesq(target, preds)
```

**Solução**: Verificar ordem dos argumentos em todos os Lightning modules.

### PESQ muito alto no teste real (> 4.0)

**Problema**: Possivelmente argumentos invertidos ou augmentação não aplicada

**Solução**:
1. Verificar que augmentação está sendo aplicada
2. Verificar ordem dos argumentos
3. Verificar que está comparando clean vs noisy (não noisy vs noisy)

### STOI fora do intervalo [0, 1]

**Problema**: Erro de cálculo ou dados corrompidos

**Solução**:
1. Verificar formato dos tensores
2. Verificar sample rate
3. Verificar que áudio não está corrompido

### SI-SDR negativo

**Problema**: Ruído extremamente alto ou erro de cálculo

**Solução**:
1. Verificar nível de ruído aplicado
2. Verificar ordem dos argumentos
3. Verificar normalização do áudio

## Arquivos Salvos

Com `--save-samples`, o script salva:

```
test_metrics_output/
├── clean.wav    # Áudio limpo (referência)
└── noisy.wav    # Áudio com ruído aplicado
```

Você pode ouvir esses arquivos para validar que:
- clean.wav está limpo
- noisy.wav tem ruído perceptível
- Ruído não é excessivo

## Uso em CI/CD

Para validação automática:

```bash
# Teste básico
python test_metrics.py

# Teste com verificação de erro
python test_metrics.py --sanity-check || exit 1
```

## Comparação com Correção PESQ

### Antes da Correção

```python
# Código antigo (ERRADO)
val_pesq = self.val_pesq(preds, target)

# Resultado: PESQ sempre ~1.0
```

### Depois da Correção

```python
# Código corrigido
val_pesq = self.val_pesq(target, preds)

# Resultado: PESQ varia conforme qualidade (1.0 - 4.5)
```

## Validação Durante Treinamento

Para verificar métricas durante o treinamento:

1. **Treine por alguns epochs**:
```bash
python train.py --config configs/your_config.yaml
```

2. **Verifique TensorBoard**:
```bash
tensorboard --logdir logs/
```

3. **Procure por**:
   - `val/pesq`: Deve variar entre 2.0 - 4.0
   - `val/stoi`: Deve variar entre 0.7 - 0.95
   - `val/si_sdr`: Deve aumentar durante treinamento

## Troubleshooting

### torch-audiomentations não instalado

```bash
pip install torch-audiomentations
```

O script funciona sem, mas usa ruído gaussiano simples.

### Arquivo de áudio não encontrado

Especifique um arquivo:

```bash
python test_metrics.py --audio path/to/your/audio.wav
```

Ou crie um arquivo de teste:

```bash
python -c "import torch, torchaudio; torchaudio.save('test_audio.wav', torch.randn(1, 16000), 16000)"
python test_metrics.py --audio test_audio.wav
```

### Erro de sample rate

PESQ só suporta 8kHz ou 16kHz. O script reamostra automaticamente.

## Referências

- **PESQ**: ITU-T P.862 (intervalo: -0.5 a 4.5)
- **STOI**: Short-Time Objective Intelligibility (intervalo: 0.0 a 1.0)
- **SI-SDR**: Scale-Invariant Signal-to-Distortion Ratio (dB)

## Documentação Relacionada

- [TorchMetrics Audio](https://torchmetrics.readthedocs.io/en/stable/audio/perceptual_evaluation_speech_quality.html)
- [torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations)

## Status

✅ Script testado e funcionando corretamente
✅ Validação de métricas implementada
✅ Suporte a teste de sanidade
✅ Amostras de áudio salvas para inspeção manual
