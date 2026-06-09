# Guia de Learning Rate Schedulers

Este guia explica os diferentes schedulers de learning rate disponíveis nos configs otimizados com LR decay.

## 📁 Arquivos

- `configs/train_xvector_optimized_stage1_lr_decay.yaml` - Stage 1 com LR scheduler
- `configs/train_xvector_optimized_stage2_lr_decay.yaml` - Stage 2 com LR scheduler

## 🎯 Schedulers Disponíveis

### 1. CosineAnnealingWarmRestarts (RECOMENDADO) ⭐

**Quando usar**: Treinamentos longos (500-1000+ épocas) onde você quer explorar diferentes regiões do espaço de loss.

**Como funciona**: 
- Reduz o LR seguindo uma curva cosine
- Reinicia periodicamente o LR para o valor inicial
- Permite escapar de mínimos locais

**Parâmetros**:
```yaml
cosine_annealing_warm_restarts:
  T_0: 50        # Reinicia a cada 50 épocas
  T_mult: 2      # Multiplica período por 2 após cada reinício (50, 100, 200, ...)
  eta_min: 1.0e-07  # LR mínimo antes de reiniciar
```

**Exemplo de comportamento**:
- Épocas 0-50: LR vai de 5e-5 → 1e-7 (curva cosine)
- Época 51: LR volta para 5e-5 (reinício!)
- Épocas 51-150: LR vai de 5e-5 → 1e-7 (100 épocas, devido a T_mult=2)
- Época 151: LR volta para 5e-5 (reinício!)
- Épocas 151-350: LR vai de 5e-5 → 1e-7 (200 épocas)
- E assim por diante...

**Vantagens**:
- ✅ Excelente para treinamentos muito longos
- ✅ Ajuda a escapar de mínimos locais
- ✅ Não precisa de tuning de hiperparâmetros

**Desvantagens**:
- ❌ Pode ser instável se T_0 for muito pequeno
- ❌ Picos de LR podem causar divergência se o modelo já estiver bem treinado

---

### 2. ReduceLROnPlateau

**Quando usar**: Quando você quer reduzir o LR apenas quando o modelo para de melhorar.

**Como funciona**:
- Monitora uma métrica (ex: val_loss)
- Reduz o LR se a métrica não melhorar por N épocas

**Parâmetros**:
```yaml
reduce_on_plateau:
  monitor: val_loss
  mode: min           # 'min' para loss, 'max' para accuracy
  factor: 0.5         # Reduz LR para 50% do valor atual
  patience: 10        # Espera 10 épocas sem melhora
  min_lr: 1.0e-07     # LR mínimo
  verbose: true
```

**Exemplo de comportamento**:
- LR inicial: 5e-5
- Se val_loss não melhorar por 10 épocas → LR = 2.5e-5
- Se val_loss não melhorar por mais 10 épocas → LR = 1.25e-5
- E assim por diante até atingir min_lr

**Vantagens**:
- ✅ Muito estável
- ✅ Adapta-se automaticamente ao progresso do treinamento
- ✅ Seguro para usar

**Desvantagens**:
- ❌ Pode ser conservador demais
- ❌ Depende da qualidade da validação

---

### 3. ExponentialLR

**Quando usar**: Quando você quer uma redução gradual e constante do LR.

**Como funciona**:
- Multiplica o LR por `gamma` a cada época
- LR decresce exponencialmente

**Parâmetros**:
```yaml
exponential:
  gamma: 0.995    # LR multiplicado por 0.995 a cada época
```

**Exemplo de comportamento** (com gamma=0.995):
- Época 0: LR = 5.0e-5
- Época 100: LR ≈ 3.0e-5
- Época 200: LR ≈ 1.8e-5
- Época 500: LR ≈ 4.1e-6
- Época 1000: LR ≈ 3.4e-7

**Vantagens**:
- ✅ Simples e previsível
- ✅ Funciona bem para treinamentos médios (200-500 épocas)

**Desvantagens**:
- ❌ Requer tuning do gamma
- ❌ LR pode ficar muito pequeno muito rápido

---

### 4. CosineAnnealingLR

**Quando usar**: Quando você sabe exatamente quantas épocas vai treinar.

**Como funciona**:
- Reduz o LR seguindo uma curva cosine
- Atinge o LR mínimo na época T_max

**Parâmetros**:
```yaml
cosine_annealing:
  T_max: 1000      # Número total de épocas
  eta_min: 1.0e-07 # LR mínimo
```

**Exemplo de comportamento** (T_max=1000):
- Época 0: LR = 5.0e-5
- Época 250: LR ≈ 2.5e-5
- Época 500: LR ≈ 1.0e-7 (mínimo atingido no meio)
- Época 750: LR ≈ 2.5e-5
- Época 1000: LR = 1.0e-7

**Vantagens**:
- ✅ Redução suave e gradual
- ✅ Funciona muito bem se T_max for bem escolhido

**Desvantagens**:
- ❌ Precisa saber quantas épocas vai treinar
- ❌ Sem reinícios (diferente do CosineAnnealingWarmRestarts)

---

## 🎨 Visualização dos Schedulers

```
CosineAnnealingWarmRestarts:
LR │     ╱╲          ╱╲              ╱╲
   │    ╱  ╲        ╱  ╲            ╱  ╲
   │   ╱    ╲      ╱    ╲          ╱    ╲
   │  ╱      ╲    ╱      ╲        ╱      ╲
   └─╱────────╲──╱────────╲──────╱────────╲──────> Época
     0    50   100   150   250    350

ReduceLROnPlateau:
LR │──────┐      ┐       ┐
   │      │      │       │
   │      └──────┘       │
   │                     └──────
   └────────────────────────────────────> Época

ExponentialLR:
LR │╲
   │ ╲___
   │     ╲___
   │         ╲___
   │             ╲___
   └──────────────────────────────────> Época

CosineAnnealingLR:
LR │    ╱╲
   │   ╱  ╲
   │  ╱    ╲
   │ ╱      ╲
   └╱────────╲────────────────────────> Época
   0         T_max
```

---

## 🚀 Como Usar

### 1. Escolher o scheduler no config

Edite o arquivo `train_xvector_optimized_stage1_lr_decay.yaml`:

```yaml
lr_scheduler:
  type: cosine_annealing_warm_restarts  # Escolha aqui
  
  # Configure os parâmetros do scheduler escolhido
  cosine_annealing_warm_restarts:
    T_0: 50
    T_mult: 2
    eta_min: 1.0e-07
```

### 2. Executar o treinamento

```bash
# Stage 1
python train_xvector.py --config=configs/train_xvector_optimized_stage1_lr_decay.yaml --stage stage1

# Stage 2
python train_xvector.py --config=configs/train_xvector_optimized_stage2_lr_decay.yaml --stage stage2
```

### 3. Monitorar o LR

O learning rate será logado automaticamente no TensorBoard e WandB graças ao callback `LearningRateMonitor`.

**TensorBoard**:
```bash
tensorboard --logdir experiments/exp_xvector_optimized_stage1_lr_decay
```

Procure por: `lr-AdamW` no dashboard

**WandB**:
Acesse: https://wandb.ai/freds0/CleanUNet2_XVector_Optimized
O LR será plotado automaticamente nos gráficos.

---

## 📊 Recomendações por Cenário

### Treinamento Longo (1000+ épocas)
→ **CosineAnnealingWarmRestarts** com T_0=50, T_mult=2

### Treinamento Médio (200-500 épocas)
→ **ExponentialLR** com gamma=0.995 ou **CosineAnnealingLR** com T_max=500

### Treinamento Curto (< 200 épocas)
→ **ReduceLROnPlateau** com patience=10

### Fine-tuning
→ **ReduceLROnPlateau** com patience=5, factor=0.5

### Experimentos / Incerto
→ **CosineAnnealingWarmRestarts** (mais robusto a hiperparâmetros)

---

## ⚠️ Notas Importantes

1. **Early Stopping**: A patience do early stopping foi aumentada para 30 épocas nos configs com LR decay, pois o scheduler pode causar flutuações temporárias na loss.

2. **LR Monitor**: O callback `LearningRateMonitor` está ativo e loga o LR a cada época.

3. **Compatibilidade**: Os configs funcionam tanto com quanto sem LR scheduler. Se não houver `lr_scheduler` no config, o treinamento usa LR fixo.

4. **Warm Restarts**: Se usar `CosineAnnealingWarmRestarts`, é normal ver a loss aumentar momentaneamente após cada reinício. Isso é esperado!

---

## 🔧 Troubleshooting

**Problema**: Loss diverge após reinício do LR
→ **Solução**: Aumente T_0 ou reduza o LR inicial

**Problema**: LR fica muito pequeno muito rápido
→ **Solução**: Para ExponentialLR, aumente gamma (ex: 0.998). Para outros, aumente eta_min

**Problema**: Modelo não melhora mesmo com scheduler
→ **Solução**: Tente ReduceLROnPlateau que é mais conservador

**Problema**: Quero desabilitar o scheduler
→ **Solução**: Comente ou remova a seção `lr_scheduler` do config

---

## 📚 Referências

- [PyTorch LR Schedulers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- [CosineAnnealingWarmRestarts Paper (SGDR)](https://arxiv.org/abs/1608.03983)
- [PyTorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/common/optimization.html#learning-rate-scheduling)
