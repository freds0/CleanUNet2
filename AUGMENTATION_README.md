# Data Augmentation e Validação - Guia de Configuração

Este projeto agora suporta:
1. **Data augmentation** on-the-fly durante o treinamento, com busca recursiva de arquivos de ruído e conversão automática de taxa de amostragem
2. **Configuração flexível de validação** com duas opções: arquivo separado ou split automático

---

## 📋 Índice

1. [Data Augmentation](#data-augmentation)
   - Busca Recursiva de Arquivos
   - Conversão Automática de Taxa de Amostragem
   - Augmentations Disponíveis
   - Como Configurar Augmentations
2. [Configuração de Validação](#configuração-de-validação)
   - Opção 1: Arquivo Separado
   - Opção 2: Split Automático
3. [Exemplos de Configuração](#exemplos-de-configuração)
4. [Como Testar](#como-testar)

---

## Data Augmentation

### ✅ Busca Recursiva de Arquivos
- Quando você especifica um diretório em `background_paths` ou `ir_paths`, o sistema busca **recursivamente** todos os arquivos de áudio (.wav, .flac, .mp3, .ogg)
- Você pode especificar um único diretório ou uma lista de diretórios/arquivos

### ✅ Conversão Automática de Taxa de Amostragem
- Os arquivos de ruído são **automaticamente reamostrados** para a taxa de amostragem configurada (16kHz por padrão)
- A conversão é feita **on-the-fly** pela biblioteca `torch_audiomentations`
- Não é necessário pré-processar os arquivos de ruído

### ✅ Augmentations Disponíveis
1. **AddBackgroundNoise**: Adiciona ruído de fundo
2. **ApplyImpulseResponse**: Aplica reverberação de sala
3. **Gain**: Ajusta o ganho do áudio
4. **AddColoredNoise**: Adiciona ruído colorido (pink, brown, etc.)
5. **LowPassFilter**: Filtro passa-baixa
6. **HighPassFilter**: Filtro passa-alta
7. **BandPassFilter**: Filtro passa-faixa

---

## Configuração de Validação

O projeto oferece **duas formas** de configurar o conjunto de validação:

### Opção 1: Arquivo/Lista Separada (Recomendado)

Use esta opção quando você tem arquivos distintos para treino e validação:

```yaml
data:
  train_list_path: "filelists/train.csv"
  val_list_path: "filelists/test.csv"  # Arquivo separado
```

**Vantagens:**
- Controle total sobre quais arquivos vão para validação
- Validação consistente entre experimentos
- Útil quando você já tem um benchmark/test set definido

### Opção 2: Split Automático (Porcentagem)

Use esta opção quando você quer dividir automaticamente o conjunto de treino:

```yaml
data:
  train_list_path: "filelists/train.csv"
  val_split: 0.1  # 10% para validação, 90% para treino
  # NÃO especifique val_list_path neste caso
```

**Vantagens:**
- Simples quando você tem apenas um arquivo de dados
- Fácil experimentar com diferentes proporções (0.1, 0.15, 0.2, etc.)
- Split determinístico (sempre os mesmos arquivos com seed fixo)

**Importante:**
- O split usa seed fixo (42) para garantir reprodutibilidade
- Augmentations são aplicadas **apenas no conjunto de treino**, não na validação
- Você DEVE especificar `val_list_path` OU `val_split`, mas não ambos
- Se ambos forem especificados, `val_list_path` terá prioridade

### Exemplos de Configuração de Validação

#### Exemplo 1: 10% para validação
```yaml
data:
  data_dir: "/datasets/VoiceBank-DEMAND-16k/"
  train_list_path: "filelists/all_data.csv"
  val_split: 0.1  # 10%
```

#### Exemplo 2: 15% para validação
```yaml
data:
  train_list_path: "filelists/train.csv"
  val_split: 0.15  # 15%
```

#### Exemplo 3: Arquivo separado (padrão)
```yaml
data:
  train_list_path: "filelists/train.csv"
  val_list_path: "filelists/validation.csv"
```

---

## Como Configurar Augmentations

### 1. Prepare seus Arquivos de Ruído

Organize seus arquivos de ruído em um diretório. A estrutura pode ser qualquer uma, pois a busca é recursiva:

```
/path/to/noise/
├── ambient/
│   ├── crowd.wav
│   └── traffic.wav
├── industrial/
│   ├── machine1.wav
│   └── machine2.wav
└── office.wav
```

### 2. Configure o arquivo `configs/train.yaml`

Edite a seção `augmentations` no arquivo de configuração:

```yaml
data:
  data_dir: "/home/fred/Projetos/DATASETS/VoiceBank-DEMAND-16k/"
  train_list_path: "filelists/train.csv"
  val_list_path: "filelists/test.csv"
  batch_size: 30
  num_workers: 8
  persistent_workers: true
  segment_size: 24576

  # Configure suas augmentations aqui
  augmentations:
    # Adicionar ruído de fundo
    - name: "AddBackgroundNoise"
      params:
        background_paths: "/path/to/your/noise/directory"  # MUDE ISSO!
        min_snr_in_db: 3.0    # SNR mínimo (mais ruído)
        max_snr_in_db: 30.0   # SNR máximo (menos ruído)
        p: 0.5                # 50% de chance de aplicar

    # (Opcional) Aplicar reverberação
    - name: "ApplyImpulseResponse"
      params:
        ir_paths: "/path/to/impulse_responses/"
        p: 0.3

    # (Opcional) Ajustar ganho aleatoriamente
    - name: "Gain"
      params:
        min_gain_in_db: -15.0
        max_gain_in_db: 5.0
        p: 0.3
```

### 3. Desabilitar Augmentations (opcional)

Para desabilitar completamente as augmentations, defina como lista vazia ou null:

```yaml
data:
  augmentations: []  # ou use "null"
```

## Exemplos de Configuração

### Exemplo 1: Apenas Ruído de Fundo
```yaml
augmentations:
  - name: "AddBackgroundNoise"
    params:
      background_paths: "/datasets/noise/DEMAND"
      min_snr_in_db: 5.0
      max_snr_in_db: 20.0
      p: 0.8
```

### Exemplo 2: Múltiplas Augmentations
```yaml
augmentations:
  - name: "AddBackgroundNoise"
    params:
      background_paths: "/datasets/noise/DEMAND"
      min_snr_in_db: 3.0
      max_snr_in_db: 30.0
      p: 0.5

  - name: "ApplyImpulseResponse"
    params:
      ir_paths: "/datasets/RIRs"
      p: 0.3

  - name: "AddColoredNoise"
    params:
      min_snr_in_db: 10.0
      max_snr_in_db: 30.0
      min_f_decay: -2.0  # pink noise
      max_f_decay: 2.0
      p: 0.2
```

### Exemplo 3: Lista de Diretórios
```yaml
augmentations:
  - name: "AddBackgroundNoise"
    params:
      background_paths:
        - "/datasets/noise/DEMAND"
        - "/datasets/noise/FSDnoisy18k"
        - "/datasets/noise/AudioSet"
      min_snr_in_db: 5.0
      max_snr_in_db: 25.0
      p: 0.7
```

## Como Testar

### Teste a Configuração de Validação

Execute o script de teste para verificar se sua configuração de validação está correta:

```bash
# Testar configuração de arquivo YAML
python test_validation_config.py --config configs/train.yaml

# Testar split automático específico
python test_validation_config.py --split 0.1 \
  --train_list filelists/train.csv \
  --data_dir /path/to/data
```

O script irá:
- ✅ Verificar se a configuração é válida
- ✅ Instanciar o DataModule
- ✅ Fazer setup dos datasets
- ✅ Mostrar o número de amostras em treino e validação
- ✅ Calcular a proporção real do split

### Teste a Busca Recursiva de Ruído

Execute o script de teste para verificar quantos arquivos de ruído serão encontrados:

```bash
python test_augmentation.py --noise_dir /path/to/your/noise/directory
```

### Teste a Configuração de Augmentation

Teste se suas augmentations estão configuradas corretamente:

```bash
python test_augmentation.py --config configs/train.yaml
```

### Teste com Arquivos Reais

Execute o script de augmentation standalone:

```bash
python augmentation.py \
  --config configs/train.yaml \
  --input_dir dataset/wavs \
  --output_dir output_augmented \
  --search_pattern "*.wav"
```

## Notas Importantes

### Data Augmentation
1. **Validação**: As augmentations são aplicadas **apenas** no conjunto de treinamento, não na validação
2. **Performance**: A busca recursiva é feita apenas uma vez na inicialização
3. **Memória**: Os arquivos de ruído são carregados sob demanda pela biblioteca
4. **Taxa de Amostragem**: A conversão é feita automaticamente - você pode misturar arquivos com diferentes sample rates
5. **Formatos Suportados**: .wav, .flac, .mp3, .ogg

### Configuração de Validação
1. **Split Determinístico**: O split automático usa seed fixo (42) para reprodutibilidade
2. **Prioridade**: Se `val_list_path` e `val_split` forem especificados, `val_list_path` tem prioridade
3. **Obrigatório**: Você DEVE especificar pelo menos uma das duas opções
4. **Proporções Típicas**:
   - 10% (0.1): Padrão para datasets pequenos/médios
   - 15% (0.15): Boa escolha para datasets médios
   - 20% (0.2): Para datasets grandes onde você quer mais amostras de validação
5. **Augmentations**: Sempre aplicadas apenas no treino, nunca na validação (independente do modo)

## Troubleshooting

### "No audio files found in directory"
- Verifique se o caminho está correto
- Verifique se há arquivos de áudio no diretório
- Verifique as permissões de leitura

### "Memory Error"
- Reduza o `batch_size`
- Reduza o número de arquivos de ruído
- Use menos augmentations simultaneamente

### Augmentations não estão sendo aplicadas
- Verifique se `p` (probabilidade) está > 0
- Verifique se o caminho do diretório de ruído está correto
- Veja os logs durante o treinamento - deve mostrar "Audio augmentation enabled"
