# Changelog - Validação e Data Augmentation

## ✅ Implementações Concluídas

### 1. Data Augmentation com Busca Recursiva

#### Arquivo: `augmentation.py`
**Mudanças:**
- ✅ Adicionada função `get_audio_files_recursively()` para busca recursiva em diretórios
- ✅ Suporte para múltiplas extensões de áudio (.wav, .flac, .mp3, .ogg)
- ✅ Expansão automática de diretórios em listas de caminhos
- ✅ Busca recursiva funciona tanto para `background_paths` quanto para `ir_paths`

**Exemplo de uso:**
```python
# Antes: Tinha que listar todos os arquivos manualmente
background_paths: ["/noise/file1.wav", "/noise/file2.wav", ...]

# Agora: Apenas especifique o diretório
background_paths: "/noise/directory"  # Busca recursiva automática!
```

#### Arquivo: `spec_dataset.py`
**Mudanças:**
- ✅ Import do `AudioAugmenter` adicionado
- ✅ Augmenter habilitado e funcional (não está mais comentado)
- ✅ Augmentations aplicadas no áudio ruidoso antes do cálculo do espectrograma
- ✅ Conversão on-the-fly de taxa de amostragem (pela biblioteca torch_audiomentations)

---

### 2. Configuração Flexível de Validação

#### Arquivo: `lightning_modules/data_module.py`
**Mudanças:**
- ✅ Suporte para duas opções de validação:
  - **Opção 1**: `val_list_path` - arquivo separado (comportamento original)
  - **Opção 2**: `val_split` - porcentagem para split automático (NOVO!)
- ✅ Validação automática de parâmetros (erro claro se nenhum for especificado)
- ✅ Split determinístico com seed fixo (42) para reprodutibilidade
- ✅ Augmentations aplicadas apenas no treino, nunca na validação
- ✅ Mensagens informativas durante o setup

**Novo comportamento:**
```python
# Opção 1: Arquivo separado (original)
CleanUNetDataModule(
    data_dir="/datasets/VB-DEMAND/",
    train_list_path="train.csv",
    val_list_path="test.csv"  # Arquivo separado
)

# Opção 2: Split automático (NOVO!)
CleanUNetDataModule(
    data_dir="/datasets/VB-DEMAND/",
    train_list_path="train.csv",
    val_split=0.1  # 10% para validação
)
```

---

### 3. Arquivos de Configuração Atualizados

#### `configs/train.yaml`
- ✅ Comentários explicativos sobre as duas opções de validação
- ✅ Seção de augmentations com exemplos (desabilitada por padrão)
- ✅ Configuração padrão usando arquivo separado

#### `configs/train_with_augmentation.yaml` (NOVO)
- ✅ Exemplo completo com todas as augmentations habilitadas
- ✅ Comentários explicando cada parâmetro
- ✅ Múltiplas opções de augmentation demonstradas

#### `configs/train_with_auto_split.yaml` (NOVO)
- ✅ Exemplo usando split automático (val_split)
- ✅ Configuração limpa e fácil de entender
- ✅ Ideal para quem tem apenas um arquivo de dados

---

### 4. Scripts de Teste Criados

#### `test_augmentation.py`
**Funcionalidades:**
- ✅ Testa busca recursiva em um diretório
- ✅ Mostra estatísticas de arquivos encontrados (contagem por extensão)
- ✅ Testa configuração de augmentations do YAML
- ✅ Valida caminhos de diretórios de ruído/IR
- ✅ Tenta instanciar o AudioAugmenter

**Uso:**
```bash
# Testar busca recursiva
python test_augmentation.py --noise_dir /path/to/noise

# Testar configuração YAML
python test_augmentation.py --config configs/train.yaml
```

#### `test_validation_config.py` (NOVO)
**Funcionalidades:**
- ✅ Testa configuração de validação de arquivo YAML
- ✅ Verifica se val_list_path ou val_split está configurado
- ✅ Instancia o DataModule e faz setup
- ✅ Mostra tamanhos dos datasets e proporção real
- ✅ Suporte para teste manual de split

**Uso:**
```bash
# Testar configuração YAML
python test_validation_config.py --config configs/train.yaml

# Testar split específico
python test_validation_config.py --split 0.15 \
  --train_list filelists/train.csv \
  --data_dir /path/to/data
```

---

### 5. Documentação

#### `AUGMENTATION_README.md` (Atualizado)
**Seções:**
- ✅ Índice completo com links
- ✅ Data Augmentation: busca recursiva, conversão automática
- ✅ **Configuração de Validação**: novas opções explicadas
- ✅ Exemplos de configuração para ambos os modos
- ✅ Guia de testes para validação e augmentation
- ✅ Notas importantes sobre ambas as funcionalidades
- ✅ Troubleshooting

---

## 📊 Resumo de Funcionalidades

| Funcionalidade | Status | Descrição |
|---------------|--------|-----------|
| Busca Recursiva de Ruído | ✅ | Busca automática em subdiretórios |
| Conversão Taxa Amostragem | ✅ | On-the-fly pela biblioteca |
| Augmentations Habilitadas | ✅ | Não está mais comentado |
| Split Automático Validação | ✅ | Nova opção com porcentagem |
| Arquivo Separado Validação | ✅ | Opção original mantida |
| Scripts de Teste | ✅ | Dois scripts completos |
| Documentação | ✅ | README detalhado |
| Exemplos de Config | ✅ | 3 arquivos YAML diferentes |

---

## 🚀 Como Começar a Usar

### Para Data Augmentation:

1. **Configure o diretório de ruído** em `configs/train.yaml`:
   ```yaml
   augmentations:
     - name: "AddBackgroundNoise"
       params:
         background_paths: "/seu/diretorio/de/ruido"
         min_snr_in_db: 3.0
         max_snr_in_db: 30.0
         p: 0.5
   ```

2. **Teste a configuração**:
   ```bash
   python test_augmentation.py --config configs/train.yaml
   ```

3. **Treine normalmente**:
   ```bash
   python train.py --config configs/train.yaml
   ```

### Para Split Automático de Validação:

1. **Configure o split** em `configs/train.yaml`:
   ```yaml
   data:
     train_list_path: "filelists/train.csv"
     val_split: 0.1  # 10% para validação
     # Comente ou remova val_list_path
   ```

2. **Teste a configuração**:
   ```bash
   python test_validation_config.py --config configs/train.yaml
   ```

3. **Treine**:
   ```bash
   python train.py --config configs/train.yaml
   ```

---

## 📝 Notas Técnicas

### Conversão de Taxa de Amostragem
- Feita pela biblioteca `torch_audiomentations`
- Automática e on-the-fly
- Não precisa pré-processar os arquivos de ruído
- Taxa alvo definida em `model.sample_rate` (padrão: 16kHz)

### Split Determinístico
- Usa `torch.Generator().manual_seed(42)`
- Garante que o mesmo split é usado sempre
- Importante para reprodutibilidade de experimentos

### Augmentations e Validação
- Augmentations sempre aplicadas **apenas no treino**
- Validação sempre usa dados **sem augmentation**
- Independente de usar val_list_path ou val_split

---

## 🐛 Troubleshooting

### "No audio files found in directory"
- Verifique se o caminho do diretório está correto
- Verifique se há arquivos .wav, .flac, .mp3 ou .ogg no diretório
- Use o script de teste: `python test_augmentation.py --noise_dir /path`

### "Must specify val_list_path OR val_split"
- Você precisa configurar pelo menos uma das opções
- Use val_list_path para arquivo separado
- Use val_split para split automático

### Augmentations não estão sendo aplicadas
- Verifique os logs: deve aparecer "Audio augmentation enabled"
- Certifique-se que `p` (probabilidade) > 0
- Teste com: `python test_augmentation.py --config configs/train.yaml`

---

## 📧 Suporte

Para problemas ou dúvidas:
1. Consulte `AUGMENTATION_README.md`
2. Execute os scripts de teste
3. Verifique os logs durante o treino
