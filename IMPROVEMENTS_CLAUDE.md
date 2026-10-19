# Propostas de melhorias arquiteturais: CleanUNet2‑Mamba

*Autor: Claude (Opus 5.5) · 2026‑10‑12 · branch `feature/mamba`*

Este documento lista mudanças de arquitetura para avaliar experimentalmente no CleanUNet2‑Mamba. Cada proposta traz:
- a motivação, apoiada em medições deste repositório;
- a hipótese a testar;
- o custo estimado;
- o desenho do experimento;
- o critério de sucesso.

Marcações usadas no texto:

| Marca | Significado |
|---|---|
| ✔ | Medido neste repositório (scripts e números na seção 1) |
| ◇ | Estimativa, a confirmar |
| ⚑ | Já existe como opção de configuração; falta só rodar o experimento |

---

## 0. Resumo: as prioridades

| # | Proposta | Motivo principal | Implementação | GPU (triagem) |
|---|---|---|---|---|
| 1 | **A0** Upsampler exato e rápido | O upsampler é ~83% do tempo de um passo do híbrido ✔. A reformulação exata é 4,3× mais rápida ✔ | ~30 linhas, pesos inalterados | nenhuma |
| 2 | **A5** Controles (LR, sementes, condição embaralhada) | Sem eles não dá para atribuir ganhos ao ramo espectral | ~20 linhas | ~2 h |
| 3 | **C1** Normalização causal | Toda a perda do streaming (−0,16 PESQ ✔) vem da normalização causal | ~40 linhas | 0 h (só inferência) a 1 h |
| 4 | **A1** Upsampler sensível à frequência | O upsampler atual não consegue ponderar bandas de frequência ✔ (análise) e reduz tudo a 1 canal | ~80 linhas | ~1 h por variante ◇ |
| 5 | **A2/A3** Fusão no gargalo e multiescala | Hop da STFT (256) = stride total do U‑Net (2⁸): a fusão é natural e barata | ⚑ / ~60 linhas | ~1 h por variante ◇ |
| 6 | **D2** Termo de SI‑SDR na perda | O ajuste fino subiu o PESQ (+0,27) mas baixou o SI‑SDR (−0,37 dB, p≈1e‑14) ✔ | ~15 linhas | ~1 h |
| 7 | **B1** CleanSpecNet residual ou máscara | Hoje o ramo espectral regride a magnitude diretamente, sem ligação da entrada para a saída | ~20 linhas | ~30 min |
| 8 | **C2** Gargalo BiMamba (modo offline) | Mede o custo da causalidade; pode partir dos pesos do DNS sem perda inicial | ~40 linhas | ~1 h |

---

## 1. Evidências medidas

### 1.1 Resultados no teste (VoiceBank‑DEMAND, 824 arquivos) ✔

Todos estes resultados usam o **modelo só de onda**: CleanUNet com gargalo Mamba, igual ao CleanUMamba‑E8, 41,4M parâmetros.

| Sistema | PESQ‑WB | STOI | SI‑SDR | CSIG | CBAK | COVL |
|---|---|---|---|---|---|---|
| Ruidoso | 1,97 | 0,921 | 8,4 | 3,49 | 2,55 | 2,74 |
| CleanUMamba‑E8 (DNS), zero‑shot | 2,82 | 0,942 | 18,9 | 4,18 | 3,59 | 3,55 |
| Estágio A (E8 ajustado 20 épocas) | **3,09** | **0,953** | 18,6 | **4,57** | **3,67** | **3,90** |
| Estágio A, streaming (16 ms de passo) | 2,93 | 0,945 | 17,4 | 4,46 | 3,57 | 3,76 |

CSIG, CBAK e COVL usam a implementação Python do CMGAN/CleanUMamba. No áudio ruidoso ela dá ~0,1 acima dos valores publicados (feitos em MATLAB), então só compare essas três métricas entre resultados deste repositório.

### 1.2 Híbrido × só onda (validação, 770 arquivos, preliminar) ✔

PESQ‑WB por época. O híbrido usa o mesmo ramo de onda iniciado do DNS, mais o CleanSpecNet‑Mamba e FiLM com inicialização zero.

| Época | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Só onda (estágio A) | 2,666 | 2,644 | 2,662 | 2,673 | 2,674 | 2,671 | 2,677 | 2,726 | 2,699 | 2,701 | 2,722 |
| Híbrido | 2,638 | 2,684 | 2,680 | 2,672 | 2,712 | 2,670 | 2,714 | 2,738 | 2,703 | 2,742 | 2,732 |
| Diferença | −0,03 | +0,04 | +0,02 | 0,00 | +0,04 | 0,00 | +0,04 | +0,01 | 0,00 | +0,04 | +0,01 |

A diferença média é **+0,015**: pequena, ruidosa e com um fator de confusão. O ramo de onda treina com LR efetivo de 3e‑5 no híbrido (1e‑4 × 0,3) e de 5e‑5 no estágio A (ver A5).

### 1.3 Onde está o custo ✔

**FLOPs**, contados com `torch.utils.flop_counter` num forward de 2 s, batch 1:

| Modelo | Total | CleanUNet | CleanSpecNet | SpecUpsampler |
|---|---|---|---|---|
| Híbrido Mamba | 24,4 GMAC | 21,3 (87%) | 1,39 (6%) | 1,72 (7%) |
| Híbrido Transformer (E0) | 24,7 GMAC | 22,0 (89%) | 0,92 (4%) | 1,72 (7%) |
| Só onda Mamba | 21,3 GMAC | 21,3 | — | — |

As operações elementares do scan do Mamba não entram nessa contagem.

**Tempo por passo** (forward + backward, bf16, batch 2, com a GPU dividida com outro treino):

| Variante | Tempo |
|---|---|
| Híbrido completo | 145 ms |
| Híbrido com o upsampler trocado por uma operação trivial | 24 ms |
| Só o CleanUNet | 18 ms |
| Só o CleanSpecNet | 6 ms |
| Só o upsampler, em bf16 | 121 ms (o forward sozinho: 108 ms) |
| Só o upsampler, em fp32 | 56 ms |

**Memória do upsampler:** pico de 1,8 GB com batch 2, ou seja ~14 GB com batch 16 ◇. Isso explica os ~17,7 GB de GPU usados pelos treinos híbridos.

**Conclusão:** o upsampler representa 7% das contas, mas ~83% do tempo. O gargalo é de implementação: a `ConvTranspose2d` com stride igual ao kernel no tempo cai num kernel cuDNN lento em bf16, com ativações enormes de (B, 32, 513, 16T). O resto do híbrido custa só ~1,3× o modelo só de onda.

### 1.4 Limitação de projeto do SpecUpsampler (análise do código) ✔

As duas `ConvTranspose2d` usam kernel 3 com stride 1 na frequência, ou seja, os mesmos pesos para todas as bandas. Depois delas vem a média sobre as 513 bandas. Consequências:
- O sinal de condicionamento de cada amostra é, essencialmente, a média de uma mesma função não linear aplicada a vizinhanças locais do espectro.
- O módulo **não consegue dar pesos diferentes a bandas diferentes**: energia a 200 Hz e a 6 kHz contam igual.
- Toda a informação espectral chega ao CleanUNet como **um único canal**.

### 1.5 Desempenho por SNR de entrada (teste) ✔

**PESQ‑WB:**

| SNR de entrada | < 5 dB (n=218) | 5–10 (n=209) | 10–15 (n=206) | ≥ 15 (n=191) |
|---|---|---|---|---|
| Ruidoso | 1,46 | 1,74 | 2,15 | 2,61 |
| Zero‑shot | 2,43 | 2,76 | 2,96 | 3,19 |
| Estágio A | 2,68 | 3,02 | 3,23 | 3,50 |
| Streaming | 2,66 | 2,92 | 3,03 | 3,16 |

**SI‑SDR (dB):**

| SNR de entrada | < 5 dB | 5–10 | 10–15 | ≥ 15 |
|---|---|---|---|---|
| Zero‑shot | 16,3 | 18,6 | 19,9 | 21,2 |
| Estágio A | 16,0 | 17,8 | 19,4 | 21,4 |
| Streaming | 16,1 | 17,5 | 18,0 | 18,0 |

O que esses números mostram:
- **Há mais espaço para melhorar em SNR baixo:** PESQ 2,68 abaixo de 5 dB, contra 3,50 acima de 15 dB. Depois do estágio A, 58 arquivos ainda ficam com PESQ < 2,0.
- **O ajuste fino trocou SI‑SDR por PESQ** justamente nas faixas de SNR baixo e médio. A perda é L1 + MR‑STFT, sem nenhum termo de SI‑SDR.
- **A perda do streaming cresce com o SNR:** −0,02, −0,10, −0,20 e −0,34 de PESQ nas quatro faixas. O SI‑SDR satura em ~18 dB. Esse é o padrão de um **ganho variando no tempo**, como seria de esperar da normalização acumulada.

### 1.6 Origem da perda do streaming ✔

Com a normalização desligada, o streaming reproduz o modo offline exatamente:
- em float64, a diferença é de 2,5e‑14;
- em fp32/CUDA sobre fala real, a concordância é acima de 40 dB.

Logo, a perda de −0,16 PESQ (p≈2,5e‑50) vem **inteira da normalização causal**. O streamer divide a entrada pela média acumulada dos desvios‑padrão por quadro, enquanto o modelo foi treinado com o desvio‑padrão global da utterance.

### 1.7 Poder estatístico ✔

- Desvio‑padrão das diferenças de PESQ pareadas por arquivo: ~0,33, o que dá um erro padrão de 0,012 nos 824 arquivos do teste. Diferenças de ~0,03 se distinguem do ruído entre arquivos.
- A **variância entre sementes ainda não foi medida** (proposta A5b). Até lá, diferenças < 0,03 entre execuções únicas não são conclusivas.

---

## 2. Propostas

Formato de cada proposta: **Motivação** · **Mudança** · **Hipótese** · **Custo** · **Experimento** · **Sucesso** · **Riscos**.

Salvo indicação, as comparações usam:
- treino de 20 épocas, partindo dos pesos do DNS no ramo de onda;
- seleção do checkpoint pelo PESQ de validação;
- uma semente na triagem e três na confirmação.

### Grupo A: ligação entre os ramos (fusão)

#### A0. Upsampler exato e rápido (engenharia, pré‑requisito)
- **Motivação:** 1.3.
- **Mudança:** com stride igual ao kernel no tempo (16), cada `ConvTranspose2d` equivale exatamente a uma `Conv2d` com kernel (3, 1), com os pesos invertidos na frequência e reorganizados, seguida de um rearranjo dos canais no tempo (pixel shuffle). Os parâmetros continuam os mesmos, então os checkpoints seguem compatíveis.
- **Verificado em protótipo:**
  - diferença máxima em relação ao original: 1,0e‑5 ✔;
  - forward + backward em bf16 com batch 4: 238 → 55 ms ✔.

  O núcleo do protótipo (válido quando `stride[1] == kernel[1]` e `padding[1] == 0`):
  ```python
  def convT_as_conv(x, convT):                      # x: (B, Cin, F, T) -> (B, Cout, F, T*kt)
      W = convT.weight                              # (Cin, Cout, kf, kt)
      Cin, Cout, kf, kt = W.shape
      Wc = W.flip(2).permute(1, 3, 0, 2).reshape(Cout * kt, Cin, kf, 1)
      y = F.conv2d(x, Wc, padding=(kf - 1 - convT.padding[0], 0))      # (B, Cout*kt, F, T)
      B, _, Fq, T = y.shape
      y = y.view(B, Cout, kt, Fq, T).permute(0, 1, 3, 4, 2).reshape(B, Cout, Fq, T * kt)
      return y + convT.bias.view(1, -1, 1, 1)
  ```
- **Efeito esperado:** passo do híbrido de ~145 para ~52 ms (≈2,8×), estimado a partir das medições da seção 1.3 ◇.
- **Experimento:** nenhum para métricas. Basta um teste de regressão (equivalência numérica) e cronometrar uma época.
- **Sucesso:** época do híbrido ≤ 3,5 min (hoje ~8 min).

#### A1. Upsampler sensível à frequência, leve e multicanal
- **Motivação:** 1.3 e 1.4.
- **Mudança:** um novo módulo (`upsampler.type: freq_proj`) com três camadas:
  1. `Conv1d(513 → C, k=1)`: pesos próprios para cada banda;
  2. `ConvTranspose1d(C, C, 16, stride 16)` + GELU;
  3. `ConvTranspose1d(C, K, 16, stride 16)`, produzindo (B, K, 256·T), cortado no tamanho L.
  
  Duas formas de injetar o resultado no ramo de onda:
  - **(i) FiLM**, com o `Conditioner` aceitando `cond_channels = K` (ele já aceita);
  - **(ii) concatenação na primeira camada do encoder:** `channels_input = 1 + K`. Os pesos de `encoder.0.0` são estendidos com zeros para os K canais novos, então o modelo começa idêntico ao ramo pré‑treinado.
- **Hipótese H1:** um condicionamento mais rico dá híbrido ≥ estágio A + 0,03 de PESQ na validação, com o ganho concentrado em SNR < 10 dB.
- **Custo** ◇: ~0,03 GMAC por 2 s (contra 1,7 do original) e ativações ~50× menores.
- **Experimento:** K ∈ {1, 8} × forma ∈ {FiLM, concatenação} = 4 execuções de triagem, comparadas com o híbrido atual (R3) e com o estágio A no mesmo LR (A5a).
- **Riscos:** deixa de reproduzir exatamente o artigo do CleanUNet2. O upsampler original continua disponível como `upsampler.type: original`.

#### A2. Fusão no gargalo ⚑ (com variantes)
- **Motivação:** com hop 256 = 2⁸, o quadro *j* da STFT e o token *j* do gargalo andam no mesmo passo (ver `../PLANO_CleanUNet2_Mamba.md`, seção 2.1). A opção já existe como `bottleneck_fusion: true` (FiLM com inicialização zero).
- **Variantes:**
  - (a) FiLM (já implementado);
  - (b) concatenação + `Linear(2·d → d)`, iniciada como identidade na metade do ramo de onda;
  - (c) **ajuste de alinhamento** de ∈ {−1, 0, +1} quadro. Com `alignment: center`, o centro do quadro *j* fica em 256j+128, enquanto o campo receptivo do token *j* ([256j, 256j+766)) está centrado em 256j+383, cerca de um quadro adiante;
  - (d) **fusão só no gargalo**, sem o condicionamento na forma de onda (`use_waveform_conditioning: false`). Seria o híbrido mais barato: ~24 ms por passo, contra 18 ms do modelo só de onda (seção 1.3).
- **Hipótese H2:** a fusão no gargalo iguala ou supera o condicionamento na forma de onda, a uma fração do custo.
- **Custo:** +0,5M parâmetros; computação desprezível.

#### A3. Fusão multiescala
- **Mudança:** além do gargalo, aplicar FiLM com inicialização zero nas entradas dos dois níveis seguintes do decoder (stride 128 e 64), repetindo as features do CleanSpecNet 2× e 4×. Requer ganchos por nível no `CleanUNet.forward` (`decoder_hooks`).
- **Hipótese H3:** melhora sobretudo as faixas de SNR baixo (1.5).
- **Experimento:** A2(a) contra A2(a)+A3.

#### A4. Pós‑filtro espectral (fusão tardia)
- **Mudança:**
  1. Calcular `Y = STFT(ŷ)` da saída do CleanUNet.
  2. Montar as entradas `[compress|Y|, Ŝ (saída do CleanSpecNet)]`.
  3. Passar por uma rede pequena (conv + 2 camadas Mamba no tempo, d=128) que produz uma máscara limitada `M = 1 + α·tanh(Δ)`, com Δ começando em zero (identidade).
  4. A saída final é `ŷ' = iSTFT(M ⊙ Y)`.
- **Hipótese H4:** reduz o ruído residual (CBAK e SSNR sobem), sobretudo ruído estacionário em frequências altas.
- **Custo:** baixo em computação. Em streaming, acrescenta a latência da janela da STFT.
- **Riscos:** artefatos de iSTFT; a fase não é corrigida.

#### A5. Controles (necessários para interpretar o grupo A)
- **(a) LR igualado:** repetir o estágio A com o LR do ramo de onda igual ao do híbrido (3e‑5), ou o híbrido com 5e‑5.
- **(b) Variância entre sementes:** 3 sementes do estágio A (~30 min cada) para estimar σ entre execuções.
- **(c) Condição embaralhada:** o ramo de onda recebe o espectrograma de *outra* utterance do batch (uma permutação). Se o PESQ ficar igual ao do híbrido normal, o condicionamento não está carregando informação útil.
- **Implementação:** a flag `shuffle_condition` no CleanUNet2, usada só no treino e na avaliação do controle.

### Grupo B: ramo espectral (CleanSpecNet)

#### B1. Saída residual ou por máscara
- **Motivação:** a `output_layer` regride a magnitude comprimida diretamente, sem nenhuma ligação da entrada para a saída.
- **Mudança** (`output_mode`):
  - `residual`: `Ŝ = relu(X + Δ)`, com a última camada iniciada em zero (começa como identidade);
  - `mask`: `Ŝ = compress(M ⊙ |X|)`, com `M = β·σ(m)` e β ≈ 1,5;
  - `direct`: o comportamento atual.
- **Hipótese H5:** convergência mais rápida e menor L1 espectral, o que melhora o condicionamento.
- **Experimento:** estágio B (só o ramo espectral, 10 épocas) comparando o L1 de validação; o melhor modo segue para o híbrido.

#### B2. Backbone tempo‑frequência (dual‑path Mamba)
- **Motivação:** hoje as 513 bandas viram canais de uma `Conv1d`, e a estrutura de frequência se perde logo na primeira camada.
- **Mudança** (`backbone.type: tf_mamba`):
  1. Front‑end `Conv2d` 1→32 canais, reduzindo a frequência a ~129 bandas (dois strides de 2).
  2. N = 4 blocos alternando: Mamba ao longo do tempo para cada banda (causal) e BiMamba ao longo da frequência para cada quadro (a frequência não precisa ser causal).
  3. Decoder `ConvTranspose2d` de volta a 513 bandas.
- **Hipótese H6:** melhor estimativa espectral, sobretudo em SNR baixo. Arquiteturas da literatura que modelam tempo e frequência explicitamente (como as da família MP‑SENet/SEMamba) costumam ir bem no VoiceBank. Isso é uma hipótese, não um número a reproduzir.
- **Custo** ◇: ~16k tokens por exemplo por bloco, com d=32–64; é preciso medir o tempo antes de escalar.
- **Riscos:** memória e tempo. Só depois de A0/A1.

#### B3. Representação da entrada ⚑ (parcial)
- Compressão `log1p` contra potência 0,3 (⚑ `spectrogram.compression: pow`).
- Entrada complexa comprimida (parte real e imaginária, 2×F canais), para o ramo espectral ver a fase.
- Barato: 2 a 3 execuções do estágio B.

#### B4. Causalidade separada e versão bidirecional offline ⚑ (parcial)
- Separar as flags `conv_causal` e `backbone_causal`. Isso permite reproduzir exatamente o CleanUNet2 original (atenção causal com convoluções não causais).
- Medir o **custo da causalidade** no ramo espectral com `cleanunet2_mamba_e8_bidir.yaml` ⚑.

#### B5. Ramo espectral com estimativa de fase (decoders de magnitude e de fase)
- Produz uma estimativa espectral completa, o que permite combinar as duas estimativas (onda e espectro) por pesos aprendidos em tempo‑frequência.
- Custo alto; fica para depois de B2.

### Grupo C: ramo de onda (CleanUNet‑Mamba)

#### C1. Normalização causal (fechar o gap do streaming)
- **Motivação:** 1.5 e 1.6.
- **Passo 1, sem treino** (só no `StreamingCleanUNet`), avaliar o modelo do estágio A em streaming com:
  - (i) a média acumulada do std por quadro (a atual);
  - (ii) RMS com média móvel exponencial, τ ∈ {0,5, 1, 2} s;
  - (iii) aquecimento seguido de congelamento: estimar nos primeiros 0,5 s e depois fixar;
  - (iv) o std acumulado verdadeiro, sobre todas as amostras já vistas.
- **Passo 2, com treino:** treinar com a mesma normalização causal (`normalization.type: cumulative | ema`), ou sem normalização e com augmentação de ganho de ±12 dB.
- **Sucesso:** gap do streaming ≤ 0,05 de PESQ, sem dependência do SNR.

#### C2. Gargalo bidirecional (modo offline)
- **Mudança:** um `BiMambaMixer` no gargalo (a classe já existe em `sequence.py`).
  - O ramo direto recebe os pesos do DNS.
  - O ramo reverso começa como cópia, com `out_proj` zerado. Assim o modelo começa idêntico ao causal.
  - O carregador precisa remapear `mixer.*` → `mixer.fwd.*`.
- **Hipótese H7:** ganho offline de pelo menos +0,05 de PESQ, o que quantifica o custo da causalidade no ramo de onda.

#### C3. Mamba em múltiplas escalas
- **Mudança:** uma camada Mamba em cada um dos dois níveis mais profundos do decoder (stride 128 e 64), projetando 768 → 256 canais, com `out_proj` zerado.
- **Custo** ◇: +3 a 5M parâmetros. O streaming continua viável (um token por nível a cada 128 ou 64 amostras), mas exige estado por nível.
- **Hipótese:** modelar o contexto longo também em resolução temporal mais fina ajuda nos transientes.

#### C4. Skip connections com porta, ou projeção residual ⚑ (parcial)
- **Porta:** `x + g ⊙ skip`, com `g = σ(conv1x1([x, skip]))`, viés inicial +4 (g ≈ 1, preserva os pesos pré‑treinados).
- **Projeção residual:** `residual_projection` ⚑ já existe, mas é iniciada aleatoriamente, o que quebraria a inicialização do DNS. É preciso iniciá‑la como identidade.

#### C5. Escala do Mamba ⚑
- Variações: `n_layers` ∈ {3, 5}, `d_state` ∈ {16, 64, 128}, `expand` ∈ {2, 4}, e Mamba2 ⚑.
- Só no ramo de onda, do zero, 30 épocas (~40 min cada). Isso complementa as execuções só‑onda MHA × Mamba da fila 2.

#### C6. Latência: E6, E7 e E8 ⚑ (parcial)
- E6: 190 amostras = 12 ms de antecipação; existe checkpoint do DNS (`CleanUMamba-3N-E6-high`).
- E7: 382 amostras = 24 ms.
- E8: 766 amostras = 48 ms.
- Objetivo: a curva PESQ × latência, também em streaming.

#### C7. Ablações baratas ⚑
- `glu_activation` (SiLU/GELU) e `encoder_norm: batch` (do CleanUNet_MambaSSM).
- Só treinando do zero, porque os pesos pré‑treinados assumem a GLU com sigmoid e nenhuma normalização.

### Grupo D: objetivos de treino ligados à arquitetura

#### D1. Perda de consistência entre os ramos
- `L1(compress|STFT(ŷ)|, sg(Ŝ))`: a saída de onda passa a concordar com a estimativa espectral. Também vale testar a versão simétrica.
- Pesos ∈ {0,1, 0,5}.

#### D2. Termo de SI‑SDR
- **Motivação:** 1.5 e 1.7. O PESQ subiu +0,27 enquanto o SI‑SDR caiu −0,37 dB.
- **Mudança:** somar `λ·(−SI‑SDR)` à perda, com λ ∈ {0,01, 0,05, 0,1} na escala adequada.
- **Saída do experimento:** a fronteira PESQ × SI‑SDR.

#### D3. Discriminador de métrica (estilo MetricGAN)
- Uma rede que aprende a prever o PESQ e é usada como discriminador.
- Risco de artefatos e de instabilidade. Prioridade baixa.

### Grupo E: eficiência e uso em tempo real

#### E1. Destilação: híbrido offline → modelo só de onda causal
- **Professor:** o melhor híbrido offline (B4 ou C2 bidirecional).
- **Aluno:** o modelo só de onda causal, E8 ou E6.
- **Perdas:** L1 + MR‑STFT contra o áudio limpo, mais L1 contra a saída do professor. Opcionalmente, KD de features nas skips, como no CleanUMamba (`cleanumamba/src/util/util.py:259-290`).
- **Hipótese:** o aluno fecha pelo menos metade da diferença entre os dois.

#### E2. Poda estruturada (portar o CleanUMamba)
- Os nomes dos módulos já são compatíveis (`tsfm_Mamba_layers`, `norm_f`).
- Tamanhos‑alvo: 5M, 2M e 1M parâmetros, com KD.

#### E3. Streaming do híbrido
- `Spectrogram(alignment="causal")` ⚑, mais o `step` do Mamba no CleanSpecNet e um buffer de STFT quadro a quadro.
- Avaliar `n_fft` 512 (janela de 32 ms) para conter a latência.

---

## 3. Protocolo experimental

1. **Dados:** treino com 10.802 arquivos, validação com 770 (p226/p287) e teste com 824. Seleção pelo PESQ‑WB de validação; o teste só é usado uma vez, ao final.
2. **Triagem:** uma semente.
   - 20 épocas quando o ramo de onda parte dos pesos do DNS; 30 épocas quando o modelo treina do zero.
   - Passa para a confirmação o que superar a referência pareada (mesma inicialização, LR e número de épocas) em ≥ 0,03 de PESQ na validação, ou em ≥ 2σ entre sementes, quando A5b estiver medido.
3. **Confirmação:** 3 sementes. Reportar média ± desvio.
4. **Testes estatísticos:** Wilcoxon pareado por arquivo no teste (PESQ, STOI e SI‑SDR), com correção de Holm.
5. **Relatório obrigatório para cada variante:**
   - parâmetros, GMAC por segundo de áudio, tempo por passo e memória de pico;
   - RTF e latência algorítmica;
   - resultados por faixa de SNR (as quatro faixas da seção 1.5);
   - métricas em streaming, quando a variante for causal.
6. **Comparações de eficiência:** comparar com orçamento igual de épocas e também de tempo de GPU.
7. **Ferramentas existentes:**
   - `evaluate.py` (métricas por arquivo e `summary.json`);
   - `scripts/show_metrics.py` (curvas de treino);
   - `torch.utils.flop_counter` (FLOPs).

## 4. Roteiro e orçamento de GPU (RTX 4090)

Referências de tempo por época:
- só onda: ~1,3 min ✔;
- híbrido atual: ~8 min ✔;
- híbrido com A0: ~3 min ◇;
- híbrido com A1 ou fusão só no gargalo: ~1,7–2 min ◇.

| Fase | Conteúdo | GPU ◇ | Depende de |
|---|---|---|---|
| 0 | A0; C1 passo 1 (só inferência); A5 (a, b, c); terminar a fila atual | ~3 h | — |
| 1 | Fusão: A1 (4 variantes), A2 (b, c, d), A3 | ~6–8 h | A0, A5 |
| 2 | Ramo espectral: B1, B3, B4; depois B2 | ~6–10 h | Fase 1 |
| 3 | Ramo de onda: C1 passo 2, C2, C5, C6, C4/C7 | ~8–12 h | A5b |
| 4 | Objetivos: D1, D2 (varredura de λ); D3 opcional | ~3–5 h | Fases 1–3 |
| 5 | Implantação: E1, E3, E2 | ~8–12 h | Fases 1–4 |
| — | Confirmação com 3 sementes dos finalistas | +~50% | — |

## 5. Onde mexer no código

| Proposta | Arquivos | Configuração nova | Teste a acrescentar |
|---|---|---|---|
| A0 | `cleanunet/cleanunet2.py` (`SpecUpsampler.forward`) | — | equivalência com `ConvTranspose2d` |
| A1 | `cleanunet2.py` (nova classe); `cleanunet.py` (`channels_input > 1`); `checkpoint_io.py` (estender `encoder.0.0` com zeros) | `upsampler: {type: freq_proj, channels: 64, out_channels: K}`; `conditioning_type: encoder_concat` | inicialização zero ⇒ saída igual ao ramo de onda |
| A2 | `cleanunet2.py` (`BottleneckFusion`) | `bottleneck_fusion: {type: film\|concat, shift: 0}`; `use_waveform_conditioning` | idem |
| A3 | `cleanunet.py` (`decoder_hooks`); `cleanunet2.py` | `multiscale_fusion: [256, 128, 64]` | idem |
| A4 | novo `cleanunet/postfilter.py` | `postfilter: {...}` | identidade na inicialização |
| A5c | `cleanunet2.py` | `shuffle_condition: true` | — |
| B1 | `cleanspecnet.py` | `output_mode: direct\|residual\|mask` | identidade na inicialização (residual) |
| B2 | novo `cleanunet/tf_mamba.py` | `backbone: {type: tf_mamba, ...}` | shapes; causalidade no tempo |
| B3 | `spectral.py` | `compression: pow`; `representation: complex` | — |
| B4 | `cleanspecnet.py` | `conv_causal`, `backbone_causal` | causalidade |
| C1 | `streaming.py`; `cleanunet2.py` | `normalization: {type: global\|cumulative\|ema\|warmup, tau}` | streaming = offline com a mesma normalização |
| C2 | `sequence.py`; `checkpoint_io.py` | `bottleneck: {bidirectional: true}` | identidade na inicialização contra o modelo causal |
| C3 | `cleanunet.py` | `decoder_mamba_levels: [...]` | causalidade; streaming |
| C4 | `cleanunet.py` | `skip_gate: true` | identidade aproximada na inicialização |
| D1, D2 | `lightning_modules/cleanunet_module.py` | `loss.weight_consistency`, `loss.weight_sisdr` | — |
| E1 | `lightning_modules/` (professor congelado) | `teacher: {architecture, checkpoint}` | — |
| E3 | `streaming.py` | — | streaming = offline (híbrido) |

Princípio comum: **todo módulo novo começa como identidade**, com a última camada zerada ou uma porta perto de 1. Assim cada variante parte exatamente do modelo de referência pré‑treinado, e a comparação mede o efeito do módulo em vez de uma pior inicialização.

## 6. Pontos de decisão e riscos

**Decisões depois da fila atual:**

| Resultado da fila | Decisão |
|---|---|
| Híbrido com pesos do DNS (R3) − estágio A no mesmo LR < 0,03, e o controle A5c igual a R3 | O condicionamento atual não ajuda. Priorizar A1/A2 e considerar remover o condicionamento na forma de onda (A2d) |
| Mamba híbrido do zero (R5) ≥ Transformer (E0), com custo igual ou menor | Mamba vira o backbone padrão nas fases seguintes |
| C1, passo 1, já fecha o gap do streaming | Não é preciso retreinar com normalização causal |

**Riscos:**
- **VoiceBank é pequeno** (9,4 h): variantes maiores (B2, C3) podem sofrer overfitting. Mitigação: partir dos pesos do DNS, usar augmentação e early stopping na validação.
- **Generalização para outros dados:** se o conjunto de teste do DNS (sem reverberação) for baixado, avaliar os finalistas nele sem retreino.
- **Métricas compostas:** a implementação em Python dá valores diferentes dos publicados (seção 1.1). Comparar só dentro deste repositório.
