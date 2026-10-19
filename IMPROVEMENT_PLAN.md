# Plano de ablação: condicionamento e fusão CleanSpecNet → CleanUNet

*Autor: Claude (Opus 5.5) · 2026‑10‑13, atualizado em 2026‑10‑16 · branch `feature/mamba` · detalha os Grupos A (fusão) e B (CleanSpecNet) de `IMPROVEMENTS_CLAUDE.md`*

| Marca | Significado |
|---|---|
| ✔ | Medido neste repositório |
| ◇ | Estimativa, a confirmar |
| ⚑ | Já existe no código ou na configuração |

---

## 0. Resumo

**Estado em 2026‑10‑19, 15h.** A fila Y (14.9) terminou: verificações sem treino e um conjunto novo com 80 locutores não vistos (vctk_unseen), que confirma os resultados do teste. A fila X (14.8) terminou em 2026‑10‑16 às 10:53, sem falhas, e a análise complementar (14.8.2) revisou a justificativa do S30hp. Ela cobriu o streaming com valor inicial, o DNSMOS, uma página de escuta e o discriminador restrito à banda acima de 100 Hz (S30hp). A fila W (seção 14: melhor sistema e modelo mais simples e robusto) terminou às 04:43: resultados em 14.4–14.6 e análise em 14.7.

- **Melhor sistema (A):** a mistura sem treino da soup do **S30** com o ramo espectral bidirecional **W5** (w = 0,25, escolhido na val_unseen) dá **PESQ 3,375** no teste. Com as 3 sementes do S30 separadas, dá **3,372 ± 0,009**.
  - São +0,164 sobre o RX_C3 (3,212) e +0,212 sobre o K0extfull (3,164).
  - STOI 0,956, SI‑SDR 18,85 dB, COVL 4,15. Tem 60,9M parâmetros e só funciona offline.
  - Não depende da polaridade e tem a menor cauda de falhas: 0,4% dos arquivos ficam mais de 0,1 abaixo do K0extfull.
  - **Para uso, a versão com o S30hp (14.8, X4)** dá PESQ 3,326, mas tem o maior PESQ sem < 100 Hz (3,469) e o maior SI‑SDR (19,00 dB) de todos os sistemas, além de DNSMOS OVRL 3,219 contra 3,206.
- **Modelo simples e robusto (B):** a **soup do S30** dá **3,335** no teste (3 sementes: 3,328 ± 0,011).
  - É uma única rede só de onda, com 41,4M parâmetros, sem fusão e compatível com streaming.
  - Receita: o K0 com ruído externo, mais o discriminador métrico (peso 0,3), a inversão de polaridade do par de treino (p = 0,5) e 30 épocas.
  - Ressalva: o ganho vem de uma perda que mira o PESQ. STOI, SI‑SDR, segSNR e WSS não melhoram (14.5).
  - **Para uso, a soup do S30hp (14.8, X4):** o mesmo, com o discriminador só acima de 100 Hz.
    - PESQ 3,280 (−0,055).
    - O mesmo PESQ‑NB do S30 (+0,001): o ganho na banda da fala é igual.
    - SI‑SDR +0,21 dB, segSNR +0,28 e WSS −0,19 sobre o S30, com o SI‑SDR igual ao do controle. No vctk_unseen (14.9), o SI‑SDR (+0,17 dB) e o segSNR (+0,25) se repetem em 98–100% dos 80 locutores; o WSS, não (−0,01).
    - O +0,017 de PESQ sem < 100 Hz sobre o S30 é a métrica que o S30hp otimiza e não se confirma no PESQ‑NB (14.8.2).
    - DNSMOS OVRL +0,016 sobre o S30 e metade da variância entre sementes.
- **Polaridade (14.3):** os modelos treinados com o babble do MUSAN quase deixam de funcionar com a entrada invertida (K0extfull: 3,18 → 1,82; RX_C3: 3,22 → 1,96). Inverter o par de treino com p = 0,5 corrige isso sem custo. Dar sinal aleatório aos locutores do babble não corrige.
- **O RX_C3 deixa de ser recomendado.** Ele perde para a soup do S30 por 0,124, depende da polaridade, e colocá‑lo no ensemble piora o resultado.
- **Streaming:** o S30 perde 0,25 no teste em streaming. Quase toda a perda vem do aquecimento da escala causal da entrada (14.7.4).
  - No primeiro segundo, a escala estimada fica cerca de 3× pequena.
  - Com escala fixa ou com valor inicial, o streaming da soup do S30 vai de 3,164 para 3,335 ou 3,325 na val_unseen, sem treino. O offline dá 3,348.
  - **Confirmado no teste (14.8, X1):** com valor inicial, o streaming dá 3,323, contra 3,335 offline e 3,085 antes. A opção está em `cleanunet/streaming.py` (`prior_std`, `prior_frames`).
- **Análise (14.7):**
  - A val_unseen ordena os sistemas como o teste: Spearman 0,978 em 52 sistemas. A val antiga dá 0,66.
  - Quase 60% do ganho do discriminador métrico (+0,099 de +0,170) e toda a perda de SI‑SDR estão abaixo de 100 Hz. O filtro de entrada do PESQ‑WB ainda deixa passar a faixa de 50–100 Hz.
  - Sem essa faixa na referência e na saída, o S30 ganha +0,078 sobre o controle e +0,053 sobre o RX_C3, e a mistura mantém +0,040 sobre o S30. A ordem dos sistemas não muda.
- **Fila X (14.8):**
  - **DNSMOS:** não confirma o ganho do discriminador.
    - S30 − controle dá OVRL −0,026 e BAK −0,037.
    - A diferença vem da faixa < 100 Hz: o mesmo conteúdo que o PESQ‑WB premia soa ao DNSMOS como ruído de fundo.
    - O DNSMOS está no teto: o controle já empata com a referência limpa.
  - **S30hp:** separa os dois efeitos do discriminador.
    - Mantém o ganho na banda da fala: PESQ‑NB +0,067 sobre o controle, igual ao do S30 (+0,066). Sem < 100 Hz dá +0,094 contra +0,077, mas essa diferença não se confirma (14.8.2).
    - Corta ~60% do efeito abaixo de 100 Hz. O resto vem do corte suave do filtro.
  - **Análise complementar (14.8.2):**
    - O PESQ‑NB (300–3400 Hz) confirma o ganho do discriminador na banda da fala, igual no S30 e no S30hp.
    - O teste e a val_unseen têm 2 locutores cada. A parte do ganho abaixo de 100 Hz varia 5× entre os 4 locutores (0,03 a 0,15); os outros efeitos têm o mesmo sinal nos 4.
  - **Recomendação:** S30 e S30 + W5 para o número de benchmark (PESQ‑WB); S30hp e S30hp + W5 para uso.
  - **Escuta:** a página (X5) está publicada e ainda não tem respostas (conferido em 2026‑10‑19). O bloco B decide se a faixa de 50–100 Hz é audível.
- **Fila Y (14.9):**
  - **80 locutores não vistos (vctk_unseen):** os números ficam a ≤ 0,03 dos do teste, com a mesma ordem. O ganho do discriminador no PESQ‑NB (+0,055) aparece em 100% dos locutores, igual no S30 e no S30hp. A mistura com o W5 melhora todas as métricas em 85–100% dos locutores.
  - **O S30hp não ganha PESQ do S30 em banda nenhuma** (PESQ‑NB −0,001; sem < 100 Hz +0,008, em 51% dos locutores). Ganha SI‑SDR e segSNR.
  - **Peso da mistura do S30hp + W5:** w = 0,25, como no S30. **Mistura com o ramo causal (Bext):** só +0,011 a +0,018 no teste, um terço do W5; a mistura em streaming cai para prioridade baixa.

Resumo da fila V (2026‑10‑15):

- **Resposta do V1:** uma mistura sem treino do K0extfull com o Bext (w = 0,25) dá **3,209 ± 0,005** no teste, empatada com o RX_C3 (3,212 ± 0,008). A fusão aprendida não acrescenta nada no ruído novo; o sistema offline recomendado passa a ser a mistura.
- **Análise complementar (12.11):** o RX_C3 tem uma cauda de falhas que a mistura não tem: 5,9% dos arquivos ficam mais de 0,1 abaixo do K0extfull, contra 0,5%. Essa cauda não aparece no SI‑SDR nem no STOI. O PESQ de arquivos isolados é instável, mas as médias não mudam com um PESQ robusto. A média das formas de onda do RX_C3 e da mistura dá **3,238** no teste e 3,253 na val_unseen.
- **V2–V7:** condition dropout não reduziu a dependência; nenhuma fusão de streaming nem a destilação passou do δ_min; nenhum ramo espectral novo passou do limiar (o bidirecional chegou mais perto); a receita só `noise` rendeu +0,007 no teste; com 30 épocas o K0ext melhora +0,018 e o RX_C3 nada.

Resumo anterior à fila V:

- **Melhor resultado no teste:** **PESQ 3,212 ± 0,008** (3 sementes) ✔, com o híbrido **RX_C3**: P3 + S5 + P8, ramo espectral Bext e treino com ruído externo (MUSAN).
  - São +0,048 sobre o controle só de onda com o mesmo treino (K0extfull, 3,164) e +0,127 sobre o controle antigo (K0full, 3,085).
  - O ganho aparece nas 3 sementes, nos 5 ruídos do teste e em todas as faixas de SNR.
  - O RX_C3 só funciona offline (S5 e P8), tem 53,8M parâmetros (contra 41,4M) e dependência 2,06: com a condição de outro arquivo, o PESQ cai 2 pontos.
- **Maior ganho isolado:** treinar o ramo de onda com ruído externo rende +0,079 no teste, sem custo na inferência ✔.
- **Streaming:** a única fusão compatível com streaming (P3) perde 0,013 no teste, mesmo com o Bext ✔. O melhor modelo para streaming continua sendo só de onda (K0ext).
- **O que destravou a fusão:** um ramo espectral que generaliza para ruído novo. Com o B++, treinado só com o VB‑DEMAND, os híbridos perderam (P3: −0,074) ou empataram (C3: −0,014) no teste (Fase 6) ✔.
- **Lição de método:** contra o controle, a validação antiga (com os mesmos 10 tipos de ruído do treino) errou o sinal do resultado do teste em 4 de 5 comparações. A val_unseen (ruído não visto) acertou as 5 ✔.
  - A seleção passa a usar a val_unseen (seção 3.4).
  - Os rankings das Fases 1–4 não valem para ruído novo.
- **Pergunta em aberto mais importante:** quanto do ganho do RX_C3 vem da fusão aprendida e quanto viria de uma simples mistura dos dois modelos, sem treino? É o experimento V1 (seção 13). **Respondida: nada (seção 12.9).**

**Plano original (2026‑10‑13):**

- **Problema.** O CleanUNet2‑Mamba híbrido empata com o modelo só de onda no teste: PESQ 3,096 contra 3,095 ✔. O ramo espectral acrescenta 11M parâmetros e dobra o tempo de inferência sem trazer ganho.
- **Causa provável.** Toda a saída do CleanSpecNet chega ao CleanUNet como um **ganho escalar sobre a forma de onda**. Esse ganho é gerado por um FiLM de 8 parâmetros, que o treino deixa quase em zero ✔ (seção 1.2).
- **Achado da implementação.** O próprio CleanSpecNet quase não limpa: a magnitude que ele estima, ressintetizada com a fase ruidosa, fica perto do áudio ruidoso ✔ (seções 1.4 e 6.1). Sem um ramo espectral bom, nenhuma fusão tem o que transmitir.
- **Objetivo.** Encontrar uma forma de fusão que faça o híbrido superar o modelo só de onda, com triagem por treinos curtos de ~10 min por variante ◇, e **melhorar a arquitetura do CleanSpecNet** (trilha N, seção 6) para que ele traga informação útil.
- **Método:**
  1. Um **protocolo curto comum** (seção 3): parte dos dois ramos já treinados, treina só a fusão com os ramos congelados e depois tudo junto por 5 épocas.
  2. **Um eixo por vez**, cada fase partindo da vencedora da anterior: onde injetar (Fase 1), o que injetar (Fase 2), como combinar (Fase 3) e como treinar (Fase 4).
  3. Toda variante é comparada com um **controle só de onda que recebe exatamente o mesmo treino extra**. Outros controles separam "informação do CleanSpecNet" de "parâmetros a mais": espectro ruidoso, condição embaralhada e oráculo.
  4. Em paralelo à fusão, a **trilha N** melhora o CleanSpecNet: primeiro medido sozinho (barato), depois dentro do híbrido. O melhor ramo espectral (B++) passa a valer da Fase 2 em diante.
  5. Os finalistas são confirmados com 3 sementes e o protocolo completo. O conjunto de teste é usado uma única vez, no fim.
- **Custo:** ~75 execuções, ~21 h de GPU no total ◇ (cerca de três noites de fila). Quase todas as variantes dispensam o `SpecUpsampler`. Sem ele, o passo do híbrido custa ~1,3× o do modelo só de onda ✔, em vez de ~8×.

---

## 1. Ponto de partida

### 1.1 Resultados no teste (824 arquivos) ✔

| Modelo | Início | Parâm. | PESQ‑WB | STOI | SI‑SDR | COVL |
|---|---|---|---|---|---|---|
| Só onda, Mamba E8 (estágio A) | DNS | 41,4M | 3,095 | 0,953 | 18,57 | 3,90 |
| Híbrido Mamba, estágios A→B→C1→C2 | DNS | 52,6M | 3,096 | 0,954 | 18,57 | 3,91 |
| Híbrido Mamba, um único treino | DNS | 52,6M | 3,089 | 0,954 | 18,55 | 3,89 |
| Só onda, Mamba | zero, 30 ép. | 41,4M | 2,668 | 0,938 | 16,61 | 3,46 |
| Híbrido Mamba | zero, 30 ép. | 52,6M | 2,650 | 0,939 | 16,38 | 3,45 |
| Híbrido Transformer (E0) | zero, 30 ép. | 58,6M | 2,595 | 0,938 | 16,20 | 3,38 |

### 1.2 Por que o condicionamento atual não funciona ✔

```
|STFT| (513 x T) -> CleanSpecNet -> S_hat (513 x T)
  -> SpecUpsampler: 2x ConvTranspose2d (same weights for every band) -> mean over 513 bands -> c(t), 1 channel
  -> FiLM: x'(t) = (1 + gamma(t)) * x(t) + beta(t),  [gamma, beta] = Conv1d(1 -> 2, k=3)(c)   <- 8 parameters
  -> CleanUNet(x')
```

1. **Gargalo de informação.** A média em frequência deixa um único canal, então nenhuma banda recebe peso próprio.
2. **Operador fraco.** Um ganho escalar sobre a forma de onda não filtra frequências: só muda o volume ao longo do tempo.
3. **Pesos perto de zero.** Depois do treino, o RMS dos pesos do `film_gen` vai de 0,005 (estágio C2) a 0,06 (treino do zero), e o do bias é 0,02. (A inicialização zerada em si não trava o aprendizado: a última camada recebe gradiente desde o 1º passo, e as anteriores logo depois.)
4. **Evidência direta.** O estágio C1 (só a "cola" treina, com os ramos congelados) terminou com PESQ de validação 2,7548, praticamente o ponto de partida (2,755).

### 1.3 Fatos de arquitetura que orientam a fusão ✔

| Ponto do CleanUNet E8 | Stride (amostras) | Taxa | Canais |
|---|---|---|---|
| encoder nível 1 | 2 | 8 kHz | 64 |
| encoder nível 2 | 4 | 4 kHz | 128 |
| encoder nível 3 | 8 | 2 kHz | 256 |
| encoder nível 4 | 16 | 1 kHz | 512 |
| encoder níveis 5–8 | 32–256 | 500–62,5 Hz | 768 |
| gargalo | 256 | 62,5 Hz | 768 → 512 (3 camadas Mamba) → 768 |

- **Hop da STFT (256) = stride total do U‑Net (2⁸).** O quadro *j* da STFT e o token *j* do gargalo andam juntos. Num nível de stride 2^ℓ, cada quadro cobre 256/2^ℓ posições.
- **`spec_hidden`**, a saída do backbone do CleanSpecNet, tem 512 dimensões: o mesmo `d_model` do gargalo.
- **Lookahead.** O token *j* vê as amostras [256j, 256j+766), ou seja 48 ms. Com `alignment: center`, o quadro *j* usa [256j−384, 256j+640).
  - Fundir o quadro *j* no token *j* **não aumenta** o lookahead.
  - Fundir o quadro *j*+1 aumenta em ~8 ms.
  - Com `alignment: causal`, o quadro *j* usa só [256j−768, 256j+256).
- **Custo por passo** (forward + backward, batch 2):

  | Variante | Tempo |
  |---|---|
  | Só onda | 18 ms |
  | Só CleanSpecNet | 6 ms |
  | Híbrido sem upsampler | 24 ms |
  | Híbrido completo | 145 ms |

- **Tempo por época, incluindo a validação:** só onda 0,85 min; híbrido completo 6,5 min; só CleanSpecNet ~0,2 min.

### 1.4 Achado da implementação: o CleanSpecNet treinado quase não limpa ✔

Medido com `scripts/diagnose_spec_branch.py` em 30 arquivos de validação. O erro é |·|^0,3 da magnitude contra a limpa, num domínio comum a todos os modelos. "+ fase ruidosa" é o PESQ da magnitude estimada ressintetizada com a fase do áudio ruidoso.

| Sistema | Erro de magnitude | Erro em SNR ≥ 12,5 dB | PESQ (+ fase ruidosa) |
|---|---|---|---|
| Áudio ruidoso | 0,137 | 0,081 | 1,57 (o próprio sinal) |
| Ramo de onda (estágio A) | **0,047** | 0,037 | 2,84 (saída do modelo) |
| CleanSpecNet do estágio B (10 épocas) | 0,117 | 0,116 | 1,14 |
| Magnitude limpa (oráculo) | 0 | 0 | 3,31 |

Na métrica log1p, os CleanSpecNet de todos os híbridos já treinados (C2, DNS, do zero, E0) ficaram entre 0,058 e 0,067, contra 0,064 do áudio ruidoso, e todos dão PESQ ~1,2 com a fase ruidosa. Isso quer dizer:
- **O ramo espectral nunca teve informação útil a oferecer ao ramo de onda**, o que explica boa parte do empate da seção 1.1.
- Em SNR alto ele **piora** a entrada: o erro fica acima do do áudio ruidoso.

Causas prováveis:
- **Saída direta sem atalho da entrada:** a rede precisa reconstruir as 513 bandas a partir de um gargalo de 64 canais.
- **Compressão log1p quase linear:** as magnitudes normalizadas são pequenas, então o log1p praticamente não comprime e as bandas de baixa energia quase não pesam na perda.
- **Generalização para sequências longas:** num teste rápido com saída por máscara, o erro crescia nos trechos além de 2 s (a duração dos recortes de treino); processado em blocos de 2 s, o erro caía.

**Consequência para o plano:** o D0 deixou de ser "treinar o B por mais tempo" e passou a ser uma **comparação de 7 candidatos de CleanSpecNet** (Fase 0, seção 5), e a arquitetura do CleanSpecNet ganhou uma trilha própria (seção 6). As opções implementadas são `cleanspecnet_params.output_mode` (`direct`, `residual`, `mask`) e `spectrogram.compression` (`log1p`, `pow` com expoente 0,3). O B+ é escolhido pelo PESQ da magnitude estimada com a fase ruidosa, em utterances inteiras.

### 1.5 O que os resultados ensinam sobre a triagem ✔

- **Diferenças abaixo de 0,02 não são confiáveis.** Treinados do zero, o híbrido Mamba ficou à frente do modelo só de onda na validação (2,284 contra 2,277) e atrás no teste (2,650 contra 2,668).
- **O ranking no início de um treino do zero engana.** Na época 10 de 30, o híbrido Mamba era o 3º de 4 na validação e terminou em 1º. Por isso a triagem usa ajuste fino a partir de ramos já treinados, e não treinos curtos do zero.
- **No fim do cosseno, a validação é estável.** Nas 3 últimas épocas do estágio C2, o PESQ de validação variou 0,002. Usar o checkpoint **final** (LR → 0), e não a "melhor época", evita escolher o máximo de uma série ruidosa.
- **Treinos curtos são válidos neste protocolo.** Toda variante começa exatamente igual ao estágio A, porque a fusão é iniciada como identidade. A diferença entre variantes mede só o que a fusão acrescenta.
- **Variância entre sementes ✔.** δ_min ficou em 0,02 em todos os casos.

  | Controle | val | val_unseen | teste |
  |---|---|---|---|
  | K0 (protocolo curto) | σ = 0,005 | σ = 0,001 | — |
  | K0ext (protocolo curto) | σ = 0,004 | σ = 0,002 | — |
  | K0full (protocolo completo) | σ = 0,002 | σ = 0,001 | σ = 0,007 |
  | K0extfull (protocolo completo) | σ = 0,004 | σ = 0,004 | σ = 0,008 |

- **A validação antiga não prevê ruído novo ✔** (atualização de 2026‑10‑15). Ela tem os mesmos 10 tipos de ruído do treino.
  - Nas 50 variantes curtas, a ordem dela foi oposta à da val_unseen (Spearman −0,62).
  - Contra o controle, ela errou o sinal do teste em 4 de 5 comparações.
- **A val_unseen acerta o sinal, mas não o tamanho.** Acertou o sinal nas 5 comparações, mas exagerou as perdas da Fase 6: −0,16 nela, contra −0,07 e −0,01 no teste.
- **Para ordenar híbridos entre si, as duas validações concordaram com o teste:** C3 ficou acima de P3 na Fase 6 e na Fase RT.

---

## 2. Hipóteses

| # | Hipótese | Teste | Fase |
|---|---|---|---|
| H1 | O condicionamento atual não carrega informação: trocar a condição pela de outro arquivo não muda o resultado | D3 | 0 |
| H2 | Injetar features por canal na taxa de quadros (gargalo) permite aproveitar o CleanSpecNet: ganho sobre o controle **e** dependência da condição > 0 | P3, P4 | 1 |
| H3 | Injetar nos níveis finos (encoder, decoder, atalhos) ajuda mais em SNR baixo, onde importa a informação por banda | P1, P2, P6, P7, por faixa de SNR | 1 |
| H4 | O que o ramo espectral *removeu* (a máscara Ŝ − log1p\|X\|) é mais útil que o espectro limpo estimado | S2 × S3 | 2 |
| H5 | O ganho cresce com a qualidade do CleanSpecNet, e o oráculo (espectro limpo verdadeiro) dá o teto | Q1, Q2, K3 | 2 |
| H6 | Uma fusão que depende também do conteúdo do CleanUNet (concatenação, atenção cruzada) supera o FiLM, que depende só da condição | O1–O7 | 3 |
| H7 | O benefício é maior quando o ramo de onda é mais fraco (treino do zero, modelo E6) | Z1, Z2 | 5 |
| H8 | O front‑end de 64 canais do CleanSpecNet é um gargalo de informação: alargá‑lo reduz o erro de magnitude, sobretudo em SNR alto | N1a, N1b | N |
| H9 | Modelar a frequência com pesos compartilhados (dual‑path TF ou sub‑banda) dá um salto de qualidade do ramo espectral com menos parâmetros | N2, N3 | N |
| H10 | Um CleanSpecNet melhor sozinho aumenta o ganho do híbrido (relação monótona entre o PESQ isolado e o Δ sobre o K0) | N‑d, Q1, Q2 | N |
| H11 | Treinar com sequências mais longas elimina a deriva do erro depois de 2 s | N7a | N |

---

## 3. Protocolo curto (PC)

### 3.1 Ponto de partida
- **Ramo de onda:** estágio A (`logs/stageA_cleanunet_mamba/checkpoints/best-epoch016-pesq2.755.ckpt`, PESQ de validação 2,755) ✔.
- **Ramo espectral: B+**, o melhor dos 7 candidatos do D0 (seção 1.4), retreinado por 100 épocas (~20 min ◇). As configurações dele (modo de saída e compressão) valem para todas as variantes; ficam gravadas em `results/abl/bplus.json`.
- **Fusão iniciada como identidade**, com a última camada zerada. No passo 0, toda variante reproduz o estágio A exatamente (verificado por teste unitário).
- **`conditioning_type: none`:** sem `SpecUpsampler` e sem FiLM na forma de onda. A exceção são as variantes que usam o condicionamento original (F0 = K1).

### 3.2 Etapas

| Etapa | O que treina | Duração | LR | Perda |
|---|---|---|---|---|
| PC‑1: aquecimento da fusão | só a fusão (ramos congelados) | 1000 passos (~1,5 época) | 1e‑3, warmup de 5%, cosseno | onda (L1 + MR‑STFT) |
| PC‑2: treino conjunto | tudo | 5 épocas (3375 passos) | fusão 1e‑4; ramos 2e‑5 (`lr_scale` 0,2); cosseno até 1% | híbrida: onda + 1,0 × L1 espectral do CleanSpecNet |

Batch 16, segmentos de 2 s e a augmentação de `base.yaml`.

**Custo por variante** ◇, sem o upsampler:

| Parte | Tempo |
|---|---|
| PC‑1 | ~2 min |
| PC‑2 | ~6 min |
| Avaliações (normal e com a condição embaralhada) | ~2 min |
| **Total** | **~10 min** |

Com o upsampler original (só a referência F0), a variante leva ~45 min, ou ~15 min com o A0 do `IMPROVEMENTS_CLAUDE.md`.

### 3.3 Controles

| ID | Controle | Pergunta |
|---|---|---|
| K0 | Só onda (estágio A) com o mesmo PC‑2: LR 2e‑5, 5 épocas, perda de onda. **3 sementes** | Quanto rende só o treino extra? Qual é o desvio‑padrão entre sementes (σ)? |
| K1 = F0 | Híbrido atual (FiLM na forma de onda), com o PC completo | Onde fica o condicionamento atual no mesmo protocolo? |
| K2 | Melhor fusão alimentada com o espectro **ruidoso** (CleanSpecNet desligado) | O ganho vem da limpeza feita pelo CleanSpecNet ou só de uma visão tempo‑frequência da entrada? |
| K3 | Melhor fusão alimentada com o espectro **limpo verdadeiro** (oráculo, no treino e na avaliação) | Qual é o teto dessa forma de fusão? |
| — | Condição de **outro arquivo** na avaliação (sem treino extra) | O modelo depende de fato da condição? |

### 3.4 Métricas e regra de decisão
- **Métrica principal:** PESQ‑WB de validação (770 arquivos, falantes p226/p287, utterances inteiras), do checkpoint **final** do PC‑2.
- **Atualização (2026‑10‑15): a métrica principal passa a ser o PESQ‑WB na val_unseen.**
  - A val_unseen usa os mesmos 770 arquivos limpos, misturados com 5 ambientes do DEMAND que o VB‑DEMAND não usa: field, hallway, park, river e washing.
  - A validação antiga fica como guarda.
  - O motivo está na seção 1.5.
  - O teste já foi usado duas vezes (Fases 6 e RT). Daqui em diante ele fica só para a confirmação final.
- **Métricas de guarda:**
  - SI‑SDR: não pode cair mais de 0,3 dB em relação ao K0;
  - STOI.
- **Diagnósticos, registrados para toda variante:**
  - **dependência** = PESQ(normal) − PESQ(com a condição de outro arquivo, cortada ou repetida até o comprimento certo);
  - PESQ por SNR de entrada: 0, 5, 10 e 15 dB, os SNRs do conjunto de treino;
  - parâmetros extras, ms por passo, memória de pico e lookahead.
- **Limiar:** δ_min = max(0,02; 2σ), com σ medido no K0 (Fase 0).
- **Promoção em dois passos:**
  1. Roda 1 semente. Se Δ = PESQ − média(K0) ≥ δ_min **e** a dependência ≥ 0,02, roda a 2ª semente.
  2. Promove se a média das duas sementes ≥ δ_min **e** o intervalo de 95% do bootstrap pareado por arquivo (contra o K0) excluir zero.
- **Empate** (diferença < δ_min): fica a variante mais barata em parâmetros, tempo e lookahead.
- **Comparações múltiplas:** com ~50 variantes avaliadas na mesma validação, o maior Δ obtido só por acaso fica perto de 2,5σ ◇. Por isso, nada entra no relatório final sem passar pela Fase 6.

---

## 4. Espaço de design

**Pontos de injeção, nomeados pelo stride:**

| Nome | Onde fica |
|---|---|
| `enc_s{2…256}` | saída de um bloco do encoder |
| `skip_s*` | conexão de atalho (skip) |
| `bott_pre` | depois de `tsfm_conv1`, antes do Mamba; é onde age o `bottleneck_hook` atual ⚑ |
| `bott_post` | depois do Mamba, antes de `tsfm_conv2` |
| `dec_s*` | entrada de um bloco do decoder, já somado o atalho |
| `out` | saída; fusão tardia no domínio tempo‑frequência |

| Eixo | Opções | Padrão (até ser decidido) |
|---|---|---|
| **Onde (P)** | entrada (atual); níveis finos do encoder; `bott_pre`; `bott_post`; decoder grosso ou fino; atalhos; `out` | decidido na Fase 1 |
| **O quê (S)** | `spec_hidden` (512); Ŝ (513 bandas); máscara Ŝ − log1p\|X\|; concatenação; estimativa no tempo iSTFT(Ŝ, fase ruidosa). Controles: ruidoso, oráculo | `spec_hidden` |
| **Como (O)** | operadores O1–O9 da tabela abaixo | O2 (FiLM limitado) |
| **Gerador (G)** | só `Linear`; tronco causal (Conv k=3 → GELU → Conv 1×1, 256 canais) + uma cabeça por ponto; tronco maior | tronco de 256 canais + cabeças |
| **Reamostragem (R)** | repetir o quadro; interpolação linear entre os quadros *j*−1 e *j* (não aumenta o lookahead); `ConvTranspose1d` aprendida | repetir |
| **Alinhamento (L)** | deslocamento de −1, 0 ou +1 quadro; STFT `center` ou `causal` | 0, `center` |
| **Treino (T)** | aquecimento de 0, 1000 ou 3000 passos; LR da fusão; gradiente no CleanSpecNet (stop‑gradient, peso λ_spec); inicialização | 1000 passos; LR 1e‑3 / 1e‑4; ponta a ponta com λ_spec = 1; só a última camada zerada |

### 4.1 Operadores

Notação: *h* são as features do CleanUNet no ponto (B, C, T_ℓ); *c* é a condição reamostrada para T_ℓ; *g* é o gerador; Z(·) é uma camada com pesos zerados na inicialização.

| ID | Operador | Fórmula | Começa como identidade? | Referência |
|---|---|---|---|---|
| O0 | FiLM escalar na forma de onda (atual) | x' = (1+γ)·x + β, um γ e um β por amostra | sim (gerador todo zerado) | CleanUNet2 |
| O1 | FiLM por canal | h' = (1+γ)⊙h + β, com [γ; β] = Z(g(c)) | sim | FiLM (Perez et al., 2018) |
| O2 | FiLM limitado | h' = (1 + tanh γ)⊙h + β | sim | — |
| O3 | adaLN‑Zero | h' = h + α⊙[(1+γ)⊙Norm(h) + β], com α, γ, β = Z(g(c)) | sim | DiT (Peebles & Xie, 2023) |
| O4 | Soma com zero‑conv | h' = h + Z(g(c)) | sim | ControlNet (Zhang et al., 2023) |
| O5 | Concatenação + MLP residual | h' = h + Z(MLP([h; g(c)])) | sim | — |
| O6 | Atenção cruzada com porta, só na taxa de quadros | h' = h + tanh(a)·Attn(q = h, kv = c dos quadros j−W…j), a = 0 no início | sim | Flamingo (porta tanh); HT‑Demucs (atenção entre os ramos) |
| O7 | Troca bidirecional entre as pilhas Mamba dos dois ramos | a cada camada: h_onda += Z(f(h_spec)) e h_spec += Z(f(h_onda)) | sim | Hybrid Demucs / HT‑Demucs |
| O8 | Fusão tardia no domínio TF | ŷ' = iSTFT(M⊙STFT(ŷ)), com M = 1 + tanh(Z(rede([log\|Y\|, Ŝ, log\|X\|]))) | sim | — |
| O9 | Porta nos atalhos | skip' = skip⊙(1 + tanh(Z(g(c)))) | sim | Attention U‑Net (Oktay et al., 2018) |

Diferenças que motivam a Fase 3:
- **O1 e O2** modulam *h* só em função da condição.
- **O5 e O6** combinam *h* e *c*, então a correção pode depender do que o CleanUNet já extraiu (H6).
- **O7** também deixa o CleanSpecNet usar as features do ramo de onda.

---

## 5. Fases e experimentos

### Fase 0: infraestrutura, diagnósticos e piso de ruído (~1,7 h ◇)

| ID | O quê | Custo | Para que serve |
|---|---|---|---|
| D0 | 7 candidatos de CleanSpecNet, 30 épocas cada: saída {direta, residual, máscara} × compressão {log1p, pow 0,3}, mais máscara/pow com recortes de 4 s. Escolha pelo PESQ da magnitude estimada + fase ruidosa (validação inteira). O vencedor é retreinado por 100 épocas (B+) | ~1 h | Ramo espectral usado em todo o plano (seção 1.4) |
| D1 | Qualidade do ramo espectral sozinho: iSTFT(expm1(Ŝ) · fase ruidosa) para B e B+. Teto: magnitude limpa com fase ruidosa | ~5 min, sem treino | Situa o CleanSpecNet entre o áudio ruidoso e o oráculo |
| D2 | Complementaridade. (a) Erro de log‑magnitude por banda e por SNR para o ruidoso, Ŝ e \|STFT(ŷ_A)\|. (b) Teto "melhor dos dois em cada bin", com a fase de ŷ_A. (c) Mistura sem treino exp((1−w)·log\|Y_A\| + w·log\|Ŝ\|), w ∈ {0,25; 0,5} | ~10 min, sem treino | Se Ŝ não for melhor que a saída do ramo de onda em nenhuma região, a fusão tem pouco a acrescentar. Informa P8 |
| D3 | Dependência dos híbridos já treinados (C2, DNS, do zero): avaliar com a condição embaralhada e com a fusão desligada | ~5 min | Testa H1 (esperado: ΔPESQ ≈ 0) |
| K0 | Controle só onda no PC‑2, sementes 1, 2 e 3 | 3 × ~5 min | σ e δ_min |
| K1 | F0 (híbrido atual) com o PC completo | ~45 min (~15 com A0) | Referência |
| I0 | Sanidade: no passo 0, toda variante nova reproduz o estágio A | testes unitários | Garante a comparação justa |

### Fase 1: onde injetar (~2 h ◇)

Fixos nesta fase: sinal `spec_hidden`, operador O2, tronco de 256 canais, repetição de quadros, deslocamento 0.

| ID | Pontos | Parâm. extras ◇ | Hipótese ou motivo |
|---|---|---|---|
| P1 | `enc_s2` | ~0,5M | O 1º conv do encoder é um banco de filtros aprendido: um ganho por canal ali equivale a um ganho por banda |
| P2 | `enc_s2`, `s4`, `s8`, `s16` | ~1,0M | H3 |
| P3 | `bott_pre` ⚑ (o `BottleneckFusion` atual, sem o FiLM na onda) | ~0,7M | H2; a variante mais barata |
| P4 | `bott_post` | ~0,7M | A informação entra depois da modelagem temporal e vai direto ao decoder |
| P5 | `dec_s256`, `s128`, `s64`, `s32` | ~2,0M | Decoder, níveis grossos |
| P6 | `dec_s16`, `s8`, `s4`, `s2` | ~1,0M | H3; níveis que reconstroem os detalhes finos |
| P7 | Porta nos 8 atalhos (O9) | ~1,5M | O ruído vaza pelos atalhos, que o gargalo não controla |
| P8 | `out`: fusão tardia TF (O8) | ~1,5M | Usa a complementaridade medida no D2. A STFT/iSTFT eleva o lookahead total para ~112 ms (1790 amostras), então só vale offline |
| P9 | Combinação dos 2 melhores entre P1 e P8 | — | Os ganhos se somam? |

### Fase 2: o que injetar (~2 h ◇), no melhor ponto P\*

| ID | Condição | Pergunta |
|---|---|---|
| S1 | `spec_hidden` (vencedora da Fase 1) | — |
| S2 | Ŝ (513 bandas, com projeção aprendida) | Features internas ou saída do CleanSpecNet? |
| S3 | Máscara m = Ŝ − log1p\|X\| | H4 |
| S4 | [`spec_hidden`; m] | As duas se complementam? |
| S5 | Estimativa no tempo, iSTFT(magnitude do CleanSpecNet · e^{j∠X}), injetada no 1º nível do encoder por uma conv strided própria com saída zerada. Não é um 2º canal de `encoder.0.0`, para poder treinar no PC‑1 com o CleanUNet congelado | Um sinal físico, na taxa de amostras, funciona melhor? A iSTFT eleva o lookahead total para ~112 ms |
| K2 | log1p\|X\| (ruidoso, CleanSpecNet desligado) | Controle: só a visão tempo‑frequência |
| K3 | log1p\|S\| limpo (oráculo), mesma projeção de S2 | Teto (H5) |
| Q1 | S1 com o CleanSpecNet B (10 épocas) no lugar do B+ | Ganho em função da qualidade do ramo espectral (H5) |
| Q2 | S1 com o CleanSpecNet bidirecional (`cleanunet2_mamba_e8_bidir` ⚑, treinado como o B+) | Teto prático offline; +~20 min para treinar o ramo |

Se a trilha N (seção 6) rodar antes, a Fase 2 usa o B++ no lugar do B+, e o Q1 e o Q2 ficam incorporados ao N‑d.

### Fase 3: como combinar (~2,5 h ◇), com P\* e S\*

| ID | Variante | Condição para rodar |
|---|---|---|
| O1 | FiLM por canal, sem limite | — |
| O3 | adaLN‑Zero | — |
| O4 | Soma com zero‑conv | — |
| O5 | Concatenação + MLP residual | — |
| O6 | Atenção cruzada causal com porta, janela W = 8 quadros (128 ms) | P\* na taxa de quadros |
| O7 | Troca bidirecional entre as pilhas Mamba (exploratória; ~150 linhas de código) | P\* no gargalo |
| G1 | Gerador só com `Linear` | — |
| G3 | Tronco de 512 canais, k = 5 | — |
| R2 | Interpolação linear entre os quadros *j*−1 e *j* | P\* com níveis finos (risco de modulação audível a 62,5 Hz) |
| R3 | `ConvTranspose1d` depthwise aprendida | P\* com níveis finos |
| L−1, L+1 | Deslocamento de −1 ou +1 quadro (+1 acrescenta ~8 ms de lookahead) | P\* na taxa de quadros |
| Lc | STFT `causal` | Mede o custo de deixar o ramo espectral pronto para streaming |

### Fase 4: como treinar a fusão (~1,3 h ◇)

| ID | Variante | Pergunta |
|---|---|---|
| T1 | Sem aquecimento (direto ao PC‑2) | O PC‑1 é necessário? |
| T2 | Aquecimento de 3000 passos | Mais aquecimento ajuda? |
| T3 | LR da fusão de 5e‑4 no PC‑2 | A fusão aprende devagar demais? |
| T4 | Stop‑gradient na condição: o CleanSpecNet treina só pela perda espectral | O gradiente da onda ajuda ou atrapalha o CleanSpecNet? |
| T5 | λ_spec = 0: o CleanSpecNet vira um extrator de features livre | A perda espectral é necessária? |
| T6 | λ_spec = 0,3 | Ponto intermediário |
| T7 | Última camada com inicialização aleatória pequena (N(0; 10⁻³)) em vez de zero (`zero_init: small`) | O gradiente imediato no tronco acelera o aprendizado da fusão? |
| T8 | Currículo de oráculo no PC‑1: usa a condição limpa com probabilidade p, de 0,5 a 0 (exploratória) | Ensina a fusão a ler a condição antes de a condição real ficar ruidosa |

### Fase 5: regimes com ramo de onda mais fraco (~2 h ◇)

| ID | Experimento | Custo ◇ | Observação |
|---|---|---|---|
| Z1 | Do zero, 30 épocas: melhor fusão × só onda (✔ val 2,277 / teste 2,668) × híbrido F0 (✔ val 2,284 / teste 2,650) | ~40 min (1,2 min/época) | Treino longo de propósito, porque rankings no início de um treino do zero enganam (1.5) |
| Z2 | E6 (27,2M parâmetros, lookahead de 12 ms): estágio A‑E6 partindo do DNS E6 (20 épocas), depois PC com a melhor fusão × PC‑controle | ~40 min | Exige STFT `causal` (lookahead ≤ 16 ms). Com a STFT centrada, o lookahead passaria de 12 para ~40 ms |
| Z3 | (Opcional) modelo leve `cleanunet_mamba_light` do zero, com e sem fusão | ~30 min | Verificar antes a compatibilidade entre o hop e o stride total |

### Fase 6: confirmação (~3 h ◇)

- **Quem:** as 2 melhores configurações e o K0.
- **Protocolo completo:** estágio C1 (2000 passos) + estágio C2 (10 épocas) a partir de A + B+, com **3 sementes** cada.
- **Teste** (824 arquivos), uma única vez:
  - média ± desvio entre sementes;
  - Wilcoxon pareado por arquivo, com correção de Holm;
  - resultados por faixa de SNR;
  - dependência da condição;
  - parâmetros, RTF e lookahead.
- **Se a variante for causal:** registrar o lookahead total. O streaming do híbrido ainda não existe (E3 do `IMPROVEMENTS_CLAUDE.md`).

### Fases acrescentadas durante a execução (C, U, R e RT) ✔

| Fase | O quê | Por quê |
|---|---|---|
| C | Combinações offline sobre o P3, no protocolo curto: C1 = P3 + S5, C2 = P3 + P8, C3 = P3 + S5 + P8 | S5 e P8 empataram com o P3 (Fase 2 e N‑x). Os ganhos se somam? |
| 6 com `--finalists auto` | P3 + B++ e o melhor offline da Fase C (C3), com 3 sementes, o protocolo completo e o teste | Confirmação |
| U | Toda execução com resultado na validação é reavaliada, sem treino, na **val_unseen** (`scripts/make_val_unseen.py`): os falantes p226/p287 misturados com os 5 ambientes do DEMAND que o VB‑DEMAND não usa, nos SNRs do teste | Os ganhos da validação não apareceram no teste |
| R | Ramo espectral treinado com **ruído externo**, depois o K0ext e os híbridos RX com o mesmo ruído (detalhes abaixo) | Um ramo espectral que generaliza faz a fusão generalizar? |
| RT | RX_S5P8 (S5 + P8 sem o P3) no protocolo curto; depois o protocolo completo da Fase 6, com 3 sementes, para K0extfull, RX_P3 e RX_C3, avaliados no teste, na val e na val_unseen | Confirmação da Fase R no teste |

**Ruído externo (`noise_bank.py`, opção `data.ext_noise`):**
- Com probabilidade p, o ruído do recorte é trocado por um trecho do MUSAN (CC BY 4.0).
- O SNR é sorteado entre −5 e 20 dB, com o nível de fala medido pela P.56.
- O trecho vem de três fontes, com pesos 0,6 / 0,25 / 0,15:
  - `noise`: 6,2 h;
  - babble de 3 a 7 falantes, montado a partir de `speech` (60 h);
  - `music`: 43 h.
- Os ambientes do DEMAND ficam de fora: tanto os 5 do teste quanto os 5 da val_unseen.

**Fase R, passo a passo:**
1. Treina dois ramos espectrais, com p = 0,5 e p = 0,8, e escolhe pela média do PESQ isolado na val e na val_unseen.
2. Retreina o vencedor por 100 épocas. Esse é o **Bext**.
3. Treina o K0ext: o K0 com o mesmo ruído, 3 sementes.
4. Treina os híbridos RX_P3, RX_S5 e RX_C3, com o Bext e o mesmo ruído.
5. Treina o RX_P3vb: com o Bext, mas só com os dados do VB‑DEMAND.

---

## 6. Trilha N: arquitetura do CleanSpecNet

O D1 (seção 1.4) mostrou que a fusão não tem o que transmitir enquanto o CleanSpecNet não limpar. Esta trilha melhora o ramo espectral **em si**. Cada variante é medida primeiro sozinha (treino de ~10 min e avaliação de ~2 min) e só as melhores entram no híbrido.

**Estado em 2026‑10‑13:** N0–N9 implementados e testados (`tests/test_specnet.py`, 28 testes). O N10 não foi implementado, porque depende de baixar os dados do DNS. A trilha roda automaticamente com `scripts/run_ablation.py --phases N Nd` (seção 6.5).

**Atualização em 2026‑10‑15:**
- A trilha foi executada em 13/10, e os resultados estão na seção 12.2. O B++ é o N1b com 100 épocas.
- O ramo espectral com ruído externo (Bext, Fase R) usa a mesma arquitetura.
- O N10 continua sem implementação. O ruído externo veio do MUSAN, não do DNS.

### 6.1 Diagnóstico da arquitetura atual ✔

```
compressed |X| (513 x T)
  -> input_layer: Conv1d(513 -> 513, k=3)                                     0.79M
  -> 5 blocks: Conv1d(-> 64, k=4) -> ReLU -> Conv1d(64 -> 128, k=4) -> GLU     0.36M   <- 64 channels
  -> Linear(64 -> 512)                                                        0.03M
  -> 5 Mamba layers (d = 512, causal)                                         9.22M (82%)
  -> output_layer: Conv1d(512 -> 513, k=3)                                    0.79M
  = 11.2M parameters (Mamba version)
```

| Problema | Evidência | Consequência |
|---|---|---|
| **Gargalo de 64 canais antes da modelagem temporal.** Cada quadro de 513 bandas vira 64 números já no 1º bloco; o backbone de 512 dimensões recebe uma projeção de posto ≤ 64 | código ✔ | A estrutura harmônica fina não passa: com 15,6 Hz por banda, os harmônicos de uma voz com F0 de 100–300 Hz ficam a cada ~6–20 bandas. A saída tende a um envelope suave |
| **Frequência tratada como canais.** Não há pesos compartilhados ao longo da frequência: um padrão (harmônico, formante) precisa ser aprendido em cada posição separadamente | código ✔ | Pouca eficiência de dados, num corpus de só 9,4 h |
| **82% dos parâmetros no backbone**, que só recebe a projeção de 64 dimensões | contagem ✔ | Capacidade mal distribuída |
| **Saída direta** e **compressão log1p quase linear** | seção 1.4 ✔ | Tratados pelo D0 (modos `residual`/`mask`, compressão `pow`) |
| **Só magnitude**, sem fase na entrada nem na saída | código ✔ | O ramo não gera um sinal completo sozinho; a fase ruidosa limita o S5 e o P8 |
| **Deriva em sequências longas** (variante com máscara, 3 épocas) | seção 1.4 ✔ | O erro cresce depois de 2 s, a duração dos recortes de treino |
| **Janela de 64 ms** (n_fft 1024) | configuração ✔ | Boa resolução em frequência, mas espalha transientes no tempo (pré‑eco) |

**Primeiro resultado do D0** (30 arquivos de validação). É o mesmo CleanSpecNet, com LR 5e‑4 e 30 épocas em vez de 1e‑4 e 10:

| Sistema | Erro de magnitude | PESQ | STOI | SI‑SDR |
|---|---|---|---|---|
| Áudio ruidoso | 0,137 | 1,57 | 0,858 | 6,87 |
| CleanSpecNet B antigo + fase ruidosa | 0,117 | 1,14 | 0,674 | −3,13 |
| D0 `direct_log1p` + fase ruidosa | 0,093 | 1,79 | 0,826 | 6,15 |
| Ramo de onda (estágio A) | 0,047 | 2,84 | 0,928 | 17,71 |
| Magnitude limpa + fase ruidosa (teto) | 0 | 3,31 | 0,974 | 18,93 |

Mais treino ajuda, mas o resultado continua perto do áudio ruidoso, com STOI e SI‑SDR até abaixo dele. O problema é estrutural, não só de otimização.

### 6.2 Propostas

| ID | Proposta | Mudança | Hipótese | Implementação | Treino ◇ |
|---|---|---|---|---|---|
| N0 | Modo de saída e compressão | `direct`/`residual`/`mask` × `log1p`/`pow` | Atalho da entrada + compressão eficaz | ✔ feito (D0, em execução) | ~10 min por candidato |
| N1a | Front‑end mais largo | `conv_hidden_dim` 64 → 256 e `num_conv_layers` 5 → 2 | H8 | ⚑ só configuração | ~10 min |
| N1b | Codificador por quadro | Trocar os blocos conv por `Linear(513 → 512)` + conv causal depthwise no tempo | H8, versão mais simples | ✔ `frontend: linear` | ~10 min |
| N2 | **Dual‑path tempo‑frequência com Mamba** | Encoder 2D (B, C, F, T) com redução de 4× em frequência; 4 blocos com Mamba **causal no tempo** (por banda) e Mamba **bidirecional na frequência** (por quadro, o que não fere a causalidade); decoder 2D com atalhos do encoder e máscara limitada. 16 canais na resolução cheia e 64 nos níveis reduzidos | H9; 0,98M parâmetros ✔ | ✔ `cleanunet/tf_mamba.py` | ~2 min/época ✔ (~60 min) |
| N3 | Banda cheia + sub‑banda | Modelo atual (com N1) + um modelo pequeno, compartilhado entre bandas, que vê a banda *f* ± *k* e a saída da banda cheia; a saída é uma correção da estimativa de banda cheia, zerada no início | H9, alternativa mais leve | ✔ `cleanunet/subband.py` | ~1 min/época ✔ (~30 min) |
| N4 | Saída com fase | Decoder de magnitude (máscara) + decoder de fase (atan2 de duas saídas). Perdas: magnitude, espectro complexo, fase anti‑wrapping (instantânea, atraso de grupo e frequência instantânea, ponderadas pela magnitude limpa) e consistência | O ramo passa a gerar um sinal completo; o S5 passa a usar a fase prevista | ✔ `predict_phase: true` + `loss.spec_terms` | — |
| N5 | Entrada com fase | [\|X\|^0,3, cos∠X, sin∠X] | A fase ruidosa carrega pistas temporais (atraso de grupo, frequência instantânea) | ✔ `input_features: mag_phase`. No modelo 1‑D, a 1ª conv passa a ter 3 × 513 canais: 19,4M parâmetros em vez de 11,2M | — |
| N6a | Janela de 32 ms | n_fft = win = 512 e hop 256 (257 bandas): continua 1 quadro por token | Menos pré‑eco e menos bandas (mais barato). O quadro passa a olhar só 384 amostras à frente, em vez de 640 | ⚑ só configuração | ~8 min |
| N6b | Hop de 128 | n_fft 512 e hop 128; a fusão agrega 2 quadros por token | Melhor resolução temporal no ramo espectral | ✔ média dos quadros dentro de cada posição (`fusion.py`) | ~15 min |
| N7a | Sequências longas | Recortes de 6 s formados por 3 utterances concatenadas (crossfade de 10 ms); 10 épocas com batch 6, ≈ o mesmo volume de áudio de 30 épocas de recortes de 2 s | H11 | ✔ `data.concat_utterances` | ~15 min |
| N7b | Normalização causal da entrada | Divide a magnitude pela média acumulada no tempo (média em frequência), como no FullSubNet | Estatísticas estáveis ao longo do tempo | ✔ `input_norm: cumulative` | ~10 min |
| N8a | Perda no tempo | L1 + MR‑STFT de iSTFT(\|Ŝ\|·e^{j∠X}) contra o sinal limpo (peso 0,2), somadas ao L1 espectral | Otimiza diretamente o que o D1 mede | ✔ `loss.spec_terms: {l1: 1, wave: 0.2}` | ~12 min |
| N8b | Perdas complexa e de consistência | Junto com o N4: `{l1: 1, phase: 0.3, complex: 0.1, consistency: 0.1}` (+ `wave: 0.2` se o N8a ajudar) | — | ✔ | — |
| N9 | Escala | Dois pontos por arquitetura: 1‑D (8 camadas; d = 256), TF (6 blocos; 96 canais), sub‑banda (3 camadas; d = 64) | Redistribuir a capacidade depois de N1/N2 | ✔ `--with_nc` | 2 × 10–60 min |
| N10 | Mais dados (opcional) | Pré‑treinar o CleanSpecNet no DNS Challenge, de onde já veio o ramo de onda | A assimetria de dados entre os ramos (centenas de horas × 9,4 h) limita o espectral | download grande ◇ | horas |

Referências para N2–N5: SEMamba (Chao et al., SLT 2024), MP‑SENet (Lu et al., Interspeech 2023), FullSubNet (Hao et al., ICASSP 2021), DPCRN (Le et al., Interspeech 2021) e TF‑GridNet (Wang et al., 2023).

**Notas de desenho:**
- **Causalidade.** No N2 e no N3, a frequência pode ser bidirecional, porque só usa o quadro atual. O tempo continua causal, então o ramo segue compatível com streaming. O teste de causalidade (perturbar quadros futuros e conferir os passados) vale para todas as variantes.
- **Interface com a fusão.** Uma variante 2D (N2) precisa de um vetor por quadro para o `spec_hidden`. Proposta ◇: média em ~16 sub‑bandas e projeção de C·16 para 512. Os sinais `spec` e `mask` funcionam sem mudança.
- **Alinhamento.** O N6a mantém hop 256, ou seja, 1 quadro por token. O N6b exige agregar quadros na fusão.

### 6.3 Protocolo da trilha (PN)

1. **Treino isolado** do CleanSpecNet, como no estágio B: 30 épocas, LR 5e‑4 com cosseno, perda espectral (mais os termos do N8 quando for o caso) e as configurações de saída e compressão do B+.
2. **Avaliação isolada** nos 770 arquivos de validação (~2 min), com `scripts/diagnose_spec_branch.py`:
   - PESQ, STOI e SI‑SDR da magnitude estimada com a fase ruidosa (no N4, também com a fase prevista);
   - erro de magnitude por banda e por faixa de SNR;
   - **deriva**: erro depois de 4 s dividido pelo erro nos primeiros 2 s, nos arquivos com mais de 4 s;
   - **complementaridade** com o estágio A: fração dos bins, ponderada pela energia, em que Ŝ fica mais perto do limpo que \|STFT(ŷ_A)\|, e o teto "melhor dos dois".
3. **Promoção:** as 2 melhores pelo PESQ isolado vão para o teste no híbrido. Empate: decide a complementaridade, depois o custo.
4. **Teste no híbrido (N‑d):** cada CleanSpecNet promovido entra na fusão vencedora da Fase 1, com o protocolo curto da seção 3, e é comparado com o K0 e com a mesma fusão usando o B+. **Este é o critério final**: um CleanSpecNet melhor sozinho só interessa se aumentar o ganho do híbrido.

**Metas na validação** ◇:

| Medida | Melhor até agora | Meta |
|---|---|---|
| PESQ isolado (+ fase ruidosa) | 1,79 | ≥ 2,3 |
| Erro de magnitude por faixa de SNR | acima do ruidoso em SNR alto | abaixo do ruidoso em todas as faixas |
| Deriva | > 1 (máscara, 3 épocas) | ≤ 1,1 |
| Teto "melhor dos dois" | 3,13 com o B antigo (30 arquivos) | ≥ ramo de onda + 0,2 |
| Ganho do híbrido sobre o K0 | — | ≥ δ_min, com dependência > 0,02 |

### 6.4 Etapas

| Etapa | Variantes | GPU ◇ | Depende de |
|---|---|---|---|
| N‑a: configuração e mudanças pequenas | N1a, N1b, N6a, N7a, N7b, N8a, um fator por vez sobre o B+ | ~1,5 h | B+ (Fase 0) |
| N‑b: arquiteturas novas | N2 e N3 com as melhores opções da N‑a; depois N4 + N5 sobre a melhor das duas | ~3 h | N‑a |
| N‑c: escala e dados (opcional) | N9; o N10 se houver dados do DNS | ~0,5 h (+ N10) | N‑b |
| N‑d: teste no híbrido | Os 2 melhores CleanSpecNets na fusão vencedora da Fase 1 (protocolo curto); repetir os 2–3 melhores pontos da Fase 1 com o novo ramo | ~1 h | Fase 1 e N‑b |

**Posição no cronograma:** depois da Fase 1 e antes da Fase 2. O melhor CleanSpecNet da trilha (**B++**) passa a ser o ramo espectral das Fases 2–6, e o Q1 e o Q2 da Fase 2 ficam incorporados ao N‑d.

### 6.5 Execução automática (`scripts/run_ablation.py --phases N Nd`)

| Passo | Regra |
|---|---|
| Referência | O candidato vencedor do D0 com o **mesmo orçamento** das variantes N (30 épocas), avaliado pelo mesmo script |
| N‑a | Cada opção é treinada sozinha sobre o B+. Uma opção é **mantida** se o PESQ isolado subir ≥ 0,05 sobre a referência |
| N‑b | Se 2 ou mais opções forem mantidas, treina o modelo 1‑D com todas (`N_a_combo`). N2 e N3 recebem as opções genéricas mantidas (N6a, N7a, N7b, N8a); o N3 recebe também o melhor front‑end. N4 e N4+N5 são treinados sobre a melhor arquitetura até ali |
| N‑c | Só com `--with_nc`: as duas variantes de escala da melhor arquitetura |
| B++ | A melhor variante, se superar a referência em ≥ 0,05, é retreinada com 10/3 do seu orçamento (como o B+: 30 → 100 épocas) e gravada em `results/abl/bplusplus.json`. Caso contrário, o B++ é o próprio B+ |
| N‑d | Os 3 melhores pontos da Fase 1 com o B++ (`--winner auto` escolhe a vencedora pelo PESQ de validação), mais o 2º melhor ramo espectral (checkpoint de 30 épocas, sem retreino) no ponto vencedor. Comparação com o K0 e com os mesmos pontos usando o B+ em `results/abl/phaseNd.md` |

Resultados isolados ficam em `results/abl/N/<variante>/diagnostics.json`, com PESQ, STOI e SI‑SDR, erro por banda e por faixa de SNR, deriva e complementaridade.

---

## 7. Árvore de decisão

| Observação | Conclusão | Próximo passo |
|---|---|---|
| D3: dependência ≈ 0 nos híbridos atuais | H1 confirmada | Seguir para a Fase 1 |
| Alguma variante da Fase 1 ≥ δ_min, com dependência > 0 | A fusão certa funciona | Fases 2–4 refinam a vencedora |
| Nenhuma variante passa, mas o oráculo K3 dá ≥ +0,1 | O limite é a qualidade do CleanSpecNet, não a fusão | Trilha N (seção 6), depois repetir os melhores pontos da Fase 1 e a Fase 2 com o B++ |
| Nenhuma variante passa e o K3 dá < +0,05 | Neste dataset, o ramo de onda (E8 + DNS) já extrai essa informação sozinho | Fase 5. Se também falhar: relatar o resultado negativo e simplificar (remover o ramo espectral, ou usá‑lo só como professor, E1) |
| K2 (espectro ruidoso) ≈ melhor variante | O ganho vem da visão tempo‑frequência, não da limpeza feita pelo CleanSpecNet | Trocar o CleanSpecNet por um codificador TF leve |
| Ganho só em SNR baixo | Útil, mas o VB‑DEMAND tem pouco SNR baixo | Avaliar também em dados mais difíceis (teste do DNS) |
| O B+ já atinge PESQ isolado ≥ 2,3 | O D0 resolveu o essencial | Trilha N reduzida: N‑d e, se sobrar tempo, o N2 |
| O PESQ isolado sobe muito (N2/N3), mas o ganho do híbrido continua < δ_min | O ramo espectral ficou bom, mas não traz informação nova ou a fusão não a aproveita | Olhar a complementaridade (D2). Se for alta: Fase 3 (operadores) e P8. Se for baixa: o ramo de onda já sabe o mesmo |
| Nenhuma variante da trilha N passa de PESQ isolado ~2 | O limite está fora da arquitetura | Revisar STFT, normalização e perda (N6–N8) e conferir o pipeline antes de tentar arquiteturas maiores |
| Os ganhos na validação não aparecem no teste (Fase 6) ✔ | A validação tem os mesmos ruídos do treino, e a fusão confia demais num ramo espectral que só viu o VB‑DEMAND | Validar com ruído não visto (Fase U); treinar o ramo espectral e o híbrido com ruído externo (Fase R) |
| Com ruído externo, o híbrido ganha nas duas validações ✔ | A fusão funciona quando o ramo espectral generaliza | Confirmar no teste (Fase RT) |
| A mistura sem treino K0ext + Bext empata com o RX_C3 (V1) | A fusão aprendida não se justifica | Sistema offline = K0ext + Bext + mistura. Pesquisa de fusão só para o que a mistura não faz: streaming e destilação (V6) |
| O RX_C3 supera a mistura por ≥ δ_min na val_unseen (V1) | A fusão aprendida acrescenta algo | Seguir com o RX_C3: condition dropout (V2) e ramo espectral melhor (V4) |

**Caminho seguido (13 a 15/10):**
1. O D3 mostrou dependência ≈ 0 nos híbridos antigos, o que confirmou a H1.
2. A Fase 1, com o B+, deu ganhos abaixo de δ_min.
3. A trilha N produziu o B++.
4. O N‑d deu +0,06 na validação.
5. As Fases 2–4 só deram empates.
6. A Fase C chegou a +0,12 na validação, com o C3.
7. A Fase 6 mostrou que esse ganho não se transfere ao teste.
8. A Fase U construiu e aplicou a validação com ruído não visto.
9. A Fase R treinou com ruído externo.
10. A Fase RT confirmou o RX_C3 no teste: +0,048 sobre o K0extfull.

---

## 8. Implementação

**Estado em 2026‑10‑15:**
- Todas as fases estão implementadas e foram executadas: 0–6, N, N‑d, N‑x, C, U, R e RT.
- A suíte de testes coleta 115 testes:

  | Arquivo | Testes |
  |---|---|
  | `test_fusion.py` | 36 |
  | `test_specnet.py` | 30 |
  | `test_models.py` | 29 |
  | `test_cleanumamba_compat.py` | 7 |
  | `test_training_pipeline.py` | 6 |
  | `test_noise_bank.py` | 4 |
  | `test_val_unseen.py` | 3 |

- Nada foi commitado ainda no `feature/mamba`.

(Até 2026‑10‑13, as Fases 0 e 1 estavam na fila e a trilha N começava automaticamente depois delas.)

| Para | Comando |
|---|---|
| Rodar as Fases 0 e 1 (já em execução) | `setsid nohup python scripts/run_ablation.py --phases 0 1 > logs/abl/queue.log 2>&1 &` |
| Rodar as Fases 2–4 depois de escolher a vencedora | `python scripts/run_ablation.py --phases 2 3 4 --winner P3_bott_pre [--signal mask] [--operator adaln_zero]` |
| Rodar a trilha N e o teste no híbrido | `python scripts/run_ablation.py --phases N Nd --winner auto [--with_nc]` |
| Rodar a Fase 6 | `python scripts/run_ablation.py --phases 6 --finalists <id1> <id2>` (ou `--finalists auto --winner P3_bott_pre`) |
| Rodar as combinações offline | `python scripts/run_ablation.py --phases C --winner P3_bott_pre` |
| Reavaliar tudo na val_unseen | `python scripts/run_ablation.py --phases U` (relatórios `results/abl/U_*.md`) |
| Rodar o treino com ruído externo e a confirmação | `python scripts/run_ablation.py --phases R RT --winner P3_bott_pre` (relatórios `results/abl/phaseR*.md`, `phaseRT*.md`) |
| Gerar a val_unseen | `python scripts/make_val_unseen.py` (escreve `val_unseen/` no diretório do dataset e `filelists/val_unseen.csv`) |
| Ver os comandos sem executar | acrescentar `--dry_run` |
| Comparar com o K0 | `python scripts/compare_runs.py --ref results/abl/K0_s{1,2,3} --runs 'results/abl/P*'` (a fila já grava `results/abl/phase<N>.md`) |
| Diagnosticar ramos espectrais | `python scripts/diagnose_spec_branch.py --wave_ckpt <A> --spec_ckpt nome=<ckpt> ...` |
| Avaliar a dependência da condição | `python evaluate.py --checkpoint <ckpt> --filelist filelists/val.csv --fast --condition shuffle\|off` |

Onde ficam os resultados:
- checkpoints e logs em `logs/abl/<id>_pc1|_pc2/`;
- avaliações em `results/abl/<id>/val` e `val_shuffle`, com a definição da variante em `variant.json`;
- a fila é reiniciável: passos concluídos são pulados e treinos interrompidos retomam do `last.ckpt`.

### 8.1 Arquivos (implementados)

| Arquivo | Mudança |
|---|---|
| `cleanunet/fusion.py` (novo) | `CondTrunk` (tronco causal na taxa de quadros); `FusionPoint` com os operadores `film`, `film_tanh`, `adaln_zero`, `add`, `gate`, `concat_mlp`, `xattn`; reamostragem `repeat`/`linear`/`convT` e `shift`; `FusionHub` (ganchos por ponto); `InputEstimateFusion` (S5); `BranchExchange` (O7, pilhas Mamba em lockstep) |
| `cleanunet/postfilter.py` (novo) | O8: máscara limitada no domínio da STFT sobre a saída do CleanUNet |
| `cleanunet/cleanunet.py` | `fusion_points()`; `forward(..., hooks=None, bottleneck_runner=None)` com ganchos em `enc_s*`, `skip_s*`, `bott_pre`, `bott_post`, `dec_s*`. O `bottleneck_hook` continua (= `bott_pre`) ⚑ |
| `cleanunet/cleanspecnet.py` | `output_mode: direct \| residual \| mask`; `forward` dividido em `encode()` e `head()` (necessário para o O7) |
| `cleanunet/spectral.py` | `complex_stft`, `istft` (overlap‑add próprio, inverte a STFT com erro < 1e‑6) e `decompress` |
| `cleanunet/cleanunet2.py` | `conditioning_type: none` (sem upsampler nem FiLM na onda); bloco `fusion:`; sinais `spec_hidden`, `spec`, `mask`, `noisy_spec`, `oracle_spec`, `oracle_mask` (ou uma lista); `forward(..., clean=, cond_noisy=, cond_clean=, fusion_enabled=)`; `oracle_prob` para o currículo T8 |
| `lightning_modules/cleanunet_module.py` | Passa `clean` ao modelo; perda espectral só quando o CleanSpecNet roda; `fusion_curriculum: {oracle_prob, steps}`. O `lr_scale` já aceita `fusion_hub`, `postfilter`, `input_estimate`, `exchange` ⚑ |
| `config_utils.py` | Overrides `model.architecture.<chave>=...` aplicados depois de resolver o YAML da arquitetura |
| `evaluate.py` | `--condition {normal,shuffle,off}`; `--fast`; arquitetura lida do checkpoint Lightning; SNR de entrada por arquivo no `metrics.csv` e resumo por faixa de SNR |
| `scripts/run_ablation.py` (novo) | Fila reiniciável das Fases 0–6 (espera memória livre na GPU, retoma após OOM) |
| `scripts/diagnose_spec_branch.py` (novo) | D0 (escolha do B+), D1 e D2 |
| `scripts/compare_runs.py` (novo) | Δ pareado por arquivo contra o K0, bootstrap de 95%, Wilcoxon + Holm, dependência, Δ por faixa de SNR |
| `configs/model/fusion/*.yaml` | `base.yaml`, `base_e6.yaml`, `F0_waveform_film`, `P1`–`P8`, `S5_input_estimate`, `O7_exchange` |
| `configs/ablation/*.yaml` | `stageB_plus` (D0/B+), `pc1`, `pc2`, `k0` |
| `tests/test_fusion.py` (novo) | Identidade no início para 21 variantes; gradiente na fusão no 1º passo; lookahead de 766 amostras preservado com a geometria E8; oráculo e condição de outro arquivo; inversão da STFT; carga dos ramos pré‑treinados |
| `noise_bank.py` (novo) | Nível de fala ativa pela P.56 (`active_speech_power`), `mix_at_snr` e `NoiseBank`: trechos de `noise`, babble e `music` do MUSAN, com pesos |
| `dataset.py`, `lightning_modules/data_module.py` | `data.ext_noise: {dir, p, snr_db, weights, babble_talkers}` (só no treino; padrão desligado) |
| `scripts/make_val_unseen.py` (novo) | Val_unseen: falantes da validação com os 5 ambientes do DEMAND que o VB‑DEMAND não usa, nos SNRs do teste (P.56, semente 0) |
| `scripts/run_ablation.py` | Fases C, U, R e RT; `--finalists auto`; `variant(..., splits=...)`; tabelas de média ± desvio por configuração (`phase*_seeds*.md`) |
| `tests/test_noise_bank.py`, `tests/test_val_unseen.py` (novos) | Nível de fala ativa (P.56) ignora as pausas; mistura no SNR pedido; o `NoiseBank` sorteia todas as fontes no comprimento pedido e recusa fonte ausente; o dataset troca o ruído só nos recortes de treino; Spearman da Fase U |

### 8.2 Configuração: cada variante é só um YAML

```yaml
# configs/model/fusion/p3_hidden_film.yaml
base: ../cleanunet2_mamba_e8.yaml
conditioning_type: none            # no SpecUpsampler, no waveform FiLM
fusion:
  signal: spec_hidden              # spec_hidden | spec | mask | noisy_spec | oracle_spec | [spec_hidden, mask]
  trunk: {channels: 256, kernel: 3, layers: 2, causal: true}
  resample: repeat                 # repeat | linear (frames j-1..j) | convT
  shift: 0                         # frames; +1 adds ~8 ms of lookahead
  zero_init: last                  # last | small
  points:
    bott_pre: {op: film_tanh}      # film | film_tanh | adaln_zero | add | concat_mlp | xattn | gate
```

```yaml
# configs/ablation/pc1.yaml - short protocol, step 1: fusion warm-up with both branches frozen
base: ../base.yaml
trainer: {max_steps: 1000, max_epochs: -1}
model:
  init:
    cleanunet_checkpoint: logs/stageA_cleanunet_mamba/checkpoints/best-epoch016-pesq2.755.ckpt
    cleanspecnet_checkpoint: logs/abl/Bplus/last.ckpt
  train_cleanunet: false
  train_cleanspecnet: false
loss: {mode: waveform}
optim: {lr: 1.0e-3}
```

```yaml
# configs/ablation/pc2.yaml - short protocol, step 2: joint fine-tuning (init = last.ckpt of PC-1)
base: ../base.yaml
trainer: {max_epochs: 5}
loss: {mode: hybrid}
optim:
  lr: 1.0e-4
  lr_scale: {clean_unet: 0.2, clean_spec_net: 0.2}
```

Cada execução recebe `model.architecture=../model/fusion/<id>.yaml`, as opções do B+ (`model.architecture.cleanspecnet_params.output_mode=...`, `model.architecture.spectrogram.compression=...`), `seed=<n>` e os diretórios próprios pela linha de comando; `scripts/run_ablation.py` monta tudo isso. O K0 usa `configs/ablation/k0.yaml` (só onda, 5 épocas, LR 2e‑5, perda de onda).

### 8.3 Arquivos da trilha N (implementados)

| Arquivo | Mudança | Propostas |
|---|---|---|
| `cleanunet/cleanspecnet.py` | `frontend: conv \| linear`; `feature_groups` (entrada com fase); `input_norm: none \| cumulative`; `predict_phase`; `run()` devolve `spec`, `hidden` e `phase`; funções comuns de saída e normalização | N0, N1b, N4, N5, N7b |
| `cleanunet/tf_mamba.py` (novo) | `TFMambaSpecNet`: encoder/decoder 2D causais no tempo, blocos Mamba tempo (causal) + frequência (bidirecional), `hidden` por sub‑bandas | N2 |
| `cleanunet/subband.py` (novo) | `SubbandSpecNet`: banda cheia (CleanSpecNet 1‑D) + sub‑banda compartilhada entre bandas | N3 |
| `cleanunet/cleanunet2.py` | `cleanspecnet_params.architecture: conv1d \| tf_mamba \| subband`; `cleanspecnet_params.input_features: mag \| mag_phase`; `spec_features()`, `spec_branch()`, `spec_to_wave()`, `target_stft()`; saída `phase` | N2–N5 |
| `losses.py` | `anti_wrapping`, `phase_losses` | N4 |
| `lightning_modules/cleanunet_module.py` | `loss.spec_terms: {l1, mse, wave, phase, complex, consistency}` (padrão `{l1: 1}` = perda original) | N4, N8 |
| `dataset.py`, `lightning_modules/data_module.py` | `data.concat_utterances`, `data.crossfade` | N7a |
| `cleanunet/fusion.py` | Média dos quadros dentro de cada posição quando hop < stride | N6b |
| `scripts/diagnose_spec_branch.py` | STFT própria de cada modelo; deriva; complementaridade; ressíntese com a fase prevista | protocolo PN |
| `scripts/run_ablation.py` | Fases `N` e `Nd`; B++ nas Fases 2–6; `--winner auto`; `--with_nc` | protocolo PN |
| `configs/model/cleanunet2_mamba_e8_tfspec.yaml`, `cleanunet2_mamba_e8_subband.yaml` | Presets com o N2 e o N3 | N2, N3 |
| `tests/test_specnet.py` (novo) | Causalidade no tempo de todas as variantes (com e sem fase), identidade inicial dos modos residual e sub‑banda, números de bandas arbitrários, normalização acumulada, perdas de fase, termos da perda espectral (com gradiente na cabeça de fase), agregação de quadros na fusão e exemplos concatenados | todas |

**Custo medido por passo** (batch 16, 2 s, bf16, com outro treino na GPU) ✔:

| Ramo espectral | Parâmetros | ms/passo | Memória de pico |
|---|---|---|---|
| 1‑D atual (B+) | 11,2M | 13 | 0,5 GB |
| N1a (front‑end de 256 canais) | 12,8M | 13 | 0,5 GB |
| N1b (front‑end linear) | 10,3M | 11 | 0,5 GB |
| N2 (dual‑path TF) | 0,98M | 173 | 6,3 GB |
| N2 com fase (N4) | 1,0M | 194 | 6,8 GB |
| N3 (sub‑banda) | 11,2M | 92 | 3,5 GB |
| 1‑D com fase na entrada e na saída (N4 + N5) | 19,4M | 11 | 0,7 GB |

Nos modelos 1‑D, a época (~20 s) é dominada pela leitura dos dados, não pela rede.

---

## 9. Orçamento

| Fase | Execuções | GPU ◇ |
|---|---|---|
| 0 | D0–D3, K0 × 3, K1 | ~1,7 h (~1 h com A0) |
| 1 | P1–P9, mais a 2ª semente das aprovadas | ~2 h |
| 2 | S2–S5, K2, K3, Q1, Q2 | ~2 h |
| 3 | ~12 variantes | ~2,5 h |
| 4 | T1–T8 | ~1,3 h |
| 5 | Z1, Z2 (+ Z3) | ~2 h |
| 6 | 3 configurações × 3 sementes, protocolo completo + teste | ~3 h |
| N | N‑a (6 variantes), N‑b (3–5), retreino do B++, N‑d (4), N‑c (opcional) | ~6–8 h (o N2 custa ~1 h; o B++ com o N2 ~3,5 h) |
| **Total** | **~75** | **~21 h** |

Cada fase cabe numa fila noturna. As Fases 0 e 1 juntas levam ~4 h e já respondem à pergunta principal: existe alguma forma de fusão que funcione com o ramo espectral atual? A trilha N responde à pergunta seguinte: com um ramo espectral bom, a fusão passa a valer a pena?

**Executado ✔ (13/10 a 15/10, 10:16):**
- 163 treinos, contando PC‑1 e PC‑2 separadamente, e 311 avaliações.
- Cada fila coube em meio dia ou numa noite.
- Uma variante curta levou ~7–10 min; uma semente do protocolo completo, ~10–15 min.
- A GPU foi compartilhada com outro usuário. Só a Fase R passou ~2 h esperando memória livre.

---

## 10. Riscos e cuidados

- **Efeito pequeno.** Com o ramo de onda já em 3,09 depois do DNS e do ajuste fino, ganhos realistas devem ficar ≤ 0,05 ◇. Sementes, controles e a medida de dependência são indispensáveis. Medido ✔: o RX_C3 ganhou +0,048 sobre o K0extfull no teste.
- **Seleção na validação.** Com muitas variantes, a validação acaba "usada" pela seleção. Só a Fase 6 (3 sementes + teste) vale para o relatório.
  - **Aconteceu ✔, e de um jeito pior que o previsto.** O problema não foi só o acaso de muitas comparações: a validação tem os mesmos tipos de ruído do treino e premiou variantes que não generalizam (seção 1.5).
- **Uso repetido do teste.** O teste já entrou em duas decisões (Fases 6 e RT). A seleção agora usa a val_unseen, e o teste fica para uma única confirmação final dos próximos finalistas.
- **Dependência alta.** O RX_C3 perde 2 pontos de PESQ com a condição de outro arquivo. Se o ramo espectral falhar num ruído que não conhece, não há rede de segurança (V2, seção 13).
- **Dados externos e comparação com a literatura.**
  - O ramo de onda vem do DNS, e o K0ext e os RX usam MUSAN.
  - Na comparação com números publicados no VB‑DEMAND, separar "só VB‑DEMAND" de "com dados externos".
  - O K0full (3,085) é o controle sem ruído externo.
- **Overfitting.** O VB‑DEMAND tem só 9,4 h de áudio. Módulos maiores (O6, O7, P5) podem decorar o treino; acompanhar a diferença entre treino e validação.
- **Lookahead e latência.** Registrar para cada variante. P8, S5 (iSTFT) e L+1 aumentam o lookahead, então valem só para o modo offline ou precisam ser reportadas à parte.
- **Artefatos de taxa de quadro.** Um γ constante por 16 ms nos níveis finos pode criar modulação audível a 62,5 Hz. Por isso existem as variantes R2/R3, e vale ouvir os exemplos que o treino já registra (`log_audio_examples`).
- **GPU compartilhada.** A fila espera memória livre e retoma do `last.ckpt` (já implementado ⚑).
- **Custo do N2/N3.** Sequências por banda (tempo) e por quadro (frequência) multiplicam o número de sequências do Mamba. Medir a época antes de enfileirar várias variantes; reduzir a frequência (4×) e as bandas (N6a) se for preciso.
- **Ramo espectral bom, mas redundante.** Um CleanSpecNet forte pode aprender o mesmo que o ramo de onda. A taxa de complementaridade (protocolo PN) mostra isso antes do teste no híbrido.
- **Comparabilidade.** O K0 e todas as variantes partem do mesmo checkpoint do estágio A, com o mesmo número de passos e o mesmo LR no ramo de onda.

---

## 11. Relação com `IMPROVEMENTS_CLAUDE.md`

| Proposta de lá | Neste plano |
|---|---|
| A1 (upsampler multicanal) | Substituída por P1 e S2 com O1: a mesma ideia, sem o upsampler |
| A2 (fusão no gargalo) | P3/P4 com O1/O5 e L±1 |
| A3 (fusão multiescala) | P2, P5, P6 |
| A4 (pós‑filtro espectral) | P8 / O8 |
| A5 (controles) | K0–K3 e a medida de dependência |
| C4 (porta nos atalhos) | P7 / O9 |
| B1 (saída residual ou por máscara) | N0. No D0, venceu a máscara com log1p |
| B2 (backbone dual‑path TF com Mamba) | N2 |
| B3 (representação da entrada) | N0 (compressão `pow`), N5 (fase), N6 (janela e hop) |
| B4 (causalidade separada, versão bidirecional) | Q2; no N2, frequência bidirecional com tempo causal |
| B5 (estimativa de fase) | N4 |

- O **A0** passa a ser necessário só para a referência F0 (K1), a única que ainda usa o upsampler original.
- O Grupo B virou a trilha N (seção 6), que roda logo depois da Fase 1.

---

## 12. Registro de resultados ✔

Convenções desta seção:
- Δ é a diferença de PESQ‑WB contra a média das sementes do controle indicado.
- "Dep." é a dependência: PESQ normal menos PESQ com a condição de outro arquivo.
- As tabelas completas, com intervalos de 95%, p de Holm e faixas de SNR, estão em `results/abl/*.md`.

### 12.1 Fases 0 e 1: ramo espectral B+, protocolo curto, val antiga

- **K0** (3 sementes): 2,760 na val (σ = 0,005, então δ_min = 0,02) e 3,092 na val_unseen.
- **D0 → B+:** o vencedor foi a saída por máscara com log1p, PESQ isolado (com a fase ruidosa) de 1,97. Os outros candidatos:

  | Candidato | PESQ isolado |
  |---|---|
  | `mask_pow` | 1,97 |
  | `mask_pow` com recortes de 4 s | 1,94 |
  | `residual_pow` | 1,92 |
  | `direct_log1p` | 1,89 |
  | `residual_log1p` | 1,75 |
  | `direct_pow` | 1,31 |

- **D3 e K1:** a fusão original (F0) tem dependência 0,000 e Δ de −0,003. Isso confirma a H1.
- **Fase 1:** com o B+, nenhum ponto chegou a δ_min.

| ID | Ponto | Δ val | Dep. |
|---|---|---|---|
| P3 | `bott_pre` | +0,008 | 0,01 |
| P2 | encoder fino | +0,007 | 0,01 |
| P5 | decoder grosso | +0,004 | 0,04 |
| P4 | `bott_post` | +0,004 | 0,02 |
| P6 | decoder fino | +0,003 | 0,00 |
| P8 | pós‑filtro TF | +0,003 | 1,18 |
| P1 | `enc_s2` | −0,001 | 0,00 |
| P7 | portas nos atalhos | −0,005 | 0,00 |

A combinação de pontos (P9) foi feita depois, na Fase C.

### 12.2 Trilha N: ramo espectral sozinho, val antiga, magnitude estimada com a fase ruidosa

| Variante | PESQ isolado | Observação |
|---|---|---|
| Referência: B+ com 30 épocas | 1,97 | |
| N1a, front‑end de 256 canais | 2,34 | |
| **N1b, front‑end linear** | **2,49** | Mantida |
| N6a, janela de 32 ms | 2,09 | |
| N7a, recortes de 6 s | 1,73 | |
| N7b, normalização acumulada | 2,07 | |
| N8a, perda no tempo | 1,89 | |
| N‑a, opções combinadas | 2,48 | |
| N2, dual‑path TF | 2,43 | 0,98M parâmetros, ~13× mais lento por passo |
| N3, sub‑banda | 2,48 | |
| N4, fase na saída | divergiu (NaN) | Corrigido com atan2 de gradiente limitado: N4fix 2,31 (2,07 com a fase prevista) |
| N4 + N5, fase na entrada e na saída | 1,85 | N45fix 2,22 (2,27 com a fase prevista) |
| **B++ = N1b com 100 épocas** | **2,556** | Ver abaixo |

Medidas do B++:
- deriva de 0,97;
- teto "melhor dos dois" de 3,145;
- a mistura sem treino com o estágio A (w = 0,5) dá 2,864, +0,10 sobre o estágio A.

Todas as metas do protocolo PN na validação foram atingidas (seção 6.3):
- o PESQ isolado passou de 2,3;
- o erro de magnitude ficou abaixo do ruidoso em todas as faixas de SNR;
- a deriva ficou abaixo de 1,1;
- o teto ficou 0,39 acima do ramo de onda;
- no híbrido, o N‑d P3 deu +0,059, com dependência de 0,37.

### 12.3 Híbridos com o B++: protocolo curto, 1 semente, Δ contra o K0

| ID | O quê | Δ val | Δ val_unseen | Dep. (val) |
|---|---|---|---|---|
| K3 | Oráculo: espectro limpo no P3 | +0,267 | +0,208 | 1,09 |
| C3 | P3 + S5 + P8 | +0,121 | −0,127 | 1,70 |
| C1 | P3 + S5 | +0,073 | −0,128 | 1,15 |
| S5 | Estimativa no tempo no `enc_s2` | +0,071 | −0,015 | 1,30 |
| L+1 | P3 deslocado em +1 quadro | +0,066 | −0,109 | 0,39 |
| C2 | P3 + P8 | +0,064 | −0,123 | 1,54 |
| N‑d P2 | Encoder fino | +0,061 | −0,101 | 0,27 |
| G1 / G3 | Gerador linear / tronco de 512 canais | +0,061 / +0,057 | −0,117 / −0,123 | 0,45 / 0,39 |
| **N‑d P3** | `bott_pre`, a base das Fases 2–4 | +0,059 | −0,116 | 0,37 |
| T1–T8 | Treino da fusão | +0,040 a +0,058 | −0,102 a −0,146 | 0,30 a 0,44 |
| O1, O3, O4, O5 | Operadores | +0,045 a +0,055 | −0,114 a −0,134 | 0,36 a 0,45 |
| S2 / S4 | Ŝ / [`spec_hidden`; máscara] | +0,055 / +0,053 | −0,073 / −0,141 | 0,24 / 0,36 |
| N‑d P5 | Decoder grosso | +0,051 | −0,176 | 0,68 |
| Q2 | CleanSpecNet bidirecional | +0,045 | −0,070 | 0,62 |
| N‑x P8 | Pós‑filtro | +0,037 | −0,089 | 1,55 |
| O6 | Atenção cruzada | +0,023 | −0,056 | 0,17 |
| N‑d, 2º ramo | N‑a combinado (30 épocas) no P3 | +0,021 | −0,032 | 0,05 |
| L−1 / Lc | −1 quadro / STFT causal | +0,018 / +0,012 | −0,118 / −0,056 | 0,25 / 0,17 |
| S3 | Máscara | +0,016 | −0,025 | 0,06 |
| O7 | Troca entre as pilhas Mamba | +0,001 | −0,021 | 0,02 |
| K2 | Espectro ruidoso | −0,002 | −0,006 | 0,00 |

Nas 50 execuções curtas, a correlação de Spearman entre os Δ da val e os da val_unseen é de −0,62. Quem mais ganhou na val foi quem mais perdeu com ruído novo.

### 12.4 Fase 6 (B++, só dados do VB‑DEMAND, protocolo completo, 3 sementes) e Fase 5

| Config. | val | val_unseen | Teste | Dep. (teste) |
|---|---|---|---|---|
| K0full | 2,767 | 3,099 | 3,085 ± 0,007 | — |
| P3 + B++ | 2,822 (+0,055) | 2,940 (−0,158) | 3,011 ± 0,007 (−0,074) | 0,32 |
| C3 + B++ | 2,893 (+0,126) | 2,944 (−0,154) | 3,071 ± 0,003 (−0,014) | 1,98 |

Diagnóstico no teste:
- Com a condição desligada, o P3 (semente 1) sobe de 3,018 para 3,107. A condição do B++ atrapalha no teste.
- Da val para o teste, o B++ sozinho sobe +0,19 e o ramo de onda sobe +0,33.
- Na val, os ganhos do P3 e do C3 se concentravam em babble (+0,15 e +0,29), em meeting e em ssn. O babble de treino é feito de vozes do VCTK.
- No teste, o P3 perde nos 5 ruídos. O C3 perde 0,07 em SNR < 2,5 dB.

Fase 5:
- **Z1** (do zero, 30 épocas, teste): P3 + B++ fez 2,656, contra 2,668 do só onda e 2,650 do híbrido antigo. É um empate.
- **Z2** (E6): +0,017 na val e −0,039 na val_unseen.

### 12.5 Fase R: ruído externo, protocolo curto

Ramo espectral sozinho, com a fase ruidosa:

| Ramo | val | val_unseen | Mistura sem treino com o estágio A, w = 0,5 (val / val_unseen) |
|---|---|---|---|
| Estágio A (ramo de onda, referência) | 2,760 | 3,101 | — |
| B++ (VB‑DEMAND, 100 épocas) | 2,556 | 2,627 | 2,864 / 3,039 |
| R_ext50 (p = 0,5, 30 épocas) | 2,415 | 2,880 | 2,785 / 3,163 |
| R_ext80 (p = 0,8, 30 épocas) | 2,379 | 2,867 | 2,765 / 3,157 |
| **Bext = R_ext50 com 100 épocas** | 2,483 | **2,902** | 2,818 / **3,177** |

O Bext na val_unseen, comparado com o estágio A:
- O erro de log‑magnitude fica perto: 0,047 contra 0,043.
- O Bext fica mais perto do sinal limpo em 61% dos bins, ponderados pela energia.
- O teto "melhor dos dois" chega a 3,443, ou +0,34.
- A mistura com w = 0,25 dá 3,168.

O K0ext (3 sementes), contra o K0, faz −0,011 na val e +0,077 na val_unseen. O ganho vem sobretudo de park e washing.

Híbridos (1 semente), com Δ contra o K0ext:

| ID | O quê | Δ val | Δ val_unseen | Dep. (val / val_unseen) |
|---|---|---|---|---|
| RX_C3 | P3 + S5 + P8 | +0,079 | +0,052 | 1,56 / 1,95 |
| RX_S5P8 (Fase RT) | S5 + P8, sem o P3 | +0,063 | +0,023 | 1,61 / 1,99 |
| RX_S5 | S5 | +0,044 | +0,029 | 1,17 / 1,53 |
| RX_P3 | P3 | +0,024 | +0,012 | 0,09 / 0,11 |
| RX_P3vb | P3 com o Bext, mas o híbrido treinado só com o VB‑DEMAND (Δ contra o K0) | +0,019 | +0,002 | 0,06 / 0,03 |

Por tipo de ruído, o RX_C3 ganha:
- nos 5 ambientes da val_unseen, de +0,04 a +0,06;
- nos 10 ruídos da val, de +0,04 (car) a +0,15 (babble).

O C3 com o B++ perdia nos 5 da val_unseen.

### 12.6 Fase RT: ruído externo e Bext, protocolo completo, 3 sementes

| Config. | Parâm. | val | val_unseen | **Teste** | STOI | SI‑SDR | CSIG | CBAK | COVL | Dep. (teste) |
|---|---|---|---|---|---|---|---|---|---|---|
| K0full (VB‑DEMAND) | 41,4M | 2,767 | 3,099 | 3,085 ± 0,007 | 0,953 | 18,48 | 4,57 | 3,66 | 3,89 | — |
| K0extfull | 41,4M | 2,754 | 3,185 | 3,164 ± 0,008 | 0,953 | 18,72 | 4,61 | 3,71 | 3,96 | — |
| RX_P3 | 52,4M | 2,777 | 3,181 | 3,151 ± 0,005 | 0,953 | 18,64 | 4,60 | 3,70 | 3,95 | 0,22 |
| **RX_C3** | 53,8M | 2,830 | 3,220 | **3,212 ± 0,008** | 0,954 | 18,96 | 4,64 | 3,75 | 4,01 | 2,06 |

**RX_C3 contra o K0extfull, no teste: +0,048.**
- Por semente: +0,056, +0,048 e +0,039, todas com p de Holm < 1e‑30.
- Por ruído: de +0,044 a +0,054 em cada um dos 5.
- Por SNR: o menor ganho é em SNR ≥ 12,5 dB (+0,02 a +0,03). Em SNR < 2,5 dB o ganho é de +0,03 a +0,05; com o B++, o C3 perdia 0,07 nessa faixa.

**RX_P3 contra o K0extfull: −0,013.**
- Por semente: −0,008, −0,014 e −0,017, significativo só em uma.
- Perde nos 5 ruídos.

**K0extfull contra o K0full: +0,079.** Ganha nos 5 ruídos do teste, de +0,06 a +0,10.

### 12.7 Validações contra o teste

| Comparação | Δ val | Δ val_unseen | Δ teste |
|---|---|---|---|
| P3 + B++ − K0full (Fase 6) | +0,055 | −0,158 | −0,074 |
| C3 + B++ − K0full (Fase 6) | +0,126 | −0,154 | −0,014 |
| K0extfull − K0full | −0,013 | +0,086 | +0,079 |
| RX_P3 − K0extfull | +0,022 | −0,004 | −0,013 |
| RX_C3 − K0extfull | +0,076 | +0,035 | +0,048 |
| K0 receita `noise` − K0extfull (VF) | +0,008 | +0,015 | +0,007 |
| Mistura K0extfull + Bext − RX_C3 (V1) | −0,019 | +0,009 | −0,002 |

### 12.8 Análise (2026‑10‑15)

1. **A diversidade de ruído foi a alavanca principal, mais que a arquitetura.**
   - Sozinho, o ruído externo no ramo de onda rendeu +0,079 no teste. Isso é mais do que qualquer variante de fusão das Fases 1–4, e nenhuma delas se sustentou com ruído novo.
   - O ramo espectral precisou do mesmo remédio: o PESQ isolado na val_unseen foi de 2,63 para 2,90.
   - O híbrido também precisa de ruído diverso no próprio treino. O RX_P3vb usa o Bext, mas foi treinado só com o VB‑DEMAND, e ficou em +0,002.
2. **A fusão só ajuda quando o ramo espectral generaliza, e o ganho vem das fusões offline.**
   - S5 e P8 usam a estimativa do ramo espectral diretamente.
   - O P3 sozinho perde no teste, mas dentro do C3 contribui: sem ele, o C3 perde 0,017 na val e 0,029 na val_unseen (1 semente).
3. **Os dois ramos se complementam muito, e a fusão aproveita pouco disso.** Na val_unseen, contra o estágio A:
   - o Bext é mais preciso em 61% dos bins;
   - o teto "melhor dos dois" fica +0,34 acima do ramo de onda;
   - o oráculo K3 dá +0,21;
   - o RX_C3 tira +0,035 sobre o K0extfull.
4. **Pelos números que já existem, uma mistura sem treino rende tanto quanto a fusão aprendida.**
   - Com o estágio A, a mistura dá +0,076 na val_unseen (Bext, w = 0,5) e +0,051 no teste (B++, w = 0,25).
   - O RX_C3 dá +0,035 e +0,048 sobre o K0extfull.
   - As bases são diferentes (estágio A × K0extfull) e a complementaridade pode cair com uma base mais forte. Por isso o V1 vem antes de qualquer outra coisa.
5. **O RX_C3 é frágil.** Com a condição de outro arquivo, perde 2,06 pontos. Nada garante o comportamento num ruído em que o Bext falhe.
6. **Streaming:** nenhuma fusão compatível com streaming ganha no teste. O melhor modelo com 48 ms de lookahead é o K0ext.
7. **Método:** só a val_unseen acompanhou o sinal do teste (seção 12.7). Os rankings das Fases 1–4 foram feitos na val antiga e não devem orientar novas escolhas.

### 12.9 Fila V (seção 13), executada em 2026‑10‑15, 14:31–17:55 ✔

`scripts/run_ablation.py --phases V1 V3 V2 V5 V6 V4 V7 VF --winner P3_bott_pre`. Tabelas em `results/abl/V*.md`; decisões em `results/abl/V*.json`. Δ em PESQ‑WB; o controle é o K0ext curto (val_unseen 3,169, val 2,749), salvo indicação.

**V1: mistura sem treino × fusão aprendida (protocolo completo, 3 sementes).** O peso escolhido na val_unseen foi w = 0,25 (3,229, contra 3,218 com w = 0,5).

| Sistema | Parâmetros | val | val_unseen | teste |
|---|---|---|---|---|
| K0extfull | 41,4M | 2,754 ± 0,004 | 3,185 ± 0,004 | 3,164 ± 0,008 |
| Mistura K0extfull + Bext (w = 0,25) | 51,7M | 2,811 ± 0,004 | **3,229 ± 0,003** | 3,209 ± 0,005 |
| RX_C3 | 53,8M | **2,830 ± 0,016** | 3,220 ± 0,018 | **3,212 ± 0,008** |

- Mistura − RX_C3: −0,019 na val, +0,009 na val_unseen, −0,002 no teste (no teste, o IC 95% de cada semente inclui zero). **`fusion_justified = false`.**
- No teste, a mistura ganha do K0extfull nas 3 sementes (+0,040 a +0,049) e tem CSIG/COVL iguais aos do RX_C3, com menos variância entre sementes.
- A fusão aprendida só ganha da mistura nos tipos de ruído vistos no treino (val antiga).
- Protocolo curto: mistura K0ext + Bext +0,045 a +0,048 na val_unseen, contra +0,052 do RX_C3 curto. O **RX_P8** sozinho dá +0,004.

**V3: receita de ruído do ramo de onda (K0 curto).**

| Receita | Δ val_unseen | Δ val |
|---|---|---|
| só `noise` (MUSAN noise, p = 0,5) | **+0,027** | +0,007 |
| p = 0,8 | +0,018 | −0,003 |
| SNR 0–20 dB | +0,013 | +0,007 |
| mais babble | +0,002 | +0,002 |
| p = 0,3 | −0,010 | +0,010 |

- Só `noise`, 3 sementes curtas: +0,020 na val_unseen. Passou e foi ao VF.
- **VF (protocolo completo, 3 sementes):** 3,171 ± 0,006 no teste contra 3,164 do K0extfull (+0,007; sementes +0,002 / +0,014 / +0,005, só uma significativa). Na val_unseen, +0,015. Ganho real, mas pequeno: não vale trocar a receita da base.

**V2: condition dropout no RX_C3 (curto).**

| Run | val_unseen | com a condição de outro arquivo | com a fusão desligada | dependência |
|---|---|---|---|---|
| RX_C3 | 3,221 | 1,269 | 3,057 | 1,95 |
| p = 0,15 | 3,209 | 1,309 | **3,196** | 1,90 |
| p = 0,30 | 3,207 | 1,365 | 3,185 | 1,84 |

- Perde 0,012–0,014 do Δ e **não reduz a dependência** (condição errada continua derrubando o PESQ para ~1,3). Reprovado.
- Efeito colateral: com a fusão desligada, o modelo treinado com p = 0,15 fica em 3,196, acima do K0ext (3,169). O ramo de onda aprende a funcionar sem condição, mas não a ignorar uma condição errada.

**V5: fusões compatíveis com streaming (curto, Bext, ruído externo).** Nenhuma passou: S2 +0,014, P2 +0,014, P3 +0,012, G1 +0,008, O4 +0,007, P3 com dropout +0,003, P5 +0,000. Todas abaixo de δ_min na val_unseen, apesar de +0,018 a +0,029 na val antiga.

**V6: destilação do RX_C3 no modelo só de onda (curto).** λ = 1: +0,020 na val_unseen, +0,011 na val; λ = 0,5: +0,016 / +0,010. Abaixo de δ_min por pouco; ganho do mesmo tamanho que o da receita `noise` do V3, que no teste rendeu só +0,007.

**V4: ramo espectral sozinho (protocolo PN, ruído externo).** Referência: R_ext50, PESQ isolado 2,880 na val_unseen.

| Ramo | sozinho (val_unseen) | mistura com o estágio A, w = 0,5 (val_unseen) |
|---|---|---|
| R_ext50 | 2,880 | 3,163 |
| R_ext50_final (= Bext) | 2,902 | 3,177 |
| N3 sub‑banda | 2,875 | 3,146 |
| profundo (8 camadas) | 2,876 | 3,161 |
| **bidirecional** | **2,915** | **3,190** |

Nenhum atingiu o limiar de +0,05. O bidirecional, só com o treino curto, já supera o Bext final (+0,013 sozinho e na mistura); não é causal.

**V7: 30 épocas (1 semente, seed 1).**

| Run | 10 épocas val_unseen | 30 épocas val_unseen |
|---|---|---|
| K0extfull | 3,182 | 3,200 (+0,018) |
| RX_C3 | 3,238 | 3,238 (+0,000) |

O ramo de onda ainda melhora com mais épocas; o RX_C3 já saturou. A distância cai de +0,056 para +0,038.

### 12.10 Análise da fila V

1. **A fusão aprendida não se justifica no ruído novo.** A mistura sem treino dos dois modelos treinados separadamente empata com o RX_C3 no teste (−0,002) e na val_unseen (+0,009), com menos parâmetros, sem treino conjunto e sem a fragilidade da dependência de 2. O RX_C3 só leva vantagem nos ruídos vistos no treino (val antiga, +0,019), o que sugere que o treino conjunto ajusta a fusão aos ruídos do treino.
2. **O sistema offline recomendado** passa a ser **K0extfull + Bext + mistura (w = 0,25)**: 3,209 ± 0,005 no teste. Para melhorá‑lo, os caminhos são os dois ramos separados: o ramo espectral (o bidirecional é o candidato mais forte, se a latência de utterance inteira for aceitável) e o ramo de onda (mais épocas: +0,018 com 30).
3. **Streaming:** nenhuma fusão causal, nem a destilação, passou do δ_min. O melhor modelo para streaming continua sendo o K0ext, eventualmente com 30 épocas e a receita `noise` (ganhos pequenos e possivelmente somáveis, não testados juntos).
4. **Condition dropout** ensina o modelo a funcionar sem condição, não a resistir a uma condição errada. Para reduzir a dependência seria preciso treinar com condições corrompidas (de outro arquivo ou com ruído), o que perde o sentido diante do item 1.
5. **Método:** a val antiga continuou otimista com tudo que envolve fusão (V5: +0,018 a +0,029 na val, ~0 na val_unseen; V1: favorece o RX_C3). A val_unseen superestimou o V3 `noise` (+0,015 contra +0,007 no teste), mas acertou o sinal; a diferença mistura − RX_C3 ficou dentro do ruído nos dois conjuntos.

### 12.11 Análise complementar da fila V (2026‑10‑15, noite) ✔

Feita com as saídas regeneradas dos 4 sistemas (K0extfull, RX_C3, mistura, K0 com a receita `noise`; 3 sementes; teste e val_unseen). As saídas reproduzem exatamente os PESQ das tabelas. Os scripts ficaram fora do repositório.

**Por ruído (média das 3 sementes, Δ contra o K0extfull):**

| Ruído | Mistura | RX_C3 | Mistura − RX_C3 |
|---|---|---|---|
| teste: bus | +0,018 | +0,044 | −0,026 |
| teste: cafe / living / office / psquare | +0,046 a +0,060 | +0,044 a +0,054 | −0,001 a +0,010 |
| val_unseen: washing | +0,020 | +0,046 | −0,026 |
| val_unseen: river / park | +0,059 / +0,063 | +0,024 / +0,044 | +0,035 / +0,020 |
| val_unseen: field / hallway | +0,039 / +0,040 | +0,037 / +0,025 | +0,002 / +0,015 |

O RX_C3 só leva vantagem nos ruídos estacionários de baixa frequência (ônibus e máquina de lavar). Nos ruídos naturais não estacionários (rio, parque), a mistura é melhor.

**Cauda de falhas do RX_C3.** Por arquivo, contra o K0extfull:

| | Mistura (teste) | RX_C3 (teste) | Mistura (val_unseen) | RX_C3 (val_unseen) |
|---|---|---|---|---|
| Arquivos com Δ < −0,1 | 0,5% | 5,9% | 0,4% | 8,3% |
| Percentil 1 do Δ | −0,08 | −0,47 | −0,09 | −0,47 |

- A mediana do ganho é igual nos dois (~+0,04 a +0,05). O RX_C3 ganha mais vezes por pouco e perde algumas vezes por muito.
- Nos arquivos da cauda, as outras métricas quase não pioram: ΔSI‑SDR −0,02 dB e ΔSTOI −0,005. Nos demais arquivos, o ΔSI‑SDR é +0,26 dB.
- A falha já aparece no PC‑1 (só a fusão treinada, ramos congelados). Vem da fusão, não do ajuste conjunto.

**Instabilidade do PESQ.** Nos piores arquivos, o PESQ‑WB muda até 1,6 ponto com transformações neutras:
- Acrescentar 100 amostras de silêncio no início do sinal limpo e do processado: no p257_196, o RX_C3 vai de 2,59 para 4,21.
- Trocar 3 ms (50 amostras) do RX_C3 pelos do K0: recupera 1,6 ponto. A diferença entre os dois nesse trecho fica ~35 dB abaixo da fala.
- O K0 também é afetado: no p257_030, o PESQ dele cai de 2,53 para 1,76 com o mesmo silêncio extra.

Com o PESQ calculado como média sobre 5 quantidades de silêncio no início (0, 40, 100, 160 e 240 amostras):
- **As médias e as conclusões não mudam.** Mistura − RX_C3 = −0,002 no teste e +0,007 na val_unseen. O RX_C3 tem +0,049 e a mistura +0,047 sobre o K0extfull no teste. A receita `noise` tem +0,005 no teste e +0,015 na val_unseen.
- **A cauda do RX_C3 continua:** 5,6% dos arquivos no teste e 8,1% na val_unseen. Cerca de metade deles falha nas 3 sementes (21 no teste, 22 na val_unseen).
- **O teste é bem mais instável que a val_unseen:** 4,5% dos arquivos variam mais de 0,5 entre as quantidades de silêncio, contra 0,2% na val_unseen.

Conclusão: a cauda é real para o PESQ, mas não aparece no SI‑SDR nem no STOI. Só uma audição dos 21 arquivos do teste que falham nas 3 sementes diz se ela é audível.

**Variância entre sementes (desvio padrão do PESQ):**

| | val_unseen | teste |
|---|---|---|
| Mistura | 0,003 | 0,005 |
| RX_C3 | 0,018 | 0,008 |

**Médias de formas de onda (sem treino, pesos 0,5 / 0,5):**

| Combinação | teste | val_unseen |
|---|---|---|
| RX_C3 + mistura | **3,238 ± 0,004** | **3,253 ± 0,010** |
| RX_C3 + K0extfull | 3,220 ± 0,002 | 3,238 ± 0,010 |
| Mistura + K0 receita `noise` | 3,207 ± 0,006 | 3,231 ± 0,002 |

- A média RX_C3 + mistura ganha +0,026 sobre o RX_C3 no teste. Os dois conjuntos concordam, e não houve ajuste de peso.
- Os ganhos por arquivo da mistura e do RX_C3 têm correlação de só 0,60 (teste) e 0,45 (val_unseen). Os erros dos dois sistemas são diferentes o bastante para que a média os compense.
- Dois modelos de onda quase iguais (K0ext e K0 com a receita `noise`) não somam nada.
- Custo: K0 + RX_C3 + Bext, ~105M parâmetros e duas passadas, offline. É um ensemble, não uma arquitetura.

---

## 13. Próximos experimentos (proposta de 2026‑10‑15)

**Estado:** V1–V7 executados em 2026‑10‑15 (seção 12.9). V4d e V8 não foram executados: exigem baixar outro corpus, e o candidato óbvio (ESC‑50) vem do Freesound, como o MUSAN, cujos IDs não são listados, então não há como garantir ruído inédito.

**Regras para todos:**
- A seleção usa a val_unseen. Na val antiga, a variante não pode cair mais de 0,04 (2·δ_min) contra o controle.
- O Δ é sempre contra o controle com o mesmo treino: o K0ext, ou o K0 de uma receita nova.
- Na triagem, 1 semente; no protocolo completo, 3.
- O teste fica só para a confirmação final.

### Prioridade 1: próxima fila, pouco ou nenhum código

| ID | Experimento | Pergunta | Custo ◇ |
|---|---|---|---|
| **V1** | **Controle de ensemble.** Mistura sem treino do K0extfull (3 sementes) com o Bext, w = 0,25 e 0,5, com a fase do ramo de onda (`scripts/diagnose_spec_branch.py --wave_ckpt logs/abl/K0extfull_s*/last.ckpt --spec_ckpt Bext=logs/abl/N_R_ext50_final/last.ckpt`), na val, na val_unseen e, com o w escolhido, no teste. Também uma execução curta do **RX_P8**: só o P8, com o Bext e o ruído externo | A fusão aprendida (RX_C3) supera uma simples mistura dos mesmos dois modelos? | ~1 h, quase tudo inferência |
| **V3** | **Receita de ruído do ramo de onda.** K0 curto com p = 0,3 e 0,8 (hoje 0,5); SNR de 0 a 20 dB (hoje −5 a 20); pesos só `noise` e com mais babble (0,4 / 0,4 / 0,2; hoje 0,6 / 0,25 / 0,15) | O p = 0,5 foi escolhido pelo ramo espectral. Qual receita é a melhor para o ramo de onda? | ~6 execuções curtas, ~45 min |

**Decisões:**
- **V1, se a mistura ficar a menos de δ_min do RX_C3 na val_unseen:** a fusão aprendida não se justifica.
  - O sistema offline passa a ser K0ext + Bext + mistura, sem treino conjunto, com os dois ramos treinados separadamente.
  - A pesquisa de fusão fica só para o que a mistura não faz: streaming (V5) e destilação (V6).
- **V1, se o RX_C3 ficar ≥ δ_min acima da mistura:** a fusão aprendida acrescenta algo. Seguir com V2 e V4 sobre o RX_C3.
- **V3, se uma receita superar o K0ext por ≥ δ_min:** ela vira a base dos próximos híbridos. Talvez seja preciso retreinar o Bext com ela.

### Prioridade 2: código novo ou dependente do V1/V3

| ID | Experimento | Pergunta / hipótese | Custo ◇ |
|---|---|---|---|
| **V2** | **Condition dropout no RX_C3.** Com probabilidade p por lote, o passo de treino roda com `fusion_enabled=False` (P3, S5 e P8 desligados), com p = 0,15 e 0,3. O ramo espectral continua recebendo a perda espectral. Se funcionar, uma versão por exemplo (máscara por item em cada caminho da fusão) vem depois | A dependência cai de ~2 para menos de 0,5 sem perder o Δ? Com a fusão desligada, o PESQ deve ficar perto do K0ext em vez de desabar. A regularização pode até ajudar na val_unseen | ~1 h de código e testes, mais ~30 min de GPU |
| **V4** | **Ramo espectral melhor para o sistema offline**, no protocolo PN com ruído externo, escolhido pelo PESQ isolado na val_unseen e depois testado no RX_C3. Candidatos: (a) N3 sub‑banda, que tende a generalizar melhor e empatou com o N1b na val antiga; (b) escala (N9) e mais épocas; (c) CleanSpecNet bidirecional, só para o modo de utterance inteira, que é outra classe de latência; (d) outro corpus de ruído além do MUSAN, sem DEMAND, o que exige download | O oráculo K3 (+0,21) e o teto "melhor dos dois" (+0,34) mostram espaço. Um Bext melhor aumenta o ganho do híbrido (H10)? | ~1–3 h por candidato |
| **V5** | **Nova triagem das fusões compatíveis com streaming na val_unseen**, com o Bext e o ruído externo: P2, P5, S2, O4/G1, Lc e o P3 com condition dropout | Existe fusão compatível com streaming que ganhe com ruído novo? Só vale se o híbrido em streaming for objetivo | ~8 execuções curtas, ~1,5 h |
| **V6** | **Destilação do RX_C3 no modelo só de onda.** O professor é o RX_C3 congelado; o aluno começa no K0extfull. Perda: onda contra o sinal limpo + λ · L1 contra a saída do professor, com o ruído externo | Parte dos +0,048 passa para um modelo de 41,4M parâmetros, com 48 ms de lookahead e sem ramo espectral? É o caminho mais provável para um modelo de streaming melhor que o K0ext | ~1–2 h de código, mais ~30 min de GPU |
| **V7** | **Treino mais longo:** K0ext e RX_C3 com 30 épocas no protocolo completo, 1 semente cada | O ruído externo deixa o treino mais difícil. O ganho do RX_C3 cresce ou some com mais épocas? | ~2–3 h |
| V8 (opcional) | **Segundo conjunto de ruído não visto**, por exemplo NOISEX‑92 ou ESC‑50, com os falantes do teste | A val_unseen passou a ser usada para seleção, então um conjunto independente fortalece a conclusão final | Download, mais ~10 min |

**Confirmação final:** os vencedores do V1–V7, com 3 sementes, o protocolo completo e o teste uma única vez.

**O que não repetir:** triagens de operador, ponto de injeção ou treino da fusão escolhidas pela val antiga.

**Ordem sugerida:**
1. Uma fila com V1 e V3, que não precisam de código.
2. Enquanto ela roda, implementar o V2. Depois, rodar o V2.
3. Escolher entre V4 e V6 conforme o resultado do V1.

---

## 14. Melhor sistema e modelo mais simples e robusto (fila W, 2026‑10‑15)

**Objetivos:**
- **(A) Melhor resultado:** o maior PESQ no teste, offline, sem limite de custo.
- **(B) Modelo mais simples e robusto:** uma única rede só de onda (41,4M parâmetros, a do K0ext), sem fusão e compatível com streaming. Robusto quer dizer:
  - estável entre sementes;
  - sem cauda de arquivos com falha;
  - bom em ruído novo, inclusive com SNR baixo.

**Ponto de partida (teste, 3 sementes):**

| Sistema | Parâmetros | Teste |
|---|---|---|
| K0extfull (só onda, 10 épocas) | 41,4M | 3,164 ± 0,008 |
| Mistura K0extfull + Bext (w = 0,25) | 51,7M | 3,209 ± 0,005 |
| RX_C3 | 53,8M | 3,212 ± 0,008 |
| Média das saídas RX_C3 + mistura (12.11) | ~105M | 3,238 ± 0,004 |

**Regras:**
- **Seleção** na val_unseen, com a val antiga como guarda (−0,04) e δ_min = 0,02, como na seção 13.
- **Guardas contra "PESQ hacking"** para as perdas novas, na val_unseen: ΔSTOI ≥ −0,003 e ΔSI‑SDR ≥ −0,5 dB.
- **TTA** só entra se ganhar ≥ 0,01 na val_unseen, porque multiplica o custo.
- **Soups** são sempre avaliadas, porque não custam nada.
- **Teste** só na fase final (WF).

### 14.1 Experimentos

| ID | Experimento | Pergunta / decisão | Custo ◇ |
|---|---|---|---|
| **W0** | Conjunto **val_unseen_low**: os ruídos da val_unseen a −5 e 0 dB (770 arquivos, 77 por condição; uma mistura que saturaria é reescalada junto com uma cópia do sinal limpo) | Teste de estresse de robustez | feito |
| **W1** | **Sem treino**, sobre checkpoints existentes: (a) **soup de pesos** das 3 sementes do K0extfull e do K0ext; (b) média das saídas das 3 sementes, como teto da soup; (c) **TTA** com inversão de sinal e com 4 deslocamentos da entrada (0, 64, 128 e 192 amostras) no K0extfull; (d) peso w da mistura com o Bext (0,15 / 0,25 / 0,35 / 0,5); (e) peso α do RX_C3 na média RX_C3 + mistura (0,3 a 0,7) | Melhorias de graça? Escolhe w, α e a TTA → `W1.json` | ~30 min |
| **W2** | **Perda do ramo de onda** (K0 curto, semente 1, ruído externo): (a) MSE de magnitude comprimida \|X\|^0,3 (n_fft 400) com peso 10 e 30; (b) **discriminador métrico** (MetricGAN+, como no CMGAN e no MP‑SENet) com peso 0,3 e 1,0 | Uma perda mais alinhada à percepção melhora o PESQ em ruído novo sem custo na inferência? | ~35 min |
| **W3** | **Destilação da média RX_C3 + mistura** (o melhor sistema, α e w do W1) no modelo só de onda, λ = 1 e 2 (K0 curto) | Um professor melhor e sem a cauda do RX_C3 passa mais ganho que o V6 (+0,020)? | ~15 min |
| **W4** | **Receita final do modelo simples:** os vencedores do W2 e do W3 e, se ambos existirem, a combinação deles (curto). Fica a melhor que passar; sem vencedor, a receita é o K0ext puro | → `W4.json` | ~10 min |
| **W5** | **Bext bidirecional** com o orçamento do Bext (100 épocas; o V4_bidir foi o único candidato acima do Bext) e as misturas com o K0extfull (4 pesos) | Escolhe ramo espectral e w para o sistema offline → `W5.json` | ~50 min |
| **WF** | **Final, 3 sementes, sementes intercaladas, avaliado no teste, na val_unseen, na val_unseen_low e na val:** (B) K0ext com 30 épocas (controle; a semente 1 é o V7_K0ext30) e a receita do W4 com 30 épocas (S30), mais as soups; (A) mistura de S30 com o ramo espectral do W5 e a média α·RX_C3 + (1−α)·mistura (com a TTA do W1), ao lado da média da análise 12.11 | Números finais dos dois objetivos | ~3–4 h |
| **WR** | **Relatório de robustez** (`results/abl/W_robustness.md`): por sistema, desvio entre sementes, PESQ robusto no teste, % de arquivos > 0,1 abaixo do K0extfull, pior tipo de ruído, val_unseen_low e streaming dos modelos só de onda | Qual é o modelo mais robusto? | ~40 min |

**Calibração dos pesos das perdas.** Medida com o K0extfull_s1 em recortes de treino com ruído externo, pela norma do gradiente:
- **Base (L1 + MR‑STFT):** o gradiente vem quase todo do MR‑STFT.
- **Magnitude comprimida:** cada unidade de peso dá ~0,044 vez o gradiente da base, então os pesos 10 e 30 equivalem a ~0,45 e ~1,3 vez.
- **Gerador do discriminador métrico** (depois de 500 passos do discriminador): ~0,3 vez a base, então os pesos 0,3 e 1,0 equivalem a ~9% e ~30%.
- **Termo complexo comprimido:** ficou de fora, porque o gradiente explode nos bins quase sem energia (7,6 vezes o da base).

**Custo de treino** (K0 com 16 recortes de 2 s por lote): base ~16 it/s; destilação com o professor‑ensemble 12 it/s e 4,3 GB; discriminador métrico ~6 it/s (o PESQ dos 16 recortes roda num pool de 8 processos).

### 14.2 Implementação

| Arquivo | Mudança |
|---|---|
| `losses.py` | `CompressedMagnitudeLoss` |
| `cleanunet/metric_disc.py` | `MetricDiscriminator`, `batch_pesq` (pool de processos), perdas do gerador e do discriminador |
| `cleanunet/blend.py` | Mistura por exemplo, usada pelo `evaluate.py` e pelos professores da destilação |
| `lightning_modules/cleanunet_module.py` | `loss.cmag`; `loss.metric_gan` (otimização manual, dois otimizadores, recorte do gradiente no módulo, `mgan_step` salvo no checkpoint); `distill.teachers` (média ponderada de professores, cada um com mistura opcional) |
| `train.py` | Sem `gradient_clip_val` no Trainer quando o módulo usa otimização manual |
| `evaluate.py` | `--tta flip/shift/flipshift`, `--robust_pesq` (coluna `pesq_wb_r`), funções de escrita reutilizáveis |
| `evaluation/metrics.py` | `robust_pesq_wb` (média sobre 0, 40, 100, 160 e 240 amostras de silêncio no início) |
| `scripts/make_soup.py` | Soup de checkpoints Lightning da mesma arquitetura |
| `scripts/ensemble_outputs.py` | Média ponderada de áudios salvos, com métricas no formato do `evaluate.py` |
| `scripts/robustness_report.py` | Relatório de robustez a partir de um JSON de sistemas |
| `scripts/make_val_unseen.py` | `--rescale` para SNRs baixos |
| `scripts/run_ablation.py` | Fases W0–W5, WF e WR |
| `tests/test_w_experiments.py` | 14 testes, inclusive os da polaridade (14.3); a suíte inteira passa (134) |

**Validação antes do lançamento:**
- Dry run com 13 treinos, 198 avaliações e 75 scripts.
- Teste das regras de decisão com resultados falsos, em dois cenários (com e sem vencedores; GAN barrado pela guarda de SI‑SDR).
- Testes de fumaça na GPU do discriminador métrico (500 passos: a perda do discriminador cai de 0,41 para 0,01) e da destilação com o professor‑ensemble.

**Execução:** `logs/abl/run_W.sh` → `logs/abl/queue_W.log`, lançado em 2026‑10‑15 às 21:11.

**Primeiro resultado:** a soup das 3 sementes do K0extfull dá 2,760 na val e 3,189 na val_unseen, contra médias das sementes de 2,754 e 3,185. Ela supera cada semente isolada, com o custo de um modelo.

### 14.3 Achado durante a fila: dependência de polaridade (2026‑10‑15, 21:30) ✔

A TTA com inversão de sinal derrubou o K0extfull em 0,24 na val_unseen (W1). Isso levou a um teste direto: o mesmo modelo com a entrada invertida e a saída invertida de volta. Medido em 300 arquivos da val_unseen para o K0extfull e o estágio A e em 150 para os demais:

| Modelo | Ruído de treino | PESQ normal | PESQ com a entrada invertida |
|---|---|---|---|
| Estágio A | só VB‑DEMAND | 3,095 | 3,095 |
| K0_s1 (5 épocas) / K0full_s1 (10 épocas) | só VB‑DEMAND | 3,097 / 3,109 | 3,074 / 3,008 |
| V3_noise | MUSAN só `noise` | 3,207 | 3,190 |
| K0ext_s1 (5 épocas) | MUSAN noise / babble / música | 3,181 | **1,976** |
| K0extfull_s1 (10 épocas) | idem | 3,187 | **1,494** (STOI 0,57; SI‑SDR 0,95 dB) |
| V3_babble | mais babble | 3,180 | **1,528** |

**Interpretação:**
- O babble do MUSAN é feito de fala de outro corpus, e cada locutor entra com a polaridade original da gravação. Como a forma de onda da fala é assimétrica, o modelo aprendeu um atalho: manter a fala com a polaridade do VCTK e remover a outra.
- Com a entrada invertida, a fala‑alvo passa a ter a "polaridade do babble" e é apagada.
- O teste do VB‑DEMAND não revela isso, porque a polaridade dele é constante.
- O babble do próprio VB‑DEMAND é feito de vozes do VCTK, por isso o K0 só com VB quase não sofre. Mesmo assim, ele perde 0,10 com 10 épocas.

**Consequências:**
- Todos os modelos de onda treinados com o ruído externo padrão herdam o atalho: K0ext, K0extfull, RX_C3 e as misturas baseadas neles. O Bext é invariante por construção, porque usa só a magnitude.
- Para o objetivo (B), robustez, isso precisa ser corrigido.

**Correções implementadas** (desligadas por padrão, para não alterar as execuções da fila em andamento):
- `augment_polarity: p`, no módulo: inverte o par de treino inteiro com probabilidade p. Os professores da destilação veem sempre a polaridade original; o sinal é aplicado à saída deles.
- `data.ext_noise.random_sign: true`, no `noise_bank.py`: dá sinal aleatório a cada locutor do babble e a cada trecho de ruído ou música.
- `evaluate.py --invert`: avalia com a entrada invertida.
- Testes: a suíte passa (134).

**Fase WP** (inserida antes do W5 e do WF; a fila foi reiniciada às 21:45 sem perder trabalho):
- **Candidatos** (K0 curto, semente 1): `WP_flip` (inversão do par, p = 0,5), `WP_sign` (sinal aleatório nas fontes), `WP_both` (os dois). Também o V3_noise (sem babble nem música; 3 sementes).
- **Regra:** a perda com a entrada invertida deve ser ≤ 0,05, o Δ na val_unseen contra o K0ext deve ser ≥ −δ_min, e a val serve de guarda. Entre os que passam, fica o melhor Δ. Um vencedor de 1 semente é confirmado com as sementes 2 e 3.
- **Efeito nos finais (WF):** a correção escolhida entra no controle (K0ext30p, 30 épocas, 3 sementes novas) e na receita S30. O V7_K0ext30 (sem correção) fica como referência.
- **Efeito no WR:** o relatório ganha a coluna "PESQ com a entrada invertida" na val_unseen, para todos os sistemas.
- **Expectativa:** o sinal aleatório sozinho não deve bastar, porque a fala‑alvo continuaria sempre com a mesma polaridade.

### 14.4 Resultados da fila W (2026‑10‑15 21:45 → 2026‑10‑16 04:43) ✔

A fila terminou sem falhas. O S30_s1 caiu uma vez com segfault em código nativo, perto da época 24, e foi retomado do último checkpoint. Os números dele ficam entre os das outras duas sementes.

**W1 (sem treino, val_unseen):**
- **Soup** das 3 sementes do K0extfull: +0,005 (3,189 contra a média 3,185), com o custo de um modelo. A **média das saídas** das 3 sementes dá +0,020, com o custo de três.
- **TTA rejeitada:** a inversão de sinal perde 0,24 (é o efeito da polaridade, 14.3) e os deslocamentos perdem 0,02.
- Mistura K0extfull + Bext: w = 0,35 (3,230). Média RX_C3 + mistura: α = 0,4 (3,254).

**W2–W4** (K0 curto, semente 1; Δ contra o K0ext com 3 sementes, que tem val 2,749 e val_unseen 3,169):

| Variante | Δ val_unseen | Δ val | ΔSTOI | ΔSI‑SDR | Decisão |
|---|---|---|---|---|---|
| Magnitude comprimida, peso 10 | +0,030 | +0,029 | +0,0003 | +0,04 dB | passa |
| Magnitude comprimida, peso 30 | +0,029 | +0,030 | +0,0006 | −0,01 dB | passa |
| **Discriminador métrico, peso 0,3** | **+0,068** | +0,061 | 0,0000 | −0,41 dB | **vencedor do W2** |
| Discriminador métrico, peso 1,0 | +0,099 | +0,090 | 0,0000 | −1,07 dB | barrado pela guarda de SI‑SDR |
| Destilação do ensemble (W3), λ = 1 | +0,026 | +0,022 | +0,0005 | +0,16 dB | passa |
| Destilação do ensemble (W3), λ = 2 | +0,030 | +0,025 | +0,0004 | +0,15 dB | vencedor do W3 (o V6 deu +0,020) |
| Discriminador 0,3 + destilação 2 (W4) | +0,056 | +0,048 | +0,0007 | +0,13 dB | perde para o discriminador sozinho |

- **Receita do modelo simples** (`W4.json`): discriminador métrico com peso 0,3.
- O ganho do discriminador é uniforme nas faixas de SNR (+0,06 a +0,07). A combinação com a destilação recupera o SI‑SDR, mas perde 0,012 de PESQ.

**WP: correção da polaridade** (K0 curto; a perda é o PESQ normal menos o PESQ com a entrada invertida, na val_unseen):

| Variante | Sementes | Δ val_unseen | Δ val | Perda com a entrada invertida | Decisão |
|---|---|---|---|---|---|
| K0ext, sem correção | 3 | — | — | 0,87 | — |
| **WP_flip** (inversão do par, p = 0,5) | 1 → 3 | +0,000 → +0,001 | −0,008 → −0,004 | −0,011 | **vencedor** |
| WP_sign (sinal aleatório nas fontes) | 1 | −0,005 | −0,001 | 0,90 | não corrige |
| WP_both (os dois) | 1 | −0,005 | −0,008 | −0,007 | passa, mas perde para o WP_flip |
| V3_noise (sem babble nem música) | 3 | +0,020 | +0,007 | 0,08 (semente 3: 0,18) | não passa |

- **A expectativa de 14.3 se confirmou:** o sinal aleatório nas fontes não basta, porque a fala‑alvo continua sempre com a polaridade do VCTK. Só a inversão do par inteiro remove o atalho.
- **A correção é de graça:** +0,001 na val_unseen e −0,004 na val, com 3 sementes.
- **Tirar o babble também não resolve:** o V3_noise ainda depende da polaridade em uma das 3 sementes.

**W5: ramo espectral bidirecional com 100 épocas** (`W5_bidir_final`):
- Sozinho, dá 2,967 na val_unseen (Bext: 2,902; V4_bidir curto: 2,915) e 2,552 na val (Bext: 2,483).
- Nas misturas com o K0extfull (3 sementes, val_unseen), o melhor peso é w = 0,5, com 3,249. O Bext dá 3,230 com w = 0,35: +0,019.
- É bidirecional, portanto só offline.

**WF: finais** (30 épocas, 3 sementes intercaladas; a correção de polaridade entra no controle e no S30; o teste tem 824 arquivos):

| Sistema | Parâmetros | Teste | STOI | SI‑SDR | COVL | val_unseen | val_unseen_low | val |
|---|---|---|---|---|---|---|---|---|
| K0extfull (10 épocas, ponto de partida) | 41,4M | 3,164 ± 0,008 | 0,9534 | 18,72 | 3,96 | 3,185 | 2,434 | 2,754 |
| K0ext30p (30 épocas + polaridade, controle) | 41,4M | 3,161 ± 0,001 | 0,9542 | 18,71 | 3,96 | 3,201 | 2,445 | 2,757 |
| K0ext30p soup | 41,4M | 3,165 | 0,9543 | 18,74 | 3,97 | 3,204 | 2,452 | 2,762 |
| **S30** (discriminador 0,3 + polaridade, 30 épocas) | 41,4M | **3,328 ± 0,011** | 0,9542 | 18,44 | 4,10 | 3,344 | 2,565 | 2,890 |
| **S30 soup** | 41,4M | **3,335** | 0,9544 | 18,47 | 4,10 | 3,348 | 2,572 | 2,897 |
| Mistura S30 + W5, w = 0,5 (peso do W5) | 60,9M | 3,353 ± 0,006 | 0,9569 | 19,05 | 4,13 | 3,345 | 2,631 | 2,946 |
| **Mistura S30 + W5, w = 0,25** (escolhido na val_unseen, 14.5) | 60,9M | **3,372 ± 0,009** | 0,9562 | 18,83 | 4,15 | 3,370 | 2,626 | 2,945 |
| **S30 soup + W5, w = 0,25** | 60,9M | **3,375** | 0,9564 | 18,85 | 4,15 | 3,373 | 2,631 | 2,951 |
| RX_C3 | 53,8M | 3,212 ± 0,008 | 0,9536 | 18,96 | 4,01 | 3,220 | 2,514 | 2,830 |
| 0,4·RX_C3 + 0,6·mistura S30 (w = 0,5) | ~115M | 3,335 ± 0,008 | 0,9561 | 19,13 | 4,12 | 3,333 | 2,633 | 2,943 |
| Média RX_C3 + mistura K0extfull (12.11) | ~105M | 3,238 ± 0,004 | 0,9541 | 18,97 | 4,03 | 3,253 | 2,535 | 2,851 |

- **O discriminador métrico é o grande ganho:** o S30 fica +0,167 acima do controle com o mesmo treino (K0ext30p). O ganho aparece nas 3 sementes, nos 5 ruídos do teste (+0,15 a +0,18) e em todas as faixas de SNR (+0,15 a +0,19).
- **30 épocas (com a correção de polaridade) não mudam o teste:** o K0ext30p fica em 3,161, contra 3,164 do K0extfull. Na val_unseen, ganham +0,016.
- **As soups ganham de +0,003 a +0,007 sobre a média das sementes em todos os conjuntos, com o custo de um modelo.**
- **O RX_C3 atrapalha o ensemble:** com ele, o resultado cai de 3,353 para 3,335 no teste. Ele só complementava a mistura antiga, feita com o K0extfull.
- **Sobre o uso do teste:** o w = 0,5 foi avaliado no teste dentro do WF. O w = 0,25 foi escolhido depois, só pela val_unseen, e avaliado no teste uma única vez.

**WR: robustez** (relatório completo em `results/abl/W_robustness_final.md`; o da fila está em `W_robustness.md`):

| Sistema | Arquivos do teste > 0,1 abaixo do K0extfull | Pior 1% do Δ por arquivo (teste) | Perda com a entrada invertida (val_unseen) | Teste em streaming (offline → streaming) |
|---|---|---|---|---|
| K0extfull | — | — | **1,36** | 3,154 → 2,868 (semente 1) |
| K0ext30, sem correção (semente 1) | — | — | **1,98** | — |
| K0ext30p | 4,1% | −0,361 | 0,000 | 3,163 → 2,911 (semente 1) |
| K0ext30p soup | 3,9% | −0,212 | −0,001 | 3,165 → 2,920 |
| S30 | 0,6% | −0,063 | +0,001 | 3,331 → 3,065 (semente 1) |
| **S30 soup** | 0,6% | −0,057 | 0,000 | **3,335 → 3,085** |
| Mistura S30 + W5, w = 0,25 | **0,2%** | **−0,026** | 0,000 | só offline |
| S30 soup + W5, w = 0,25 | 0,4% | −0,026 | −0,001 | só offline |
| RX_C3 | 5,9% | −0,472 | **1,26** | só offline |
| 0,4·RX_C3 + 0,6·mistura S30 | 0,8% | −0,084 | 0,10 | só offline |
| Média 12.11 | 1,6% | −0,163 | **1,19** | só offline |

- **Polaridade:** todo sistema com o K0extfull ou o RX_C3 herda o atalho. Com a correção, nenhum sistema perde mais de 0,002.
- **Cauda de falhas:** o S30 e as misturas com ele quase não têm cauda. O K0ext30p tem uma cauda de 4% contra o K0extfull, sem mudar a média; o S30, treinado do mesmo jeito mais o discriminador, não tem.
- **SNR baixo (val_unseen_low, −5 e 0 dB):** a soup do S30 ganha +0,138 sobre o K0extfull, e as misturas ganham +0,19 a +0,20. O pior ruído de cada sistema ainda fica acima do K0extfull.
- **Streaming:** todos os modelos de onda perdem de 0,25 a 0,29 no teste (ver 14.5).

### 14.5 Análises depois da fila (2026‑10‑16) ✔

**Peso da mistura para o S30.** O w = 0,5 do W5 foi escolhido com o K0extfull. Refeito para o S30 sem treino, na val_unseen (média das 3 sementes; a val serve de guarda):

| w | 0 (S30) | 0,15 | **0,25** | 0,35 | 0,5 |
|---|---|---|---|---|---|
| val_unseen | 3,344 | 3,367 | **3,370** | 3,365 | 3,345 |
| val | 2,890 | 2,932 | 2,945 | 2,951 | 2,946 |
| STOI (val_unseen) | 0,9236 | 0,9256 | 0,9266 | 0,9273 | 0,9280 |
| SI‑SDR (val_unseen) | 18,34 | 18,59 | 18,72 | 18,82 | 18,92 |

- Com o S30, que já é mais forte em PESQ, o peso ideal do ramo espectral cai. O w = 0,25 ganha +0,026 sobre o S30 sozinho (acima do δ_min) e passa na guarda da val.
- No teste, o w = 0,25 dá 3,372 ± 0,009: +0,044 sobre o S30 e +0,019 sobre o w = 0,5 (Wilcoxon por arquivo: p = 3·10⁻⁶⁰ e 2·10⁻²²). STOI e SI‑SDR sobem com w, então w = 0,5 é a escolha quando importam mais que o PESQ.
- A val_unseen acertou de novo o sentido do resultado no teste.

**O ganho do discriminador aparece fora do PESQ?** Médias das sementes no teste. Menor é melhor no WSS e no LLR.

| Sistema | PESQ‑WB | PESQ‑NB | STOI | SI‑SDR | segSNR | WSS | LLR |
|---|---|---|---|---|---|---|---|
| K0ext30p | 3,161 | 3,755 | 0,9542 | 18,71 | 9,40 | 3,88 | 0,289 |
| S30 | 3,328 | 3,821 | 0,9542 | 18,44 | 9,04 | 4,07 | 0,291 |
| Mistura S30 + W5, w = 0,25 | 3,372 | 3,859 | 0,9562 | 18,83 | 9,38 | 3,66 | 0,267 |
| Mistura S30 + W5, w = 0,5 | 3,353 | 3,853 | 0,9569 | 19,05 | 9,62 | 3,36 | 0,269 |
| RX_C3 | 3,212 | 3,781 | 0,9536 | 18,96 | 9,64 | 3,76 | 0,281 |

- **No S30, o ganho fica no PESQ.** Contra o K0ext30p, o PESQ‑WB sobe 0,167 e o PESQ‑NB só 0,066. O STOI não muda, e o SI‑SDR (−0,27 dB), o segSNR (−0,36 dB) e o WSS (+0,19) pioram um pouco.
- **O CSIG, o CBAK e o COVL não são evidência independente.** As três são combinações lineares com o PESQ (0,603, 0,478 e 0,805 por ponto de PESQ). Tirado esse termo, o S30 fica 0,03 abaixo no CSIG, 0,02 abaixo no CBAK e igual no COVL.
- **As misturas melhoram tudo de uma vez.** Ganham PESQ, STOI, WSS e LLR. O SI‑SDR e o segSNR voltam ao nível do K0ext30p com w = 0,25 e passam dele com w = 0,5.
- **Conclusão:** o PESQ do S30 é robusto ao deslocamento do sinal (PESQ robusto 3,329 contra 3,328), mas confirmar que a melhora é perceptível pede uma medida que não seja o PESQ: uma escuta ou um preditor de MOS sem referência (DNSMOS, UTMOS).

**Origem da perda em streaming.** Soup do S30, val_unseen, quadro a quadro com `cleanunet/streaming.py`:

| Normalização da entrada | PESQ | STOI | SI‑SDR |
|---|---|---|---|
| Offline (referência) | 3,348 | 0,9239 | 18,38 |
| Streaming, normalização causal atual (média corrente do desvio padrão por quadro) | 3,164 | 0,9136 | 17,21 |
| Streaming com a normalização do arquivo inteiro (não causal) | 3,339 | 0,9239 | 18,38 |

- O processamento quadro a quadro é praticamente exato (−0,009). **Quase toda a perda vem da estimativa causal da escala**, como já se via no estágio A (−0,16).
- **Por quê:** no início do arquivo há pouco sinal, a escala estimada fica pequena e a fala chega ao modelo amplificada além do que ele viu no treino. A média corrente demora a se ajustar.
- Uma variante com o RMS corrente foi testada em 9 arquivos e não ajudou (3,171 contra 3,189).

### 14.6 Conclusões e próximos passos

**(A) Melhor resultado: soup do S30 + ramo espectral W5, mistura com w = 0,25.**
- **Teste:** PESQ 3,375; STOI 0,956; SI‑SDR 18,85 dB; CSIG 4,72; CBAK 3,81; COVL 4,15.
- **Custo:** 60,9M parâmetros, offline, sem fusão aprendida. São duas redes treinadas separadamente, combinadas pela mistura de magnitudes de `cleanunet/blend.py`.
- **Robustez:** não depende da polaridade, a cauda de falhas é de 0,4% e tem 2,631 na val_unseen_low.
- **Variante:** com w = 0,5, troca 0,02 de PESQ por +0,2 dB de SI‑SDR e +0,0007 de STOI.

**(B) Modelo simples e robusto: soup do S30.**
- **Arquitetura e uso:** uma rede só de onda de 41,4M parâmetros, compatível com streaming.
- **Teste:** 3,335 (3 sementes: 3,328 ± 0,011).
- **Robustez:** não depende da polaridade, a cauda de falhas é de 0,6% e tem +0,138 em SNR baixo.
- **Receita:** `configs/ablation/k0.yaml` com `trainer.max_epochs=30`, `data.ext_noise` (MUSAN, p = 0,5, SNR de −5 a 20 dB), `augment_polarity=0.5` e `loss.metric_gan={weight: 0.3}`. Depois, `scripts/make_soup.py` sobre as 3 sementes.
- **Ressalvas:** o ganho é de PESQ (14.5), e o streaming perde 0,25 por causa da normalização.
  - Metade do ganho está na faixa de 50–100 Hz (14.7.2).
  - A perda do streaming tem correção sem treino na val_unseen (14.7.4).
- **Alternativa conservadora**, sem perda que mire o PESQ: a soup do K0ext30p, com 3,165 e as outras métricas iguais às do K0extfull.

**Próximos passos propostos:**
1. **Normalização para streaming** (objetivo B). *Atualizado em 14.7.4: uma escala fixa ou uma estimativa corrente com valor inicial resolve sem treino na val_unseen. O texto abaixo fica como registro.* O mais promissor é ajustar o S30 por algumas épocas com a normalização causal simulada no treino: a mesma escala corrente por quadro de `streaming.py`, aplicada ao recorte. A alternativa é um estimador causal melhor, por exemplo com valor inicial e ataque rápido, avaliado na val_unseen. Meta: recuperar a maior parte dos 0,18 da val_unseen.
2. **Validação perceptiva do discriminador métrico:** DNSMOS ou UTMOS (o download dos modelos segue a regra de arquivos não confiáveis) ou uma escuta ABX curta comparando S30 com K0ext30p.
3. **Ramo espectral causal** para uma mistura em streaming. O W5 é bidirecional.
4. **Estabilidade do treino com o discriminador:** houve 1 segfault em código nativo em ~4 h de treino com o pool de PESQ (`fork`). Trocar para `forkserver` é uma opção barata se voltar a acontecer.

### 14.7 Análise dos resultados (2026‑10‑16) ✔

Scripts e dados: `results/abl/W_analysis/` (ver o README). No teste, só entram sistemas que já tinham sido avaliados nele, e nenhuma escolha nova foi feita com ele. Os diagnósticos de streaming usam a val_unseen.

**14.7.1 O protocolo é confiável?**
- **A val_unseen prevê o teste.** Nos 52 sistemas avaliados nas duas, o Spearman é 0,978 e o Pearson 0,951 (teste ≈ 0,80 · val_unseen + 0,65, com resíduo de 0,031). A val antiga, com os ruídos do treino, dá Spearman 0,66 nos mesmos sistemas. A troca do critério de seleção (seção 10) estava certa.
- **Os efeitos estão bem acima do ruído.** Intervalos de 95% por bootstrap sobre os arquivos do teste, com a média das sementes:

| Comparação | ΔPESQ | IC 95% | Arquivos que melhoram | ΔSI‑SDR |
|---|---|---|---|---|
| S30 − K0ext30p (o discriminador) | +0,167 | [+0,157; +0,178] | 96% | −0,27 dB (pior em 96%) |
| Mistura w 0,25 − S30 | +0,044 | [+0,037; +0,051] | 75% | +0,39 dB (melhor em 99%) |
| S30 − RX_C3 | +0,117 | [+0,104; +0,130] | 84% | −0,52 dB |
| K0ext30p − K0extfull (30 épocas + polaridade) | −0,002 | [−0,009; +0,004] | 53% | −0,01 dB |
| Soup do S30 − média das sementes | +0,007 | [+0,005; +0,010] | 68% | +0,04 dB |

- **30 épocas sozinhas não ajudam no teste** (−0,002 no teste, +0,016 na val_unseen). O que mudou o S30 foi a perda.
- **O ganho do discriminador é uniforme:** +0,15 a +0,19 nos 4 níveis de SNR e +0,15 (bus) a +0,18 (office) nos ruídos.
- **Locutores.** O ganho varia por locutor sem seguir o gênero.
  - No teste: p232 (M) +0,13 e p257 (F) +0,20.
  - Na val_unseen: p226 (M) +0,21 e p287 (F) +0,09.
  - Com 2 locutores por partição, a generalização para locutores novos é o ponto mais fraco do benchmark.
- **O discriminador aumenta a variância entre sementes:** ±0,011 no S30 contra ±0,0013 no K0ext30p. A soup absorve essa variância.

**14.7.2 O que o discriminador mudou no sinal.** Atribuição no teste: parte‑se da saída da soup do K0ext30p e troca‑se uma região tempo‑frequência por vez pela saída da soup do S30 (STFT 512/128, depois iSTFT). O ganho total é +0,170.

| Região trocada | PESQ | Δ | SI‑SDR |
|---|---|---|---|
| Nenhuma (K0ext30p) | 3,165 | — | 18,74 |
| 0–100 Hz | 3,264 | **+0,099** | **18,49** |
| 100–200 Hz / 200–300 Hz | 3,182 / 3,172 | +0,017 / +0,006 | 18,72 / 18,74 |
| 300–3400 Hz | 3,213 | +0,048 | 18,74 |
| Acima de 3,4 kHz | 3,169 | +0,004 | 18,74 |
| Quadros de 0 a 20 dB abaixo do máximo | 3,215 | +0,049 | 18,66 |
| Quadros de 20 a 40 dB abaixo (fala fraca, transições) | 3,277 | **+0,111** | 18,59 |
| Quadros mais de 40 dB abaixo (pausas) | 3,172 | +0,007 | 18,71 |
| Só a magnitude / só a fase | 3,313 / 3,191 | +0,148 / +0,026 | 18,52 / 18,73 |
| Tudo (S30) | 3,335 | +0,170 | 18,47 |

- **Quase 60% do ganho e toda a perda de SI‑SDR estão abaixo de 100 Hz**, sobretudo nos quadros de fala fraca: a faixa < 300 Hz só desses quadros vale +0,078. A troca inversa confirma: o S30 com a faixa 0–100 Hz do K0ext30p perde 0,097.
- **Por que essa faixa pesa tanto no PESQ‑WB.**
  - O filtro de entrada do P.862.2 (pacote `pesq`) é um passa‑altas de 2ª ordem: −12 dB em 50 Hz e −3 dB em 100 Hz, em relação à banda passante.
  - O alinhamento de nível usa só a potência acima de 300 Hz.
  - Então o conteúdo entre 50 e 100 Hz entra direto na perturbação.
  - Abaixo de 50 Hz não importa: tirar esse conteúdo das saídas mal muda a diferença entre os sistemas (+0,162).
- **É uma faixa difícil e com pouca fala.**
  - 88% da energia do ruído do teste está abaixo de 100 Hz. O SNR de entrada é −16 dB em 0–50 Hz e −0,2 dB em 50–100 Hz.
  - Na referência limpa, 50–100 Hz tem 1,6% da energia no p232 (região do F0 masculino) e 0,15% no p257.
- **Não é mais supressão de ruído.** Pelo filtro‑sombra (G = Y/X aplicado à fala e ao ruído), o S30 atenua o ruído um pouco menos (9,7 contra 9,9 dB). Ele também deixa mais resíduo nas pausas: −13,1 contra −14,2 dB em relação à entrada.
- **Avaliação sem a faixa abaixo de 100 Hz.** Passa‑altas de fase zero em 100 Hz na referência e na saída:

| Sistema | PESQ | PESQ sem < 100 Hz | SI‑SDR sem < 100 Hz |
|---|---|---|---|
| Soup do K0ext30p | 3,165 | 3,337 | 25,96 |
| RX_C3 (3 sementes) | 3,211 | 3,363 | 25,94 |
| Soup do S30 | 3,335 | 3,415 | 25,89 |
| Soup do S30 + W5, w 0,25 | 3,375 | 3,455 | 26,11 |

  - **S30 − K0ext30p** cai de +0,170 para **+0,078** [+0,073; +0,084]. 92% dos arquivos melhoram, com só −0,07 dB de SI‑SDR. Por locutor: +0,072 e +0,083.
  - **S30 − RX_C3** cai de +0,124 para +0,053 [+0,045; +0,061].
  - **A mistura − S30** fica igual (+0,040): o ganho da mistura está na banda da fala.
  - **A mistura − RX_C3** dá +0,092 [+0,084; +0,100], e o SI‑SDR passa a favorecer a mistura (+0,17 dB).
  - A ordem dos sistemas não muda.
- **O SI‑SDR deste benchmark é dominado pelo que está abaixo de 100 Hz.**
  - Sem essa faixa, o SI‑SDR vai de ~18,7 para ~26 dB em todos os sistemas, e as diferenças entre eles caem para ≤ 0,2 dB.
  - A vantagem de SI‑SDR do RX_C3 (+0,5 dB) vem quase toda dessa faixa: sem ela, a diferença para o S30 cai para 0,04 dB.
  - Um passa‑altas de fase zero em 50 Hz só na saída dá +0,8 a +0,9 dB de SI‑SDR ao K0ext30p e ao S30, sem mudar o PESQ.
  - Um passa‑altas causal derruba o SI‑SDR por causa da fase: com 4ª ordem em 50 Hz, ele cai para ~5 dB.
- **Leitura.**
  - Metade do ganho do discriminador (+0,08) é melhora na banda da fala, sem custo de SI‑SDR acima de 100 Hz.
  - A outra metade (+0,09) está entre 50 e 100 Hz, que o PESQ‑WB pesa muito e cuja relevância perceptiva é pequena. Alto‑falantes de celular e de notebook nem reproduzem essa faixa.
  - A escuta deve olhar essa faixa, com fones.

**14.7.3 Mistura com o W5.**
- **O peso fixo é quase ótimo.** Na val_unseen, o melhor w por arquivo (entre 0 e 0,5) daria 3,398, só +0,028 sobre o w = 0,25 fixo. Uma regra pelo SNR verdadeiro daria +0,002. Uma porta aprendida tem pouco espaço.
- **O ramo espectral ajuda mais com SNR baixo.**
  - Na val_unseen, o melhor w é 0,35 em 2,5 dB e 0,15 em 12,5 e 17,5 dB.
  - No teste, a mistura ganha +0,061 em 2,5 dB e +0,027 em 17,5 dB.
  - Na val_unseen_low, ganha +0,061.
- **O ganho de PESQ da mistura vem dos quadros de fala e se espalha pelas faixas.** O maior pedaço é +0,014, em 300–3400 Hz, e das pausas vem só +0,002. Já a recuperação de SI‑SDR (+0,38 dB) vem quase toda de < 100 Hz (+0,32).

**14.7.4 Streaming.** Soup do S30, val_unseen.
- **A rede tolera bem um erro estático de escala.** Offline, com a escala da entrada multiplicada por c:

| c | 0,5 | 0,71 | 0,85 | 1 | 1,18 | 1,41 | 2 |
|---|---|---|---|---|---|---|---|
| ΔPESQ | −0,058 | −0,014 | −0,003 | 0 (3,348) | −0,002 | −0,011 | −0,042 |

- **A estimativa corrente erra muito no começo.** Em relação ao desvio padrão do arquivo inteiro, ela vale (mediana, com o intervalo dos percentis 10–90):
  - 0,29 (0,14–0,63) aos 100 ms;
  - 0,30 aos 500 ms;
  - 0,78 (0,69–0,87) no fim.

  As frases têm mediana de 2,3 a 2,6 s, então boa parte de cada arquivo entra amplificada cerca de 3×.
- **O viés final não explica a perda.** Offline, fixando por arquivo a razão final, a perda é −0,010. Com a média geométrica da trajetória, é −0,031.
- **A perda está no primeiro segundo.** SNR por trecho, estimativa corrente contra oráculo:
  - 0,5–1 s: 15,4 contra 20,3 dB;
  - 1–1,5 s: 18,6 contra 19,8 dB;
  - depois de 2 s: 12,3 contra 12,5 dB.
- **Correções sem treino.** Streaming quadro a quadro; os valores a priori vêm da val, não da val_unseen:

| Escala da entrada | PESQ | STOI | SI‑SDR |
|---|---|---|---|
| Offline | 3,348 | 0,9239 | 18,38 |
| Corrente, como está hoje | 3,164 | 0,9136 | 17,21 |
| Arquivo inteiro (oráculo, não causal) | 3,339 | 0,9239 | 18,38 |
| **Fixa** (0,075, a mediana da val) | **3,335** | 0,9235 | 18,25 |
| **Corrente com valor inicial** (62 pseudo‑quadros, ≈ 1 s, em 0,060: a mediana da estimativa final na val) | **3,325** | 0,9232 | 18,51 |

- **A escala fixa só serve com nível de entrada calibrado.** No VB‑DEMAND, o desvio padrão da entrada varia só 1,8–2× entre os percentis 10 e 90. A versão com valor inicial continua adaptativa.
- **O ajuste fino com a normalização causal simulada (14.6, passo 1) deixa de ser necessário neste benchmark.** Falta a confirmação única no teste, onde o streaming dá hoje 3,085.

**14.7.5 Consequências.**
- **As escolhas (A) e (B) se mantêm.** A ordem dos sistemas resiste à exclusão de < 100 Hz.
- **Ao relatar o ganho do discriminador**, separar a parte da banda da fala (+0,08) da parte de 50–100 Hz (+0,09). Vale reportar o PESQ sem < 100 Hz como métrica secundária.
- **(B) fica viável em streaming** quase com a qualidade offline e sem treino, se o teste confirmar.

Próximos passos, em ordem:
1. **Confirmar no teste, uma vez, o streaming da soup do S30** com a escala corrente com valor inicial, e com a fixa como referência (~5 min). Depois, levar o valor inicial para `cleanunet/streaming.py` como opção.
2. **Escuta ABX curta** do S30 contra o K0ext30p, com fones, prestando atenção à faixa de 50–100 Hz. E/ou DNSMOS/UTMOS, seguindo a regra de arquivos não confiáveis.
3. **Se a parte de 50–100 Hz for julgada irrelevante**, treinar o discriminador com o PESQ calculado sobre sinais filtrados acima de 100 Hz, para a perda mirar só a banda da fala.
4. **Ramo espectral causal** para uma mistura em streaming (sem mudança).
5. **Porta aprendida para o peso da mistura:** o teto é +0,03, então a prioridade é baixa.

### 14.8 Fila X: os próximos passos de 14.7 (2026‑10‑16)

Pedido do usuário: "executar os experimentos propostos". Foram três frentes: o streaming com valor inicial no teste e a opção no código; a validação perceptiva (DNSMOS e uma página de escuta); e o discriminador restrito à banda acima de 100 Hz (S30hp).

**Implementação.** Testes em `tests/test_x_experiments.py` (6 novos); a suíte tem 140 testes e passa.
- **Streaming com valor inicial.** `cleanunet/streaming.py`: `StreamingCleanUNet` e `stream_denoise` ganharam `prior_std` e `prior_frames`.
  - A média corrente do desvio padrão por quadro começa de `prior_frames` pseudo‑quadros em `prior_std`.
  - `prior_frames = inf` fixa a escala.
  - Sem esses parâmetros, o resultado é o de antes, bit a bit.
  - Opções no `evaluate.py`: `--stream_prior_std` e `--stream_prior_frames`.
- **PESQ sem < 100 Hz.** `evaluation/metrics.py` ganhou `highpass` (Butterworth de 4ª ordem, ida e volta, fase zero) e a métrica `pesq_wb_hp`: o PESQ‑WB com a referência e a estimativa filtradas acima de 100 Hz. Ativa com `evaluate.py --band_limited`.
- **Discriminador restrito, `loss.metric_gan.highpass_hz`** (`cleanunet/metric_disc.py` e o módulo).
  - O alvo do discriminador passa a ser o PESQ dos dois sinais filtrados.
  - O discriminador vê os sinais filtrados pelo mesmo |H|², aplicado pela FFT, de forma diferenciável.
  - Assim, a perda métrica não age abaixo de 100 Hz.
- **DNSMOS.** `evaluation/dnsmos.py` reimplementa o `dnsmos_local.py` da Microsoft: P.835 (SIG, BAK, OVRL) e P.808, com janelas de 9,01 s. O código baixado não é executado.
  - Os modelos ONNX estão em `/home/fred/DATASETS/DNSMOS_dl_20261016/` (sha256 `269fbebd…` do `sig_bak_ovr.onnx`, `9246480c…` do `model_v8.onnx`).
  - O script roda com `python -I`.
  - No teste, a referência limpa dá OVRL 3,221, SIG 3,511, BAK 4,038 e P808 3,533. O ruidoso dá 2,697, 3,346, 3,126 e 3,053: os valores conhecidos do VB‑DEMAND.
- **Lançadores e saídas.**
  - `logs/abl/run_X.sh`: o S30hp.
  - `logs/abl/run_X_eval.sh`: o streaming e o PESQ sem < 100 Hz dos sistemas existentes, em diretórios novos `*_bl`.
  - `logs/abl/run_X_dnsmos.sh`: o DNSMOS.
  - Log em `logs/abl/queue_X.log`, áudio em `results/abl/X_audio/` e DNSMOS em `results/abl/X_dnsmos/`.
  - Scripts de análise, relatório (`x_report.txt`) e a página de escuta: `results/abl/X_analysis/` (ver o README).

**X1. Streaming com valor inicial (teste, uma rodada por configuração).** Os valores a priori vêm da val, sem ajuste na val_unseen nem no teste:
- valor inicial: 0,0604 com 62 pseudo‑quadros (≈ 1 s), a mediana da estimativa final na val;
- escala fixa: 0,075, a mediana do desvio padrão do arquivo inteiro na val.

Na val_unseen, a implementação reproduz o diagnóstico de 14.7.4: 3,3253 e 3,3346.

| Soup | Offline | Streaming atual | Com valor inicial | Escala fixa |
|---|---|---|---|---|
| S30, PESQ | 3,335 | 3,085 | **3,323** | 3,317 |
| S30, STOI / SI‑SDR | 0,9544 / 18,47 | 0,9416 / 17,26 | 0,9536 / 18,70 | 0,9546 / 18,39 |
| K0ext30p, PESQ | 3,165 | 2,920 | **3,151** | 3,142 |
| K0ext30p, STOI / SI‑SDR | 0,9543 / 18,74 | 0,9405 / 17,34 | 0,9535 / 18,89 | 0,9544 / 18,68 |

- **O valor inicial recupera ~95% da perda nos dois modelos**, sem treino. A escala fixa recupera ~93%. Na val_unseen a ordem entre as duas se inverte (S30: 3,325 contra 3,335), mas ambas ficam a ≤ 0,03 do offline.
- **Recomendado: o valor inicial**, que continua adaptativo. A escala fixa depende de um nível de entrada calibrado.
- **(B) em streaming fica com 3,323 no teste**, contra 3,335 offline (STOI 0,9536, SI‑SDR 18,70). O passo 1 de 14.6 (ajuste fino com normalização causal) deixa de ser necessário.

**X2. DNSMOS no teste (824 arquivos).** Diferenças pareadas por arquivo com IC 95% por bootstrap, e a fração de arquivos que melhoram:

| Sistema | SIG | BAK | OVRL | P808 |
|---|---|---|---|---|
| Referência limpa | 3,511 | 4,038 | 3,221 | 3,533 |
| Ruidoso | 3,346 | 3,126 | 2,697 | 3,053 |
| Soup do K0ext30p (controle) | 3,512 | 4,045 | 3,225 | 3,437 |
| Soup do S30 | 3,503 | 4,008 | 3,198 | 3,447 |
| S30 com a faixa < 100 Hz do controle | 3,505 | 4,044 | 3,217 | 3,436 |
| Controle com a faixa < 100 Hz do S30 | 3,509 | 4,010 | 3,204 | 3,448 |
| Soup do S30 + W5, w 0,25 | 3,503 | 4,026 | 3,206 | 3,461 |
| RX_C3 (semente 1) | 3,512 | 4,036 | 3,219 | 3,461 |
| Soup do S30, streaming atual | 3,475 | 4,043 | 3,193 | 3,449 |
| Soup do S30, streaming com valor inicial | 3,499 | 4,030 | 3,205 | 3,443 |

- **O DNSMOS não confirma o ganho do discriminador.** S30 − controle:
  - OVRL −0,026 [−0,029; −0,024], com só 22% dos arquivos melhorando;
  - BAK −0,037 [−0,040; −0,034];
  - SIG −0,009;
  - P808 +0,010 [+0,008; +0,012].
- **A diferença vem da faixa abaixo de 100 Hz.** Devolver ao S30 a faixa < 100 Hz do controle recupera +0,019 de OVRL e +0,035 de BAK (91% dos arquivos). Pôr no controle a faixa do S30 tira −0,020 e −0,035. Acima de 100 Hz, S30 e controle ficam praticamente iguais no DNSMOS (OVRL −0,008). O mesmo conteúdo de 50–100 Hz que rende +0,10 de PESQ‑WB soa ao DNSMOS como um pouco mais de ruído de fundo.
- **O DNSMOS está perto do teto.** O controle já empata com a referência limpa (3,225 contra 3,221). As diferenças entre sistemas (≤ 0,03) são pequenas perto do ganho sobre o ruidoso (+0,5) e provavelmente abaixo do que um ouvinte nota. Por isso o DNSMOS não decide; a escuta decide.
- **Os outros pares têm efeitos pequenos.**
  - Mistura − S30: OVRL +0,008, BAK +0,018.
  - S30 − RX_C3: OVRL −0,021.
  - Streaming com valor inicial − streaming atual: OVRL +0,012, SIG +0,024. O aquecimento da escala prejudicava o sinal de fala.

**X3. PESQ sem < 100 Hz, 3 sementes (`*_bl`).** Média ± desvio padrão amostral entre as sementes:

| Sistema | Teste: PESQ | Teste: sem < 100 Hz | val_unseen: PESQ | val_unseen: sem < 100 Hz |
|---|---|---|---|---|
| K0ext30p | 3,161 ± 0,001 | 3,333 ± 0,002 | 3,201 ± 0,002 | 3,381 ± 0,002 |
| RX_C3 | 3,211 ± 0,008 | 3,363 ± 0,005 | 3,220 ± 0,018 | 3,382 ± 0,013 |
| S30 | 3,328 ± 0,011 | 3,410 ± 0,006 | 3,344 ± 0,012 | 3,443 ± 0,008 |
| S30 + W5, w 0,25 | 3,372 ± 0,009 | 3,452 ± 0,006 | 3,370 ± 0,010 | 3,472 ± 0,006 |

- **S30 − K0ext30p sem < 100 Hz:** +0,077 [+0,072; +0,082] no teste, com 93% dos arquivos melhorando, e +0,062 na val_unseen. Com o PESQ completo, era +0,167 e +0,143.
- **As soups dão os mesmos números da análise de 14.7.2**, o que confirma a implementação.

**X4. Discriminador restrito à banda acima de 100 Hz (S30hp).**
- **Receita:** a do S30 com `loss.metric_gan={weight: 0.3, highpass_hz: 100}`, 3 sementes e soup.
- **Treino:** as 3 sementes treinaram sem falhas nem retomadas.
- **Avaliação:** com `--band_limited`. ± é o desvio padrão amostral entre as sementes.

| Teste | PESQ | PESQ sem < 100 Hz | SI‑SDR | PESQ‑NB | CSIG / CBAK / COVL | segSNR | WSS |
|---|---|---|---|---|---|---|---|
| K0ext30p (controle) | 3,161 ± 0,001 | 3,333 ± 0,002 | 18,71 | 3,755 | 4,61 / 3,71 / 3,96 | 9,40 | 3,88 |
| S30 | 3,328 ± 0,011 | 3,410 ± 0,006 | 18,44 | 3,821 | 4,68 / 3,77 / 4,10 | 9,04 | 4,07 |
| **S30hp** | 3,274 ± 0,005 | **3,427 ± 0,004** | 18,65 | 3,822 | 4,66 / 3,76 / 4,05 | 9,32 | 3,88 |
| Soup do S30 | 3,335 | 3,415 | 18,47 | 3,825 | 4,69 / 3,77 / 4,10 | 9,07 | 4,06 |
| Soup do S30hp | 3,280 | 3,430 | 18,68 | 3,825 | 4,67 / 3,76 / 4,06 | 9,35 | 3,86 |
| Soup do S30 + W5, w 0,25 | **3,375** | 3,455 | 18,85 | 3,862 | 4,72 / 3,81 / 4,15 | 9,41 | 3,65 |
| Soup do S30hp + W5, w 0,25 | 3,326 | **3,469** | **19,00** | 3,861 | 4,71 / 3,80 / 4,11 | 9,61 | **3,52** |

O STOI é 0,954 sem o W5 e 0,956 com ele em todas as linhas. Diferenças pareadas no teste, por arquivo e com a média das sementes:

| Comparação | PESQ | PESQ sem < 100 Hz | SI‑SDR |
|---|---|---|---|
| S30hp − S30 | −0,054 [−0,062; −0,047], 18% melhoram | **+0,017** [+0,013; +0,021], 63% | **+0,21 dB**, 94% |
| S30hp − K0ext30p | +0,113 [+0,106; +0,120], 95% | +0,094 [+0,089; +0,099], 93% | −0,06 dB, 23% |
| S30 − K0ext30p (referência) | +0,167 [+0,157; +0,178], 96% | +0,077 [+0,072; +0,082], 93% | −0,27 dB, 4% |
| Soups com o W5: S30hp − S30 | −0,049 [−0,056; −0,043], 18% | +0,014 [+0,012; +0,017], 64% | +0,15 dB, 93% |

Na val_unseen o padrão se repete:
- S30hp − S30: −0,052, +0,014 e +0,21 dB.
- S30hp − controle: +0,091, +0,076 e +0,00 dB.

- **A restrição funcionou, mas não por completo.**
  - **O SI‑SDR volta ao do controle:** −0,06 dB no teste e +0,00 na val_unseen, contra −0,27 e −0,21 dB do S30. segSNR e WSS também voltam ao nível do controle.
  - **O erro em 50–100 Hz também volta.** A potência do erro relativa à da referência, somada no teste, fica em −12,5 dB, contra −12,6 dB no controle e −12,0 dB no S30.
  - **Resta um pedaço abaixo de 100 Hz.** Pela troca de faixas de 14.7.2 (`x_hp_swap.py`):
    - posta no controle, a faixa < 100 Hz do S30hp ainda vale +0,041 de PESQ (a do S30 valia +0,099);
    - trocá‑la pela do controle tira 0,043 do S30hp (no S30, tirava 0,097).
  - **Esse resíduo vem do corte suave.**
    - O filtro (Butterworth de 4ª ordem, aplicado ida e volta) atenua só 6 dB em 100 Hz, 8,5 dB em 94 Hz e 17 dB em 80 Hz.
    - A janela da STFT leva para os bins abaixo de 100 Hz parte do que está logo acima.
    - O PESQ sem < 100 Hz usa o mesmo filtro e também conta essa transição.
- **Sem < 100 Hz, o S30hp fica +0,094 acima do controle, contra +0,077 do S30.** Aqui estava escrito que o ganho na banda da fala aumentou. **Revisto em 14.8.2:** o PESQ‑NB dá o mesmo ganho aos dois, e o PESQ filtrado é o próprio alvo do discriminador do S30hp.
- **O custo é o PESQ‑WB completo.**
  - São −0,054 em relação ao S30. CSIG e COVL caem junto, porque são regressões sobre o PESQ.
  - Ainda assim, ficam +0,113 sobre o controle, com 95% dos arquivos melhorando.
- **Robustez.**
  - **Variância entre sementes menor:** ± 0,005, contra ± 0,011 do S30.
  - **Não depende da polaridade:** 3,295 na val_unseen e 3,299 com a entrada invertida.
  - **SNR baixo:** 2,534 na val_unseen_low, contra 2,565 do S30 e 2,445 do controle.
  - **Cauda de falhas:** 1,1% dos arquivos ficam mais de 0,1 abaixo do K0extfull (0,8% na soup). O S30 tem 0,6% e o controle K0ext30p, 4,1%.
  - **Streaming com valor inicial:** a soup dá 3,273 no teste, a −0,007 do offline. O S30 fica a −0,012.
- **DNSMOS no teste:**

| Sistema | SIG | BAK | OVRL | P808 |
|---|---|---|---|---|
| Soup do K0ext30p (controle) | 3,512 | 4,045 | 3,225 | 3,437 |
| Soup do S30 | 3,503 | 4,008 | 3,198 | 3,447 |
| Soup do S30hp | 3,508 | 4,034 | 3,215 | 3,438 |
| Soup do S30hp + W5, w 0,25 | 3,507 | 4,046 | 3,219 | 3,450 |
| Soup do S30hp, streaming com valor inicial | 3,501 | 4,050 | 3,218 | 3,428 |

  - **S30hp − S30:**
    - OVRL +0,016 [+0,014; +0,018], com 73% dos arquivos melhorando;
    - BAK +0,025, com 82%;
    - P808 −0,009.
  - **S30hp − controle:** OVRL −0,010 [−0,012; −0,008], BAK −0,012, P808 +0,001 (não significativo).
  - **O S30hp recupera ~60% do déficit do S30**, na mesma proporção em que caiu o efeito da troca de faixas (de +0,099 para +0,041). O que falta acompanha o resíduo abaixo de 100 Hz.
  - **Com o W5:** a mistura do S30hp − a do S30 dá OVRL +0,013 e BAK +0,019.
  - **O DNSMOS não vê ganho do discriminador nem na banda da fala** (S30hp − controle: OVRL −0,010, P808 +0,001). Como o controle já empata com a referência limpa, ele não separa sistemas desse nível. Aqui ele serve só para detectar degradação, como o BAK abaixo de 100 Hz.

**X5. Página de escuta cega:** https://claude.ai/artifact/GqL1QYn6mBzMSc4KHuijXm (privada, para ouvir com fones).
- **Conteúdo:** 22 tentativas, com arquivos do teste em WAV.
  - **Bloco A:** 8 de preferência, soup do S30 contra a do K0ext30p.
  - **Bloco B:** 8 ABX, S30 contra o S30 com a faixa < 100 Hz do controle. Testa se essa faixa é audível.
  - **Bloco C:** 6 de preferência, a mistura com o W5 contra o S30.
- **Seleção dos arquivos:** entre os percentis 75 e 95 do ΔPESQ por arquivo (PESQ robusto nos blocos A e C), equilibrados por locutor. A posição de cada sistema e o X do ABX foram sorteados uma vez, na geração das tentativas.
- **Respostas:** um documento por resposta na coleção `respostas` do banco do artefato (uid, tentativa, bloco, arquivo, ruído, SNR, ordem, resposta, acerto ou escolha, tempo). Para ler: ArtifactData `list`.
- **Arquivos:** `listen_audio.py` gera os WAVs, e a página está em `results/abl/X_analysis/listen/`.
- **Situação em 2026‑10‑16, 11:15:** nenhuma resposta ainda.

**14.8.1 Conclusões da fila X.**
1. **Streaming resolvido sem treino.**
   - Com a escala corrente com valor inicial (`--stream_prior_std 0.0604 --stream_prior_frames 62`), o streaming fica a ≤ 0,014 do offline no teste: S30 3,323, S30hp 3,273, K0ext30p 3,151. Antes, perdia 0,25.
   - O DNSMOS do streaming com valor inicial fica igual ao do offline ou um pouco acima (OVRL +0,003 a +0,007).
2. **O discriminador métrico tem dois efeitos.**
   - **Na banda da fala**, um ganho de PESQ que se mantém sem < 100 Hz e no PESQ‑NB (+0,066 no S30, +0,067 no S30hp), sem custo de SI‑SDR.
   - **Em 50–100 Hz**, um conteúdo que o PESQ‑WB premia (+0,10). Ele custa SI‑SDR (−0,27 dB), e o DNSMOS o lê como ruído de fundo (BAK −0,035).
   - O S30hp separa os dois: mantém o primeiro e corta ~60% do segundo.
3. **Recomendação.**
   - **Para o número de benchmark (PESQ‑WB)**, continuam a soup do S30 + W5 (3,375) e a soup do S30 (3,335), como em 14.6. A ressalva é explícita: ~0,09 desse ganho é a faixa de 50–100 Hz.
   - **Para uso, a soup do S30hp como (B) e a soup do S30hp + W5 como (A).** Elas são melhores em todas as outras métricas.
     - **(B)** PESQ 3,280; sem < 100 Hz 3,430; SI‑SDR 18,68 dB; DNSMOS OVRL 3,215. A variância entre sementes é a metade da do S30.
     - **(A)** PESQ 3,326; sem < 100 Hz 3,469 e SI‑SDR 19,00 dB, os maiores de todos os sistemas; WSS 3,52, o menor; DNSMOS OVRL 3,219.
   - A escolha entre S30 e S30hp é uma questão sobre a faixa de 50–100 Hz, e é isso que o bloco B da escuta testa.
     - Se a faixa não for audível, o S30hp fica com a vantagem nas métricas restantes.
     - Se for audível e a versão do S30 for preferida, o PESQ‑WB estava certo, e vale o S30.
4. **O DNSMOS está no teto deste benchmark:** detecta degradação, mas não ordena sistemas que já igualam a referência limpa.

**Próximos passos:**
1. **Escuta** (depende do usuário): a página de X5 leva ~10 min, com fones.
2. **Corte mais íngreme ou mais alto no discriminador** (8ª ordem, ou 150 Hz), para tirar o resíduo de +0,04 abaixo de 100 Hz: ~3 h para 3 sementes. Só vale se a escuta mostrar que essa faixa importa.
3. **Ramo espectral causal**, para uma mistura em streaming (sem mudança). Não entrou na fila X.
4. **Porta aprendida para o peso da mistura:** teto de +0,03, prioridade baixa. Não entrou na fila X.

**14.8.2 Análise complementar (2026‑10‑16, noite).** Sem treino nem inferência nova: só os resultados por arquivo já gravados. Script `results/abl/X_analysis/x_extra.py`, saída `x_extra.txt`.

**PESQ‑NB como verificação da banda da fala.** O PESQ‑NB só avalia 300–3400 Hz e não é o alvo de nenhum dos dois discriminadores. É do mesmo algoritmo, então não substitui a escuta. Diferenças pareadas no teste, por arquivo e com a média das sementes:

| Comparação | PESQ‑WB | Sem < 100 Hz | PESQ‑NB | SI‑SDR | segSNR | WSS (menor é melhor) |
|---|---|---|---|---|---|---|
| S30 − K0ext30p | +0,167 | +0,077 | +0,066 [+0,062; +0,070], 94% | −0,27 dB | −0,36 | +0,19 |
| S30hp − K0ext30p | +0,113 | +0,094 | +0,067 [+0,063; +0,071], 95% | −0,06 dB | −0,08 | −0,00 |
| S30hp − S30 | −0,054 | +0,017 | +0,001 [−0,001; +0,002], 46% | +0,21 dB | +0,28 | −0,19 |

- **O ganho do discriminador na banda da fala é real e igual nos dois modelos:** +0,066 e +0,067 de PESQ‑NB sobre o controle. Ele aparece nas 3 sementes (t de 28 e 66 entre as médias por semente), nos 5 ruídos e nos 4 SNRs do teste.
- **O +0,017 do S30hp sobre o S30 sem < 100 Hz não aparece no PESQ‑NB.** O PESQ filtrado é o alvo do discriminador do S30hp, como o PESQ‑WB é o do S30, e cada modelo ganha na métrica que otimiza. O S30hp mantém o ganho do S30 na banda da fala, sem aumentá‑lo.
- **As vantagens do S30hp sobre o S30 fora do PESQ:** SI‑SDR +0,21 dB, segSNR +0,28, WSS −0,19, DNSMOS OVRL +0,016 e BAK +0,025, com metade da variância entre sementes.
- **Contra o controle, o S30hp empata em tudo o que não é PESQ:**
  - STOI igual, WSS igual (−0,003), SI‑SDR −0,06 dB no teste e +0,00 na val_unseen, LLR +0,003;
  - o ganho fica só no PESQ: NB +0,067, WB +0,113;
  - o DNSMOS OVRL fica −0,010.

**Locutores.** O teste tem 2 locutores (p232 e p257) e a val_unseen outros 2 (p226 e p287). Os ICs por arquivo tratam ~800 arquivos como independentes e não medem a variação entre locutores. Médias por locutor:

| | p232 (teste) | p257 (teste) | p226 (val_unseen) | p287 (val_unseen) |
|---|---|---|---|---|
| S30 − K0, PESQ‑WB | +0,128 | +0,203 | +0,208 | +0,088 |
| S30 − K0, sem < 100 Hz | +0,070 | +0,083 | +0,062 | +0,062 |
| Parte abaixo de 100 Hz (a diferença das duas linhas) | 0,058 | 0,120 | 0,146 | 0,026 |
| S30hp − S30, sem < 100 Hz | +0,003 | +0,030 | +0,049 | −0,015 |
| S30hp − S30, SI‑SDR | +0,31 dB | +0,13 dB | +0,22 dB | +0,20 dB |
| S30hp − K0, sem < 100 Hz | +0,073 | +0,113 | +0,111 | +0,047 |
| Energia da referência limpa em 50–100 Hz (dB do total) | −12,4 | −25,9 | −12,1 | −8,3 |

- **Estáveis nos 4 locutores:**
  - o ganho sem < 100 Hz do S30 e do S30hp sobre o controle;
  - a perda de PESQ‑WB e o ganho de SI‑SDR do S30hp sobre o S30.
- **A parte abaixo de 100 Hz varia 5×** entre os locutores, de 0,026 a 0,146.
- **O +0,017 do S30hp sobre o S30 sem < 100 Hz vai de −0,015 a +0,049.** Ele acompanha a parte abaixo de 100 Hz: é maior onde o S30 ganhava mais abaixo de 100 Hz (p226 e p257). Isso reforça que vem da transição do corte suave, não da banda da fala.
- **Dentro de cada split, o locutor com menos energia em 50–100 Hz na referência tem o maior efeito abaixo de 100 Hz:** p257 no teste e p226 na val_unseen.
  - Em p257, uma voz mais aguda (energia concentrada em 200–300 Hz), a referência quase não tem energia nessa faixa (−25,9 dB do total).
  - O erro de todos os sistemas em 50–100 Hz é do tamanho da própria referência (−0,4 a −0,9 dB): o que o PESQ‑WB premia ali não é fala reproduzida.
  - Com 4 locutores, é só um indício.

**Ruído e SNR.**
- Todos os efeitos têm o mesmo sinal nos 5 ruídos e nos 4 SNRs do teste, e nos 5 ruídos da val_unseen.
- O déficit de DNSMOS do S30 cresce com o ruído:
  - OVRL −0,016 em 17,5 dB e −0,037 em 2,5 dB;
  - BAK −0,019 e −0,057.
- O S30hp reduz esse déficit à metade em cada SNR. Isso é coerente com o conteúdo de 50–100 Hz do S30 ser ruído residual.

**Nível em 50–100 Hz.**
- O S30 não põe mais energia na faixa: mediana −0,07 dB por arquivo, contra o controle. O erro na faixa sobe +0,6 dB.
- Por arquivo, nem a diferença de nível nem a de erro em 50–100 Hz prevê a parte do ganho de PESQ abaixo de 100 Hz (|r| ≤ 0,11).
- O mecanismo dentro do PESQ continua em aberto.

**Treino.** As curvas de treino do S30 e do S30hp (PESQ dos lotes, perda do discriminador, perda de onda) têm a mesma forma, sem instabilidade. As duas ainda sobem ~0,01 de PESQ nas últimas 10 épocas.

**Consequências.**
1. **A recomendação de 14.8.1 não muda, mas a justificativa sim.** O S30hp tem o mesmo ganho de PESQ na banda da fala que o S30 (PESQ‑NB) e é melhor em SI‑SDR, segSNR, WSS e DNSMOS. STOI e LLR ficam iguais. Ele não tem "mais ganho na banda da fala".
2. **Ao relatar o ganho do discriminador na banda da fala, usar o PESQ‑NB**, não o PESQ sem < 100 Hz. O PESQ‑NB não é o alvo de nenhum discriminador e não tem a transição do corte.
3. **Escuta.** O bloco A compara o S30 com o controle e mistura os dois efeitos. Como a recomendação de uso é o S30hp, valeria trocar o bloco A por S30hp contra o controle, que isola o ganho na banda da fala. Ainda não há respostas, então a troca não perde dados. Depende do usuário.
4. **Ao generalizar, contar locutores, não arquivos.** Efeitos que mudam de sinal entre os 4 locutores (como o +0,017) não devem ser reportados só com o IC por arquivo.

### 14.9 Fila Y: verificações sem treino e um conjunto com 80 locutores (2026‑10‑16 a 19) ✔

Pedido do usuário: executar os itens 1 a 3 propostos depois de 14.8.2 e baixar o VCTK para o item 7. Não houve treino.
- **Lançadores:** `logs/abl/run_Y.sh` (Y1–Y3) e `logs/abl/run_YV.sh` (vctk_unseen). Log em `logs/abl/queue_Y.log`.
- **Análise:** `results/abl/Y_analysis/` (README, `y_report.txt`, `y_vctk.txt`).
- **Diretórios novos:** `<run>/val_unseen_full` e `<run>/test_full` (todas as métricas mais o PESQ sem < 100 Hz) e `<run>/vctk_unseen`. As misturas são `WS_*` (W5) e `WB_*` (Bext).
- **Execução:** Y1–Y3 em 2026‑10‑16, 19:39–20:11, sem falhas. A avaliação no vctk_unseen parou com o fim da sessão (16/10, 20:13) e foi retomada em 19/10, 14:54–15:00, sem falhas.

**Y1. Peso da mistura com o W5 para o S30hp** (val_unseen, média das 3 sementes):

| w | 0,15 | 0,25 | 0,35 | 0,5 |
|---|---|---|---|---|
| S30hp + W5: PESQ | 3,314 | **3,318** | 3,316 | 3,301 |
| S30hp + W5: PESQ‑NB | 3,835 | 3,845 | 3,849 | 3,846 |
| S30hp + W5: SI‑SDR | 18,76 | 18,87 | 18,95 | 19,02 |
| S30 + W5: PESQ | 3,367 | **3,370** | 3,365 | 3,345 |

- **O w = 0,25 continua o escolhido para os dois.** A curva é plana entre 0,25 e 0,35: o 0,35 troca 0,002 de PESQ‑WB por +0,004 de PESQ‑NB e +0,08 dB de SI‑SDR.
- A receita de (A) não muda.

**Y2. PESQ‑NB na val_unseen** (diferenças pareadas por arquivo, média das sementes):

| Comparação | PESQ‑NB | p226 | p287 | Sem < 100 Hz: p226 / p287 |
|---|---|---|---|---|
| S30 − controle | +0,054 [+0,050; +0,058] | +0,052 | +0,055 | +0,062 / +0,062 |
| S30hp − controle | +0,053 [+0,049; +0,058] | +0,052 | +0,055 | +0,111 / +0,047 |
| S30hp − S30 | −0,001 [−0,002; +0,001] | −0,000 | −0,001 | +0,049 / −0,015 |

- **Confirma 14.8.2 nos outros 2 locutores:** o ganho na banda da fala é o mesmo nos dois modelos e quase não varia entre locutores.
- O PESQ sem < 100 Hz, o alvo do S30hp, é o que varia.

**Y3. Mistura com o Bext: teto offline de uma mistura em streaming.**
- **O Bext** (`N_R_ext50_final`) é o ramo espectral com backbone unidirecional e convoluções causais. A STFT (1024/256) ainda é centrada.
- **Peso escolhido na val_unseen:** w = 0,15 para o S30 e para o S30hp.
- **Ganho da mistura sobre o modelo sozinho**, no teste e com as soups:

| Mistura | PESQ‑WB | PESQ‑NB | SI‑SDR | STOI |
|---|---|---|---|---|
| S30 + W5 (w 0,25) | +0,040 | +0,036 | +0,38 dB | +0,002 |
| S30 + Bext (w 0,15) | +0,011 | +0,014 | +0,12 dB | +0,001 |
| S30hp + W5 (w 0,25) | +0,046 | +0,037 | +0,32 dB | +0,002 |
| S30hp + Bext (w 0,15) | +0,018 | +0,015 | +0,09 dB | +0,001 |

- **O ramo causal dá cerca de um terço do ganho do bidirecional.** E isso é o teto offline: em streaming, a normalização causal e a latência da STFT tirariam mais.
- **Uma mistura em streaming renderia no máximo +0,01 a +0,02 de PESQ**, com +20M parâmetros e código novo. O passo "ramo espectral causal" cai para prioridade baixa.

**YV. vctk_unseen: 80 locutores não vistos.**
- **Construção** (detalhes no README de `Y_analysis`):
  - fonte: VCTK 0.92 (CC BY 4.0, MD5 conferido), microfone 1;
  - locutores: os 80 fora do VB‑DEMAND (48 mulheres e 32 homens; sotaques sobretudo americano, escocês, irlandês e canadense), 10 frases cada;
  - processamento: 16 kHz e pico 0,493, como no VB‑DEMAND (`scripts/prepare_vctk_clean.py`, com `python -I`);
  - ruído: os 5 ambientes DEMAND da val_unseen, nos SNRs do teste (P.56), com 40 arquivos por condição (`scripts/make_val_unseen.py --seed 2`).
- **Calibração em p232 e p257:**
  - o microfone 1 reproduz o VB‑DEMAND (correlação 0,994; o microfone 2 dá 0,946);
  - o espectro abaixo de 200 Hz coincide dentro de 0,4 dB;
  - os arquivos do 0.92 têm em média 0,26 s a mais de silêncio inicial.
- **Dificuldade:** o PESQ do ruidoso é 1,971 (no teste, 1,97).
- **STOI:** o valor absoluto é baixo (0,83). Alguns locutores têm um ronco forte abaixo de 50 Hz na gravação limpa, e ele atrapalha a remoção de quadros silenciosos do STOI. As diferenças pareadas não são afetadas.

| Sistema (soup) | vctk_unseen: PESQ | Sem < 100 Hz | PESQ‑NB | SI‑SDR | Teste: PESQ |
|---|---|---|---|---|---|
| Controle K0ext30p | 3,153 | 3,373 | 3,773 | 17,48 | 3,165 |
| S30 | 3,325 | 3,447 | 3,828 | 17,30 | 3,335 |
| S30hp | 3,256 | 3,453 | 3,827 | 17,47 | 3,280 |
| S30 + W5 (w 0,25) | **3,365** | 3,483 | **3,862** | 17,63 | 3,375 |
| S30hp + W5 (w 0,25) | 3,300 | **3,489** | 3,861 | **17,74** | 3,326 |

Com as 3 sementes (média ± desvio padrão amostral): K0ext30p 3,148 ± 0,005; S30 3,321 ± 0,013; S30hp 3,252 ± 0,005.

**Diferenças pareadas.** IC 95% por bootstrap de locutores e a fração dos 80 locutores que melhoram. No WSS, menor é melhor.

| Comparação | PESQ‑WB | Sem < 100 Hz | PESQ‑NB | SI‑SDR | segSNR | WSS |
|---|---|---|---|---|---|---|
| S30 − controle | +0,173 [+0,159; +0,187], 100% | +0,074, 100% | +0,055 [+0,051; +0,059], 100% | −0,17 dB, 4% | −0,29, 0% | +0,08 [+0,04; +0,12], 40% |
| S30hp − controle | +0,104 [+0,094; +0,115], 100% | +0,082, 98% | +0,055 [+0,050; +0,059], 100% | −0,00 dB, 48% | −0,04, 20% | +0,08 [+0,03; +0,12], 36% |
| S30hp − S30 | −0,069 [−0,078; −0,060], 0% | +0,008 [+0,002; +0,015], 51% | −0,001 [−0,003; +0,001], 49% | **+0,17 dB**, 98% | **+0,25**, 100% | −0,01 [−0,06; +0,06], 56% |
| S30 + W5 − S30 (soups) | +0,040 [+0,030; +0,051], 89% | +0,036, 90% | +0,034, 99% | +0,33 dB, 94% | +0,31, 98% | −0,32, 85% |
| S30hp + W5 − S30hp (soups) | +0,044 [+0,034; +0,055], 90% | +0,036, 90% | +0,034, 100% | +0,28 dB, 94% | +0,24, 91% | −0,29, 88% |

O STOI das misturas sobe +0,007, em 99% dos locutores; nas outras comparações fica igual. Os efeitos com sinal claro (PESQ‑WB e PESQ‑NB sobre o controle; PESQ‑WB, SI‑SDR e segSNR do S30hp − S30) têm o mesmo sinal nos 5 ruídos, nos 4 SNRs e nos dois gêneros.

**14.9.1 Conclusões.**
1. **Os resultados do teste valem para 80 locutores.**
   - Os números absolutos e as diferenças ficam a ≤ 0,03 dos do teste, com a mesma ordem de sistemas.
   - Com 80 locutores, o IC por locutor fica quase igual ao IC por arquivo. O problema de 14.8.2 era ter só 2 locutores por conjunto.
2. **O ganho do discriminador na banda da fala é geral.**
   - PESQ‑NB +0,055 sobre o controle, igual no S30 e no S30hp, positivo em todos os 80 locutores (mínimo +0,018).
   - Ele cai com o SNR: +0,070 em 2,5 dB e +0,039 em 17,5 dB.
3. **O S30hp não ganha PESQ do S30 em banda nenhuma.**
   - PESQ‑NB −0,001 (49% dos locutores) e sem < 100 Hz +0,008 (51%).
   - Suas vantagens são SI‑SDR (+0,17 dB, 98% dos locutores) e segSNR (+0,25, 100%), e ele empata com o controle no SI‑SDR.
   - O WSS do teste não se repete: −0,01 contra o S30, e os dois ficam +0,08 acima do controle.
4. **A mistura com o W5 é a melhora mais robusta:** todas as métricas melhoram, em 85–100% dos locutores (PESQ‑NB em 99–100%, STOI em 99%).
5. **A parte abaixo de 100 Hz do ganho do S30 é a principal fonte de variação entre locutores.**
   - Ela vai de −0,005 a +0,298, com média +0,099.
   - O desvio padrão do ganho entre locutores é 0,066 no PESQ‑WB, 0,025 sem < 100 Hz e 0,019 no PESQ‑NB.
   - Ela é maior nos locutores com menos energia em 50–100 Hz na fala limpa: Spearman −0,28 (p = 0,01), e −0,45 no resíduo do S30hp. O indício de 14.8.2 se confirma, mas a correlação é moderada.
   - Nas mulheres é +0,110; nos homens, +0,082. O ganho no PESQ‑NB é igual nos dois (+0,055).
   - **No S30hp − S30, o ganho sem < 100 Hz aparece onde o S30 ganhava mais abaixo de 100 Hz** (Spearman −0,73). Como o PESQ‑NB não muda (desvio entre locutores de 0,008), esse ganho vem da transição do corte ou de fora de 300–3400 Hz, não do meio da banda da fala.
6. **A recomendação de 14.8.1 não muda.**
   - Para o número de benchmark: a soup do S30 + W5 (teste 3,375; vctk_unseen 3,365) e a soup do S30 (3,335; 3,325).
   - Para uso: as versões com o S30hp. Elas têm o mesmo ganho na banda da fala, sem o efeito de 50–100 Hz e sem a perda de SI‑SDR. O custo é −0,07 de PESQ‑WB (e de CSIG, CBAK e COVL, que seguem o PESQ).
   - A escuta continua decidindo entre os dois.
7. **Para relatar o ganho do discriminador**, usar o PESQ‑NB ao lado do PESQ‑WB: +0,055 em 80 locutores, contra +0,17 de PESQ‑WB que varia de +0,05 a +0,37 entre locutores.

**Próximos passos:**
1. **Escuta** (depende do usuário): ainda sem respostas em 2026‑10‑19. A sugestão de trocar o bloco A para S30hp contra o controle continua de pé.
2. **Conjunto independente para o relatório final.** O vctk_unseen usa os ruídos da val_unseen, que serviu para escolher os pesos. Repetir com os 5 ambientes DEMAND do teste (download do Zenodo) daria um conjunto sem nenhuma escolha feita nele. Opcional.
3. **DNSMOS no vctk_unseen**, para ver o efeito no BAK com 80 locutores: ~20 min, opcional.
4. **Prioridade baixa:** o corte mais íngreme (só se a escuta mostrar que a faixa importa), o ramo espectral causal (Y3) e a porta aprendida.
