# Evolução do pipeline de classificação de laudos de aferição (Task 05)

**Branch:** `task05_melhorias_pipeline_modelos`
**Data:** julho/2026
**Público-alvo deste documento:** qualquer pessoa da equipe, mesmo sem background técnico em Machine Learning.

## Resumo em uma frase

Ajustamos o sistema que tenta "adivinhar" o resultado de um laudo de aferição de medidor (`CODRSTAFER`) para focar apenas nos resultados mais comuns, usar os dois algoritmos que historicamente funcionaram melhor (CatBoost e XGBoost) com ajustes para lidar melhor com classes raras, e implementamos duas técnicas (SMOTE e ADASYN) para "balancear" os dados — mas **não foi possível rodar o pipeline completo com os dados reais da Energisa neste ambiente de trabalho**, porque o arquivo de dados (`resultado_laudo_afericao.xlsx`) não estava disponível. Tudo foi implementado, testado com dados fictícios (sintéticos) gerados especificamente para validar o código, e documentado de forma transparente abaixo: o que é resultado real e o que é proposta ainda não comprovada com os dados verdadeiros.

---

## ⚠️ Aviso importante sobre disponibilidade de dados

Antes de tudo, é essencial deixar claro o que pôde e o que não pôde ser feito nesta tarefa:

- O arquivo `src/machine_learning/resultado_laudo_afericao.xlsx` (dataset bruto, com os laudos reais) **não existe neste ambiente**. Ele nunca foi versionado no Git (está fora do controle de versão) e não foi encontrado em disco.
- O arquivo `model/prepared_training_dataset.csv` (dataset já preparado, que permitiria treinar sem o Excel) **também não existe** neste ambiente.
- Por isso, **nenhuma execução com dados reais foi feita nesta tarefa**. Os únicos números "reais" que aparecem neste documento vêm de um resultado anterior, já existente no repositório antes desta tarefa começar (arquivos `model/preparation_metadata.json` e `model/classification/classification_details.json`, gerados em outra máquina/ambiente em 21/07/2026).
- Para validar que todo o código novo funciona corretamente (sem erros), criamos um conjunto de **testes automatizados com dados fictícios** (pasta `tests/`), que geram um dataset sintético com o mesmo "formato" do problema real (várias classes, sendo poucas dominantes e muitas raras) e rodam o pipeline inteiro de ponta a ponta. Esses testes **passam e comprovam que o código funciona**, mas os números/métricas que eles produzem **não têm significado de negócio** — são apenas para garantir que a lógica está correta.
- **Assim que o dataset real estiver disponível**, o pipeline pode ser executado diretamente (`python src/main.py`) para obter os números reais equivalentes aos que estão marcados como "pendente" abaixo.

Ao longo do documento, cada resultado é marcado com um destes três selos:

- 🟢 **REAL** — número obtido de uma execução com os dados verdadeiros da Energisa (executada antes desta tarefa, arquivo já existia no repositório).
- 🟡 **SINTÉTICO (smoke test)** — número obtido rodando o código novo com dados fictícios, apenas para provar que o código funciona. Não usar para decisões de negócio.
- 🔴 **PENDENTE** — funcionalidade implementada e revisada, mas ainda não executada com dados reais porque o dataset não estava disponível.

---

## O que foi feito

### Fase 1 — Foco nos modelos que funcionam melhor e nas classes que importam

1. **Branch dedicado.** Todo o trabalho foi feito em um branch novo, `task05_melhorias_pipeline_modelos`, criado a partir do branch principal.

2. **Menos modelos, mais foco.** O sistema testava 4 algoritmos diferentes (CatBoost, XGBoost, k-NN e SVM) toda vez que rodava. Como CatBoost e XGBoost historicamente entregam os melhores resultados (ver tabela de métricas abaixo) e k-NN/SVM são mais lentos, o pipeline padrão agora roda **só CatBoost e XGBoost**. k-NN e SVM continuam existindo no código (podem ser reativados por configuração), só não rodam mais automaticamente.

3. **Relatório de distribuição das classes + "expurgo" das classes raras.** O sistema agora gera, automaticamente, uma tabela e um gráfico mostrando quantos registros existem de cada resultado possível (classe) do laudo, tanto **antes** quanto **depois** de descartar as classes raras — isso ajuda a enxergar visualmente o desbalanceamento. Foi criada uma regra nova e configurável: qualquer classe que represente **menos de 15% do total de linhas** é removida do treinamento. Essa é uma mudança de critério importante:
   - **Antes:** só se removia uma classe se ela tivesse menos de 2 exemplos no total (ou seja, praticamente nada era removido).
   - **Agora:** removemos qualquer classe com menos de 15% de representatividade — um critério bem mais rigoroso, focado em manter apenas os resultados mais comuns.

   ⚠️ **Efeito esperado e intencional:** como existem dezenas de classes possíveis (o dataset real tinha 50), é matematicamente inevitável que a maioria delas tenha menos de 15% de participação cada. Ou seja, esse novo critério deve sobrar **poucas classes dominantes** (possivelmente entre 2 e 5, a depender de como os dados reais estão distribuídos) — isso é exatamente o que foi pedido: fazer o modelo focar nos resultados mais frequentes, em vez de tentar (e falhar) em prever dezenas de resultados raros.
   - O limiar de 15% é uma constante nomeada e configurável no código (`MIN_CLASS_PERCENTAGE_THRESHOLD`, em `src/machine_learning/feature_engineering.py`), podendo ser ajustado facilmente se a equipe achar necessário depois de ver os números reais.

4. **Ajuste dos "botões" internos dos modelos (hiperparâmetros).** Foram propostos novos valores de configuração para CatBoost e XGBoost, pensados especificamente para melhorar o desempenho nas classes minoritárias (as que sobram após o expurgo, mas que ainda podem estar em desvantagem numérica frente à classe mais comum). Também foi implementada uma busca automática de hiperparâmetros (`RandomizedSearchCV`), que testa várias combinações e escolhe a melhor. 🔴 Como não havia dados reais disponíveis, essa busca **não pôde ser executada de fato** sobre o problema real — os novos valores foram implementados como novo "padrão de fábrica" do sistema, com a justificativa documentada em código, mas ainda **não comprovados** com números reais.

### Fase 2 — Balanceando os dados artificialmente (SMOTE e ADASYN)

1. Foram implementadas duas técnicas conhecidas de balanceamento de classes:
   - **SMOTE**: cria exemplos sintéticos (fictícios, mas plausíveis) das classes minoritárias, "inventando" pontos intermediários entre exemplos reais já existentes.
   - **ADASYN**: parecido com o SMOTE, mas dá mais atenção às classes minoritárias que são mais difíceis de separar das demais.
2. As duas técnicas são aplicadas **apenas nos dados de treino** — nunca nos dados de teste/validação, para não "inflar" artificialmente os resultados. Isso é configurável (dá para escolher SMOTE, ADASYN ou nenhum dos dois, para comparação).
3. Foi adicionada uma validação mais robusta chamada **validação cruzada estratificada (`StratifiedKFold`, k=5)**: em vez de testar o modelo em um único "corte" dos dados, ele é testado 5 vezes, em pedaços diferentes, e a média/desvio dos resultados é reportada. Isso deixa a avaliação mais confiável e menos dependente de "sorte" na divisão dos dados.
4. 🔴 Assim como a busca de hiperparâmetros, **essas técnicas não puderam ser testadas com os dados reais** por falta do arquivo de dados. O código foi validado com dados fictícios (ver seção de testes).

### Fase 3 — Este documento

Este arquivo markdown reúne o resumo de tudo o que foi feito, os números disponíveis, e as recomendações — você está lendo agora.

---

## O que foi de fato executado (e como)

Como o dataset real não estava disponível, criamos testes automatizados (`tests/`) com um dataset **sintético** (fictício, gerado por código, com ~600 linhas e 13 classes simuladas, imitando o desbalanceamento do problema real) para garantir que:

- o expurgo de classes por percentual funciona corretamente;
- o pipeline de preparação de dados gera os artefatos de "antes/depois" esperados;
- os dois modelos padrão (CatBoost/XGBoost) treinam e avaliam sem erros;
- os três cenários de resampling (sem resampling, SMOTE, ADASYN) rodam corretamente;
- a validação cruzada estratificada funciona e reporta média/desvio por métrica;
- a busca de hiperparâmetros roda e escolhe uma configuração.

Todos os 12 testes automatizados passam (`python -m pytest tests -v`). Durante essa validação, identificamos e corrigimos **3 bugs pré-existentes** no pipeline de preparação de dados (não relacionados diretamente ao pedido desta tarefa, mas que impediam a validação e afetariam qualquer uso futuro com configurações diferentes das já testadas anteriormente):

1. Colunas "esparsas" (resultado de codificação de categorias) eram salvas com valor ausente (`NaN`) em vez de zero, o que quebraria técnicas como SMOTE/ADASYN.
2. A escolha entre duas técnicas de redução de dimensionalidade (PCA vs. TruncatedSVD) podia escolher a técnica errada quando havia poucas colunas mas alguma coluna categórica — quebrando o processamento.
3. Uma etapa interna de codificação de categorias não conseguia gerar nomes de coluna corretamente em um cenário específico (redução de dimensionalidade desligada).

Nenhuma dessas correções muda o comportamento do pipeline padrão de produção (que sempre usou a configuração que "escondia" esses bugs), mas eram necessárias para conseguir validar as novidades desta tarefa e evitam que os mesmos problemas apareçam no futuro.

---

## Tabela "antes x depois": distribuição das classes

### Dados reais (histórico, já existente no repositório antes desta tarefa)

| | Antes do expurgo | Depois do expurgo (critério **antigo**: contagem mínima = 2) |
|---|---|---|
| Total de linhas | 35.318 | 35.309 |
| Total de classes | 50 | 41 |
| % de linhas retidas | 100% | 99,97% |

🟢 **REAL**, mas obtido com o critério **antigo** (contagem absoluta), não com o novo critério de 15%.

🔴 **PENDENTE:** não temos a distribuição percentual completa das 50 classes reais (quantas linhas cada uma tinha), então **não é possível calcular hoje quantas classes sobreviveriam ao novo limiar de 15% no dataset real**. Isso só pode ser respondido rodando `python src/main.py` (ou `prepare_training_dataset(...)`) com o arquivo `resultado_laudo_afericao.xlsx` disponível — o resultado ficará automaticamente salvo em `model/exploration/class_distribution_after_purge_CODRSTAFER.csv`.

**O que sabemos, com razoável confiança, sobre o resultado esperado:** dado que havia 50 classes disputando 100% dos dados, e o critério antigo já indicava que pelo menos 9 classes tinham *apenas 1 registro cada* (ou seja, frações irrisórias, muito abaixo de 15%), é praticamente certo que o número de classes remanescentes após o corte de 15% será **pequeno — provavelmente entre 2 e 6 classes**, exatamente como o solicitado (focar nos resultados mais frequentes).

### Dados sintéticos (smoke test, para ilustrar o mecanismo)

| | Antes do expurgo | Depois do expurgo (novo critério: 15%) |
|---|---|---|
| Total de linhas | 600 | 486 |
| Total de classes | 13 | 3 |
| % de linhas retidas | 100% | 81,0% |

🟡 **SINTÉTICO** — apenas para demonstrar que o mecanismo de expurgo por percentual funciona como esperado: das 13 classes fictícias, sobraram as 3 mais frequentes (41,3%, 24,3% e 15,3% de participação cada), e 81% das linhas originais foram mantidas mesmo removendo 10 das 13 classes.

---

## Tabela comparativa de métricas

Todas as métricas usam média **macro** (cada classe pesa igual, independentemente do tamanho) — importante porque é justamente nas classes minoritárias que o modelo historicamente performa mal, e a média macro evidencia isso (diferente da accuracy simples, que esconde o problema).

### 🟢 Baseline REAL (antes desta tarefa — 4 modelos, critério antigo de expurgo por contagem, sem SMOTE/ADASYN, sem validação cruzada)

| Algoritmo | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (macro) | PR-AUC (macro) |
|---|---|---|---|---|---|---|
| CatBoost | 93,1% | 26,3% | 25,6% | 25,3% | 98,7% | 36,2% |
| XGBoost | 92,6% | 25,0% | 25,6% | 24,4% | 96,9% | 36,5% |
| k-NN | 91,9% | 25,1% | 21,8% | 22,3% | 72,8% | 26,7% |
| SVM | 93,0% | 26,4% | 22,8% | 22,7% | 98,4% | 34,9% |

**Leitura simples:** a accuracy (acerto geral) parece ótima (~93%), mas é enganosa — como poucas classes dominam o dataset, "acertar sempre a classe mais comum" já dá uma accuracy alta. As métricas que realmente importam aqui (precision/recall/F1 macro) estão baixas (22%–26%), confirmando que o modelo tem muita dificuldade com as classes raras — exatamente o problema que esta tarefa tenta resolver.

### CatBoost/XGBoost com hiperparâmetros ajustados (Fase 1, dataset já com expurgo de 15%)

🔴 **PENDENTE** — não executado com dados reais (dataset indisponível). Os novos hiperparâmetros propostos estão implementados como padrão em `src/machine_learning/classification/models.py` (`DEFAULT_CATBOOST_PARAMS`, `DEFAULT_XGBOOST_PARAMS`), com o racional documentado em comentários no código, mas sem validação empírica.

### Com SMOTE / ADASYN (Fase 2)

🔴 **PENDENTE** — não executado com dados reais, pelo mesmo motivo.

### Validação cruzada estratificada (StratifiedKFold, k=5)

🔴 **PENDENTE** — não executado com dados reais.

### 🟡 Ilustração com dados SINTÉTICOS (prova de que o código roda de ponta a ponta com todas as funcionalidades novas juntas)

*Números sem significado de negócio — servem apenas para mostrar que o pipeline roda sem erros com hiperparâmetros ajustados, resampling e validação cruzada todos ativos ao mesmo tempo.*

| Algoritmo | Resampling | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | F1 macro (CV, k=3) |
|---|---|---|---|---|---|---|---|---|
| CatBoost | nenhum | 87,8% | 86,0% | 86,2% | 85,9% | 97,3% | 93,9% | 84,4% ± 0,8pp |
| CatBoost | SMOTE | 86,7% | 84,3% | 85,1% | 84,6% | 97,4% | 94,1% | 83,9% ± 0,5pp |
| CatBoost | ADASYN | 86,7% | 83,7% | 85,1% | 84,2% | 97,0% | 93,3% | 83,5% ± 0,6pp |
| XGBoost | nenhum | 87,8% | 86,9% | 86,2% | 86,3% | 97,2% | 93,7% | 84,5% ± 0,3pp |
| XGBoost | SMOTE | 88,8% | 87,8% | 87,4% | 87,3% | 97,0% | 93,6% | 83,5% ± 1,6pp |
| XGBoost | ADASYN | 87,8% | 86,9% | 86,2% | 86,3% | 96,1% | 90,5% | 84,6% ± 0,2pp |

(Obtido rodando `pytest tests/test_full_pipeline_default_pca_smoke.py -v -s`, com dados fictícios de 600 linhas / 13 classes / 3 classes retidas.)

---

## Conclusões e recomendações

1. **A limitação mais importante desta entrega é a ausência do dataset real.** Todo o código das Fases 1 e 2 foi implementado, revisado e testado (com dados fictícios), mas os números que realmente importam para decisão de negócio — quantas classes reais sobrevivem ao corte de 15%, e se isso de fato melhora recall/F1/ROC-AUC das classes minoritárias — **só existirão depois que alguém rodar `python src/main.py` com o arquivo `resultado_laudo_afericao.xlsx` (ou o `model/prepared_training_dataset.csv`) disponível no ambiente**.

2. **Recomendação imediata:** assim que o dataset real estiver disponível, rodar o pipeline com a configuração padrão (expurgo de 15%, CatBoost/XGBoost com os novos hiperparâmetros) e comparar com a tabela "baseline real" acima. Se as métricas macro ainda estiverem abaixo de ~70% (critério sugerido no enunciado desta tarefa), ativar `resampling_strategies=(None, "smote", "adasyn")` e `enable_cross_validation=True` em `ClassificationConfig` para obter a comparação completa.

3. **Sobre o limiar de 15%:** é uma decisão de negócio, não só técnica. Um limiar tão alto vai necessariamente reduzir bastante o número de classes que o modelo tenta prever (pelo desenho matemático do problema — muitas classes, cada uma pequena). Isso é positivo para a qualidade das previsões nas classes que sobrarem, mas significa que **laudos com resultados raros deixarão de ser cobertos pelo modelo automatizado** e provavelmente precisarão de tratamento manual ou de uma estratégia complementar (ex.: um "modelo residual" para as classes descartadas, ou uma categoria "outros/revisão manual"). Recomendamos validar esse trade-off com quem entende o processo de negócio antes de colocar em produção.

4. **Sobre os hiperparâmetros e o resampling:** foram implementados de forma correta e testável, mas como "propostas bem fundamentadas, não comprovadas". Recomendamos fortemente rodar a busca de hiperparâmetros (`enable_hyperparameter_search=True`) e comparar SMOTE vs. ADASYN vs. nenhum, com os dados reais, antes de assumir que os novos padrões são de fato melhores que os anteriores.

5. **Bugs corrigidos:** as três correções de bugs latentes no pipeline de preparação (ver seção acima) são independentes do resultado do modelo, mas importantes para a robustez do sistema — recomendamos incluí-las mesmo que a equipe decida não seguir adiante com as demais mudanças desta tarefa.

---

## Resumo rápido: o que é real vs. o que é proposta

| Item | Status |
|---|---|
| Branch, restrição a CatBoost/XGBoost, README atualizado | 🟢 Código implementado e mesclado no branch |
| Relatório de distribuição de classes (mecanismo) | 🟢 Implementado e testado (sintético) |
| Expurgo por percentual (15%, configurável) | 🟢 Implementado e testado (sintético) |
| Quantas classes reais sobrevivem a 15%? | 🔴 Pendente de dados reais |
| Novos hiperparâmetros CatBoost/XGBoost | 🟡 Implementados como novo padrão; 🔴 não validados com dados reais |
| Busca automática de hiperparâmetros | 🟢 Implementada e testada (sintético); 🔴 não executada com dados reais |
| SMOTE / ADASYN | 🟢 Implementados e testados (sintético); 🔴 não executados com dados reais |
| StratifiedKFold (k=5) | 🟢 Implementado e testado (sintético); 🔴 não executado com dados reais |
| Métricas finais de negócio (recall/F1/ROC-AUC reais pós-mudanças) | 🔴 Pendente de dados reais |
