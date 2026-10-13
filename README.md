# Análise de sentimento em textos: da preparação à API

Como transformar uma coleção de mensagens curtas em um sinal que um produto pode consumir? Este projeto percorre o ciclo de vida de uma solução de NLP: prepara textos rotulados, compara configurações de modelo com validação cruzada, registra experimentos e disponibiliza inferência por HTTP.

O resultado é uma API que classifica textos em **negativo** ou **positivo**, acompanhada por uma pipeline reproduzível e artefatos que permitem inspecionar como o modelo foi treinado e avaliado.

## O que o projeto demonstra

- **Engenharia de dados para texto:** remoção de duplicatas, limpeza e normalização de tweets e geração de dados processados.
- **Treinamento com avaliação cuidadosa:** divisão estratificada entre treino e holdout, seguida de validação cruzada estratificada para comparar hiperparâmetros.
- **Busca automatizada:** `BayesSearchCV` ajusta um pipeline TF-IDF + Regressão Logística, incluindo o peso das classes e a frequência de termos.
- **Rastreabilidade:** DVC descreve o estágio de treinamento e seus artefatos; MLflow registra parâmetros, métricas e o modelo.
- **Entrega consumível:** FastAPI expõe predições individuais e em lote, confiança estimada, documentação OpenAPI e health check.

## Fluxo de ponta a ponta

```text
data/train.csv
    └── pipeline/data_pipeline.py
          ├── remove duplicatas e limpa tweets
          └── processed_data/clean_tweets_full.parquet
                └── pipeline/model_validation_cv.py
                      ├── separação estratificada: treino / holdout
                      ├── TF-IDF + Regressão Logística
                      ├── busca Bayesiana + validação cruzada
                      ├── métricas e execução MLflow
                      └── models/bayes_search_best_estimator.joblib
                            └── main.py (FastAPI)
```

O conjunto de dados usa as colunas `text` e `sentiments`. Na preparação, o rótulo original `4` é convertido em `1`; o rótulo `0` permanece como `0`. Portanto, a API retorna `0` para negativo e `1` para positivo.

## Como executar

### 1. Preparar o ambiente

O projeto requer Python 3.12 ou superior e usa `uv` para sincronizar o ambiente:

```powershell
uv sync
```

A preparação dos dados brutos espera `data/train.csv`. Para executá-la:

```powershell
uv run python -m pipeline.data_pipeline
```

Essa etapa usa spaCy com o modelo `en_core_web_sm`, além de BeautifulSoup e lxml. A API também usa Uvicorn como servidor ASGI. Esses componentes não estão declarados atualmente em `pyproject.toml`; instale-os e baixe o modelo spaCy antes de executar a preparação e o serviço:

```powershell
uv add spacy beautifulsoup4 lxml uvicorn
uv run python -m spacy download en_core_web_sm
```

Se os dados processados já estiverem disponíveis pelo DVC, configure o remote do projeto e recupere-os:

```powershell
uv run dvc pull
```

### 2. Treinar e avaliar

Execute o estágio gerenciado pelo DVC:

```powershell
uv run dvc repro bayesian_search
```

Ou rode o treinamento diretamente:

```powershell
uv run python -m pipeline.model_validation_cv
```

Os parâmetros da busca, seed, métrica de seleção, configuração do MLflow e caminhos dos artefatos estão em `params.yaml`. A configuração atual usa uma divisão de holdout de 20%, três folds e oito iterações de busca.

O holdout fica fora da busca de hiperparâmetros: é usado para uma avaliação final depois que a melhor configuração é selecionada. Isso reduz o risco de ajustar o modelo com base nos resultados do conjunto de avaliação.

### 3. Servir predições

Após gerar ou recuperar o artefato do modelo:

```powershell
uv run python main.py
```

A API fica disponível em `http://localhost:8000`. Alternativamente, inicie-a com Uvicorn:

```powershell
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

## API

O Swagger interativo está em [`/docs`](http://localhost:8000/docs), o ReDoc em [`/redoc`](http://localhost:8000/redoc) e o schema OpenAPI em [`/openapi.json`](http://localhost:8000/openapi.json).

| Método | Rota | Finalidade |
| --- | --- | --- |
| `GET` | `/health` | Verifica a disponibilidade e informa se o modelo está carregado. |
| `GET` | `/model/info` | Informa o tipo do estimador e as classes disponíveis. |
| `POST` | `/predict` | Classifica um texto e retorna classe, rótulo, confiança e texto original. |
| `POST` | `/predict/batch` | Classifica de 1 a 50 textos em uma chamada. |

### Predição individual

```powershell
curl.exe -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"I love this product\"}"
```

Exemplo de resposta:

```json
{
  "sentiment": 1,
  "label": "positive",
  "confidence": 0.78,
  "text": "I love this product"
}
```

### Predição em lote

```powershell
curl.exe -X POST http://localhost:8000/predict/batch `
  -H "Content-Type: application/json" `
  -d "{\"texts\":[\"I love this product\",\"I hate this service\"]}"
```

Os campos de entrada são validados: o texto individual aceita de 1 a 5.000 caracteres; o lote aceita de 1 a 50 textos não vazios. A confiança é a probabilidade estimada pelo classificador para a classe prevista, não uma garantia de acerto.

## Pipeline, experimentos e artefatos

O estágio `bayesian_search`, definido em `dvc.yaml`, declara dependências, parâmetros, saídas do modelo e métricas. Com um remote DVC configurado, `dvc pull` recupera os dados e `dvc repro` reproduz o estágio quando necessário. Para compartilhar os artefatos, configure o remote no ambiente de desenvolvimento e use `dvc push`.

O MLflow registra execuções no experimento `sentiment-bayes-search`, usando o tracking URI configurado em `params.yaml`.

| Artefato | Conteúdo |
| --- | --- |
| `models/bayes_search_best_estimator.joblib` | Pipeline selecionado e refitado sobre os dados de treino; consumido pela API. |
| `metrics/bayes_search_summary.json` | Seed, configuração, melhores parâmetros e score de validação cruzada. |
| `metrics/bayes_search_cv_results.json` | Resultados dos candidatos avaliados na busca. |
| `metrics/holdout_metrics.json` | Acurácia, relatório de classificação e matriz de confusão no holdout. |
| `splits/train_split.parquet` e `splits/test_split.parquet` | Partições usadas no treinamento e avaliação final. |

O modelo salvo é refitado sobre o conjunto de treino, não sobre todos os dados rotulados. O holdout permanece como avaliação final e não deve ser usado para escolher hiperparâmetros.

## Decisões e próximos passos

A combinação TF-IDF + Regressão Logística cria uma referência leve, interpretável e adequada para classificação de texto. A busca Bayesiana explora hiperparâmetros dentro de uma validação cruzada estratificada, e as métricas do holdout ajudam a avaliar o resultado fora dos folds usados na seleção.

Como próximos passos para evolução do serviço, fazem sentido automatizar testes de contrato da API e da pipeline, alinhar a normalização usada na inferência com toda a transformação aplicada aos dados de treino, adicionar autenticação e métricas operacionais e definir uma estratégia de promoção/rollback de versões de modelo. Essas capacidades não fazem parte da implementação atual.
