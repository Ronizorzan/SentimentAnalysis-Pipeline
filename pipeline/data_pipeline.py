import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from core.utils import clean_tweets, elapsed_time, logger


@elapsed_time
def load_and_process_data():
    # === Carregamento dos dados brutos
    train_file_path = Path("data") / "train.csv"
    train_data = pd.read_csv(train_file_path,
                            engine="python",
                            header=None,
                            encoding="latin-1")


    cols = ['sentiment', 'id', 'date', 'query', 'user', 'text']

    # Nomeamos as colunas adequadamente
    train_data.columns = cols

    # Tratamos os dados em uma nova variável
    new_data = train_data.copy()

    # Removemos as duplicatas para obtermos métricas mais precisas e evitar viés no modelo
    new_data.drop_duplicates(subset=["text"], inplace=True)
    logger.info(f"Total de duplicatas removidas: {train_data.shape[0] - new_data.shape[0]}")

    # Para nossas análises, só precisamos da coluna 'text' e 'sentiment'
    data = new_data.copy()
    data = new_data.loc[:, ["text", "sentiment"]]


    # Separação das variáveis dependentes e independentes
    text = data.loc[:, "text"]
    sentiment = data.loc[:, "sentiment"]
    logger.info(f"Dados tratados com sucesso. Total de registros mantidos após processamento: {text.shape[0]}")
    
    return text, sentiment


@elapsed_time
def clean_tweets_with_logs(text):
    new_tweet = [clean_tweets(tweet) for tweet in text]
    logger.info("Tweets processados com sucesso")
    return new_tweet



@elapsed_time
def save_data(tweets, labels):
    path = "processed_data"
    os.makedirs(path, exist_ok=True)

    with open(Path(path) / "clean_tweets_full.parquet", "wb") as f:
        clean_tweets_df = pd.DataFrame({"text": tweets, "sentiments": labels})
        clean_tweets_df.to_parquet(f, engine="pyarrow", index=False)
    logger.info(f"Arquivo salvo em: {Path(path) / 'clean_tweets_full.parquet'}")



def main():
    text, sentiment = load_and_process_data()
    tweets = clean_tweets_with_logs(text)
    labels = sentiment.copy()
    labels[labels == 4] = 1
    save_data(tweets, labels)


if __name__ == "__main__":
    main()