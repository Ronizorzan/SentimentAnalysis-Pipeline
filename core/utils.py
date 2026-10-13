import logging
import functools
import time
import re
import string

# === Compilação prévia das expressões (para maximizar o desempenho) ===
nlp = None


# === Compilação prévia das expressões (para maximizar o desempenho) ===
url_pattern = re.compile(r"https?://\S+|www\.\S+")
user_pattern = re.compile(r"@\w+")
non_alpha_pattern = re.compile(r"[^a-zA-Z.!?]")
multi_space_pattern = re.compile(r"\s+")


# === Configuração do logger ===
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Decorator para medir o tempo de execução de uma função ===
def elapsed_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        initial_time = time.perf_counter()

        result = func(*args, **kwargs)

        end_time = time.perf_counter()

        elapsed_time = end_time - initial_time

        logger.info(f"Função {func.__name__} executada em {elapsed_time:.2f} segundos")

        return result

    return wrapper




# === Limpeza e remoção de StopWords ===
def clean_tweets(tweet):    
    global nlp

    if nlp is None:
        import spacy

        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    from bs4 import BeautifulSoup

    
    # Remoção de HTML
    tweet = BeautifulSoup(tweet, "lxml").get_text()
        
    
    tweet = url_pattern.sub(" ", tweet)
    tweet = user_pattern.sub(" ", tweet)
    tweet = non_alpha_pattern.sub(" ", tweet)
    tweet = multi_space_pattern.sub(" ", tweet)
    
    # Normalização
    tweet = tweet.lower().strip()
    
    # Tokenização com spaCy
    doc = nlp(tweet)
    words = [
        token.lemma_ for token in doc 
        if token.text not in string.punctuation
    ]
    
    # Reconstrução do texto limpo
    cleaned_tweet = " ".join(words)
        
    
    return cleaned_tweet