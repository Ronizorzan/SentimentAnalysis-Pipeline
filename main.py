import html
import re
from pathlib import Path
from typing import Any, Literal

import joblib
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

MODEL_PATH = Path(__file__).resolve().parent / "models" / "bayes_search_best_estimator.joblib"


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Texto de entrada para análise de sentimento.")

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("O campo 'text' não pode estar vazio.")
        return cleaned


class PredictionResponse(BaseModel):
    sentiment: int
    label: str
    confidence: float
    text: str


class BatchPredictionRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=50)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("A lista 'texts' não pode estar vazia.")
        for item in value:
            if not item.strip():
                raise ValueError("Cada texto da lista deve conter conteúdo válido.")
        return value


app = FastAPI(
    title="Sentiment Analysis API",
    version="1.0.0",
    description="API para classificação de sentimento em textos usando o modelo treinado do projeto NLP.",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

model: Any = None


def preprocess_text(text: str) -> str:
    """Aplica limpeza básica para manter a entrada compatível com o pipeline do modelo."""
    decoded = html.unescape(text)
    decoded = re.sub(r"https?://\S+|www\.\S+", " ", decoded, flags=re.IGNORECASE)
    decoded = re.sub(r"@\w+|#", " ", decoded)
    decoded = re.sub(r"[^a-zA-Z0-9\s.!?]", " ", decoded)
    decoded = re.sub(r"\s+", " ", decoded).strip().lower()
    return decoded


def load_model() -> Any:
    global model

    if model is not None:
        return model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Arquivo do modelo não encontrado em: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    return model


@app.on_event("startup")
def startup_event() -> None:
    """Carrega o modelo treinado ao iniciar a aplicação."""
    load_model()


@app.get("/health", status_code=status.HTTP_200_OK)
def healthcheck() -> dict[str, Any]:
    """Health check simples para verificação da disponibilidade da API."""
    return {
        "status": "ok",
        "service": "sentiment-analysis-api",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.get("/model/info", status_code=status.HTTP_200_OK)
def model_info() -> dict[str, Any]:
    """Retorna metadados do modelo carregado."""
    estimator = load_model()
    metadata = {
        "model_type": type(estimator).__name__,
        "classes": getattr(estimator, "classes_", []).tolist() if hasattr(estimator, "classes_") else [0, 1],
    }
    return metadata


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_text(payload: PredictionRequest) -> PredictionResponse:
    """Classifica o sentimento de um único texto."""
    try:
        estimator = load_model()
        cleaned_text = preprocess_text(payload.text)
        probabilities = estimator.predict_proba([cleaned_text])[0]
        class_index = int(probabilities.argmax())
        predicted_class = estimator.classes_[class_index]
        label = "positive" if int(predicted_class) == 1 else "negative"

        return PredictionResponse(
            sentiment=int(predicted_class),
            label=label,
            confidence=float(probabilities[class_index]),
            text=payload.text,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - guard para falhas inesperadas
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar a predição: {exc}",
        ) from exc


@app.post("/predict/batch", status_code=status.HTTP_200_OK)
def predict_batch(payload: BatchPredictionRequest) -> dict[str, Any]:
    """Classifica diversos textos em lote."""
    try:
        estimator = load_model()
        texts = [preprocess_text(item) for item in payload.texts]
        probabilities = estimator.predict_proba(texts)
        results: list[dict[str, Any]] = []

        for index, probs in enumerate(probabilities):
            class_index = int(probs.argmax())
            predicted_class = int(estimator.classes_[class_index])
            label = "positive" if predicted_class == 1 else "negative"
            results.append(
                {
                    "index": index,
                    "sentiment": predicted_class,
                    "label": label,
                    "confidence": float(probs[class_index]),
                    "text": payload.texts[index],
                }
            )

        return {
            "count": len(results),
            "results": results,
        }
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - guard para falhas inesperadas
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar o lote: {exc}",
        ) from exc


def main() -> None:
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
