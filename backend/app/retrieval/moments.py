from __future__ import annotations

import math
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Tuple


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[\w]+", text.casefold())
    return {word for word in words if len(word) > 2}


def _text(moment: Dict[str, Any]) -> str:
    return " ".join([
        str(moment.get("title") or ""),
        " ".join(moment.get("signals") or []),
        str(moment.get("transcript") or ""),
    ])


@lru_cache(maxsize=1)
def _embedding_model():
    model_name = os.getenv("VISIONINSIGHT_EMBEDDING_MODEL", "")
    if not model_name:
        return None
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def retrieve(question: str, moments: List[Dict[str, Any]], limit: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
    if not moments:
        return "none", []
    query_tokens = _tokens(question)
    lexical = []
    for moment in moments:
        document_tokens = _tokens(_text(moment))
        overlap = len(query_tokens & document_tokens)
        lexical.append(overlap / math.sqrt(max(1, len(query_tokens) * len(document_tokens))))

    scores = lexical
    method = "lexical"
    try:
        model = _embedding_model()
        if model is not None:
            vectors = model.encode([question] + [_text(moment) for moment in moments], normalize_embeddings=True)
            scores = [float(vectors[0] @ vector) for vector in vectors[1:]]
            method = "semantic"
    except Exception:
        # Local/offline installations remain usable without the embedding model.
        pass

    ranked = sorted(zip(scores, moments), key=lambda pair: (-pair[0], pair[1]["start_sec"]))
    threshold = 0.26 if method == "semantic" else 0.05
    return method, [
        {"score": round(score, 3), "moment": moment}
        for score, moment in ranked[:limit] if score >= threshold
    ]
