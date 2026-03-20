"""
model.py
--------
Builds the BERT token classification model.
Kept minimal — all training logic lives in trainer.py.
"""

from transformers import AutoModelForTokenClassification
from config import LABEL2ID, ID2LABEL


def build_model(model_name: str):
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )
    return model
