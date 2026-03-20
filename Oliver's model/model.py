from transformers import AutoModelForTokenClassification


def build_model(model_name: str, label2id, id2label):
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )
    return model