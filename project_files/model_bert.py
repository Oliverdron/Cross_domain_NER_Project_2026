
from transformers import BertForTokenClassification

class BertNER:
    def __init__(self, num_labels):
        self.model = BertForTokenClassification.from_pretrained(
            "bert-base-cased",
            num_labels=num_labels
        )

    def get_model(self):
        return self.model