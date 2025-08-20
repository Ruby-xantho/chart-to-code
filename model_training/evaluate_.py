# evaluate_.py

import evaluate
import numpy as np

# Load metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

def compute_metrics(predictions, references):
    # BLEU expects tokenized lists
    bleu_preds = [pred.split() for pred in predictions]
    bleu_refs = [[ref.split()] for ref in references]

    # BLEU score
    bleu_scores = bleu.compute(predictions=bleu_preds, references=bleu_refs)

    # ROUGE score
    rouge_scores = rouge.compute(predictions=bleu_preds, references=bleu_refs)

    return {
        "bleu": bleu_scores["bleu"],
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"]
    }
