import os
import numpy as np
from pydantic import BaseModel
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)

from main import get_datasets
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from util import print_colored

def main():
    model = AutoModelForSequenceClassification.from_pretrained(f'sentiment-analysis-Social')
    tokenizer = AutoTokenizer.from_pretrained(f'sentiment-analysis-{category}')

    test_tokenized_dataset = preprocessing_dataset(dataset=test_tokenized_dataset, tokenizer=tokenizer)

    # Prédiction sur l'ensemble de test
    test_predictions = trainer.predict(test_tokenized_dataset)
    test_scores = test_predictions.predictions
    test_predicted_classes = np.argmax(test_scores, axis=1)

    # Affichage des résultats sur l'ensemble de test
    print("Classes prédites sur l'ensemble de test :", test_predicted_classes)

    # Calcul de l'accuracy sur l'ensemble de test
    test_labels = test_predictions.label_ids
    test_accuracy = np.sum(test_predicted_classes == test_labels) / len(test_labels)
    print("Accuracy sur l'ensemble de test :", test_accuracy)

if __name__ == "__main__":
    main()