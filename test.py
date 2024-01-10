import pandas as pd
from util import print_colored
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def predict_random(model, tokenizer):
    review_data = pd.read_csv('Restaurant.csv')
    for _, row in review_data.iterrows():
        review_text = row['review']
        category = row['category']

        if category == 'water':
            print_colored(review_text, "red")
            if model:
                # Tokeniser le texte de la revue
                inputs = tokenizer(review_text, return_tensors="pt", padding=True, truncation=True)

                # Obtenir les logits du modèle
                with torch.no_grad():
                    outputs = model(**inputs)
                
                logits = outputs.logits

                # Obtenir la classe prédite
                predicted_class = torch.argmax(logits, dim=1).item()

                
                print(f"Predicted Rating: {predicted_class}")
                print("-" * 50)
            else:
                print(f"No model found for category: {category}")

def main():
    model_name = 'EynardM/sentiment-analysis-Water'
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    predict_random(model, tokenizer)

if __name__ == "__main__":
    main()
