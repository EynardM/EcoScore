import os
import pandas as pd
from objects import Model
from util import print_colored

DATA_LOCATION = 'Data/'

def init_models():
    models = []
    for filename in os.listdir(DATA_LOCATION):
        if filename.endswith(".csv"):
            csv_path = os.path.join(DATA_LOCATION, filename)
            df = pd.read_csv(csv_path)
            category_name = os.path.splitext(filename)[0].split('.')[0]
            model = Model(name=category_name, dataframe=df)
            model.plot_word_cloud()
            model.plot_distribution()
            models.append(model)
    return models

def fine_tuning(models):
    for model in models:
        model.fine_tune()

def predict_random(models):
    review_data = pd.read_csv(os.path.join(DATA_LOCATION, 'Restaurant.csv'))
    for _, row in review_data.iterrows():
        review_text = row['review']
        category = row['category']

        # Find the corresponding model
        selected_model = next((model for model in models if model.name == category), None)

        if selected_model:
            predicted_rating = selected_model.predict(review_text)
            print_colored(review_text, "red")
            print(f"Predicted Rating by {selected_model.name}: {predicted_rating}")
            print("-" * 50)
        else:
            print(f"No model found for category: {category}")

def main():
    models = init_models()
    print("Fine tuning models ...")
    fine_tuning(models=models)
    predict_random(models)
if __name__ == "__main__":
    main()
