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

def main():
    models = init_models()
    fine_tuning(models=models)
    
if __name__ == "__main__":
    main()
