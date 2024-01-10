import os
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding )

WORDCLOUDS_LOCATION = 'Plots/WordClouds/'
DISTRIBUTIONS_LOCATION = 'Plots/Distributions/'

class Model:
    def __init__(self, name, dataframe):
        self.name = name

        self.dataframe = dataframe
        self.dataset = self.dataframe_to_dataset(dataframe=dataframe)
        self.train_dataset = None
        self.test_dataset = None
        self.eval_dataset = None

        self.model = None
        self.tokenizer = None

    def dataframe_to_dataset(self, dataframe):
        return Dataset.from_pandas(dataframe)

    def plot_word_cloud(self, save_path=WORDCLOUDS_LOCATION):
        reviews = " ".join(self.dataset['review'])

        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(reviews)

        # Display the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')

        # Save the word cloud if a path is provided
        if save_path:
            save_path = os.path.join(save_path, f'{self.name}_wordcloud.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_distribution(self, save_path=DISTRIBUTIONS_LOCATION):
        labels = self.dataframe['label']
        label_counts = labels.value_counts()
        total_samples = len(labels)
        label_distribution = {label: count / total_samples for label, count in label_counts.items()}

        # Plotting
        labels, proportions = zip(*label_distribution.items())
        plt.bar(labels, proportions, color=['green', 'red'])
        plt.xlabel('Labels')
        plt.ylabel('Proportion')
        plt.title('Label Distribution')

        # Save or show the plot
        if save_path:
            save_path = os.path.join(save_path, f'{self.name}_distribution.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def train_test_eval_split(self, test_size=0.2, eval_size=0.1):
        train_data, test_data = train_test_split(self.dataset, test_size=test_size, random_state=42)
        test_data, eval_data = train_test_split(test_data, test_size=eval_size, random_state=42)

        self.train_dataset = train_data
        self.test_dataset = test_data
        self.eval_dataset = eval_data

    def predict(self, review_text):
        inputs = self.tokenizer(review_text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        return predicted_class

    def compute_metrics(self):
        predictions, true_labels = self.make_predictions(self.test_dataset)
        f1 = f1_score(true_labels, predictions, average='weighted')
        accuracy = accuracy_score(true_labels, predictions)
        return {"f1_score": f1, "accuracy": accuracy}
    
    @staticmethod
    def tokenize_review(sample, tokenizer):
        return tokenizer(sample['review'], padding='max_length', truncation=True)

    @staticmethod
    def split_dataset(dataset, test_size, eval_size):
        train_data, temp_data = train_test_split(dataset, test_size=(test_size + eval_size), random_state=42)
        temp_data = Dataset.from_dict(temp_data)
        val_data, test_data = train_test_split(temp_data, test_size=(test_size / (test_size + eval_size)), random_state=42)
        return Dataset.from_dict(train_data), Dataset.from_dict(test_data), Dataset.from_dict(val_data)
    
    def fine_tune(self, num_train_epochs=1, eval_size=0.1, test_size=0.2, model_name='distilbert-base-cased'):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)

        tokenized_dataset = self.dataset.map(lambda sample: self.tokenize_review(sample, tokenizer))
        train_dataset, test_dataset, val_dataset = self.split_dataset(tokenized_dataset, test_size, eval_size)

        output_dir = f'EynardM/sentiment-analysis-{self.name}'
        trainer_args = TrainingArguments(output_dir=output_dir, 
                                         num_train_epochs=num_train_epochs, 
                                         evaluation_strategy="epoch", 
                                         save_strategy="epoch", 
                                         push_to_hub=True)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        trainer = Trainer(model=model, 
                          args=trainer_args, 
                          train_dataset=train_dataset, 
                          eval_dataset=val_dataset, 
                          data_collator=data_collator, 
                          tokenizer=tokenizer, 
                          compute_metrics=self.compute_metrics)
        trainer.train()

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.eval_dataset = val_dataset
        self.model = model
        self.tokenizer = tokenizer
        