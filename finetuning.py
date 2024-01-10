import os
import numpy as np
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'false'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_config = {'protected_namespaces': ()}

TEST_SIZE = 0.2
VAL_SIZE = 0.1

def preprocessing_dataset(dataset: Dataset, tokenizer):
    return dataset.map(lambda sample: tokenize_review(sample, tokenizer))

def tokenize_review(sample, tokenizer):
    return tokenizer(sample['review'], padding='max_length', truncation=True)

def compute_metrics(eval_predictions):
    logits, labels = eval_predictions
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"f1_score": f1}

def split_dataset(dataset: Dataset):
    train_dataset, temp_dataset = train_test_split(dataset, test_size=(TEST_SIZE + VAL_SIZE), random_state=42)
    temp_dataset = Dataset.from_dict(temp_dataset)
    val_dataset, test_dataset = train_test_split(temp_dataset, test_size=(TEST_SIZE / (TEST_SIZE + VAL_SIZE)), random_state=42)
    return Dataset.from_dict(train_dataset), Dataset.from_dict(test_dataset), Dataset.from_dict(val_dataset)

def main():
    datasets_by_category = get_datasets()

    for category, dataset in datasets_by_category.items():
        model_name = 'distilbert-base-cased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        tokenized_dataset = preprocessing_dataset(dataset=dataset, tokenizer=tokenizer)
        # print_colored(tokenized_dataset, "red")
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        # print_colored(data_collator, "green")
        trainer_args = TrainingArguments(output_dir=f'sentiment-analysis-{category}',
                        num_train_epochs=2,
                        evaluation_strategy="epoch",
                        save_strategy="epoch",
                        push_to_hub=True)
        train_tokenized_dataset, test_tokenized_dataset, val_tokenized_dataset = split_dataset(dataset=tokenized_dataset)
        # print_colored(train_dataset, "magenta")
        trainer = Trainer(
            model=model,
            args=trainer_args,
            train_dataset=train_tokenized_dataset,
            eval_dataset=val_tokenized_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        trainer.train()
    
if __name__ == "__main__":
    main()
