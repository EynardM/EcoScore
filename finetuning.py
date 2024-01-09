import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datamodule import get_datasets
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2
VAL_SIZE = 0.1

def preprocessing_dataset(dataset: Dataset, tokenizer):
    return dataset.map(lambda sample: preprocessing(sample, tokenizer))

def preprocessing(sample, tokenizer):
    review_tokens = tokenizer(sample['review'], padding='max_length', truncation=True, return_tensors='pt')
    return {
        'review_tokens': review_tokens['input_ids'][0]
    }

def compute_metrics(eval_predictions):
    logits, labels = eval_predictions
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"f1_score": f1}

def split_dataset(dataset: Dataset):
    train_dataset, temp_dataset = train_test_split(dataset, test_size=(TEST_SIZE + VAL_SIZE), random_state=42)
    temp_dataset = Dataset.from_dict(temp_dataset)
    val_dataset, test_dataset = train_test_split(temp_dataset, test_size=(TEST_SIZE / (TEST_SIZE + VAL_SIZE)), random_state=42)
    return train_dataset, test_dataset, val_dataset

def main():
    model_name = 'distilbert-base-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    datasets_by_category = get_datasets()

    for category, dataset in datasets_by_category.items():
        dataset = preprocessing_dataset(dataset=dataset, tokenizer=tokenizer)
        train_dataset, test_dataset, val_dataset = split_dataset(dataset=dataset)
        trainerargs = TrainingArguments(output_dir='Model_test',
                              num_train_epochs=5,
                              evaluation_strategy="epoch",
                              save_strategy="epoch",
                              load_best_model_at_end=True,
                              push_to_hub=True)
        
        trainer = Trainer(model=model,
                    args=trainerargs,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics)
        
        trainer.train()
        break   

if __name__ == "__main__":
    main()
