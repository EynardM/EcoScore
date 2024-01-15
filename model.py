# Import necessary modules and functions from utility files
from util.imports import *
from util.locations import *
from util.parameters import *
from util.helper import *
from util.objects import *
from datamodule import load_tokenizer, load_model

# Function to split a dataset into train, test, and validation datasets
def split_dataset(dataset: Dataset):
    # Split the dataset into train and temporary datasets
    train_dataset, temp_dataset = train_test_split(dataset, test_size=(TEST_SIZE), random_state=42)
    
    # Convert the temporary dataset to a Hugging Face Dataset object
    temp_dataset = Dataset.from_dict(temp_dataset)
    
    # Further split the temporary dataset into validation and test datasets
    val_dataset, test_dataset = train_test_split(temp_dataset, test_size=(TEST_SIZE / (TEST_SIZE + VAL_SIZE)), random_state=42)
    
    return Dataset.from_dict(train_dataset), Dataset.from_dict(test_dataset), Dataset.from_dict(val_dataset)

# Function to compute evaluation metrics (F1 score)
def compute_metrics(eval_predictions):
    logits, labels = eval_predictions
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"f1_score": f1}

# Main function to execute the script
def main():
    # Set the model name (BERT base uncased)
    model_name = 'bert-base-uncased'
    
    # Load the tokenizer for the specified model
    tokenizer = load_tokenizer(model_name=model_name)

    # Iterate through specified categories
    for category in CATEGORIES:

        # Load the pre-trained model for sequence classification
        model = load_model(model_name=model_name)
        
        # Load the dataset from the specified path
        dataset_path = DATASET_PATH + category + '/'
        dataset = Dataset.load_from_disk(dataset_path=dataset_path)

        # Split the dataset into train, test, and validation datasets
        train_dataset, test_dataset, val_dataset = split_dataset(dataset=dataset)

        # Transform datasets into TrainerDataset objects
        train_dataset = TrainerDataset(dataset=train_dataset, tokenizer=tokenizer)
        test_dataset = TrainerDataset(dataset=test_dataset, tokenizer=tokenizer)
        val_dataset = TrainerDataset(dataset=val_dataset, tokenizer=tokenizer)

        # Training the model
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=400)

        trainer_args = TrainingArguments(
            output_dir=f'EynardM/sentiment-analysis-{category}',
            num_train_epochs=1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=LR,
            per_device_train_batch_size=3,
            push_to_hub=True
        )
        trainer = Trainer(
            model=model,
            args=trainer_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()

# Entry point to run the script
if __name__ == '__main__':
    main()
