# Import necessary modules and functions from utility files
from util.imports import *
from util.locations import *
from util.parameters import *
from util.helper import *

# Function to save a dataset to disk
def save_dataset(dataset: Dataset, path) -> None:
    dataset.save_to_disk(path)
    # Print a colored message indicating successful saving
    print_colored(dataset, "green")

# Function to load a tokenizer for a given model name
def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

# Function to load a pre-trained BERT model for sequence classification
def load_model(model_name):
    return AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name, num_labels=7)

# Function to load a dataset from a CSV file and preprocess it
def load_dataset(dataset_path, tokenizer=None):
    # Read CSV file into a Pandas DataFrame
    df = pd.read_csv(dataset_path)

    # Convert DataFrame to a Hugging Face Dataset
    dataset = Dataset.from_pandas(df=df)
    
    # Preprocess the dataset using the specified tokenizer
    dataset = preprocessing_dataset(dataset, tokenizer=tokenizer)

    # Set the format for torch tensors
    dataset.set_format("torch", columns=['review','rate','input_ids','token_type_ids','attention_mask'])
    return dataset

# Function to preprocess a single sample using the tokenizer
def preprocessing(sample, tokenizer, max_length=400):
    return tokenizer(sample['review'], max_length=max_length, padding='max_length', truncation=True, return_tensors='pt', return_attention_mask=True)

# Function to preprocess the entire dataset using the specified tokenizer
def preprocessing_dataset(dataset: Dataset, tokenizer):
    return dataset.map(lambda sample: preprocessing(sample, tokenizer))

# Main function to execute the script
def main():
    # Set the model name (BERT base uncased)
    model_name = 'bert-base-uncased'
    
    # Load the tokenizer for the specified model
    tokenizer = load_tokenizer(model_name)

    # Iterate through CSV files in the specified path
    for filename in os.listdir(CSV_PATH):
        if filename.endswith(".csv"):
            # Extract category from the filename
            category = filename.split('_reviews.csv')[0]
            
            # Load and preprocess the dataset
            dataset = load_dataset(dataset_path=CSV_PATH+filename, tokenizer=tokenizer)
            
            # Save the processed dataset to a specified path
            save_dataset(dataset=dataset, path=DATASET_PATH+category+'/')

# Entry point to run the script
if __name__ == '__main__':
    main()
