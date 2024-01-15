from util.imports import *
from util.locations import *
from util.parameters import *
from util.helper import *

def save_dataset(dataset: Dataset, path) -> None:
    dataset.save_to_disk(path)
    print_colored(dataset, "green")

def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

def load_model(model_name):
    return AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name, num_labels=7)

def load_dataset(dataset_path, tokenizer=None):
    df = pd.read_csv(dataset_path)

    dataset = Dataset.from_pandas(df=df)
    dataset = preprocessing_dataset(dataset, tokenizer=tokenizer)

    dataset.set_format("torch", columns=['review','rate','input_ids','token_type_ids','attention_mask'])
    return dataset

def preprocessing(sample, tokenizer, max_length=400):
    return tokenizer(sample['review'], max_length=max_length, padding='max_length', truncation=True, return_tensors='pt', return_attention_mask=True)

def preprocessing_dataset(dataset: Dataset, tokenizer):
    return dataset.map(lambda sample: preprocessing(sample, tokenizer))

def main():

    model_name = 'bert-base-uncased'
    tokenizer = load_tokenizer(model_name)

    for filename in os.listdir(CSV_PATH):
        if filename.endswith(".csv"):
            category = filename.split('_reviews.csv')[0]
            dataset = load_dataset(dataset_path=CSV_PATH+filename,tokenizer=tokenizer)
            save_dataset(dataset=dataset, path=DATASET_PATH+category+'/')

if __name__ == '__main__':
    main()