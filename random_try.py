from util.imports import *
from util.locations import *
from util.parameters import *
from util.helper import *

def get_random_review(category):
    dataset = load_from_disk(DATASET_PATH+category+'/')
    random_index = random.randint(0, len(dataset) - 1)
    review = dataset['review'][random_index]
    rate = dataset['rate'][random_index]
    return review, rate

def predict_review(model, tokenizer, review):
    encoded_review = tokenizer(review, padding='max_length', truncation=True, max_length=400, return_tensors='pt')
    logits = model(**encoded_review).logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

def main():
    user_input = input("Entrez le nombre correspondant à la catégorie choisie: ")
    category_index = int(user_input)
    category_choosed = CATEGORIES[category_index]
    print_colored(category_choosed, "cyan")
    random_review, true_rate = get_random_review(CATEGORIES[category_index]) 
    print_colored(random_review, "red")
    print('Loading model...')
    model = AutoModelForSequenceClassification.from_pretrained(f'EynardM/sentiment-analysis-{category_choosed}')
    tokenizer = AutoTokenizer.from_pretrained(f'EynardM/sentiment-analysis-{category_choosed}')
    predicted_class = predict_review(model, tokenizer, random_review)
    print_colored(predicted_class, "blue")
    print_colored(true_rate, "green")

if __name__ == '__main__':
    main()
