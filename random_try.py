# Import necessary modules and functions from utility files
from util.imports import *
from util.locations import *
from util.parameters import *
from util.helper import *

# Function to get a random review and its true rate from a specified category
def get_random_review(category):
    # Load the dataset for the specified category
    dataset = load_from_disk(DATASET_PATH + category + '/')
    
    # Generate a random index to select a random review
    random_index = random.randint(0, len(dataset) - 1)
    
    # Retrieve the random review and its true rate
    review = dataset['review'][random_index]
    rate = dataset['rate'][random_index]
    
    return review, rate

# Function to predict the sentiment class of a given review using a pre-trained model and tokenizer
def predict_review(model, tokenizer, review):
    # Tokenize the review and obtain the predicted logits from the model
    encoded_review = tokenizer(review, padding='max_length', truncation=True, max_length=400, return_tensors='pt')
    logits = model(**encoded_review).logits
    
    # Predict the sentiment class by selecting the index with the highest logit value
    predicted_class = torch.argmax(logits, dim=1).item()
    
    return predicted_class

# Main function to execute the script
def main():
    # Prompt the user to input the category index
    user_input = input("Enter the number corresponding to the chosen category: ")
    category_index = int(user_input)
    
    # Retrieve the chosen category from the list of categories
    category_chosen = CATEGORIES[category_index]
    
    # Print the chosen category in cyan color
    # print_colored(category_chosen, "cyan")
    
    # Get a random review and its true rate from the chosen category
    random_review, true_rate = get_random_review(CATEGORIES[category_index])
    
    # Print the random review in red color
    # print_colored(random_review, "red")
    
    # Load the pre-trained model and tokenizer for the chosen category
    print('Loading model...')
    model = AutoModelForSequenceClassification.from_pretrained(f'EynardM/sentiment-analysis-{category_chosen}')
    tokenizer = AutoTokenizer.from_pretrained(f'EynardM/sentiment-analysis-{category_chosen}')
    
    # Predict the sentiment class for the random review
    predicted_class = predict_review(model, tokenizer, random_review)
    
    # Print the predicted sentiment class in blue color
    # print_colored(predicted_class, "blue")
    
    # Print the true rate of the random review in green color
    # print_colored(true_rate, "green")

# Entry point to run the script
if __name__ == '__main__':
    main()
