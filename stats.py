from util.parameters import *
from util.imports import *
from util.locations import *
from util.helper import *

# Loading models and tokenizers
organic_model = AutoModelForSequenceClassification.from_pretrained(ORGANIC_MODEL_PATH)
organic_tokenizer = AutoTokenizer.from_pretrained(ORGANIC_MODEL_PATH)

climate_model = AutoModelForSequenceClassification.from_pretrained(CLIMATE_MODEL_PATH)
climate_tokenizer = AutoTokenizer.from_pretrained(CLIMATE_MODEL_PATH)

water_model = AutoModelForSequenceClassification.from_pretrained(WATER_MODEL_PATH)
water_tokenizer = AutoTokenizer.from_pretrained(WATER_MODEL_PATH)

social_model = AutoModelForSequenceClassification.from_pretrained(SOCIAL_MODEL_PATH)
social_tokenizer = AutoTokenizer.from_pretrained(SOCIAL_MODEL_PATH)

governance_model = AutoModelForSequenceClassification.from_pretrained(GOVERNANCE_MODEL_PATH)
governance_tokenizer = AutoTokenizer.from_pretrained(GOVERNANCE_MODEL_PATH)

waste_model = AutoModelForSequenceClassification.from_pretrained(WASTE_MODEL_PATH)
waste_tokenizer = AutoTokenizer.from_pretrained(WASTE_MODEL_PATH)

adverse_model = AutoModelForSequenceClassification.from_pretrained(ADVERSE_MODEL_PATH)
adverse_tokenizer = AutoTokenizer.from_pretrained(ADVERSE_MODEL_PATH)

# Storing models and tokenizers
models = {
    "organic": organic_model,
    "climate": climate_model,
    "water": water_model,
    "social": social_model,
    "governance": governance_model,
    "waste": waste_model,
    "adverse": adverse_model,
}

tokenizers = {
    "organic": organic_tokenizer,
    "climate": climate_tokenizer,
    "water": water_tokenizer,
    "social": social_tokenizer,
    "governance": governance_tokenizer,
    "waste": waste_tokenizer,
    "adverse": adverse_tokenizer,
}

def get_review_rates(review):
    rates = []
    for category, model in models.items():
        tokenizer = tokenizers[category]
        inputs = tokenizer(review, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax().item()
        rates.append(predicted_class-1)
    return rates

def get_restaurant_ratings(restaurant_reviews):
    reviews_rates = []
    for review in restaurant_reviews:
        review_rates = get_review_rates(review)
        reviews_rates.append(review_rates)
    reviews_rates = np.array(reviews_rates)
    restaurant_ratings = []
    for category_index in range(reviews_rates.shape[1]):
        category_values = reviews_rates[:, category_index]
        valid_values = category_values[category_values != -1]
        if len(valid_values) > 0:
            category_rating = np.mean(valid_values)
        else:
            category_rating = -1
        restaurant_ratings.append(category_rating)
    return restaurant_ratings 

def get_restaurant_global_rate(restaurant_ratings):
    valid_ratings = [rating for rating in restaurant_ratings if rating != -1]
    if not valid_ratings:
        return -1
    else:
        average_rating = sum(valid_ratings) / len(valid_ratings)
        return average_rating
    
def get_restaurants():
    restaurants_reviews = {}
    for filename in os.listdir(RESTAURANTS_DATA_PATH):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(RESTAURANTS_DATA_PATH, filename))
            restaurant_name = os.path.splitext(filename)[0]
            restaurants_reviews[restaurant_name] = df['review'].tolist()
    return restaurants_reviews

def main():
    restaurants_reviews = get_restaurants()
    results = []

    for restaurant, reviews in restaurants_reviews.items():
        print(f"Getting ratings predictions for restaurant {restaurant}...")
        restaurant_ratings = get_restaurant_ratings(restaurant_reviews=reviews)
        global_rate = get_restaurant_global_rate(restaurant_ratings=restaurant_ratings)

        results.append({
            'restaurant': restaurant,
            'ratings': restaurant_ratings,
            'global_rate': global_rate
        })
         
    results_df = pd.DataFrame(results)
    results_df.to_csv('Results/restaurants_predictions.csv', index=False)
    print("Ratings predictions saved!")

    print(f"Getting statistics on restaurants...")
    results_df = pd.read_csv('restaurants_predictions.csv')

    ratings_array = np.vstack(results_df['ratings'].apply(lambda x: literal_eval(x)))
    ratings_array[ratings_array == -1] = np.nan

    ratings_mean = np.nanmean(ratings_array, axis=0)
    ratings_std = np.nanstd(ratings_array, axis=0)

    valid_global_rates = results_df['global_rate'][results_df['global_rate'] != -1]

    global_rate_mean = np.nanmean(valid_global_rates)
    global_rate_std = np.nanstd(valid_global_rates)

    stats_df = pd.DataFrame({
        'category': [CATEGORIES[i] for i in range(ratings_mean.size)] + ['global_rate'],
        'mean': np.concatenate([ratings_mean, [global_rate_mean]]),
        'std': np.concatenate([ratings_std, [global_rate_std]])
    })

    stats_df.to_csv('Results/restaurants_statistics.csv', index=False)
    print("Statistics saved!")

if __name__ == '__main__':
    main()