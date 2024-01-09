import os
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def get_datasets(csv_directory='Reviews'):
    datasets_by_category = {}

    for filename in os.listdir(csv_directory):
        if filename.endswith(".csv"):
            csv_path = os.path.join(csv_directory, filename)
            df = pd.read_csv(csv_path)
            df['label'] = df['label'].map({"Positive": 2, "Negative": 1})

            dataset_dict = {
                "review": df['review'].tolist(),
                "label": df['label'].tolist(),
            }

            dataset = Dataset.from_dict(dataset_dict)

            category_name = df['category'].iloc[0]
            datasets_by_category[category_name] = dataset

    return datasets_by_category

def get_datasets_split(csv_directory='Reviews', test_size=0.2, random_state=42):
    datasets_by_category = {}

    for filename in os.listdir(csv_directory):
        if filename.endswith(".csv"):
            csv_path = os.path.join(csv_directory, filename)
            df = pd.read_csv(csv_path)
            df['label'] = df['label'].map({"Positive": 1, "Negative": 0})

            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

            train_dataset_dict = {
                "review": train_df['review'].tolist(),
                "label": train_df['label'].tolist(),
            }

            test_dataset_dict = {
                "review": test_df['review'].tolist(),
                "label": test_df['label'].tolist(),
            }

            train_dataset = Dataset.from_dict(train_dataset_dict)
            test_dataset = Dataset.from_dict(test_dataset_dict)

            category_name = df['category'].iloc[0]
            datasets_by_category[category_name] = {'train': train_dataset, 'test': test_dataset}

    return datasets_by_category

def get_word_cloud(dataset, save_path=None):
    reviews = " ".join(sample['review'] for sample in dataset)
    
    # Générer le nuage de mots
    wordcloud = WordCloud(width = 800, height = 400, 
                random_state=21, max_font_size=110).generate(reviews)

    # Afficher le nuage de mots à l'écran
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')

    # Sauvegarder le nuage de mots si un chemin est fourni
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def get_labels_distribution(dataset, save_path=None):
    labels = [sample['label'] for sample in dataset]
    label_counts = {label: labels.count(label) for label in set(labels)}
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
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    datasets_by_category = get_datasets_split()

    for category, dataset_splits in datasets_by_category.items():
        train_dataset = dataset_splits['train']
        test_dataset = dataset_splits['test']

        get_word_cloud(train_dataset, save_path=f"Plots/WordClouds/{category}_train_wordcloud.png")
        get_word_cloud(test_dataset, save_path=f"Plots/WordClouds/{category}_test_wordcloud.png")
        get_labels_distribution(train_dataset, save_path=f'Plots/LabelsDistributions/{category}_train_label_distribution_.png')
        get_labels_distribution(test_dataset, save_path=f'Plots/LabelsDistributions/{category}_test_label_distribution_.png')

if __name__ == "__main__":
    main()
