from util.imports import *
from util.locations import *

def generate_wordcloud_and_save(text, category):
    filename = f'wordcloud_{category}.png'
    wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110, background_color='white').generate(text)
    plt.figure(figsize=(10, 7))
    plt.title(f'Word cloud pour la catégorie: {category}')
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    
    save_path = os.path.join(PLOT_PATH, filename)
    plt.savefig(save_path)
    plt.close()

def plot_distribution_and_save(data, column_name, category):
    filename = f'distribution_rate_{category}.png'
    plt.figure(figsize=(8, 5))

    # Convert PyTorch Tensor to a NumPy array
    data_array = data[column_name].numpy()

    # Convert NumPy array to PyTorch Tensor
    data_tensor = torch.from_numpy(data_array)

    # Compute value counts
    unique_values, counts = torch.unique(data_tensor, return_counts=True)

    # Plot the distribution
    plt.bar(unique_values.numpy(), counts.numpy(), color='skyblue')
    plt.title(f'Distribution des notes pour la catégorie: {category}')
    plt.xlabel(column_name)
    plt.ylabel('Fréquence')

    # Save the plot
    save_path = os.path.join(PLOT_PATH, filename)
    plt.savefig(save_path)
    plt.close()

def preprocessing(sample):
    return sample['rate'].item() != -1

def main():
    for category_folder in os.listdir(DATASET_PATH):
        category_path = os.path.join(DATASET_PATH, category_folder)
        
        dataset = load_from_disk(category_path)
        print(dataset)
        dataset_filtered = dataset.filter(preprocessing)
        if len(dataset_filtered) != 0:
            all_text = ' '.join(dataset_filtered['review'])
            generate_wordcloud_and_save(all_text, category_folder)
            plot_distribution_and_save(dataset_filtered, 'rate', category_folder)

if __name__ == '__main__':
    main()
