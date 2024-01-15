# Import necessary modules and functions from utility files
from util.imports import *
from util.locations import *

# Function to generate a word cloud from the given text and save it
def generate_wordcloud_and_save(text, category):
    # Define the filename for the word cloud image
    filename = f'wordcloud_{category}.png'
    
    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110, background_color='white').generate(text)
    
    # Create and configure the plot
    plt.figure(figsize=(10, 7))
    plt.title(f'Word cloud for category: {category}')
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    
    # Save the word cloud plot
    save_path = os.path.join(PLOT_PATH, filename)
    plt.savefig(save_path)
    
    # Close the plot to release resources
    plt.close()

# Function to plot the distribution of data and save the plot
def plot_distribution_and_save(data, column_name, category):
    # Define the filename for the distribution plot
    filename = f'distribution_rate_{category}.png'
    
    # Create a plot
    plt.figure(figsize=(8, 5))

    # Convert PyTorch Tensor to a NumPy array
    data_array = data[column_name].numpy()

    # Convert NumPy array to PyTorch Tensor
    data_tensor = torch.from_numpy(data_array)

    # Compute value counts
    unique_values, counts = torch.unique(data_tensor, return_counts=True)

    # Plot the distribution
    plt.bar(unique_values.numpy(), counts.numpy(), color='skyblue')
    plt.title(f'Distribution of ratings for category: {category}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')

    # Save the distribution plot
    save_path = os.path.join(PLOT_PATH, filename)
    plt.savefig(save_path)
    
    # Close the plot to release resources
    plt.close()

# Function to filter out samples with a rate of -1
def preprocessing(sample):
    return sample['rate'].item() != -1

# Main function to execute the script
def main():
    # Iterate through category folders in the dataset path
    for category_folder in os.listdir(DATASET_PATH):
        # Construct the full path for the category
        category_path = os.path.join(DATASET_PATH, category_folder)
        
        # Load the dataset from the category path
        dataset = load_from_disk(category_path)
        
        # Filter out samples with a rate of -1
        dataset_filtered = dataset.filter(preprocessing)
        
        # Check if there are samples left after filtering
        if len(dataset_filtered) != 0:
            # Concatenate all reviews into a single string
            all_text = ' '.join(dataset_filtered['review'])
            
            # Generate word cloud and save
            generate_wordcloud_and_save(all_text, category_folder)
            
            # Plot distribution of ratings and save
            plot_distribution_and_save(dataset_filtered, 'rate', category_folder)

# Entry point to run the script
if __name__ == '__main__':
    main()
