All commands in this document should be executed from the root directory of the project folder (**"EcoScore"**).

# Working Environment

To install all the required packages for running the files, execute the following command:

**pip install -r requirements.txt**

# Datasets

## Training Datasets
To obtain the training datasets, execute: **datamodule.py**
Training datasets can be visualized in the folder: **Datasets/**

# Models
Note: Every model below has been fine-tuned based on the *'bert-base-uncased'* model.

## First Approach: One Model for Every Category
Dataset: *AllInData/reviews.csv*

### All-in method: 
We fine-tune a model in order to predict the right note for each category for a review.
To do so, you can run: **GpuScipts/Ecoscore_allin_finetune.ipynb** (with the runtime environment of your choice)

## Second Approach: One Model for one Category 
Datasets: *Data/...* (CSV files) or *Datasets/...* (Dataset objects from the datasets library)

### Try with a simple model and embeddings
We tried with a simple model if we could have good results just in predicting grades on reviews. So we tried the same model on both the dataset entirely
made with ChatGPT and the one with real reviews and ChatGPT. We noticed that it was better with the dataset with real reviews. To test it, you just need
to run the files.

### Chosen Solution for the Presentation, the best one
We fine-tune a model for each category. 
Thus, we obtain 7 models, pushed to Hugging Face's hub. 
To do so, you can run: **model.py** (local) or **GpuScripts/Ecoscore_category_finetuned_models.ipynb** (with the runtime environment of your choice)

# Results
To try the models, you can run: **random_try.py** or adapt this code to your needs
To get the statistics, run: **stats.py**

Note: These files are edited to use the best models we implemented to make this work.

# Appendices
Imports: **imports.py**
Functions: **functions.py**
Paths: **locations.py**
Parameters: **parameters.py**
