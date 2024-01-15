All commands in this document should be executed from the root directory of the project folder (**"EcoScore"**).

# Working Environment

To install all the required packages for running the files, execute the following command:

**pip install -r requirements.txt**

# Datasets

## Training Datasets
To obtain the training datasets, execute: **datamodule.py**.
Training datasets can be visualized in the folder: **Datasets/**

# Models
<span style="color:blue"> Note: Every model below has been fine-tuned based on the *'bert-base-uncased'* model. </span>

## First Approach: One Model for Every Category
Dataset: *AllInData/reviews.csv*

### <span style="color:orange">All-in method:</span>
<span style="color:orange">We fine-tune a model in order to predict the right note for each category for a review.
To do so, you can run: **GpuScipts/Ecoscore_allin_finetune.ipynb** (with the runtime environment of your choice)</span>

## Second Approach: One Model for one Category 

### First Try with a simple model and embeddings
Datasets: *Data/...* (CSV files) or *Datasets/...* (Dataset objects from the datasets library) and/or *GptData/...* (CSV files)
We tried using a simple model to determine if we could achieve good results in predicting grades for each category based on reviews. 
We applied the same model to both datasets—one entirely generated with ChatGPT and the other consisting of real reviews mixed (or not, you can decide) with ChatGPT content. 
We observed that the model performed better on the dataset containing real reviews. To test it, you just need to run the files in the folder *TryEmbeddings/...*.

### <span style="color:red"> Chosen Solution for the Presentation, the best one </span>
<span style="color:red"> Datasets: *Data/...* (CSV files) or *Datasets/...* (Dataset objects from the datasets library)
We fine-tune a model for each category. 
Thus, we obtain 7 models, pushed to Hugging Face's hub. 
To do so, you can run: **model.py** (local) or **GpuScripts/Ecoscore_category_finetuned_models.ipynb** (with the runtime environment of your choice)</span>

# Results
To try the models (of the choosen solution), you can run: **random_try.py** or adapt this code to your needs
To get the statistics, run: **stats.py**
Note: These files are edited to use the best models we implemented to make this work.

# Appendices
Imports: **imports.py**
Functions: **functions.py**
Paths: **locations.py**
Parameters: **parameters.py**
