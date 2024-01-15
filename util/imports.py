import os
import numpy as np
import random
import pandas as pd 
from datasets import Dataset, load_from_disk
from ast import literal_eval
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import warnings

# Disable parallelism in tokenization for transformers library
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ignore RuntimeWarnings related to numpy.lib.nanfunctions
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy.lib.nanfunctions")


