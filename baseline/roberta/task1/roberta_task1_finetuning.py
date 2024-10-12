#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import (
    AutoModelForCausalLM, AutoTokenizer, pipeline, BertTokenizer, LongformerTokenizerFast, 
    LongformerForSequenceClassification, Trainer, TrainingArguments, 
    LongformerConfig, RobertaTokenizer, RobertaForSequenceClassification,
    RobertaModel, RobertaTokenizer, RobertaConfig, AdamW)
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, StratifiedShuffleSplit 
from huggingface_hub import login
from tqdm import tqdm
import tensorflow as tf
from collections import Counter
import pandas as pd
import accelerate
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import nltk
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from collections import defaultdict
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, precision_recall_fscore_support, confusion_matrix
from transformers import EvalPrediction
import os
import re
import csv
import json
import joblib
import glob
import warnings
import wandb
warnings.filterwarnings('ignore')
wandb.init(project="roberta-task1-finetune", entity="mattdrive")


# In[2]:


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
print(device)


# In[3]:


tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True)


# In[4]:


df = pd.read_csv('/Users/guida/llm_argument_tasks/clean_data/task1_finetune_data_binary.csv', index_col=0)


# In[5]:


df


# In[6]:


df['combined'] = df['text'] + ' [SEP] ' + df['argument']
df.head()


# In[11]:


X = df['combined']  
y = df['label'] 


# In[15]:


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# In[8]:


class ArgumentsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# In[9]:


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


# In[14]:


all_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nTraining Fold {fold + 1}")
    
    # Split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Tokenize
    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, return_tensors='pt')
    val_encodings = tokenizer(list(X_val), truncation=True, padding=True, return_tensors='pt')
    
    # Create datasets
    train_dataset = ArgumentsDataset(train_encodings, y_train.to_numpy())
    val_dataset = ArgumentsDataset(val_encodings, y_val.to_numpy())
    
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', device_map="auto")
    
    training_args = TrainingArguments(
        output_dir=f'./results/fold-{fold}',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'./logs/fold-{fold}',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    trainer.train()

    metrics = trainer.evaluate()
    all_metrics.append(metrics)
    print(f"Fold {fold + 1} metrics:", metrics)

