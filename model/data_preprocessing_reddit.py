import torch
import torch.nn as nn
import os
from datetime import datetime
import configparser
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from summarizer import Summarizer
from scipy.special import softmax
from sklearn import metrics

from utils import CustomDataset, SentencePairClassifier, test_prediction, compute_metric, plot_graph, poissonLoss

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check that we are using 100% of GPU memory footprint support libraries/code
# from https://github.com/patrickvonplaten/notebooks/blob/master/PyTorch_Reformer.ipynb
import psutil
import humanize
import os
import GPUtil as GPU
GPUs = GPU.getGPUs()
# XXX: only one GPU on Colab and isn’t guaranteed
gpu = GPUs[0]
def printm():
    process = psutil.Process(os.getpid())
    print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize(
        process.memory_info().rss))
    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed,
                                                                                                gpu.memoryUtil*100, gpu.memoryTotal))
printm()


def load_data():
    # Load Reddit dataset
    csv_dataset_path = "fyp/text_reddit_v1.csv"
    df = pd.read_csv(csv_dataset_path) 

    #count the words of question and answer text
    df['questionText_word_count'] = df['selftext'].apply(lambda x: len(x.split()))
    df['answerText_word_count'] = df['comment'].apply(lambda x: len(x.split()))
    
    return df

def summarize(text, summarizer_model, maxlen):
    try:
        if len(text.split()) > maxlen: 
            result = summarizer_model(text, ratio=0.6, max_length=maxlen)
            #result = summarizer_model(text, max_length=maxlen)
            text = ''.join(result)
    except (AttributeError, TypeError):
        print("error: ")
        print(text.describe())
        raise AssertionError('Input variables should be strings')
        
    return text

def all_softmax(df):
    df["score"] = 0
    scores=[]
    size = 0
    scores = pd.Series() 
    upvote_sample = []
    for index in range(len(df.index)):
        upvote_sample.append(df.iloc[index]["upvote"])
    scores = scores.append(pd.Series(softmax(upvote_sample)), ignore_index=True)
    df['score'] = scores   
    return df
    

def each_softmax(df):
    df["score"] = 0
    scores=[]
    size = 0
    scores = pd.Series() 

    upvote_sample = []
    for index in range(len(df.index)):
        upvote_sample.append(df.iloc[index]["upvote"])
        if index < df.last_valid_index() and df.iloc[index]["subid"] != df.iloc[index+1]["subid"]:
            scores = scores.append(pd.Series(softmax(upvote_sample)), ignore_index=True)    
            upvote_sample = []
    df['score'] = scores
    return df

def no_softmax(df):
    df["score"] = 0

    best_cm = True
    for index in df.index:
        if best_cm and df.iloc[index]["upvote"] > 0:
            df.at[index, 'score'] = 1

        if index != df.last_valid_index() and df.iloc[index]["subid"] == df.iloc[index+1]["subid"]:
            if df.iloc[index]["upvote"] > df.iloc[index+1]["upvote"] :
                best_cm = False
        else:      # already the next question
            best_cm = True
    return df
    

def each_log(df):
    df["score"] = 0
    scores = pd.Series() 

    upvote_sample = []
    for index in range(len(df.index)):
        upvote_sample.append(df.iloc[index]["upvote"])  
        if index < df.last_valid_index() and df.iloc[index]["subid"] != df.iloc[index+1]["subid"]:
            scores = scores.append(pd.Series(np.log1p(upvote_sample)), ignore_index=True)   # log(1+x) to prevent negative infinity 
            upvote_sample = []
    df['score'] = scores
    return df

def all_log(df):
    df["score"] = df['upvote'].copy()
    df["score"] = df["score"].apply(lambda s: np.log1p(s) if s != -1 else -5)
    return df
    
def preprocess_data(df, summarize_question: bool=False, summarize_answer: bool=False, normalize: str=None, maxlen=512):
    
    
    #summzrize text
    summarizer_model = Summarizer()
    if summarize_question:
        print("summarizing question...")
        
        subid = df.iloc[0]['subid']
        summarized_selftext = summarize(df.iloc[0]['selftext'], summarizer_model, maxlen)
        for index, row in df.iterrows():
            if row['subid'] != subid:
                summarized_selftext = summarize(row['selftext'], summarizer_model, maxlen)
                subid = row['subid']               
            
            df.loc[index, 'selftext'] = summarized_selftext
                
    if summarize_answer:
        print("summarizing answer...")
        df["comment"] = df["comment"].apply(lambda x: summarize(x, summarizer_model, maxlen,) if len(x.split()) > maxlen else x)
    
    # save the file first
    df.to_csv("temp_df.csv")
    
    print("normalize:")
    #normalize upvote as score
    if normalize == "all_softmax":  # scores using softmax every answer
        print("all_softmax\n")
        df = all_softmax(df)
    elif normalize == "each_softmax": # scores using softmax for each question's answer
        print("each_softmax\n")
        df = each_softmax(df)
    elif normalize == "no_softmax":  # not using any normalization
        print("no_softmax\n")
        df = no_softmax(df)
    elif normalize == "each_log": # scores using softmax for each question's answer
        print("each_log\n")
        df = each_log(df)
    elif normalize == "all_log": # scores using all_log for each comments
        print("all_log\n")
        df = all_log(df)
        
    # drop the unwanted column
    df = df.drop(['title', 'subid', 'subreddit', 'comment_id',\
                  'questionText_word_count', 'answerText_word_count'], axis=1)
    df = df.rename(columns={'selftext': 'sentence1', 'comment': 'sentence2'})
    df = df.dropna()
    
    # remove some unbalnced data 
    df
    temp = df[(df["score"] > 0) & (df["score"] <= 2)]
    remove_n = len(temp)//2
    drop_indices = np.random.choice(temp.index, remove_n, replace=False)
    temp_subset = temp.drop(drop_indices)
    df = df.drop(temp.index).append(temp_subset)

    # Split the data
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
    
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_test = df_test.sample(frac=1).reset_index(drop=True)

    print(df_train.describe(include='all'))
    print(df_test.describe(include='all'))    
    
    if os.path.exists("demofile.txt"):
        os.remove("demofile.txt")
    else:
        print("The file does not exist")
    
    return df_train, df_test
    
    
    
    
def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    


def run():
    
    # load parameters
    print("load parameters")
    config = configparser.ConfigParser()
    config.read('fyp/comment_ranking_model/parameter.ini')
    parameters =  config['parameters']
    
    summarize_question = parameters.getboolean('summarize_question')
    summarize_answer = parameters.getboolean('summarize_answer')
    normalize = parameters.get('normalize')
    bert_model = parameters.get('bert_model')
    freeze_bert = parameters.getboolean('freeze_bert') # if True, freeze the encoder weights and only update the classification layer weights
    maxlen = parameters.getint('maxlen')  # maximum length of the tokenized input sentence pair : if greater than "maxlen", the input is truncated and else if smaller, the input is padded
    bs = parameters.getint('bs')  # batch size
    iters_to_accumulate = parameters.getint('iters_to_accumulate')  # the gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate. If set to "1", you get the usual batch size
    lr = parameters.getfloat('lr')  # learning rate
    epochs = parameters.getint('epochs')  # number of training epochs

    
    #  Set all seeds to make reproducible results
    set_seed(1)
    
    # preprocess data
    print("Loading data...")
    df = load_data()
    print("Preprocessing data...")
    df_train, df_test = preprocess_data(df, summarize_question, summarize_answer, normalize, maxlen)
    #train_filepath = "fyp/comment_ranking_model/df_train.csv"
    train_filepath = "fyp/comment_ranking_model/df_reddit_train.csv"
    #train_filepath = "fyp/comment_ranking_model/df_train_bce.csv"
    df_train.to_csv(train_filepath)
    #test_filepath = "fyp/comment_ranking_model/df_test.csv"
    test_filepath = "fyp/comment_ranking_model/df_reddit_test.csv"
    #test_filepath = "fyp/comment_ranking_model/df_test_bce.csv"
    df_test.to_csv(test_filepath)
    
    print("train data avaliable in: ", train_filepath)
    print("test data available in: ", test_filepath)
    

if __name__ == "__main__":
    run()