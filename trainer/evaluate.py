from utils import CustomDataset, SentencePairClassifier, test_prediction, compute_metric, plot_graph
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime
import configparser
from matplotlib import pyplot as plt
import os.path as path


# Load parameters
print("load parameters")
config = configparser.ConfigParser()
config.read('fyp/comment_ranking_model/parameter.ini')
parameters =  config['parameters']
    
summarize_question = parameters.getboolean('summarize_question')
summarize_answer = parameters.getboolean('summarize_answer')
normalize = parameters.get('normalize')
bert_model = parameters.get('bert_model')    
freeze_bert = parameters.getboolean('freeze_bert') # if True, freeze the encoder weights and only update the classification layer weights
maxlen = parameters.getint('maxlen') 
bs = parameters.getint('bs')  # batch size
iters_to_accumulate = parameters.getint('iters_to_accumulate')  # the gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate. If set to "1", you get the usual batch size
lr = parameters.getfloat('lr')  # learning rate
epochs = parameters.getint('epochs')  # number of training epochs
summarize = any((summarize_question, summarize_answer))


# Read test data
print("Reading test data...")
path_to_df_test = "fyp/comment_ranking_model/df_test.csv"
#path_to_df_test = "fyp/comment_ranking_model/df_reddit_test.csv"
df_test = pd.read_csv(path_to_df_test)
test_set = CustomDataset(df_test, maxlen, bert_model)
test_loader = DataLoader(test_set, batch_size=bs, num_workers=5)


# Prepare for testing
print("Prepare test data...")
path_to_text_file = f'fyp/comment_ranking_model/results/{normalize}_test.txt'  # path to the file with test_df probabilities
path_to_output_file = f'fyp/comment_ranking_model/results/{normalize}_MSELoss()_output.txt'  # path to the file with prediction probabilities
print()

#w = open(f"fyp/comment_ranking_model/results/{normalize}_test.txt", 'w')
#w.writelines(str(scores)+'\n' for scores in df_test["score"].tolist())
#w.close()

#if not path.isfile(path_to_output_file):    
if True:    
    # Load model
    path_to_model = "fyp/comment_ranking_model/models/roberta-base_lf_MSELoss()_lr_2e-06_val_loss_0.16045_ep_6.pt"
    print("Loading the weights of the model...")
    model = SentencePairClassifier(bert_model)
    if torch.cuda.device_count() > 1:  # if multiple GPUs
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path_to_model))
    model.to(device)
    print("start testing")
    test_prediction(net=model,
                    device=device,
                    dataloader=test_loader,
                    with_labels=True,  # set the with_labels parameter to False if your want to get predictions on a dataset without labels
                    result_file=path_to_output_file)


labels_test = pd.read_csv(path_to_text_file, header=None)[0]  # true labels
probs_test = pd.read_csv(path_to_output_file, header=None)[0]  # prediction probabilities
filename = f"fyp/comment_ranking_model/results/{datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}_{bert_model}_{normalize}_summarize_is_{summarize}"
print()


#compute_metric(labels_test, probs_test, filename)
#plot_graph(bert_model, normalize, labels_test, probs_test, filename)
