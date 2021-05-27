from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.cuda.amp import autocast
from torch.utils.data import Dataset
import torch.nn as nn
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader


class CustomDataset(Dataset):

    def __init__(self, data, maxlen, bert_model="albert-base-v2", with_labels=True):

        self.data = data  # pandas dataframe
        #Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model, use_fast=False)       
        
        self.maxlen = maxlen
        self.with_labels = with_labels 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent1 = str(self.data.loc[index, 'sentence1'])
        # later try to merge sent1 with presona. i.e. sent1 = sent1 + persona. create a new function
        sent2 = str(self.data.loc[index, 'sentence2'])

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(sent1, sent2, 
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=self.maxlen,  
                                      return_tensors='pt',  # Return torch.Tensor objects
                                      return_token_type_ids=True)
        
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens
        

        if self.with_labels:  # True if the dataset has labels (score)
            score = self.data.loc[index, 'score']
            return token_ids, attn_masks, token_type_ids, score  
        else:
            return token_ids, attn_masks, token_type_ids
        
        

        

        
class SentencePairClassifier(nn.Module):

    def __init__(self, bert_model="albert-base-v2", freeze_bert=False):
        super(SentencePairClassifier, self).__init__()
        #  Instantiating BERT-based model object
        #  next line is for reference
        # config = AutoConfig.from_pretrained(bert_model, max_position_embeddings=max_position_embeddings)
        # default: self.bert_layer = AutoModel.from_pretrained(bert_model)
        self.bert_layer = AutoModel.from_pretrained(bert_model, output_attentions=True)


        #  Fix the hidden-state size of the encoder outputs (If you want to add other pre-trained models here, search for the encoder output size)
        if bert_model == "albert-base-v2":  # 12M parameters
            hidden_size = 768
        elif bert_model == "albert-large-v2":  # 18M parameters
            hidden_size = 1024
        elif bert_model == "albert-xlarge-v2":  # 60M parameters
            hidden_size = 2048
        elif bert_model == "albert-xxlarge-v2":  # 235M parameters
            hidden_size = 4096
        elif bert_model == "bert-base-uncased": # 110M parameters
            hidden_size = 768
        elif bert_model == "bert-large-uncased":  # 336M parameters
            hidden_size = 1024
        elif bert_model == "roberta-base": # 125M parameters
            hidden_size = 768
        elif bert_model == "roberta-large": # 355M parameters
            hidden_size = 1024

        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Regression layer
        self.reg_layer = nn.Linear(hidden_size, 1)
        #self.cls_layer = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(p=0.5)

    @autocast()  # run in mixed precision
    def forward(self, input_ids, attn_masks, token_type_ids):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
            -token_type_ids : Tensor containing token type ids to be used to identify sentence1 and sentence2
        '''

        # Feeding the inputs to the BERT-based model to obtain contextualized representations
        cont_reps, pooler_output, attentions = self.bert_layer(input_ids, attn_masks, token_type_ids, return_dict=False)


        # Feeding to the regression layer the last layer hidden-state of the [CLS] token further processed by a
        # Linear Layer and a Tanh activation. The Linear layer weights were trained from the sentence order prediction (ALBERT) or next sentence prediction (BERT)
        # objective during pre-training.
        #output = self.relu(pooler_output)
        output = self.dropout(pooler_output)
        output = self.reg_layer(output)
        #output = self.cls_layer(output)
        #output = self.relu(output)
        
        return output, attentions
    
def poissonLoss(xbeta, y):
    """Custom loss function for Poisson model."""
    loss=torch.mean(torch.exp(xbeta)-y*xbeta)
    return loss

def get_probs_from_logits(logits):
    """
    Converts a tensor of logits into an array of probabilities by applying the sigmoid function
    """
    probs = torch.sigmoid(logits.unsqueeze(-1))
    return probs.detach().cpu().numpy()

def test_prediction(net, device, dataloader, with_labels=True, result_file=None):
    """
    Predict the score on a dataset with or without labels and print the result in a file
    """
    net.eval()
    w = open(result_file, 'w')
    predict_all = []

    with torch.no_grad():
        if with_labels:
            for seq, attn_masks, token_type_ids, _ in tqdm(dataloader):
                seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
                output, _ = net(seq, attn_masks, token_type_ids)
                predict_all += output.squeeze(-1).tolist()
        else:
            for seq, attn_masks, token_type_ids in tqdm(dataloader):
                seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
                output = net(seq, attn_masks, token_type_ids)
                predict_all += output.squeeze(-1).tolist()

    w.writelines(str(output)+'\n' for output in predict_all)
    w.close()
    print("Predictions are available in : {}".format(result_file))
    
def test_ranking():
    pass
    
    
def pairing(post_content, comment):
    df_prediction = pd.DataFrame()

    prediction = {"sentence1": post_content, "sentence2": comment, "label": np.nan}
    series_prediction = pd.Series(prediction)

    df_prediction = df_prediction.append(series_prediction, ignore_index=True)
    prediction_set = CustomDataset(df_prediction, maxlen, with_labels=False, bert_model=bert_model)
    prediction_loader = DataLoader(prediction_set, num_workers=5)

    net.eval()
    with torch.no_grad():
        for seq, attn_masks, token_type_ids in prediction_loader:
            seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
            score = net(seq, attn_masks, token_type_ids)
            return score
      
       
def compute_metric(labels_test, predict_test, filename):
    print(labels_test, predict_test)
    metric = {'mean_absolute_error': metrics.mean_absolute_error(labels_test, predict_test), 
        'mean_squared_error': metrics.mean_squared_error(labels_test, predict_test), 
        'explained_variance_score': metrics.explained_variance_score(labels_test, predict_test),
           'r2_score': metrics.r2_score(labels_test, predict_test)} 
   
    # create series from dictionary 
    metric = pd.Series(metric) 
    metric.to_csv(filename, index=True)

    
def plot_graph(bert_model, normalize=None, labels_test=None, predict_test=None, filename=None):
    plt.title(bert_model + " " + normalize) 
    plt.plot(labels_test, 'o')
    plt.plot(predict_test, 'o')
    plt.legend(["true", "pred"])
    plt.grid()
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()


def rank_comment(post: str, raw_comments: list):
    ranking_comments = []
    for raw_comment in raw_comments:
        comment = {}
        comment['content'] = raw_comment
        comment['score'] = pairing(post, raw_comment)
        ranking_comments.append(comment)
    return ranking_comments, sorted(ranked_comment, key=lambda i: i['score'], reverse=True)
    
    