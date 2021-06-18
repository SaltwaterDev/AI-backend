import torch
import torch.nn as nn
import os
from datetime import datetime
import configparser
import matplotlib.pyplot as plt
import copy
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_metric, list_metrics
from sklearn.model_selection import train_test_split
from summarizer import Summarizer
from sklearn import metrics
from matplotlib import pyplot as plt

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


    
def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    

def evaluate_loss(net, device, criterion, dataloader):
    net.eval()

    mean_loss = 0
    count = 0

    with torch.no_grad():
        for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(dataloader)):
            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
            logits = net(seq, attn_masks, token_type_ids)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            count += 1

    return mean_loss / count


def train_bert(device, net, bert_model, criterion, opti, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate):

    nb_iterations = len(train_loader)
    print_every = nb_iterations // 5  # print the training loss 5 times per epoch
    best_loss = np.Inf
    ep = 0
    best_ep = 1
    iters = []
    train_losses = []
    val_losses = []

    scaler = GradScaler()    
    
    for ep in range(epochs):
        
        net.train()
        running_loss = 0.0
        for it, (seq, attn_masks, token_type_ids, scores) in enumerate(tqdm(train_loader)):

            # Converting to cuda tensors
            seq, attn_masks, token_type_ids, scores = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), scores.to(device)
    
            # Enables autocasting for the forward pass (model + loss)
            with autocast():
                # Obtaining the logits from the model
                logits = net(seq, attn_masks, token_type_ids)

                # Computing loss
                loss = criterion(logits.squeeze(-1), scores.float())
                loss = loss / iters_to_accumulate  # Normalize the loss because it is averaged

            # Backpropagating the gradients
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()

            if (it + 1) % iters_to_accumulate == 0:
                # Optimization step
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, opti.step() is then called,
                # otherwise, opti.step() is skipped.
                scaler.step(opti)
                # Updates the scale for next iteration.
                scaler.update()
                # Adjust the learning rate based on the number of iterations.
                lr_scheduler.step()
                # Clear gradients
                opti.zero_grad()


            running_loss += loss.item()
 
            if (it + 1) % print_every == 0:  # Print training loss information
                print()
                mean_loss = running_loss / print_every
                print("Iteration {}/{} of epoch {} complete. Loss : {} "
                      .format(it+1, nb_iterations, ep+1, mean_loss))

                running_loss = 0.0
            

        val_loss = evaluate_loss(net, device, criterion, val_loader)  # Compute validation loss
        val_losses.append(val_loss)
        train_losses.append(mean_loss)

        print()
        print("Epoch {} complete! Validation Loss : {}".format(ep+1, val_loss))

        if val_loss < best_loss:
            print("Best validation loss improved from {} to {}".format(best_loss, val_loss))
            print()
            net_copy = copy.deepcopy(net)  # save a copy of the model
            best_loss = val_loss
            best_ep = ep + 1

    # Saving the model
    path_to_model='fyp/comment_ranking_model/models/{}_lf_{}_lr_{}_val_loss_{}_ep_{}.pt'.format(bert_model, criterion, lr, round(best_loss, 5), best_ep)
    #torch.save(net_copy.state_dict(), path_to_model)
    torch.save(net.state_dict(), path_to_model)
    print("The model has been saved in {}".format(path_to_model))

    #plot the model
    fig = plt.figure()
    fig.suptitle('train and val loss during training', fontsize=20)
    plt.xlabel('epochs', fontsize=18)
    plt.ylabel('loss', fontsize=16)
    plt.plot(val_losses)
    plt.plot(train_losses)
    fig.savefig(f"{criterion}_epoch_loss", bbox_inches='tight')
    fig.clf()
    

    del loss
    torch.cuda.empty_cache()
    return path_to_model



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
    print("Load preprocessed data...")
    df_train = pd.read_csv("fyp/comment_ranking_model/df_train.csv")
    df_test = pd.read_csv("fyp/comment_ranking_model/df_test.csv")
        
    
    # Creating instances of training and validation set
    print("Reading training data...")
    train_set = CustomDataset(df_train, maxlen, bert_model)
    print("Reading validation data...")
    val_set = CustomDataset(df_test, maxlen, bert_model)
    
    # Creating instances of training and validation dataloaders
    train_loader = DataLoader(train_set, batch_size=bs, num_workers=5)
    val_loader = DataLoader(val_set, batch_size=bs, num_workers=5)
    
    # Train the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = SentencePairClassifier(bert_model, freeze_bert=freeze_bert)


    if torch.cuda.device_count() > 1:  # if multiple GPUs
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    net.to(device)

    #criterion = nn.BCEWithLogitsLoss()
    criterions = [nn.MSELoss()]
    #criterion = nn.MSELoss()
    #criterion = nn.L1Loss()
    #criterion = nn.PoissonNLLLoss()
    #criterion = poissonLoss
    opti = AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    num_warmup_steps = 0 # The number of steps for the warmup phase.
    num_training_steps = epochs * len(train_loader)  # The total number of training steps
    t_total = (len(train_loader) // iters_to_accumulate) * epochs  # Necessary to take into account Gradient accumulation
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

    print("training start\n")
    for criterion in criterions:
        path_to_model = train_bert(device, net, bert_model, criterion, opti, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate)
        
    
        # evaluation
        path_to_output_file = f'fyp/comment_ranking_model/results/{normalize}_{criterion}_output.txt'

        # write test data into .txt file for seperated prediction
        w = open(f"fyp/comment_ranking_model/results/{normalize}_test.txt", 'w')
        w.writelines(str(scores)+'\n' for scores in df_test["score"].tolist())
        w.close()

        print("Reading test data...")
        test_set = CustomDataset(df_test, maxlen, bert_model)
        test_loader = DataLoader(test_set, batch_size=bs, num_workers=5)


        model = SentencePairClassifier(bert_model)
        if torch.cuda.device_count() > 1:  # if multiple GPUs
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        print()
        print("Loading the weights of the model...")
        model.load_state_dict(torch.load(path_to_model))
        model.to(device)

        print("Predicting on test data...")
        test_prediction(net=model, device=device, dataloader=test_loader, with_labels=True,  # set the with_labels parameter to False if your want to get predictions on a dataset without labels
                        result_file=path_to_output_file)
        print()
        print("Predictions are available in : {}".format(path_to_output_file))


        # compute metric and plot graph
        summarize = any((summarize_question, summarize_answer))
        scores_test = df_test['score']  # true scores
        probs_test = pd.read_csv(path_to_output_file, header=None)[0]  # prediction probabilities
        filename = f"graph_metric/{datetime.today().strftime('%Y-%m-%d-%H-%M')}_{bert_model}_{normalize}_summarize_is_{summarize}"

        compute_metric(scores_test,probs_test, filename)
        plot_graph(bert_model, normalize, scores_test, probs_test, filename)

    
    

if __name__ == "__main__":
    run()