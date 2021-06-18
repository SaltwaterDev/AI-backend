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
from scipy.special import softmax
from sklearn import metrics
from matplotlib import pyplot as plt

#for hyperparameter optimzation
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from tempfile import TemporaryFile

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


def train_bert(parameterization):
    
   # Train the model
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    global net
    net = SentencePairClassifier(bert_model, freeze_bert=freeze_bert)


    if torch.cuda.device_count() > 1:  # if multiple GPUs
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    net.to(device)

    global best_loss
    best_loss = np.Inf
    best_ep = 1
    nb_iterations = len(train_loader)
    print_every = nb_iterations // 5  # print the training loss 5 times per epoch
    iters = []
    train_losses = []
    val_losses = []
    
    t_total = (len(train_loader) // parameterization.get("iters_to_accumulate")) * epochs  # Necessary to take into account Gradient accumulation
    opti = AdamW(net.parameters(), lr=parameterization.get("lr"), weight_decay=1e-2)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

    print("epochs: ", epochs)
    print("iters_to_accumulate: ", parameterization.get("iters_to_accumulate"))
    print("lr: ", parameterization.get("lr"))
    
    scaler = GradScaler()

    for ep in range(epochs):

        net.train()
        running_loss = 0.0
        for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(train_loader)):

            # Converting to cuda tensors
            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
    
            # Enables autocasting for the forward pass (model + loss)
            with autocast():
                # Obtaining the logits from the model
                logits = net(seq, attn_masks, token_type_ids)

                # Computing loss
                loss = criterion(logits.squeeze(-1), labels.float())
                loss = loss / parameterization.get("iters_to_accumulate")  # Normalize the loss because it is averaged

            # Backpropagating the gradients
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()

            if (it + 1) % parameterization.get("iters_to_accumulate") == 0:
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
                print("Iteration {}/{} of epoch {} complete. Loss : {} "
                      .format(it+1, nb_iterations, ep+1, running_loss / print_every))

                running_loss = 0.0

                
        val_loss = evaluate_loss(net, device, criterion, val_loader)  # Compute validation loss
        val_losses.append(val_loss)
        print()
        print("Epoch {} complete! Validation Loss : {}".format(ep+1, val_loss))

        if val_loss < best_loss:
            print("Best validation loss improved from {} to {}".format(best_loss, val_loss))
            print()
            net_copy = copy.deepcopy(net)  # save a copy of the model
            best_loss = val_loss
            best_ep = ep + 1

    

    #plot the model
    plt.plot(val_losses)
    plt.savefig("epoch_val_loss", bbox_inches='tight')
    plt.clf()
    

    del loss
    torch.cuda.empty_cache()
    return best_loss

   
def run():
    
    # load parameters
    print("load parameters")
    config = configparser.ConfigParser()
    config.read('fyp/comment_ranking_model/parameter.ini')
    parameters =  config['parameters']
    
    summarize_question = parameters.getboolean('summarize_question')
    summarize_answer = parameters.getboolean('summarize_answer')
    normalize = parameters.get('normalize')
    global bert_model
    bert_model = parameters.get('bert_model')
    global freeze_bert
    freeze_bert = parameters.getboolean('freeze_bert') # if True, freeze the encoder weights and only update the classification layer weights
    maxlen = parameters.getint('maxlen')  # maximum length of the tokenized input sentence pair : if greater than "maxlen", the input is truncated and else if smaller, the input is padded
    bs = parameters.getint('bs')  # batch size
    iters_to_accumulate = parameters.getint('iters_to_accumulate')  # the gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate. If set to "1", you get the usual batch size
    lr = parameters.getfloat('lr')  # learning rate
    global epochs
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
    global train_loader 
    train_loader = DataLoader(train_set, batch_size=bs, num_workers=5)
    global val_loader
    val_loader = DataLoader(val_set, batch_size=bs, num_workers=5)
    
  
    
    # global criterion = nn.BCEWithLogitsLoss()
    global criterion
    criterion = poissonLoss
    global num_warmup_steps
    num_warmup_steps = 0 # The number of steps for the warmup phase.
    num_training_steps = epochs * len(train_loader)  # The total number of training steps
    
    
    print("optimization start (training)\n")
        
    # hyparameter optimize
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-8, 1e-2], "log_scale": True},
            {"name": "bs", "type": "range", "bounds": [4, 128]},
            {"name": "iters_to_accumulate", "type": "range", "bounds": [1, 16]},
            # more...
        ],

        evaluation_function=train_bert,
        objective_name='accuracy',
    )

    print("best parameters:")
    print(best_parameters)
    means, covariances = values
    print("means: ", means)
    print("covariances: ", covariances)

    # Saving the model
    # path_to_model='fyp/comment_ranking_model/models/{}_lr_{}_val_loss_{}_ep_{}.pt'.format(bert_model, lr, round(best_loss, 5), best_ep)
    path_to_model='fyp/comment_ranking_model/models/{}_{}.pt'.format(bert_model, datetime.today().strftime('%Y-%m-%d-%H-%M'))
    torch.save(model.state_dict(), path_to_model)
    print("The model has been saved in {}".format(path_to_model))

    best_objectives = np.array([[trial.objective_mean*100 for trial in experiment.trials.values()]])
    np.save("best_objectives", best_objectives)

    
    """
    # might delete in the future in this file
    
    # evaluation
    path_to_output_file = f'fyp/comment_ranking_model/results/{normalize}_output.txt'

    # write test data into .txt file for seperated prediction
    w = open(f"fyp/comment_ranking_model/results/{normalize}_test.txt", 'w')
    w.writelines(str(label)+'\n' for label in df_test["label"].tolist())
    w.close()
       
    
    model = SentencePairClassifier(bert_model)
    if torch.cuda.device_count() > 1:  # if multiple GPUs
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    print()
    print("Loading the weights of the model...")
    model.load_state_dict(torch.load(path_to_model))
    model.to(device)

    
    print("Reading test data...")
    test_set = CustomDataset(df_test, maxlen, bert_model)
    global test_loader
    test_loader = DataLoader(test_set, batch_size=bs, num_workers=5)
    
    print("Predicting on test data...")
    
    test_prediction(net=model, device=device, dataloader=test_loader, with_labels=True,  # set the with_labels parameter to False if your want to get predictions on a dataset without labels
                    result_file=path_to_output_file)
    print()
    print("Predictions are available in : {}".format(path_to_output_file))
    
    
    # compute metric and plot graph
    summarize = any((summarize_question, summarize_answer))
    labels_test = df_test['label']  # true labels
    probs_test = pd.read_csv(path_to_output_file, header=None)[0]  # prediction probabilities
    filename = f"graph_metric/{datetime.today().strftime('%Y-%m-%d')}_{bert_model}_{normalize}_summarize_is_{summarize}"
    
    compute_metric(labels_test,probs_test, filename)
    plot_graph(bert_model, normalize, labels_test, probs_test, filename)
"""
    
    

if __name__ == "__main__":
    run()