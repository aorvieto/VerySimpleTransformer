from models import *
from models.former import util
import ast
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchtext.legacy
import numpy as np
from argparse import ArgumentParser
import tqdm, math
from dotmap import DotMap
from utils import *
from utils.torch_utils import get_grad_norm_squared
import wandb

# Used for converting between nats and bits


def run_toy_transformer_config(config_dict,devices):

    ########### Setting Up Writer ###########
    config = DotMap(config_dict) 
    if config.use_wandb: # if we use wandb then selecting the project folder
        run=wandb.init(project="Lion and Adam",config=config_dict)

    ########### Setting Up Seed ###########
    # this does not mean the run is deterministic
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    ########### Setting Up GPU ########### 
    gpu_ids = ast.literal_eval(devices)
    torch.cuda.set_device(gpu_ids[0])
    device = 'cuda'

    ########### Upload of text dataset ########### 
    batch_size = config.optimizer.bs
    embedding_size=128
    vocab_size = 50_000
    max_length = 512
    TEXT = torchtext.legacy.data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = torchtext.legacy.data.Field(sequential=False)
    train, test = torchtext.legacy.datasets.IMDB.splits(TEXT, LABEL)
    TEXT.build_vocab(train, max_size=vocab_size - 2)
    LABEL.build_vocab(train)
    train_iter, test_iter = torchtext.legacy.data.BucketIterator.splits((train, test), batch_size=batch_size, device=util.d())
    print(f'- nr. of training examples {len(train_iter)}')
    print(f'- nr. of test examples {len(test_iter)}')

    ########### Creating the model ########### 
    # for big models, using multiple GPUs is better. for IMDB, its better to use one
    num_heads = 8
    depth = 6
    model = torch.nn.DataParallel(former.CTransformer(emb=embedding_size, heads=num_heads, depth=depth, seq_length=max_length, num_tokens=vocab_size, num_classes=2, max_pool=True),gpu_ids).cuda()
    if torch.cuda.is_available(): model.cuda()

    ########### Selecting the optimizer ########### 
    # The first two are from torch, else we use a custom one in the train loop. its very simple.
    lr = config.optimizer.lr
    if config.optimizer.method == 'adam_torch':
        opt = torch.optim.Adam(lr=lr, params=model.parameters())
    if config.optimizer.method == 'sgd_torch':
        opt = torch.optim.SGD(lr=lr, params=model.parameters(),momentum=0.9)
    else:
        #init variables for adam moving averages
        avg_grad_1, avg_grad_2 = [], []
        for p in model.parameters():
            avg_grad_1.append(None)
            avg_grad_2.append(None) 

    ########### Selecting the scheduler ########### 
    # we do not use this for now
    #sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (lr_warmup / batch_size), 1.0))

    ########### Training loop ########### 
    iteration = 0
    for e in range(config.epochs):

        print(f'\n epoch {e}')
        model.train(True)

        loss_avg = 0
        for batch in tqdm.tqdm(train_iter):

            ########### Getting gradients ########### 
            model.zero_grad()
            input = batch.text[0].to(device, non_blocking=True)
            label = (batch.label - 1).to(device, non_blocking=True)
            if input.size(1) > max_length:
                input = input[:, :max_length]
            out = model(input)
            loss = F.nll_loss(out, label)
            loss.backward()

            #logging local train stats
            if config.use_wandb: #logging
                wandb.log({"lr":lr, "train_loss_local": loss.item(), "grad_norm_square":get_grad_norm_squared(model)})

            ########### Optimizer step ########### 
            if config.optimizer.method == 'adam_torch':
                opt.step()

            elif config.optimizer.method == 'sgd_torch':
                opt.step()

            elif config.optimizer.method == 'adam': #example of writing an optimizer
                with torch.no_grad(): # very important
                    #standard adam parameters
                    beta1 = 0.9
                    beta2 = 0.999

                    # parameters have to be updated with a fpr loop (this is fast)
                    for p_idx, p in enumerate(model.parameters()):
                        grad_p = p.grad
                        square_grad_p = grad_p**2
                        if None in (avg_grad_1[p_idx], avg_grad_2[p_idx]): #init
                            avg_grad_1[p_idx] =  grad_p
                            avg_grad_2[p_idx] = square_grad_p
                        avg_grad_2[p_idx] = beta2 * avg_grad_2[p_idx] + (1-beta2)*square_grad_p
                        avg_grad_1[p_idx] = beta1 * avg_grad_1[p_idx] + (1-beta1)*grad_p

                        #parameter update
                        preconditioner = 1/(1e-8 + avg_grad_2[p_idx].sqrt())
                        new_val = p - lr * preconditioner * avg_grad_1[p_idx]
                        p.copy_(new_val)

            elif config.optimizer.method == 'sign_sgd': 
                with torch.no_grad(): 
                    # parameters have to be updated with a fpr loop (this is fast)
                    for p_idx, p in enumerate(model.parameters()):
                        new_val = p - lr * torch.sign(p.grad)
                        p.copy_(new_val)

            elif config.optimizer.method == 'signum': 
                with torch.no_grad(): 
                    #standard lion parameters
                    beta1 = 0.9

                    # parameters have to be updated with a fpr loop (this is fast)
                    for p_idx, p in enumerate(model.parameters()):
                        grad_p = p.grad
                        if avg_grad_1[p_idx] is None: #init
                            avg_grad_1[p_idx] =  grad_p
                        avg_grad_1[p_idx] = beta1 * avg_grad_1[p_idx] + (1-beta1)*grad_p

                        #parameter update
                        new_val = p - lr * torch.sign(avg_grad_1[p_idx])
                        p.copy_(new_val)

            elif config.optimizer.method == 'lion': 
                with torch.no_grad(): 
                    #standard lion parameters
                    beta1 = 0.95
                    beta2 = 0.98

                    # parameters have to be updated with a fpr loop (this is fast)
                    for p_idx, p in enumerate(model.parameters()):
                        grad_p = p.grad
                        if None in (avg_grad_1[p_idx], avg_grad_2[p_idx]): #init
                            avg_grad_1[p_idx] =  grad_p
                            avg_grad_2[p_idx] = grad_p
                        avg_grad_1[p_idx] = beta1 * avg_grad_2[p_idx] + (1-beta1)*grad_p
                        avg_grad_2[p_idx] = beta2 * avg_grad_2[p_idx] + (1-beta2)*grad_p

                        #parameter update
                        new_val = p - lr * torch.sign(avg_grad_1[p_idx])
                        p.copy_(new_val)
            else:
                print('optimizer not found')

            # updating train loss average
            with torch.no_grad():
                loss_avg =loss_avg + loss.item()/len(train_iter)
            #sch.step() # only if scheduler is used

        ########### Testing at every epoch ########### 
        with torch.no_grad():
            model.train(False)
            tot, cor= 0.0, 0.0
            for batch in test_iter:
                input = batch.text[0]
                label = batch.label - 1
                if input.size(1) > max_length:
                    input = input[:, :max_length]
                out = model(input).argmax(dim=1)
                tot += float(input.size(0))
                cor += float((label == out).sum().item())
            acc = cor / tot
            print(f'-- "test accuracy {acc:.3}')

        #saving to wandb
        if config.use_wandb:
            wandb.log({"train_loss_epoch":loss_avg, "test_acc_epoch":acc})

    ########### Closing Writer ###########  
    if config.use_wandb:
        run.finish()
        wandb.finish()
