# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 10:37:22 2019

@author: Fede
"""
import numpy as np
from hyperparams_initialization import hyperparams_initialization
from extraction_performances import extraction_performances
from tqdm import tqdm_notebook as tqdm
import torch 
import torch.nn as nn
import torch.optim as optim
import random

def IAHOS(rounds,method,limits,attempts,variables,iterations,net,
          training_set,training_labels,validation_set,validation_labels,dim):

    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    net.to(device)
    
    training_set = torch.tensor(training_set).float().view(-1,7).cuda()
    training_labels = torch.tensor(validation_set).float().view(-1,len(e)).cuda()
    
    #Learning rate
    lr = 0.02    
    #update
    train_loss_log = []
    test_loss_log = []
    Ni = 7
    No = len(e)

    for r in range(rounds):
        print ("Round ",r+1," of ",rounds)
        num_epochs=1
        if r>0:
            attempts=2
        hm,p_values = hyperparams_initialization(attempts,variables,method,limits,r)
        
        training_accuracy = []
        validation_accuracy = []
        for i in tqdm(range(iterations)):
            net = Net(Ni,i,hm,No,'train')
            loss_fn = nn.BCEWithLogitsLoss()
            net.train()
            for num_ep in range(num_epochs):
                net.zero_grad()
                out = net(training_set)
                loss = loss_fn(out,training_labels)
                loss.backward()
                for p in net.parameters():
                    p.data.sub_(p.grad.data * lr)
                train_loss_log.append(float(loss.data))
        general_perf,general_perf2=extraction_performances(validation_accuracy,training_accuracy,variables,iterations,attempts)
        del training_accuracy
        del validation_accuracy
        
        if r==0:
            old_general_perf = general_perf
            old_general_perf2 = general_perf2
            temp_general_perf=general_perf
            temp_general_perf2 = general_perf2
        else:
            new_p_values = []
            for s in range(variables):
                a = []
                a.append(p_values[s])
                b = []
                b.append(temp_p_values[s])
                c=np.concatenate((a,b),axis=1)
                new_p_values.append(np.sort(c)[0])
            p_values = new_p_values
            new_values = []
            new_values2 = []
            for t in range(len(temp_general_perf)):
                new_values.append(np.insert(temp_general_perf[t],indeces[t][1],general_perf[t]))
                new_values2.append(np.insert(temp_general_perf2[t],indeces[t][1],general_perf2[t]))
            temp_general_perf=new_values
            temp_general_perf2=new_values2
        
        indeces = []
        best_hp=[]
        final = []
        for i in range(variables):
            values = []
            index = []
            index.append(np.argsort(temp_general_perf[i])[-1])
            values.append(p_values[i][np.argsort(temp_general_perf[i])[-1]])
            final.append(p_values[i][np.argsort(temp_general_perf[i])[-1]])
            if np.argsort(temp_general_perf[i])[-1]>0:
                values.append(p_values[i][np.argsort(temp_general_perf[i])[-1]-1])
                index.append(np.argsort(temp_general_perf[i])[-1]-1)
            else:
                values.append(p_values[i][np.argsort(temp_general_perf[i])[-1]+1])
                index.append(np.argsort(temp_general_perf[i])[-1]+1)
            best_hp.append(np.sort(values))
            indeces.append(np.sort(index))
        limits = best_hp
        temp_p_values = p_values
    
    np.save('final',final)
    return temp_general_perf,temp_general_perf2,old_general_perf,old_general_perf2,final