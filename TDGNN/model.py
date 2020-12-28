import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict
from sklearn import metrics
import torch.nn.functional as F
from encoders import Encoder
from aggregators import MeanAggregator
import gc
import argparse

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""
#file="employees"
#name="origin"
dimension=128
class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc,name):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()
        self.name=name
        if name!="activation" and name!="origin":
            self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        else:
            self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim*2))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        #original
        #embeds = torch.cat((embeds[:,0],embeds[:,1]),0).unsqueeze(1)
        #mean
        if self.name=="mean":
            embeds=(embeds[:,0]+embeds[:,1])/2
            embeds = embeds.unsqueeze(1)
        #hadamard
        elif self.name=='had':
            embeds=embeds[:,0].mul(embeds[:,1])
            embeds = embeds.unsqueeze(1)
        #weight-l1
        elif self.name=="w1":
            embeds=torch.abs(embeds[:,0]-embeds[:,1])
            embeds = embeds.unsqueeze(1)
        #weight-l2
        elif self.name=="w2":
            embeds=torch.abs(embeds[:,0]-embeds[:,1]).mul(torch.abs(embeds[:,0]-embeds[:,1]))
            embeds = embeds.unsqueeze(1)
        #activation
        elif self.name=='activation':
            embeds = torch.cat((embeds[:, 0], embeds[:, 1]), 0).unsqueeze(1)
            embeds = F.relu(embeds)
        elif self.name=='origin':
            embeds = torch.cat((embeds[:, 0], embeds[:, 1]), 0).unsqueeze(1)

        scores = self.weight.mm(embeds)

        return scores.t()

    def loss(self, labels,agg1,agg2,edge):
        agg1.time = edge[0][2]
        agg2.time = edge[0][2]
        scores = F.softmax(self.forward([edge[0][0],edge[0][1]]),dim=1)
        predict_y = []
        predict_y.append(scores.cpu().detach().numpy()[0][1])
        for i in range(1, len(edge)):
            #print(i)
            agg1.time = edge[i][2]
            agg2.time = edge[i][2]
            temp = self.forward([edge[i][0], edge[i][1]])
            temp=F.softmax(temp,dim=1)
            predict_y.append(temp.cpu().detach().numpy()[0][1])
            scores=torch.cat((scores, temp), 0)
        return self.xent(scores, labels),predict_y

def load_data(input_node,input_edge_train,input_edge_test):
    #feat_data=np.genfromtxt("../../"+file+"/feature_random_"+file+".txt")[:,1:]
    feat_data = np.genfromtxt(input_node)[:, 1:]
    label_train=[]
    edge_train=[]
    adj_list={}
    adj_time={}
    for i in range(0,len(feat_data)):
        adj_list[i] = []
        adj_time[i] = []
    #with open("../../"+file+"/edge_train_"+file) as fp:
    with open(input_edge_train) as fp:
        for i, line in enumerate(fp):
            temp=line.split(" ")
            left=int(temp[0])-1
            right=int(temp[1])-1
            label_train.append(int(temp[3]))
            edge_train.append([left,right,int(temp[2])])
            if int(temp[3]) == 1:
                if left not in adj_list:
                    adj_list[left]=[right]
                    adj_time[left]=[int(temp[2])]
                else:
                    adj_list[left].append(right)
                    adj_time[left].append(int(temp[2]))
    label_test = []
    edge_test = []
    #with open("../../"+file+"/edge_test_"+file) as fp:
    with open(input_edge_test) as fp:
        for i, line in enumerate(fp):
            temp=line.split(" ")
            left=int(temp[0])-1
            right=int(temp[1])-1
            label_test.append(int(temp[3]))
            edge_test.append([left, right,int(temp[2])])
            if int(temp[3])==1:
                if left not in adj_list:
                    adj_list[left]=[right]
                    adj_time[left]=[int(temp[2])]
                else:
                    adj_list[left].append(right)
                    adj_time[left].append(int(temp[2]))
    return feat_data,edge_train,label_train,edge_test,label_test,adj_list,adj_time
def run_data(input_node,input_edge_train,input_edge_test,output_file,name):
    feat_data,edge_train,label_train,edge_test,label_test,adj_lists,adj_time=load_data(input_node,input_edge_train,input_edge_test)
    print("Finish Loading Data")
    features = nn.Embedding(len(feat_data), 1000)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    #features.cuda()
    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 1000, dimension, adj_lists,adj_time, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, dimension, adj_lists, adj_time,agg2,
                   base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5
    enc2.last=True
    graphsage = SupervisedGraphSage(2, enc2,name)
    #graphsage.cuda()
    #f=open('result_test'+name+'_'+file+"_"+str(dimension),'a+')
    f = open(output_file, 'a+')
    f.write("Training\n")
    #f.close()
    for epoch in range(0,20):
        #f = open('result_test'+name+'_'+file+"_"+str(dimension), 'a+')
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)
        optimizer.zero_grad()
        f.write("epoch "+str(epoch)+"\n")
        loss,predict_y = graphsage.loss(Variable(torch.LongTensor(label_train)),agg1,agg2,edge_train)
        print("AUC: "+str(metrics.roc_auc_score(label_train, predict_y))+"\n")
        f.write("AUC: "+str(metrics.roc_auc_score(label_train, predict_y))+"\n")
        loss.backward()
        optimizer.step()
        #f.close()
        #gc.collect()
    #f = open('result_test'+name+'_'+file+"_"+str(dimension), 'a+')
    f.write("Testing\n")
    loss, predict_y = graphsage.loss(Variable(torch.LongTensor(label_test)), agg1, agg2, edge_test)
    f.write("AUC: " + str(metrics.roc_auc_score(label_test, predict_y)) + "\n")
    predict_y1=[]
    for i in range(0,len(predict_y)):
        if predict_y[i]>0.5:
            predict_y1.append(1)
        else:
            predict_y1.append(0)

    f.write("Micro-f1 score: " + str(metrics.f1_score(label_test, predict_y1,average="micro")) + "\n")
    f.write("Macro-f1 score: " + str(metrics.f1_score(label_test, predict_y1, average="macro")) + "\n")
    f.write("recall: "+str(metrics.recall_score(label_test, predict_y1)) + "\n")
    f.close()
if __name__ == "__main__":
    array_function=['mean','had','w1','w2','origin']
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_node', '--input_node', type=str)
    parser.add_argument('-input_edge_train', '--input_edge_train', type=str)
    parser.add_argument('-input_edge_test', '--input_edge_test', type=str)
    parser.add_argument('-output_file', '--output', type=str)
    parser.add_argument('-aggregate_function', '--aggregate_function', type=str)
    parser.add_argument('-hidden_dimension', '--hidden_dimension', type=int)
    args = parser.parse_args()
    input_node = args.input_node
    input_edge_train=args.input_edge_train
    input_edge_test=args.input_edge_test
    output_file = args.output
    aggregate_function=args.aggregate_function
    dimension=args.hidden_dimension
    if aggregate_function not in array_function:
        print("----------------Please Use Right Function----------------")
    else:
        run_data(input_node,input_edge_train,input_edge_test,output_file,aggregate_function)
