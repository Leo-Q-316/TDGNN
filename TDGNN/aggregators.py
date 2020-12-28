import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import random
import numpy as np
import gc
"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False, gcn=False,time=0):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        time --- the current time
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        self.time=time
        
    def forward(self, nodes, to_neighs,time_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        time_neighs --- number of neighbors to sample with its appearance's time. No sampling if None.
        """
        samp_neighs=[]
        dic_temp={}
        dic_edge={}
        total_value={}
        nodes=np.array(nodes)
        for i in range(0,len(to_neighs)):
            samp_neighs.append([int(nodes[i])])
            dic_edge[i]={}
            dic_temp[int(nodes[i])]=1
            total_value[i]=0
            for j in range(0,len(to_neighs[i])):
                if(time_neighs[i][j]<=self.time):
                    if(to_neighs[i][j] not in dic_edge[i]):
                        dic_edge[i][to_neighs[i][j]]=math.exp((time_neighs[i][j]-self.time)/100)
                    else:
                        dic_edge[i][to_neighs[i][j]]+=math.exp((time_neighs[i][j]-self.time)/100)
                    total_value[i]+=float(math.exp((time_neighs[i][j]-self.time)/100))
                    samp_neighs[-1].append(to_neighs[i][j])
                    dic_temp[to_neighs[i][j]]=1
            if total_value[i]==0:
                total_value[i]=1
        unique_nodes_list = list([key for key in dic_temp])
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))

        # column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        # row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        # mask[row_indices, column_indices] = 1
        for key1 in dic_edge:
            for key2 in dic_edge[key1]:
                mask[key1,unique_nodes[key2]]=dic_edge[key1][key2]/total_value[key1]
        for i in range(0,len(nodes)):
            #print(mask[i][unique_nodes[nodes[i]]])
            mask[i,unique_nodes[nodes[i]]]=mask[i,unique_nodes[nodes[i]]]+1
        if self.cuda:
            mask = mask.cuda()
        #print("Type-----------------------------------------:",mask.type)
        num_neigh = mask.sum(1, keepdim=True)
        #num_neigh = mask.sum(1)
        #print("num_neigh:"+str(num_neigh))
        mask = mask.div(num_neigh)
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        #print("embed_matrix:"+str(embed_matrix))
        to_feats = mask.mm(embed_matrix)
        del mask,embed_matrix,dic_edge,dic_temp,total_value
        # gc.collect()
        return to_feats
