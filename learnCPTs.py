#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:13:38 2020

@author: francesca
"""

import os
import pandas as pd
import numpy as np
import pickle
from scipy.optimize import LinearConstraint, minimize
 
path =os.getcwd()
datapath = path + '/data/'


# Select dataset 
# 10:= artificial dataset with noise level at 10%
# 50:= artificial dataset with noise level at 50%
# 'real':= dataser built using real tags (not shown in the paper)
noise = 50
if noise == 'real':
    name = 'user_features_real' 
    outfilename = 'allcpts_real_small.pkl'
else:
    name = 'user_features_noise_at_%d_percent'%noise
    outfilename = 'allcpts%d_small.pkl'%noise
#%% Reduce dataset
movies = pd.read_pickle(datapath+'small_dataset_movies.pkl')

user_features_test_small = pd.read_pickle(datapath+name+'_test.pkl')
user_features_train_small = pd.read_pickle(datapath+name+'_train.pkl')

item_features = pd.read_pickle(datapath+'item_features.pkl')
item_features_small = item_features.loc[movies,:]
item_features_small.to_pickle(datapath+'item_features_small.pkl')

#%%
item_features = item_features
user_features = user_features_train_small

pCgivenIdict = dict()

nitems = item_features.shape[0]
for q in item_features.columns:
    feat = item_features[q]
    if q=='period':
        answers = set(feat)
        pCgivenIdict[q] = pd.get_dummies(feat)
    elif item_features[q].dtypes=='object':
        answers = len(item_features[q].iloc[0])
        pCgivenIdict[q] = item_features[q].apply(lambda x: pd.Series(x,index = range(answers)))
    elif item_features[q].dtypes=='float':
        cpt = pd.DataFrame([1-feat,feat]).transpose()  
        pCgivenIdict[q] = cpt
    else:
        print('problem with ',q )
    
f = open(datapath+'pCgivenIdict.pkl',"wb")
pickle.dump(pCgivenIdict,f)
f.close()

pCgivenIdict_small = dict()
for f in user_features.columns:
    pCgivenIdict_small[f] = pCgivenIdict[f].loc[movies,:].values

s = 1

def assign_counts(x,I,pCgivenI):
    C = pCgivenI.loc[I]
    Ca = C
    nx = np.sum(x)
    counts = np.array([[0.]*len(x)]*len(x))
    if (x*C).sum()>0:
        counts = counts + np.diag(x*C/nx)
        Ca = Ca - x*C
        x = x-x*C
        
    if (x*(1-C)).sum()>0:
        for i in np.where(x*(1-C)!=0)[0]:
            counts[i,np.where(Ca)[0]] = 1/(nx*Ca.sum())
    if counts.sum()<0.999:
        if x.sum()>0:
            print('Wrong counts for item', I)
            return
    return counts

def PQgivenC(collected_answers, pCgivenI, qtype):
    
    # multiple properties and multiple answers
    if qtype=='genre':
        na = len(collected_answers.iloc[0])
        counts = np.eye(na)*s
        for i,r  in collected_answers.iteritems():
            counts += assign_counts(r,i,pCgivenI)
        #check
        nonempty_answers = (collected_answers.apply(lambda x: np.sum(x))!=0).sum()
        if np.abs(nonempty_answers+na*s-counts.sum())>0.1: print('warning')
     
    # single property and single answers
    elif qtype=='period':
        nA = pCgivenI.shape[1]
        I = collected_answers.index.values
        pC = pCgivenI.loc[I,:]
        counts = np.array([[np.nan]*nA]*nA)
        for a in range(nA):
            counts[a,:] = pC.apply(lambda x: x*(collected_answers==a)).sum()
        counts = counts+np.eye(nA)*s    
        nonempty_answers = collected_answers.dropna().count()
        if np.abs(nonempty_answers+nA*s-counts.sum())>0.1: print('warning',q)
        
   
    elif qtype=='tag':
        nA = 2
        I = collected_answers.index.values
        pC = pCgivenI.loc[I,:]
        counts = np.array([[np.nan]*nA]*nA)
        lc = LinearConstraint(np.eye(2),np.zeros(2), np.ones(2))
        # c = minimize(lambda x: -likelihood(x,pC), 0.5,constraints=lc, method = 'trust-constr').x[0]
        c = minimize(lambda x: -likelihood(x,pC), [0,1],constraints=lc, method = 'trust-constr').x
        print(c)
        counts[0,0] = c[1]
        counts[1,0] = 1-c[1]
        counts[0,1] = c[0]
        counts[1,1] = 1-c[0]
   
        nonempty_answers = collected_answers.dropna().count()
         
    return counts.transpose()


def likelihood(x,pC):
    if (x>=1).any()|(x<=0).any():
        like = -1e20
    else:
        like = ((s/2-1)*(np.log(x[0])+np.log(x[1])+ np.log(1-x[0]) +np.log(1-x[1]))
                +np.sum(np.log(x[0]*pC.iloc[:,1]+x[1]*pC.iloc[:,0])*(1-collected_answers))
                +np.sum(np.log((1-x[0])*pC.iloc[:,1]+(1-x[1])*pC.iloc[:,0])*(collected_answers)))
    return like

counts = dict() #Tables of counts: answers are on v, properties on rows
pQgivenCdict = dict() #Tables of P(Q|C): answers are on columns, properties on rows
pQgivenIdict = dict() #Tables of P(Q|I): answers are on columns, items on rows
for q in user_features.columns:
    collected_answers = user_features[q]
    pCgivenI = pCgivenIdict[q]
    qtype = q.split('_')[0]
    counts[q] = PQgivenC(collected_answers, pCgivenI,qtype)
    
    pQgivenCdict[q] = (counts[q]/counts[q].sum(1).reshape([-1,1])) 
    pQgivenIdict[q] = np.dot(pCgivenIdict_small[q],pQgivenCdict[q])        

f = open(datapath+outfilename,"wb")
pickle.dump(pQgivenIdict,f)
f.close()