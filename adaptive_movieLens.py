import numpy as np
import pandas as pd
import pickle
import random as rn
import time
import matplotlib.pyplot as plt


def updating(prior, tables, q, a):
    likelihood = tables[q][:, a]
    joint = likelihood*prior
    if ( sum(joint)==0):
        joint = prior
    posterior = joint / sum(joint)
    return posterior
	
def entropy(mass_function):
	
	base = len(mass_function)
	np.place(mass_function, mass_function==0, 1)
	return (-mass_function*np.log(mass_function)).sum(0)/np.log(base)

def pick_question(prior, tables, askable_questions):
	
	entropies = [2 for _ in tables] # Array of the conditional entropies H(I|Q) for each question Q
	priort = prior.reshape(-1,1)
	for question in askable_questions: # Only allowed questions are considered as candidates

		conditional_table = tables[question].copy()
		joint_table = conditional_table*np.repeat(priort,conditional_table.shape[1], axis = 1)
		
		# Compute P(Q)
		question_probabilities = joint_table.sum(axis=0)/joint_table.sum() # P(Q)
		
		if (joint_table.sum(0)==0).any():
			posterior = joint_table
			posterior[:,joint_table.sum(0)==0] = priort
			posterior = posterior/posterior.sum(0)
		else: 
			posterior = joint_table/joint_table.sum(0)
            
		# Compute H(I|Q)
		entropies[question] = (entropy(posterior)*question_probabilities).sum()
	entropies = [_ if _ >= 0.0 else 2.0 for _ in entropies]
	return np.argmin([_ for _ in entropies if _ >= 0])


# MAIN 
# Select dataset 
# 10:= artificial dataset with noise level at 10%
# 50:= artificial dataset with noise level at 50%
# 'real':= dataser built using real tags (not shown in the paper)
noise = 10
# Set cpts = 'logic' to obtain the results for the Bayesian adaptive approach based on structural judgements
# Set cpts = '' to use cpts learned from data
cpts=''

if noise=='real':
    resultspath = 'results/real/'
    user_features = pd.read_pickle('data/user_features_real_small_test.pkl')
    pQgivenIdict = pickle.load( open( 'data/allcpts_real_small.pkl', "rb" ) )
else:    
    resultspath = '/results/noise%d/'%noise
    user_features = pd.read_pickle('data/user_features_noise_at_'+str(noise)+'_percent_small_test.pkl')
    pQgivenIdict = pickle.load( open('data/allcpts'+str(noise)+'_small.pkl', "rb" ) )
if cpts=='logic':
    pQgivenIdict = pickle.load( open( 'data/pCgivenIdict_small.pkl', "rb" ) )

item_features = pd.read_pickle('data/item_features_small.pkl')
n_items = pQgivenIdict['genre'].shape[0]                                  # Number of items in the catalogue

def L1distance(a, catalogue):
    if type(a)==np.ndarray:
        return (catalogue - a).apply(abs).apply(sum)
    else:
        return (catalogue-a).abs()  

for i in range(pQgivenIdict['genre'].shape[1]): 
    pQgivenIdict['genre_'+str(i)] = np.array([1-pQgivenIdict['genre'][:,i]/7,pQgivenIdict['genre'][:,i]/7]).transpose()
    item_features['genre_'+str(i)] = item_features['genre'].apply(lambda x: x[i])

del pQgivenIdict['genre']    
del item_features['genre']  

n_answers = list()  # Number of possible answers for each question  

for q,cpt in pQgivenIdict.items():
    n_answers += [cpt.shape[1]]                   # Number of possible answers for each question
n_questions = len(n_answers)

cols = ['p'+str(_) for _ in item_features.index]+['question', 'answer']

nusers = user_features.shape[0]
emptyDF = pd.DataFrame(columns = range(nusers))
methods = ['random', 'adaptive', 'similarity', 'similarity_random']
rtable = dict()
ptable = dict()

for m in methods:
    rtable[m] = emptyDF.copy()
    ptable[m] = emptyDF.copy()


for i in range(nusers):
    
    answers = user_features.iloc[i].copy()
    # transform list of genres to 19 different questions 
    for g in range(len(answers['genre'])): 
        answers['genre_'+str(g)] = answers['genre'][g]
    answers = answers.drop(index = 'genre')
    valid_answers = answers.dropna().index
    trueI = user_features.iloc[i].name
    col = "p"+str(trueI)
    print('Item:', trueI)
    answers.loc[valid_answers] = answers.loc[valid_answers].astype(int) 

    tables = list(pQgivenIdict.values())
    
################################################################################    
# Adaptive
    method = 'adaptive'     
    # Uniform Prior
    prior = np.array([1.0/n_items for _ in range(n_items)])  # P(I)
    df = pd.DataFrame(columns=cols+['entropy'])
    df.loc[0] = np.append(prior, [-1, -1, entropy(prior.copy())])
    # All the questions can be initially asked
    askable_questions = [_ for _ in range(n_questions)]
   
    start0 = time.time()
    while(len(askable_questions)>0):
     	# Pick the most informative question and remove from the candidates list
     	question_asked =  pick_question(prior, tables, askable_questions)
     	askable_questions.remove(question_asked)
     	if np.isnan(answers[question_asked]): print(question_asked); continue
     	# Update prior (to the posterior)
     	prio0 = prior
     	prior = updating(prior, tables, question_asked, answers[question_asked])
     	prior
     	# Add information to the df
     	df.loc[len(df)] = np.append(prior, [question_asked, answers[question_asked], entropy(prior.copy())])
    end = time.time()    
   
    rtable[method][i] = df.drop(columns = ['question', 'answer', 'entropy']).rank(axis=1,ascending=False)[col]
    ptable[method][i] = df[col]

    print('Final rank BN: ', rtable[method][i].iloc[-1] )
    print('Final probability BN: ', ptable[method][i].iloc[-1])

    if cpts=='logic':
       continue 
#############################################33
# Similarity 
    method = 'similarity'
    distance =  pd.Series(0, index = item_features.index)        
    prior = np.array([1.0/n_items]*n_items)  # P(I)
    dfsim = pd.DataFrame(columns=cols+['correct'])
    dfsim.loc[0] = np.append(prior, [-1, -1,-1])
    AB = 1
    A2 = 1
    B2 = 1
    for q in df.question[1:].astype(int):
        answer =answers.iloc[q]
        if np.isnan(answer): continue
        items = item_features.iloc[:,q]
        if np.isscalar(answer):
            AB += (answer*items)
            A2 += answer**2 
            B2 += items**2
        else:
            answer = user_features.iloc[[i],q]
            AB += items.apply(lambda x: answer.values[0]*x).apply(sum)
            A2 += (answer.values[0]**2).sum()
            B2 += (items**2).apply(sum)
        
        similarity =AB/np.sqrt(A2*B2)
        distance +=L1distance(answer,items)
        prior = similarity
    	
    	# Add information to the df
        dfsim.loc[len(dfsim)] = np.append(prior, [q, answers[q], items[trueI]])
    rtable[method][i] = dfsim.drop(columns = ['question', 'answer', 'correct']).rank(axis=1,ascending=False)[col]
    ptable[method][i] = dfsim[col]
 
    print('Final rank similarity: ',  rtable[method][i].iloc[-1])
    print('Final similarity: ',  ptable[method][i].iloc[-1])
    print('Time adaptive: ', end - start0)	  
    
    
################################################################################

# Random
    method = 'random'
    # Uniform Prior
    prior = np.array([1.0/n_items for _ in range(n_items)])  # P(I)
    df = pd.DataFrame(columns=cols+['entropy'])
    df.loc[0] = np.append(prior, [-1, -1, entropy(prior.copy())])
    # All the questions can be initially asked
    askable_questions = [_ for _ in range(n_questions)]
    start0 = time.time()
    while(len(askable_questions)>0):
#    	start = time.time()
     	# Pick a random question and remove from the candidates list
     	question_asked = rn.choice(askable_questions)
     	askable_questions.remove(question_asked)
     	if np.isnan(answers[question_asked]): continue
     	# Update prior (to the posterior)
     	prior = updating(prior, tables, question_asked, answers[question_asked])
    	
     	# Add information to the df
     	df.loc[len(df)] = np.append(prior, [question_asked, answers[question_asked], entropy(prior.copy())])
    end = time.time()    
    print('Time random: ', end - start0)	
    
    rtable[method][i] = df.drop(columns = ['question', 'answer', 'entropy']).rank(axis=1,ascending=False)[col]
    ptable[method][i] = df[col]  
  
#############################################33
# Similarity random
   
    method = 'similarity_random'    
    prior = np.array([1.0/n_items]*n_items)  # P(I)
    dfsimrnd = pd.DataFrame(columns=cols)
    dfsimrnd.loc[0] = np.append(prior, [-1, -1])
    AB = 1
    A2 = 1
    B2 = 1

    for q in df.question[1:].astype(int):
        answer =answers.iloc[q]
        items = item_features.iloc[:,q]
        if np.isnan(answer): continue
        if np.isscalar(answer):
            AB += (answer*items)
            A2 += answer**2 
            B2 += items**2
        else:
            answer = user_features.iloc[[i],q]
            AB += items.apply(lambda x: answer.values[0]*x).apply(sum)
            A2 += (answer.values[0]**2).sum()
            B2 += (items**2).apply(sum)
        
        similarity =AB/np.sqrt(A2*B2)
        prior = similarity
    	
    	# Add information to the df
        dfsimrnd.loc[len(dfsimrnd)] = np.append(prior, [q, answers[q]])

    rtable[method][i] = dfsimrnd.drop(columns = ['question', 'answer']).rank(axis=1,ascending=False)[col]
    ptable[method][i] = dfsimrnd[col] 
        

f = open('results/ranks%s%s.pkl'%(str(noise),cpts),"wb")
pickle.dump(rtable,f)
f.close()
f = open('results/probabilities%s%s.pkl'%(str(noise),cpts),"wb")
pickle.dump(ptable,f)
f.close()

probabilities = pd.DataFrame()
ranks = pd.DataFrame()
for m in methods:
    probabilities[m] = ptable[m].mean(1)
    ranks[m] = rtable[m].mean(1)

probabilities[['random', 'adaptive']].plot()
ranks.loc[1:,].plot()
ranks.loc[150:,].plot()
probabilities.to_csv('results/probabilities%s%s.csv'%(str(noise),cpts))
ranks.to_csv('results/ranks%s%s.csv'%(str(noise),cpts))

