import numpy as np
import pandas as pd
import networkx as nx
import csv

## Functions to extract verbatim from texts
#requires a dataframe with graph and associated texts. 

def get_verbatim(df,paragraph_size = 5,nb_exemples = 2,nb_texts = 0,store = False):
    df['weight_distance'] = df['graph'].apply(lambda x: create_weight_distance_plot(x,paragraph_size = paragraph_size))
    df['mean'] = df['weight_distance'].apply(lambda x: mean_weight_distance_plot(x))
    df['std'] = df['weight_distance'].apply(lambda x: np.std(x))
    exemples = sample_exemples_from_df(df,nb_texts=nb_texts,nb_exemples = nb_exemples,paragraph_size = paragraph_size)
    if store:
        with open(f"{store}.csv", 'w') as file:
            wrt = csv.DictWriter(file, fieldnames=exemples[0].keys())
            wrt.writeheader()
            wrt.writerows(exemples)
    return exemples

def create_weight_distance_plot(graph:nx.Graph,paragraph_size = 5):
    weight_distance = [sum([x[2]['weight'] for x in list(graph.subgraph([a for a in range(i-paragraph_size,i+paragraph_size+1)]).edges(data = True))]) for i in graph.nodes()]
    mean = np.mean(weight_distance[paragraph_size:len(weight_distance)-paragraph_size])
    weight_distance = weight_distance[paragraph_size:len(weight_distance)-paragraph_size]
    weight_distance = weight_distance-mean
    return weight_distance

def mean_weight_distance_plot(weight_distance):
    return np.mean(abs(weight_distance))

def get_exemples(weight_distance,text,paragraph_size = 5,nb_exemples = 1,mode = 'max'):
    exemples = []
    weight_distance = weight_distance[paragraph_size:len(weight_distance)-paragraph_size]
    if 'normal' in mode:
        argsort = np.argsort(abs(weight_distance))
        #argsort = np.flip(argsort)
    else:    
        argsort = np.argsort(weight_distance)
    if 'max' in mode:
        argsort = np.flip(argsort)
    while nb_exemples>0:
        ex = {}
        ex['mode'] = mode
        ex['value'] = weight_distance[argsort[0]]
        indexes = [argsort[0]-paragraph_size,argsort[0]+paragraph_size]
        ex['index'] = [i for i in range(indexes[0],indexes[1]+1)]
        ex['text'] = text[indexes[0]:indexes[1]]
        argsort = [x for x in argsort if x not in ex['index']]
        exemples.append(ex)
        nb_exemples-=1
    return exemples

def get_exemples_from_subject(subject_line,paragraph_size = 5,nb_exemples = 2):
    exemples = []
    weight_distance = subject_line['weight_distance'].values[0]
    mean = subject_line['mean'].values[0]
    std = subject_line['std'].values[0]
    text = subject_line['text'].values[0]
    if 'code' in subject_line.columns:
        code = subject_line['code'].values[0]
    else:
        code = 'EXT_TEXT'
    argsort = np.argsort(weight_distance)
    for mode in ['normal','min','max']:
        if mode == 'min':
            argsort = np.argsort(weight_distance)
        if mode == 'max':
            argsort = np.flip(np.argsort(weight_distance))
        if mode == 'normal':
            argsort = np.argsort(abs(weight_distance))
        for _ in range(nb_exemples):
            ex = {}
            ex['mode'] = mode
            ex['value'] = weight_distance[argsort[0]]
            ex['scaled_value'] = ex['value']/mean
            ex['z_score'] = ex['value']/std
            ex['mean'] = mean
            ex['code'] = code
            indexes = [argsort[0]-paragraph_size,argsort[0]+paragraph_size]
            ex['index'] = [i for i in range(indexes[0],indexes[1]+1)]
            ex['text'] = format_text_exemple(text[indexes[0]:indexes[1]])
            argsort = [x for x in argsort if x not in ex['index']]
            exemples.append(ex)
    return exemples

def sort_max_sentences(weight_distance,paragraph_size = 5,mode = 'max'):
    score= len(weight_distance)*[0]
    weight_distance = abs(weight_distance[paragraph_size:len(weight_distance)-paragraph_size])
    
    if 'normal' in mode:
        argsort = np.argsort(abs(weight_distance))
    else:    
        argsort = np.argsort(weight_distance)
    if 'max' in mode:
        argsort = np.flip(argsort)
    for idx in argsort:
        s = weight_distance[idx]
        for i in range(idx-paragraph_size,idx+paragraph_size+1):
            score[i]+=s
    return score

def sample_exemples_from_df(df,nb_texts = 0 , nb_exemples = 2,paragraph_size = 5):
    exemples = []
    if nb_texts != 0:
        df = df.sample(nb_texts)
    for line in df.iterrows():
        weight_distance = line[1]['weight_distance']
        mean = line[1]['mean']
        std = line[1]['std']
        text = line[1]['text']
        mean = line[1]['mean']
        assert len(text)>paragraph_size*2, "Text is too short for chosen paragraph size"
        if 'code' in line[1].index:
            code = line[1]['code']
        else:
            code = 'EXT_TEXT'
        argsort = np.argsort(weight_distance)
        for mode in ['min','max','normal']:
            if mode == 'min':
                argsort = np.argsort(weight_distance)
            if mode == 'max':
                argsort = np.flip(np.argsort(weight_distance))
            if mode == 'normal':
                argsort = np.argsort(abs(weight_distance))
            for _ in range(nb_exemples):
                ex = {}
                ex['mode'] = mode
                ex['value'] = weight_distance[argsort[0]]
                ex['mean'] = mean
                ex['scaled_value'] = ex['value']/mean
                ex['z_score'] = ex['value']/std
                ex['code'] = code
                indexes = [argsort[0]-paragraph_size,argsort[0]+paragraph_size]
                ex['index'] = [i for i in range(indexes[0],indexes[1]+1)]
                ex['text'] = format_text_exemple(text[indexes[0]:indexes[1]])
                argsort = [x for x in argsort if x not in ex['index']]
                exemples.append(ex)
            
    return exemples

def format_text_exemple(sentences):
    ret = ''
    for s in sentences:
        ret+=s+'.\n'
    return ret