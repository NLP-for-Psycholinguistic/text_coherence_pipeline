import numpy as np
import pandas as pd
import networkx as nx
import csv

## Functions to extract verbatim from texts
#requires a dataframe with graph and associated texts. 

def get_verbatim(df,paragraph_size = 5,nb_examples = 2,nb_texts = 0,store = False):
    df['weight_distance'] = df['graph'].apply(lambda x: create_weight_distance_plot(x,paragraph_size = paragraph_size))
    df['mean'] = df['weight_distance'].apply(lambda x: mean_weight_distance_plot(x))
    df['std'] = df['weight_distance'].apply(lambda x: np.std(x))
    examples = sample_examples_from_df(df,nb_texts=nb_texts,nb_examples = nb_examples,paragraph_size = paragraph_size)
    if store:
        with open(f"{store}.csv", 'w') as file:
            wrt = csv.DictWriter(file, fieldnames=examples[0].keys())
            wrt.writeheader()
            wrt.writerows(examples)
    return examples

def get_verbatim_score_by_treshold(df,threshold,paragraph_size = 5):
    if 'weight_distance' not in df.columns:
        df['weight_distance'] = df['graph'].apply(lambda x: create_weight_distance_plot(x,paragraph_size = paragraph_size))

    score_dict = {}
    for line in df.iterrows():
        score = 0
        weight_distance_abs = np.abs(line[1]['weight_distance'])
        l = len(weight_distance_abs)
        argsort = np.flip(np.argsort(weight_distance_abs[paragraph_size:-paragraph_size]))
        while weight_distance_abs[argsort[0]+paragraph_size]>threshold:
            indexes = list(range(argsort[0]-paragraph_size,argsort[0]+paragraph_size+1))
            score+=np.mean([weight_distance_abs[i+paragraph_size] for i in indexes])
            argsort = [x for x in argsort if x not in indexes]
        score_dict[line[1].name] = score
    df['score'] = pd.Series(score_dict)
    return df



def create_weight_distance_plot(graph:nx.Graph,paragraph_size = 5):
    weight_distance = [sum([x[2]['weight'] for x in list(graph.subgraph([a for a in range(i-paragraph_size,i+paragraph_size+1)]).edges(data = True))]) for i in graph.nodes()]
    mean = np.mean(weight_distance[paragraph_size:len(weight_distance)-paragraph_size])
    weight_distance = weight_distance[paragraph_size:len(weight_distance)-paragraph_size]
    weight_distance = weight_distance-mean
    return weight_distance

def mean_weight_distance_plot(weight_distance):
    return np.mean(abs(weight_distance))

def get_examples(weight_distance,text,paragraph_size = 5,nb_examples = 1,mode = 'max'):
    examples = []
    weight_distance = weight_distance[paragraph_size:len(weight_distance)-paragraph_size]
    if 'normal' in mode:
        argsort = np.argsort(abs(weight_distance))
        #argsort = np.flip(argsort)
    else:    
        argsort = np.argsort(weight_distance)
    if 'max' in mode:
        argsort = np.flip(argsort)
    while nb_examples>0:
        ex = {}
        ex['mode'] = mode
        ex['value'] = weight_distance[argsort[0]]
        indexes = [argsort[0]-paragraph_size,argsort[0]+paragraph_size]
        ex['index'] = [i for i in range(indexes[0],indexes[1]+1)]
        ex['text'] = text[indexes[0]:indexes[1]]
        argsort = [x for x in argsort if x not in ex['index']]
        examples.append(ex)
        nb_examples-=1
    return examples

def get_examples_from_subject(subject_line,paragraph_size = 5,nb_examples = 2):
    examples = []
    weight_distance = subject_line['weight_distance'].values[0]
    mean = subject_line['mean'].values[0]
    std = subject_line['std'].values[0]
    text = subject_line['text'].values[0]
    if 'code' in subject_line.columns:
        code = subject_line['code'].values[0]
    else:
        code = f"{subject_line[0]}" #use index as code
    argsort = np.argsort(weight_distance)
    for mode in ['normal','min','max']:
        if mode == 'min':
            argsort = np.argsort(weight_distance)
        if mode == 'max':
            argsort = np.flip(np.argsort(weight_distance))
        if mode == 'normal':
            argsort = np.argsort(abs(weight_distance))
        for _ in range(nb_examples):
            ex = {}
            ex['mode'] = mode
            ex['value'] = weight_distance[argsort[0]]
            ex['scaled_value'] = ex['value']/mean
            ex['z_score'] = ex['value']/std
            ex['mean'] = mean
            ex['code'] = code
            indexes = [argsort[0]-paragraph_size,argsort[0]+paragraph_size]
            ex['index'] = [i for i in range(indexes[0],indexes[1]+1)]
            ex['text'] = format_text_example(text[indexes[0]:indexes[1]])
            argsort = [x for x in argsort if x not in ex['index']]
            examples.append(ex)
    return examples

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

def sample_examples_from_df(df,nb_texts = 0 , nb_examples = 2,paragraph_size = 5):
    examples = []
    if nb_texts != 0:
        df = df.sample(nb_texts)
    for line in df.iterrows():
        weight_distance = line[1]['weight_distance']
        mean = line[1]['mean']
        std = line[1]['std']
        text = line[1]['text']
        mean = line[1]['mean']
        assert len(text)>(paragraph_size*2 + nb_examples*paragraph_size*3), "Text is too short for chosen paragraph size"
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
            for _ in range(nb_examples):
                ex = {}
                ex['mode'] = mode
                ex['value'] = weight_distance[argsort[0]]
                ex['mean'] = mean
                ex['scaled_value'] = ex['value']/mean
                ex['z_score'] = ex['value']/std
                ex['code'] = code
                indexes = [argsort[0]-paragraph_size,argsort[0]+paragraph_size]
                ex['index'] = [i for i in range(indexes[0],indexes[1]+1)]
                ex['text'] = format_text_example(text[indexes[0]:indexes[1]])
                argsort = [x for x in argsort if x not in ex['index']]
                examples.append(ex)
            
    return examples

def format_text_example(sentences):
    ret = ''
    for s in sentences:
        ret+=s+'.\n'
    return ret