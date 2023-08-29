import pandas as pd
import numpy as np
import re
import spacy
import networkx as nx
import yaml
from src.sim import Encoder,SimCalc
import logging

def load_config(config_path:str)->dict:
    """Load config file from path"""
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def texts_to_df_graph(dataset_text:pd.DataFrame,config:str,logger = logging)->pd.DataFrame:
    """Build a dataframe containing the text,and the embeddings of the text"""
    assert "text" in dataset_text.columns, "text column not found in dataset"

    logger.info("Loading encoder model...")
    encoder = Encoder(config)

    logger.info('cleaning text before processing...')
    dataset_text["text"] = dataset_text["text"].map(lambda x : clean_text(x,config))

    logger.info("Processing and encoding text...")
    logger.info(f"=>Using method {config['sentence_segmentation_method']}")
    dataset_text["text"] = dataset_text["text"].map(lambda x : process_for_sentence_trf(x,config))
    dataset_text["embeddings"] = dataset_text["text"].map(lambda x : encode_sentence(x,encoder))

    logger.info("Building graphs from text...")
    dataset_text["graph"] = dataset_text["text"].map(lambda x : build_graph_nodes(x))
    dataset_text["graph"] = dataset_text.apply(lambda x : build_coherence_edges(x.graph,x.embeddings),axis = 1)

    if config['scale_graph']:
        logger.info("Scaling graphs...")
        scale_df_graph(dataset_text)
    
    dataset_text.drop(columns=['Unnamed: 0'],inplace=True)
    return dataset_text


# Remove transcriptor specifities

def clean_text(text:str,config:dict)->str:
    text = re.sub('[.]{2,4}',"",text)#remove the "..." in the text as we are working with lexical indicators and we don't use the pauses 
    text = re.sub('\[SPEAKER\]',"",text) #remove the speaker token for sentence graph, so that it is not confused with a word for the similarity measure
    if config['sentence_segmentation_method'] != 'punct_comma':
        text = re.sub('[,*]',"",text) #remove commas
    text = re.sub('[/]{1,}',"",text)
    #text = re.sub("[!?]",".",text)#replace question and exclamation marks by a dot
    text = re.sub('[ ]{2,}'," ",text)#remove double (or more) spaces
    return text

#sentence splitting methods

def text_to_sentences(text :str)->list:
    text = text.replace('?','.')
    text = text.replace('!','.')
    text = text.replace('...',' ')
    text = text.split('.')
    return [sentence for sentence in text if sentence.count(' ') > 3]

def cut_sentence_by_comma(sentence:str)->list:
    sub_sentence = []
    s = []
    for elem in sentence.replace(',',' , ').split(' '):
        s.append(elem)
        if len(s)> 50 and elem ==',':
            sub_sentence.append(' '.join(s))
            s = []
    return sub_sentence

def text_to_sentences_comma(text:str)->list:
    """Split the text into sentences, then splits again based on commas if the sentence is too long"""
    sentences = text_to_sentences(text)
    sentences2 = []
    for s in sentences:
        if s.count(' ')> 100:
            #sentence is too long
            #split on commas by the middle so that we have two equal sub-sentences 
            sentences2.extend(cut_sentence_by_comma(s))
        else:
            sentences2.append(s)
    return sentences2

## Processing so the text is usable
def process_for_sentence_trf(text:str,config:dict)->list[str]:
    """Process the text so that it can be used for the sentence transformer."""
    if config['sentence_segmentation_method'] == 'punct_comma':
        sentences = text_to_sentences_comma(text)
    else:
        sentences = text_to_sentences(text)
    return sentences

def encode_sentence(sentences:str,encoder:Encoder):
    """Encode the text using the sentence transformer"""
    embeddings = []
    for line in sentences:
        embeddings.append(encoder(line))
    return embeddings

# Constructing the graph
def build_graph_nodes(text:list[str]):
    """Build a graph from a list of sentences"""
    graph = nx.Graph()
    for i in range(len(text)):
        graph.add_node(i)
    return graph

def build_coherence_edges(graph:nx.Graph, embeddings:list)->None:
    """Builds the edges of the graph by similarity, computing similarity between each sentences"""

    sim = SimCalc()
    nodes = graph.nodes
    for i,node1 in enumerate(nodes):
        for node2 in list(nodes)[i+1:]:
            graph.add_edge(node1,node2,weight = sim(embeddings[node1],embeddings[node2]))
    return graph

def scale_df_graph(df:pd.DataFrame)->pd.DataFrame:
    """Scales graph edge weights between 0 and 1 """
    total_edge_list = []
    for graph in df['graph']:
        total_edge_list.extend([x[2]['weight'] for x in graph.edges(data=True)])
    max_edge = abs(max(total_edge_list))
    min_edge = abs(min(total_edge_list))
    
    def scale_graph(g):
        for edge in g.edges():
            g.edges[edge]['weight'] = (min_edge + g.edges[edge]['weight'])/(max_edge + min_edge)
        return g
    
    df['graph'] = df['graph'].map(lambda x : scale_graph(x))