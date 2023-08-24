import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm


def load_model(config):
    if config['model'] == 'spacy_trf':
        return (spacy.load("fr_dep_news_trf"),None)
    elif config['model'] == 'sentence_sim':
        return (None,SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2"))
    elif config['model'] == 'spacy_lg':
        return (spacy.load("fr_core_news_lg"),None)
    else:
        raise ValueError("model not supported")
    
class Encoder():
    def __init__(self,config):
        self.nlp,self.model = load_model(config)
    
    def __call__(self,s1) -> float:
        if self.model == None:
            return self.nlp(s1).vector
        else:
            return self.model.encode(s1)

class SimCalc():
    def __init__(self):
        pass

    def __call__(self,v1,v2) -> float:
        return(np.sum(np.multiply(v1,v2))/(norm(v1)*norm(v2)))

