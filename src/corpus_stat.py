import pandas as pd
import numpy as np

def sentences_statistics(corpus_text_df:pd.DataFrame)-> dict:
    text_lengths = []
    sentences_lengths = []

    for text in corpus_text_df.text:
        sentences_lengths.append(len(text))
        for sentence in text:
            text_lengths.append(len(sentence.split(' ')))
    statistics = {
        'mean text size' : np.mean(text_lengths),
        'mean sentence size' : np.mean(sentences_lengths),
        'std text size' : np.std(text_lengths),
        'std sentence size' : np.std(sentences_lengths),
        'outliers sentence size' : len([length for length in sentences_lengths if length > np.mean(sentences_lengths) + 3*np.std(sentences_lengths)])/len(sentences_lengths)
    }
    return statistics