# Text Coherence Pipeline tool

A NLP tool for speech disorder detection, used in the scope of automatic text PTSD classification

## Introduction
See [this jupyter notebook](https://github.com/SB-ENSEA/text_coherence_pipeline/blob/main/draft.ipynb) for a hands-on test of the tool.

Originally written for transcribed speech, the tool computes embeddings of sentences in the text and stores them in a similarity graph.  
Using this similarity graph, we estimate the impact of each sentence on the local coherence in the chosen paragraph, allowing us to extract exemples of speech disorder.  

## Usage

With the file containing a dataframe with a text column. Each text in the corpus must be in a separate line:  

`python main_extract('file_path.csv')` or `python main_extract('file_path.pkl')`  
  
Resulting dataframe containing texts, all additional columns and graphs is stored according to the storage field of the [config](https://github.com/SB-ENSEA/text_coherence_pipeline/blob/main/config.yaml)
Using this pickle file, run :  
  
`python main_verbatim('file_with_graph_path.pkl')`  
  
Extracting specific exemples of speech disorder along with a score, the sign of the distance, the z score computed by text, and the scaled score in a dictionnary.  
This dictionnary is stored in the storage_exemple field of the config

## Computation
The similarity model is the [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) used in the [sentence_transformer](https://www.sbert.net/) python archive.  
We compute the impact of sentence i on the local similarity by using a local subgraph containing sentences between i-paragraph_size, i+ paragraph_size. 
In this subgraph, we compute for each sentence the sum of similarities between the sentence i and all other.  
Thus we have a time serie of the impact of each sentence on the local similarity.  
![Summary of the computation pipeline]()

## Speech disorders
Our speech disorder definition are taken from [andreasen et al. 1979](https://doi.org/10.1001/archpsyc.1979.01780120045006)



