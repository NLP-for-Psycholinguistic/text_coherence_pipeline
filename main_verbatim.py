from src.text_processing import load_config
import pandas as pd
import logging
import sys
from src.verbatim import get_verbatim

if __name__ == "__main__" :
    config = load_config("config.yaml")
    file_path =  sys.argv[1]
    logger = logging.getLogger()

    assert file_path.endswith(".pkl"), "file format not supported, use pickle files"

    try :
        logger.info(f"extracting graphs from {file_path}...")
        df_corpus = pd.read_pickle(file_path)
        logger.info(f"loaded {len(df_corpus)} texts and graphs")
        
        logger.info(f"extracting verbatim...")
        get_verbatim(df_corpus,paragraph_size = config['paragraph_size'],nb_exemples = config['nb_exemples'],nb_texts = config['nb_texts'],store = config['storage_exemples'])
        logger.info(f"extraction done, stored in {config['storage_exemples']}")
    except Exception as e :
        print(f"failed to store graphs because of {e}")