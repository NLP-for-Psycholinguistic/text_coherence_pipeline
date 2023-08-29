from src.text_processing import texts_to_df_graph
from src.text_processing import load_config
import pandas as pd
import logging
import sys

if __name__ == "__main__" :
     config = load_config("config.yaml")
     file_path = "/home/samuel/Bureau/journal_test.csv" #sys.argv[1]
     logger = logging.getLogger()

     try :
          logger.info(f"extracting graphs from {file_path}")
          if file_path.endswith(".csv"):
               df_texts = pd.read_csv(file_path)
          elif file_path.endswith(".pkl"):
               df_texts = pd.read_pickle(file_path)
          else :
               raise ValueError("file format not supported")
          logger.info(f"loaded {len(df_texts)} texts")
          df_texts = texts_to_df_graph(df_texts,config = config)
          df_texts.to_pickle(config['storage'])
          logger.info(f"extraction done, stored in {config['storage']}")
     except Exception as e :
          print(f"failed to store graphs because of {e}")