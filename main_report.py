import logging
import sys
import pandas as pd
from src.report import write_report
from src.text_processing import load_config

def main(file_path,config_path,logger = logging.getLogger()):
    assert file_path.endswith(".csv"), "file format not supported, use the provided csv example file"
    logger.info(f"extracting example info from {file_path}...")
    df_examples = pd.read_csv(file_path)
    logger.info(f"Writing results report")
    report_config = load_config(config_path)['report_config']
    mdFile = write_report(df_examples,report_config)
    mdFile.create_md_file()

if __name__ == "__main__" :
    logging.basicConfig(filename='extract_logs.log', level=logging.INFO)
    main(file_path =  sys.argv[1])