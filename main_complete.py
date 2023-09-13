import logging
import sys
from main_extract import main as main_extract
from main_verbatim import main as main_verbatim
from main_report import main as main_report
from src.text_processing import load_config

def main(file_path,config_path,logger = logging.getLogger()):
    config = load_config(config_path)
    logger.info(f"Launching main_extract...")
    main_extract(file_path,config_path,logger)
    logger.info(f"Launching main_verbatim...")
    main_verbatim(config['storage'],config_path,logger)
    logger.info(f"Launching main_report...")
    main_report(config['storage_examples'],config_path,logger)
    logger.info(f"Done, markdown report stored in repository root")

if __name__ == "__main__" :
    logging.basicConfig(filename='extract_logs.log', level=logging.INFO)
    main(file_path =  sys.argv[1],config_path="config.json")