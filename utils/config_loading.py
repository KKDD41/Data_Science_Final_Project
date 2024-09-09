import configparser

source_config = configparser.ConfigParser()
source_config.read('/usr/dsapp/src/sources.cfg')

TRAIN_DATA_URL = str(source_config['urls']['TRAIN_DATA_URL'])
TEST_DATA_URL = str(source_config['urls']['TEST_DATA_URL'])

PROCESSED_DATA_DIR = source_config['dirs']['PROCESSED_DATA_DIR']
RAW_DATA_DIR = source_config['dirs']['RAW_DATA_DIR']

MODELS_DIR = source_config['dirs']['MODELS_DIR']
PREDICTIONS_DIR = source_config['dirs']['PREDICTIONS_DIR']




