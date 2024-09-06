import configparser

source_config = configparser.ConfigParser()
source_config.read('../../sources.cfg')

TRAIN_DATA_URL = str(source_config['urls']['TRAIN_DATA_URL'])
TEST_DATA_URL = str(source_config['urls']['TEST_DATA_URL'])

PROCESSED_TRAIN_DATA_DIR = source_config['dirs']['PROCESSED_TRAIN_DATA_DIR']
PROCESSED_TEST_DATA_DIR = source_config['dirs']['PROCESSED_TEST_DATA_DIR']

RAW_TRAIN_DATA_DIR = source_config['dirs']['RAW_TRAIN_DATA_DIR']
RAW_TEST_DATA_DIR = source_config['dirs']['RAW_TEST_DATA_DIR']

MODELS_DIR = source_config['dirs']['MODELS_DIR']
PREDICTIONS_DIR = source_config['dirs']['PREDICTIONS_DIR']

print(TRAIN_DATA_URL)




