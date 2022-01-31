import sys
import os
from os.path import join

# Add submodule path into import paths
PROJECT_FOLDER = os.path.dirname(__file__)
print('Project folder = {}'.format(PROJECT_FOLDER))
sys.path.append(join(PROJECT_FOLDER))
# Define the dataset folder and model folder based on environment
HOME_DATA_FOLDER = join(PROJECT_FOLDER, 'data')
os.makedirs(HOME_DATA_FOLDER, exist_ok=True)
print('*' * 35, ' path information ', '*' * 35)
print('Data folder = {}'.format(HOME_DATA_FOLDER))
LAMBADA_DATASET_FOLDER = join(HOME_DATA_FOLDER, 'lambada')
os.makedirs(LAMBADA_DATASET_FOLDER, exist_ok=True)
OUTPUT_FOLDER = join(HOME_DATA_FOLDER, 'output')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
PROCESSED_FOLDER = join(HOME_DATA_FOLDER, 'processed')
print('Preprocessed Data folder = {}'.format(PROCESSED_FOLDER))
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
KG_DATA_FOLDER = join(HOME_DATA_FOLDER, 'freebase')
print('Freebase folder = {}'.format(KG_DATA_FOLDER))
PRETRAINED_MODEL_FOLDER = join(HOME_DATA_FOLDER, 'models')
os.makedirs(PRETRAINED_MODEL_FOLDER, exist_ok=True)
PRETRAINED_MODEL_FOLDER = join(PRETRAINED_MODEL_FOLDER, 'pretrained')
os.environ['PYTORCH_PRETRAINED_BERT_CACHE'] = PRETRAINED_MODEL_FOLDER
