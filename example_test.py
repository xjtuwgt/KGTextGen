from model_envs import MODEL_CLASSES
from evens import HOME_DATA_FOLDER
from os.path import join
import bert_score
import torch
import argparse
import os
from scripts.bert_score_api import bert_score_func, bert_score_visualization

# bert_score_func()
# bert_score_visualization()
import fileinput

# transxl_config, transxl_model, transxl_tokenizer = MODEL_CLASSES['transfo-xl']
#
# pretrained_trained_model_name = 'transfo-xl-wt103'
#
# tokenizer = transxl_tokenizer.from_pretrained(pretrained_trained_model_name)
# model = transxl_model.from_pretrained(pretrained_trained_model_name)
#
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# # outputs = model(**inputs)
# #
# # print(outputs[0].shape)
# model.generate()

