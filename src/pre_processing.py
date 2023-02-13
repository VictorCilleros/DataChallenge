import os

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_hub as hub
import tensorflow_datasets as tfds




class BertInputProcessor(tf.keras.layers.Layer):
  def __init__(self, tokenizer, packer):
    super().__init__()
    self.tokenizer = tokenizer
    self.packer = packer

  def call(self, inputs):
    tok1 = self.tokenizer(inputs['Caption'])
    #tok2 = self.tokenizer(inputs['sentence2'])

    packed = self.packer([tok1])

    if 'category_1' in inputs:
      return packed, inputs[['category_1','category_2','category_3','category_4']]
    else:
      return packed

def pre_preocessing_BERT(data:pd.Dataframe,max_seq_lenght:int=128):
    """
    param data : pandas.Dataframe, should contains columns = {Caption,'category_1','category_2','category_3','category_4'}
    or only columns = {Caption} of no labels are provided
    param max_seq_lenght : max lenght used for BertPacker
    returns : (data,labels) or (data) if no labels are provided in data
    """
    packer = tfm.nlp.layers.BertPackInputs(
        seq_length=max_seq_length,
        special_tokens_dict = tokenizer.get_special_tokens_dict())
    
    gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/v3/uncased_L-12_H-768_A-12"
    tf.io.gfile.listdir(gs_folder_bert)
    
    tokenizer = tfm.nlp.layers.FastWordpieceBertTokenizer(
        vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
        lower_case=True)
    #Potentially tuple (data,labels) or (data)
    bert_inputs_pre_proc = BertInputProcessor(tokenizer, packer).call(data)
    return bert_inputs_pre_proc