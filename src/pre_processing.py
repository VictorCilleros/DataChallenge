import os

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import nltk
import string
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer


nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')


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
    returns : tuple (data,labels) or (data) if no labels are provided in data
    """
    packer = tfm.nlp.layers.BertPackInputs(
        seq_length=max_seq_length,
        special_tokens_dict = tokenizer.get_special_tokens_dict())
    
    gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/v3/uncased_L-12_H-768_A-12"
    tf.io.gfile.listdir(gs_folder_bert)
    
    tokenizer = tfm.nlp.layers.FastWordpieceBertTokenizer(
        vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
        lower_case=True)
    #tuple (data,labels) or (data) if no labels provided 
    return BertInputProcessor(tokenizer, packer).call(data)




#Path : '../dictionnaire/dictionnaire.txt'
class CamembertInputProcessor():
  def __init__(self,path:str):
    self.mots = set(line.strip() for line in open(path))
    self.lemmatizer = FrenchLefffLemmatizer()
    self.french_stopwords = nltk.corpus.stopwords.words('french')

  def call(self,inputs,labels):
    df_pre_proc = self.French_Preprocess_listofSentence(inputs['Caption'])
    if labels is not None:
        return pd.concat([df_pre_proc,labels],axis=1).drop(columns='Id')
    else:
        return df_pre_proc

    #fonction de preprocessing qui va successivement : 
    #    enlever la ponctuation
    #    enlever les chiffres
    #    transformer les phrases en liste de tokens (en liste de mots)
    #    enlever les stopwords (mots n’apportant pas de sens)
    #    lemmatizer
    #    garder seulement les mots présent dans le dictionnaire
    #    enlever les majuscules
    #    reformer les phrases avec les mots restant

    def French_Preprocess_listofSentence(self,listofSentence):
        preprocess_list = []
        for sentence in listofSentence :
            sentence_w_punct = "".join([i.lower() for i in sentence if i not in string.punctuation])
            sentence_w_num = ''.join(i for i in sentence_w_punct if not i.isdigit())
            tokenize_sentence = nltk.tokenize.word_tokenize(sentence_w_num)
            words_w_stopwords = [i for i in tokenize_sentence if i not in self.french_stopwords]
            words_lemmatize = (self.lemmatizer.lemmatize(w) for w in words_w_stopwords)
            sentence_clean = ' '.join(w for w in words_lemmatize if w.lower() in self.mots or not w.isalpha())
            preprocess_list.append(sentence_clean)

        df_test = pd.DataFrame(preprocess_list,columns = {'text'})
        df_test.index.rename('id',inplace=True)
        return df_test