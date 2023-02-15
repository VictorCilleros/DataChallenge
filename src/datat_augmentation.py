#pip install translators

import pandas as pd
import translators as ts
import translators.server as tss

class TranslatorDataAugmentator():
  # api = "google"   from = 'fr',  to = 'en'
  def __init__(self,api:str,from_language:str,to_language:str):
    self.translator = translator_constructor(api=api)
    self.from_language = from_language
    self.to_language = to_language

  def translator_constructor(api):
      if api == 'google':
        return tss.google
      elif api == 'bing':
          return tss.bing
      elif api == 'baidu':
            return tss.baidu
      elif api == 'sogou':
            return tss.sogou
      elif api == 'youdao':
            return tss.youdao
      elif api == 'tencent':
            return tss.tencent
      elif api == 'alibaba':
            return tss.alibaba
      else:
            raise NotImplementedError(f'{api} translator is not realised!')

  def translate(self,x:str)->str:
     return self.translator(x, self.from_language, self.to_language)
  def translate_inv(self,x:str)->str:
      return self.translator(x, self.to_language, self.from_language)
  
  # data = df['Caption']
  def call(self,data:pd.DataFrame)->pd.DataFrame:
    translation = {'text': [] }
    for text in data : 
        translation['text'].append(self.translate_inv(self.translate(text)))
    return pd.DataFrame.from_dict(translation)
