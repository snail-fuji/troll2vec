import unittest
from preprocess.preprocess import *
import gensim.downloader as api
from tqdm import tqdm
from collections import Counter

class PreprocessTests(unittest.TestCase):
  def xtest_normalize(self):
    words = {'VERB': 'распахиваться', 'ADP': 'помимо', 'PRON': 'оное', 'ADJ': 'пятитонный', 'INTJ': 'трах', 'NOUN': 'урум', 'PART': 'так-с', 'NUM': 'семь', 'DET': 'такой-то', 'CCONJ': 'понеже', 'ADV': 'по-флотски'}
    for tag, word in words.items():
      result_tag = get_pos_tag(word)
      print(word, "_" + tag, result_tag)

  def xtest_get_all_pos_tags(self):
    russian_model = api.load("word2vec-ruscorpora-300")
    pos_tags = {}

    for word in tqdm(russian_model.vocab):
      pure_word, pos_tag = word.split("_")[0], word.split("_")[-1]
      if pos_tag not in pos_tags:
        pos_tags[pos_tag] = []
      pos_tags[pos_tag].append(pure_word)

    all_words = [word for words in pos_tags.values() for word in words]
    counter = Counter(all_words)
    word_frequency = dict(counter.most_common())

    for tag in pos_tags:
      for word in pos_tags[tag]:
        if word_frequency[word] == 1:
          pos_tags[tag] = word
          break

    print(pos_tags)

  def test_lemmatize(self):
    result = lemmatize_text("этот текст будет лемматизирован и тегирован")
    print(result)

  def test_normalize(self):
    result = normalize_text("""    
      В этом тексте не будет Больших Букв, 
      тегов, чисел 123, english words, множества       пробелов,
      пропусков строки \r\n\t и (Посторонних-Символов типа %$!^), 
      помимо нужных знаков препинания! (и то не факт?)   
    """)
    print(result)

  def test_clean_html(self):
    result = clean_html("""
      <a href='#'>В этом тексте не будет тегов, помимо <br> переводов строки</a>
    """)
    print(result)

  def test_preprocess_text(self):
    comment_text = """
      <div class="comment__content" data-target="webuiPopover0">
      Этот комментарий я взял с пикабу, он не очень большой. <br>
      Но обрабатываться должн правильно
      </div>
    """
    result = preprocess_text(comment_text)
    print(result)