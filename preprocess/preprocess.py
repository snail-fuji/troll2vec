from pymystem3 import Mystem
import re

stemmer = Mystem()

number_regex = re.compile("\d+")
symbol_regex = re.compile("[^А-Яа-яA-Za-z]+")
spaces_regex = re.compile("\s+")
english_words_regex = re.compile("[A-Za-z]+")
cleanr = re.compile('<.*?>')
linksr = re.compile("<a .+>.+</a>")

def clean_html(raw_html):
    raw_html = raw_html.replace("<br>", "\n")
    cleantext = re.sub(linksr, ' ', raw_html)
    cleantext = re.sub(cleanr, ' ', cleantext)
    return cleantext

def normalize_text(text):
    text = symbol_regex.sub(" ", text)
    text = number_regex.sub("NUMBER", text)
    text = english_words_regex.sub("ENGLISHWORD", text)
    text = spaces_regex.sub(" ", text)
    return text.lower().strip()

def get_pos_tag(word):
  analysis = stemmer.analyze(word)
  try:
      main_info = analysis[0]['analysis'][0]['gr']
      parts = {
          "SPRO": "PRON", 
          "ADV": "ADV", 
          "CONJ": "CCONJ", 
          "PART": "PART", 
          "INTJ": "INTJ", 
          "PR": "PRON", 
          "A": "ADJ",
          "S": "NOUN", 
          "V": "VERB",
          "NUM": "NUM"
      }
      for part in parts:
          if main_info.startswith(part):
              return "_" + parts[part]
  except:
      pass
  return ""

def lemmatize_text(text):
    string = []
    for word in stemmer.lemmatize(text)[0:-1]:
        string.append("{}{}".format(word, get_pos_tag(word)))
    return "".join(string)


def preprocess_text(text):
    text = clean_html(text)
    text = normalize_text(text)
    text = lemmatize_text(text)
    return text