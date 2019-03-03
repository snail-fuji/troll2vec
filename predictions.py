
from keras.models import load_model
from data_helpers import pad_sentences, build_vocab, build_input_data, clean_str
import json

import tensorflow as tf
import keras.metrics
from utils import auc_roc
from train import model_path, parameters_path

keras.metrics.auc_roc = auc_roc

test_sentences = ["All people are equal", "All people are equal except niggers", "All people are equal except white"]
model = None
graph = None
vocabulary = None
max_length = None

def prepare_graph():
  return tf.get_default_graph()

def read_model(path):
  model = load_model(path)
  return model

def read_parameters(path):
  parameters = json.load(open(path))
  return parameters["vocabulary"], parameters["max_length"]

def make_predictions(model, sentences):
  with graph.as_default():
    return model.predict(sentences).reshape(-1)

def process_sentences(sentences):
  return [clean_str(s.strip()).split() for s in sentences]

def prepare_sentences(sentences, vocabulary, max_length):
  print(sentences)
  sentences_processed = process_sentences(sentences)
  sentences_padded, _ = pad_sentences(sentences_processed, sequence_length=max_length)
  x, _ = build_input_data(sentences_padded, 0, vocabulary)
  return x

def predict(sentences):
  global vocabulary, max_length, model
  sentences = prepare_sentences(sentences, vocabulary, max_length)
  predictions = make_predictions(model, sentences)
  return predictions

def init():
  global model, vocabulary, max_length, graph
  model = read_model(model_path)
  vocabulary, max_length = read_parameters(parameters_path)
  graph = prepare_graph()  

if __name__ == "__main__":
  init()