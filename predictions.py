from keras.models import load_model
from data_helpers import pad_sentences, build_vocab, build_input_data, clean_str
import json

model_path = './model.h5py'
parameters_path = './parameters.json'
test_sentences = ["I hate this fuckin' bullshit you moron", "Hey, that sounds really nice!"]
threshold = 0.5

def read_model(path):
  model = load_model(path)
  return model

def read_parameters(path):
  parameters = json.load(open(path))
  return parameters["vocabulary"], parameters["max_length"]

def make_predictions(model, sentences):
  return model.predict(sentences).reshape(-1) > threshold

def process_sentences(sentences):
  return [clean_str(s.strip()).split() for s in sentences]

def prepare_sentences(sentences, vocabulary, max_length):
  sentences_processed = process_sentences(sentences)
  sentences_padded, _ = pad_sentences(sentences_processed, sequence_length=max_length)
  x, _ = build_input_data(sentences_padded, 0, vocabulary)
  return x

def predict(sentences):
  model = read_model(model_path)
  vocabulary, max_length = read_parameters(parameters_path)
  sentences = prepare_sentences(sentences, vocabulary, max_length)
  predictions = make_predictions(model, sentences)
  return predictions

if __name__ == "__main__":
  print(predict(test_sentences))