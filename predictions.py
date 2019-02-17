from keras.models import load_model
from data_helpers import pad_sentences, build_vocab, build_input_data, clean_str, lemmatize_text, remove_stopwords, remove_short_words
import json

model_path = './model-rus.h5py'
parameters_path = './parameters.json'
test_sentences = [""]

# remove stopwords
# remove short words

def read_model(path):
  model = load_model(path)
  return model

def read_parameters(path):
  parameters = json.load(open(path))
  return parameters["vocabulary"], parameters["max_length"]

def make_predictions(model, sentences, threshold):
  return model.predict(sentences).reshape(-1) > threshold

def process_sentences(sentences):
  return [lemmatize_text(clean_str(s.strip()))[:-2].lower().split() for s in sentences]

def prepare_sentences(sentences, vocabulary, max_length):
  sentences_processed = process_sentences(sentences)
  print(sentences_processed)
  sentences_processed = [remove_short_words(remove_stopwords(sentence)) for sentence in sentences_processed]
  sentences_padded, _ = pad_sentences(sentences_processed, sequence_length=max_length)
  x, _ = build_input_data(sentences_padded, 0, vocabulary)
  print(x)
  return x

def predict(sentences, threshold=0.5):
  model = read_model(model_path)
  vocabulary, max_length = read_parameters(parameters_path)
  sentences = prepare_sentences(sentences, vocabulary, max_length)
  predictions = make_predictions(model, sentences, threshold)
  return predictions

if __name__ == "__main__":
  print(predict(test_sentences))