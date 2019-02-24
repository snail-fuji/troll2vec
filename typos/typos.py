import distance
import numpy as np
import scipy.sparse as sps
from collections import Counter

vocabulary = None
all_words = None

unique_characters = None
jaccard_index = None
jaccard_index_lengths = None

def get_word_indices(word, unique_characters):
  char_indices = []
  for character in word:
    if character in unique_characters:
      char_indices.append(unique_characters[character])
  return char_indices

def create_jaccard_index(words):
  all_characters = [character for word in words for character in word]
  unique_characters = {
    character[0]: index 
    for index, character in enumerate(Counter(all_characters).most_common())
  }

  rows = []
  columns = []

  for index, word in enumerate(words):
    extracted_columns = get_word_indices(word, unique_characters)
    columns += extracted_columns
    rows += [index] * len(extracted_columns)

  data = np.ones(len(columns))

  return unique_characters, sps.coo_matrix(
    (data, (rows, columns)), 
    shape=(len(words), len(unique_characters))
  ).astype(bool).astype(int)

def init(init_vocabulary, test=False):
    global vocabulary, all_words, jaccard_index, soundex_words, unique_characters, jaccard_index_lengths
    if (not vocabulary) or test:
      vocabulary = init_vocabulary
      all_words = list(vocabulary.keys())
      unique_characters, jaccard_index = create_jaccard_index(all_words)
      jaccard_index_lengths = sps.linalg.norm(jaccard_index, axis=1)

def find_closest_by_jaccard(word, size=100):
    global jaccard_index, all_words, unique_characters, jaccard_index_lengths

    chars_indices = get_word_indices(word, unique_characters)
    word_indices = np.zeros(len(chars_indices))
    data = np.ones(len(chars_indices))

    word_vector = sps.coo_matrix(
      (data, (word_indices, chars_indices)), 
      shape=(1, len(unique_characters))
    ).astype(bool).astype(int)

    intersections = word_vector.dot(jaccard_index.T)
    word_length = sps.linalg.norm(word_vector)
    index_lengths = jaccard_index_lengths
    cosine_distances = intersections / (word_length * index_lengths) 
    return [all_words[index] for index in np.argsort(-cosine_distances).tolist()[0][0:size]]

def find_closest_by_levenshtein(word, selected_words):
    distances = list(distance.ifast_comp(word, selected_words))
    if not distances:
      distances = list(distance.ilevenshtein(word, selected_words, max_dist=len(word) // 2))
    if not distances:
      distances = [(0, selected_words[0])]
    return max(distances)[1]

def find_closest(word):
    global vocabulary
    if word in vocabulary:
        return vocabulary[word]
    else:
        best_words = find_closest_by_jaccard(word)
        best_word = find_closest_by_levenshtein(word, best_words)
        print(word, best_word)
        return vocabulary[best_word]
