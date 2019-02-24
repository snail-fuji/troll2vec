import unittest
import timeit
from nltk.corpus import words
from typos import typos
import random
from tqdm import tqdm
from fuzzywuzzy import fuzz
import numpy as np

class LevenschteinTestCase(unittest.TestCase):
  def test_large_index(self):
    all_tokens = words.words()
    partial_tokens = {random.choice(all_tokens): index for index in range(50000)}
    partial_tokens_inv = {i: v for v, i in partial_tokens.items()}
    typos.init(partial_tokens, test=True)
    word_pairs = []

    for index in tqdm(range(100)):
      word = random.choice(all_tokens)
      fixed_word = typos.find_closest(word)
      word_pairs.append((word, partial_tokens_inv[fixed_word]))

    print(word_pairs)
    print(np.mean([fuzz.ratio(word1, word2) for word1, word2 in word_pairs]))

  def test_large_index_for_jaccard(self):
    all_tokens = words.words()
    partial_tokens = {random.choice(all_tokens): index for index in range(50000)}
    typos.init(partial_tokens, test=True)

    for index in tqdm(range(100)):
        word = random.choice(all_tokens)
        typos.find_closest_by_jaccard(word)

  def test_create_jaccard_index(self):
    words = ["ABC", "AB", "AC", "AAB"]
    _, index_matrix = typos.create_jaccard_index(words)
    index_matrix = index_matrix.todense()
    print(index_matrix)
    assert (index_matrix == np.matrix([[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0]])).all()
  
  def test_find_closest_by_jaccard(self):
    words = {word: index for index, word in enumerate(["ABC", "AB", "AC", "AAB"])}
    typos.init(words, test=True)
    result = typos.find_closest_by_jaccard("ABD")
    print(result)