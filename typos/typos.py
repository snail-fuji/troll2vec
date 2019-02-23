import distance

vocabulary = None
all_words = None

def jaccard(set1, set2):
    return len(set1.intersection(set2)) / len(set1.union(set2))

def init(init_vocabulary):
    global vocabulary, all_words, all_sets
    if not vocabulary:
      vocabulary = init_vocabulary
      all_words = list(vocabulary.keys())
      all_sets = [set(word) for word in vocabulary]

def find_closest_by_jaccard(word, size=1000):
    global all_sets, all_words
    word_set = set(word)
    return [
      word 
      for jaccard, word in sorted([
        (-jaccard(word_set, vocabulary_word_set), vocabulary_word) 
        for vocabulary_word_set, vocabulary_word in zip(all_sets, all_words)
      ])
    ][0:size]

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
