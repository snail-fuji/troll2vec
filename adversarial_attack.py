from predictions import *
import keras.backend as K
import tensorflow as tf
from tqdm import tqdm
import numpy as np

epsilon = 0.01
iterations = 1000

if __name__ == "__main__":
  test_x = ["Hello and warm welcome to our group it is so nice to see you"]
  with tf.Session() as session:
    model = read_model(model_path)
    vocabulary, max_length = read_parameters(parameters_path)
    sentences = prepare_sentences(test_x, vocabulary, max_length)
    sentence_embeddings = model.layers[1].output
    gradients = tf.keras.backend.gradients(model.output, sentence_embeddings)[0]
    adversarial_example = sentence_embeddings + epsilon * gradients

    common_example_value = session.run(sentence_embeddings, {model.input: sentences})
    print(session.run(model.output, {sentence_embeddings: common_example_value}))

    # Generate adversarial example
    adversarial_example_value = common_example_value

    for iteration in tqdm(range(iterations)):
      adversarial_example_value = session.run(adversarial_example, {sentence_embeddings: adversarial_example_value})
    
    print(session.run(model.output, {sentence_embeddings: adversarial_example_value}))

    print(adversarial_example_value.shape)

    # Restore from embeddings
    vocabulary_inv = {v: k for k, v in vocabulary.items()}
    embedding_weights = model.layers[1].get_weights()[0]
    chosen_sentence = adversarial_example_value[0]
    words = []
    for i in range(15):
      word_vector = chosen_sentence.take(i, axis=0)
      dot_matrix = np.dot(embedding_weights, word_vector)
      modules_matrix = np.linalg.norm(embedding_weights, axis=1) * np.linalg.norm(word_vector)
      best_match = np.argmax(dot_matrix / modules_matrix)
      words += [vocabulary_inv[best_match]]
    print(" ".join(words))
