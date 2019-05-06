from troll2vec import predictions

class Troll2VecModel:
    def load(self):
        predictions.init()

    def preprocess(self, messages):
        return messages

    def predict_probabilities(self, messages):
        predictions.predict(messages)