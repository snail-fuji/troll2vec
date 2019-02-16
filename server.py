from flask import Flask, request, jsonify
import predictions

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def predict():
  messages = request.get_json(force=True)
  keys = list(messages.keys())
  values = [messages[key] for key in keys]
  toxicity = predictions.predict(messages)
  print(toxicity)
  toxic_messages = [keys[index] for index, toxic in enumerate(toxicity) if toxic]
  return jsonify(toxic_messages)

if __name__ == '__main__':
  app.run(port=5000, debug=True)