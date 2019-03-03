from flask import Flask, request, jsonify
import predictions
from flask_cors import CORS
from werkzeug.contrib.fixers import ProxyFix
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)

def delete_empty_messages(messages):
  filtered_messages = {}
  for key, message in messages.items():
    if message.strip():
      filtered_messages[key] = message
  return filtered_messages


@app.route('/api', methods=['POST'])
def predict():
  messages = request.get_json(force=True)
  messages = delete_empty_messages(messages)
  keys = list(messages.keys())
  values = [messages[key] for key in keys]
  if values:
    toxicity = predictions.predict(values)
    toxic_messages = {keys[index]: float(toxic) for index, toxic in enumerate(toxicity)}
    return jsonify(toxic_messages)
  else:
    return jsonify([])


@app.route('/save', methods=['POST'])
def save():
  client = MongoClient()
  message = request.get_json(force=True)
  client.dataset.messages.insert_one(message)
  return jsonify({"saved": True})


predictions.init()
app.wsgi_app = ProxyFix(app.wsgi_app)
if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000, debug=True)