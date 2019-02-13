from flask import Flask, request, jsonify
import predictions

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def predict():
  messages = request.get_json(force=True)["messages"]
  print(messages)
  toxicity = predictions.predict(messages)
  print(toxicity)
  return jsonify({
    "toxicity": toxicity.tolist()
  })

if __name__ == '__main__':
  app.run(port=5000, debug=True)