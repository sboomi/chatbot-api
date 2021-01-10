from flask import Flask, jsonify, request
import requests
import json
from utils.chat import ChatbotModel

app = Flask(__name__)

cb = ChatbotModel('flaskapp/utils/models/chatbot_model', 'flaskapp/utils/data/intents.json')

@app.route('/')
@app.route('/message', methods=["POST"])
def message():
    message_content = json.loads(request.data)
    
    response = cb.generate_response(message_content['data'], as_dict=True)
    app.logger.info(f"Data: {message_content}, response: {response}")

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)