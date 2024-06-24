from flask import Flask, jsonify, request

# Define the app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def test():
    return jsonify({'message': 'Hello, World from Sentiment Analysis App!'})

#  python -m flask run
#  python -m venv env  
#  .\env\Scripts\activate
#  python -m pip install flask