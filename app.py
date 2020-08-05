import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from chatbot import return_chat

app = Flask(__name__)

@app.route('my-first-chatapp-test.herokuapp.com/')
def home():
    a = return_chat('Hi')
    return render_template('index.html')

@app.route('my-first-chatapp-test.herokuapp.com/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == "POST":
        input_string = request.form['chat_input']
        if bool(input_string.strip()):
            chat_response = return_chat(str(input_string))
            return chat_response
        else:
            return 'Please enter some data'
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)