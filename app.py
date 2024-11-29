from flask import Flask, request, jsonify
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import pickle
import json

# Inicializa el Flask app
app = Flask(__name__)

# Carga el modelo y los recursos
model = load_model('chatbot_model.h5')
lemmatizer = WordNetLemmatizer()

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents = json.loads(open('intents.json').read())

# Función para limpiar la entrada del mensaje
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
    return sentence_words

# Función para crear la "bolsa de palabras"
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Función para predecir la clase
def predict_class(sentence):
    p = bow(sentence, words)
    pred = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(pred) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": r[1]})
    return return_list

# Función para obtener la respuesta de un intent
def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Ruta principal para interactuar con el chatbot
@app.route('/chat', methods=['POST'])
def chat():
    # Recibe el mensaje del usuario desde el cuerpo de la petición POST
    message = request.json['message']
    
    # Predice la clase del mensaje
    predicted_class = predict_class(message)
    tag = predicted_class[0]['intent']
    
    # Obtiene una respuesta para la clase predicha
    response = get_response(tag)
    
    # Retorna la respuesta como JSON
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
