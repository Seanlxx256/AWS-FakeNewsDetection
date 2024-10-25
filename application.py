from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import json

application = Flask(__name__)  

with open('basic_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file) 

with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)  


@application.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the incoming JSON data
        data = request.get_json(force=True)
        text = data.get('text', '')

        # Validate the input
        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "Invalid input: Please provide non-empty text."}), 400

        transformed_text = vectorizer.transform([text])
        prediction = model.predict(transformed_text)[0]  # np.str_('REAL') or np.str_('FAKE')

        # Safely convert prediction to string
        prediction_label = str(prediction)

        # Map 'FAKE' -> 1 and 'REAL' -> 0 
        prediction_int = 1 if prediction_label == 'FAKE' else 0

        # Return the prediction result as JSON
        return jsonify({"prediction": prediction_int}), 200

    except Exception as e:
        # Handle any unexpected errors
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@application.route('/', methods=['GET'])
def home():
    return "Fake News Detection API is running!", 200

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000, debug=True)
