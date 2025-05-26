import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Update this path as per your actual saved model location
model_path = os.path.join(os.getcwd(), 'sentiment_model.pkl')
text_clf = joblib.load(model_path)

label_names = ['negative', 'positive']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        review = data.get('review', '')

        if not review.strip():
            return jsonify({'error': 'Empty input'}), 400

        review_series = pd.Series([review])
        pred_label = text_clf.predict(review_series)[0]
        pred_prob = text_clf.predict_proba(review_series)[0]
        confidence = max(pred_prob)

        sentiment = label_names[pred_label]

        return jsonify({
            'sentiment': sentiment,
            'confidence': round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
