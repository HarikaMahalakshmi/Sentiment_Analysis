# Sentiment Analysis on IMDB Reviews

This project implements a **binary sentiment analysis system** that classifies IMDB movie reviews as either **positive** or **negative** using machine learning techniques.

---

## Features

- ✅ Text preprocessing with custom lemmatization  
- ✅ Modular pipeline built with Scikit-learn  
- ✅ TF-IDF vectorization (unigrams and bigrams)  
- ✅ Binary classification using Logistic Regression  
- ✅ Model evaluation: Precision, F1 Score, AUC  
- ✅ Confusion matrix visualization  

---

## Tech Stack

- Python  
- Scikit-learn  
- NLTK  
- Matplotlib & Seaborn  

---

## Project Structure

<pre> ```
  SentimentAnalysis/ ├── SentimentAnalysis/ │ ├── datapreprocessing/ │ ├── evaluation/ │ ├── models/ │ ├── dataloader/ │ └── main.py ├── sentiment_venv/ ├── data/ │ └── IMDB-Dataset.csv 
  ``` </pre>

---

## How to Run

1. Clone the repository

```bash
git clone https://github.com/HarikaMahalakshmi/Sentiment_Analysis.git
cd Sentiment_Analysis
```
2.Create and activate a virtual environment (optional but recommended)

```
python -m venv sentiment_venv
# Windows
sentiment_venv\Scripts\activate
# macOS/Linux
source sentiment_venv/bin/activate
```
3.Install Requirements
```
pip install -r requirements.txt
```

4.Run the main script
```
python SentimentAnalysis/main.py
```

## Dataset
Due to size limits, the IMDB-Dataset.csv (~366 MB) file is not included in this repository.

Download the dataset from:
Kaggle:https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Place the CSV file inside the data/ folder to enable access by the scripts.

## Flask Web Application

This project also includes a Flask-based web application that allows users to:

- Input movie reviews through a simple web interface.
- Get real-time sentiment predictions (Positive or Negative).
- View probability scores for the prediction.
- See a user-friendly visualization of results.

### How it works

1. The Flask app loads the trained sentiment analysis model saved by the pipeline.
2. When a user submits a review via the web form, the app preprocesses the input text.
3. The preprocessed text is passed to the model for prediction.
4. The prediction and confidence scores are displayed on the web page dynamically.

### Running the Flask app

Make sure you have Flask installed (`pip install flask`), then run:

```bash
python SentimentAnalysis/app.py


