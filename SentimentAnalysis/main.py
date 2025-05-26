import os
import joblib
from datapreprocessing.datapreprocessing import DataCleaning, LemmaTokenizer
from evaluation.evaluationmetrics import confusion_matrix_plot
from dataloader.dataload import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, roc_auc_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

def main():
    # Load dataset
    path = os.path.join(os.getcwd(), 'sentimentanalysis', 'data', 'IMDB-Dataset.csv')
    print(f"Loading data from: {path}")
    data = load_dataset(path)

    print(f"Data loaded. Number of samples: {len(data)}")
    print(f"Unique labels: {data['Label'].unique()}")  # Should print [0 1]

    # No further filtering or encoding since labels are already 0 and 1

    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(
        data['Reviews'], data['Label'], test_size=0.1, random_state=42
    )
    print(f"Split data: Train size = {len(x_train)}, Test size = {len(x_test)}")

    # Build pipeline
    text_clf = Pipeline([
        ('clean', DataCleaning()),
        ('vect', TfidfVectorizer(
            analyzer='word',
            tokenizer=LemmaTokenizer(),
            ngram_range=(1, 2),
            min_df=5,
            max_features=3000
        )),
        ('clf', LogisticRegression(
            penalty='l2',
            class_weight='balanced',
            C=1.0,
            solver='lbfgs',
            max_iter=100
        ))
    ])

    # Train model
    print("Training model...")
    text_clf.fit(x_train, y_train)
    print("Model training completed.")

    # Predict and evaluate
    y_predict = text_clf.predict(x_test)
    y_score = text_clf.predict_proba(x_test)[:, 1]

    print(f"Precision Score: {precision_score(y_test, y_predict)}")
    print(f"AUC Score: {roc_auc_score(y_test, y_score)}")
    print(f"F1 Score: {f1_score(y_test, y_predict)}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_predict, target_names=['negative', 'positive']))

    # Plot confusion matrix
    confusion_matrix_plot(y_test, y_predict)
    plt.show()

    # Save the model
    joblib.dump(text_clf, 'sentiment_model.pkl')
    print("Model saved as 'sentiment_model.pkl'")

if __name__ == "__main__":
    main()
