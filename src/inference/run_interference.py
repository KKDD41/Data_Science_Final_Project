import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import joblib

from src.data_preprocessing import (
    vectorize_review
)
from src.config_loading import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR
)


def load_model(
        filepath: str
):
    return joblib.load(filepath)


def test_model(
        X_test,
        Y_test,
        classifier
):
    predicted = classifier.predict(X_test)
    accuracy_score = metrics.accuracy_score(predicted, Y_test)
    cf_matrix = metrics.confusion_matrix(Y_test, predicted)
    classification_report = metrics.classification_report(Y_test, predicted)

    models_evaluation_metrics = ""
    models_evaluation_metrics += f'ComplementNB model accuracy is {round(accuracy_score * 100, 2)}%'
    models_evaluation_metrics += '\n------------------------------------------------\n'
    models_evaluation_metrics += 'Confusion Matrix:\n'
    models_evaluation_metrics += str(pd.DataFrame(cf_matrix))
    models_evaluation_metrics += '\n------------------------------------------------\n'
    models_evaluation_metrics += 'Classification Report:\n'
    models_evaluation_metrics += str(classification_report)

    with open(f"{PREDICTIONS_DIR}{classifier.__class__.__name__}_metrics.txt", 'w') as f:
        f.write(models_evaluation_metrics)

    predicted_df = test_df
    predicted_df['predicted_sentiment'] = pd.Series(predicted)
    predicted_df.to_csv(f"{PREDICTIONS_DIR}{classifier.__class__.__name__}_predictions.csv")

    return accuracy_score, cf_matrix, classification_report


if __name__ == "__main__":
    # Data & Models loading
    train_df = pd.read_csv(f'{PROCESSED_DATA_DIR}train.csv')
    test_df = pd.read_csv(f'{PROCESSED_DATA_DIR}test.csv')

    general_df = pd.concat([train_df, test_df], ignore_index=True)

    bernoulli_nb = load_model(f"{MODELS_DIR}BernoulliNB.pkl")
    SVM = load_model(f"{MODELS_DIR}SVC.pkl")
    logistic_regression = load_model(f"{MODELS_DIR}LogisticRegression.pkl")

    # Vectorization
    count_vectorizer = CountVectorizer()
    count_X_train, count_X_test, count_y_train, count_y_test = vectorize_review(
        df=general_df,
        processed_text_col_name='stemmed_review',
        train_df_len=len(train_df),
        vectorizer=count_vectorizer
    )

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test = vectorize_review(
        df=general_df,
        processed_text_col_name='stemmed_review',
        train_df_len=len(train_df),
        vectorizer=tfidf_vectorizer
    )

    # Models evaluation
    bernoulli_res = test_model(
        tfidf_X_test,
        tfidf_y_test,
        bernoulli_nb
    )

    scaler = StandardScaler(with_mean=False)
    scaler.fit(count_X_test)

    norm_count_X_test = scaler.transform(count_X_test)
    norm_count_y_test = count_y_test.apply(lambda x: 1 if x == 'positive' else 0)

    svm_results = test_model(
        norm_count_X_test,
        norm_count_y_test,
        SVM
    )

    lg_res = test_model(
        tfidf_X_test,
        tfidf_y_test,
        logistic_regression
    )


