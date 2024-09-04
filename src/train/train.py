import pandas as pd
import numpy as np
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics, svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

from preprocessing_utils import (
    _review_text_lemmatization,
    _review_text_stemming,
    _clean_review_text,
    _extract_numerical_review_info,
    _tokenize_review_text,
    _remove_outliers
)


def prepare_train_dataset(
        df: pd.DataFrame
) -> pd.DataFrame:
    df.drop_duplicates(inplace=True)
    df = _extract_numerical_review_info(df)
    df = _remove_outliers(df)
    return df


def text_preprocessing(
        df: pd.DataFrame
) -> pd.DataFrame:
    df = _clean_review_text(df)
    df = _tokenize_review_text(df)
    df = _review_text_lemmatization(df)
    df = _review_text_stemming(df)
    return df


def vectorize_review(
        df: pd.DataFrame,
        processed_text_col_name: str,
        train_df_len: int,
        vectorizer
):
    vectorized_data = vectorizer.fit_transform(df[processed_text_col_name])
    return train_test_split(
        vectorized_data,
        df['sentiment'],
        test_size=train_df_len,
        shuffle=False
    )


def train_model(
        X_train,
        Y_train,
        classifier
):
    classifier.fit(X_train, Y_train)
    joblib.dump(classifier, f'../../outputs/models/{classifier.__class__.__name__}.pkl')
    return classifier


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Data Preprocessing
    for dataset in ('train', 'test'):
        reviews_df = pd.read_csv(f"../../data/raw/final_project_{dataset}_dataset/{dataset}.csv",
                                 sep=',')

        filtered_reviews_df = prepare_train_dataset(reviews_df)
        processed_reviews_df = text_preprocessing(reviews_df)

        processed_reviews_df.to_csv(f"../../data/processed/processed_{dataset}.csv")

    # Reading preprocessed data
    train_df = pd.read_csv('../data/processed/processed_train.csv')
    test_df = pd.read_csv('../data/processed/processed_test.csv')

    general_df = pd.concat([train_df, test_df], ignore_index=True)

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

    # Model training
    BNB = BernoulliNB()
    bernoulli_nb = train_model(
        count_X_train,
        count_y_train,
        BNB
    )

    scaler = StandardScaler(with_mean=False)
    scaler.fit(count_X_train)

    norm_count_X_train = scaler.transform(count_X_train)
    norm_count_X_test = scaler.transform(count_X_test)
    norm_count_y_train = count_y_train.apply(lambda x: 1 if x == 'positive' else 0)
    norm_count_y_test = count_y_test.apply(lambda x: 1 if x == 'positive' else 0)

    SVM = svm.SVC(kernel='linear')
    SVM = train_model(
        norm_count_X_train,
        norm_count_y_train,
        SVM
    )

    LG = LogisticRegression()
    logistic_regression = train_model(
        tfidf_X_train,
        tfidf_y_train,
        LG
    )
