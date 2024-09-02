import pandas as pd
import numpy as np
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

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
        text_data: pd.Series,
        vectorizer
):
    return vectorizer.fit_transform(text_data)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Reading raw data
    for dataset in ('train', 'test'):
        reviews_df = pd.read_csv(f"../../data/raw/final_project_{dataset}_dataset/{dataset}.csv",
                                 sep=',')

        # Train data preprocessing
        filtered_reviews_df = prepare_train_dataset(reviews_df)
        processed_reviews_df = text_preprocessing(reviews_df)

        print(processed_reviews_df.head(5))

        processed_reviews_df.to_csv(f"../../data/processed/processed_{dataset}.csv")
