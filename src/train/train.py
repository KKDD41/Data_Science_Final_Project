import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

from utils.data_preprocessing import (
    text_lemmatization,
    text_stemming,
    remove_punctuation_and_stopwords,
    extract_numerical_features,
    tokenize_text,
    remove_outliers,
    vectorize_review
)
from utils.data_loading import (
    download_and_unpack_raw_datasets
)
from utils.config_loading import (
    TRAIN_DATA_URL,
    TEST_DATA_URL,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR
)


def prepare_train_dataset(
        df: pd.DataFrame
) -> pd.DataFrame:
    df.drop_duplicates(inplace=True)
    df = extract_numerical_features(df)
    df = remove_outliers(df)
    return df


def prepare_train_text(
        df: pd.DataFrame
) -> pd.DataFrame:
    df = remove_punctuation_and_stopwords(df)
    df = tokenize_text(df)
    df = text_lemmatization(df)
    df = text_stemming(df)
    return df


def train_model(
        X_train,
        Y_train,
        classifier
):
    classifier.fit(X_train, Y_train)
    joblib.dump(classifier, f'{MODELS_DIR}{classifier.__class__.__name__}.pkl')
    return classifier


if __name__ == "__main__":
    download_and_unpack_raw_datasets(
        [TRAIN_DATA_URL, TEST_DATA_URL],
        RAW_DATA_DIR
    )

    # Data Preprocessing
    for dataset in ('train', 'test'):
        reviews_df = pd.read_csv(f"{RAW_DATA_DIR}/{dataset}.csv",
                                 sep=',')

        filtered_reviews_df = prepare_train_dataset(reviews_df)
        processed_reviews_df = prepare_train_text(reviews_df)

        processed_reviews_df.to_csv(f"{PROCESSED_DATA_DIR}{dataset}.csv")

    # Reading preprocessed data
    train_df = pd.read_csv(f"{PROCESSED_DATA_DIR}train.csv")
    test_df = pd.read_csv(f"{PROCESSED_DATA_DIR}test.csv")

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
    norm_count_y_train = count_y_train.apply(lambda x: 1 if x == 'positive' else 0)

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

