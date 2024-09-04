import numpy as np
import pandas as pd
import re
import nltk
import spacy
import string
from multiprocessing import Pool
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=False)

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')


def _extract_numerical_review_info(
        df: pd.DataFrame
) -> pd.DataFrame:
    df['number_of_chars'] = df['review'].apply(len)
    df['percentage_of_signs'] = df['review'].apply(
        lambda x: sum([1 for c in x if not c.isalpha()]) / len(x) * 100)
    df['number_of_excl_marks'] = df['review'].apply(lambda x: x.count('!'))
    df['number_of_question_marks'] = df['review'].apply(lambda x: x.count('?'))
    df['number_of_ellipses'] = df['review'].apply(lambda x: x.count('...'))
    df['number_of_uppercase_words'] = df['review'].apply(
        lambda x: sum([1 for w in x.split() if re.sub(r'[^a-zA-Z]', '', w).isupper()]))

    return df


def _remove_outliers(
        df: pd.DataFrame,
        threshold: float = 1.5
) -> pd.DataFrame:
    # calculate IQR for column 'number_of_chars'
    Q1 = df['number_of_chars'].quantile(0.25)
    Q3 = df['number_of_chars'].quantile(0.75)
    IQR = Q3 - Q1

    # identify outliers
    outliers = df[
        (df['number_of_chars'] < Q1 - threshold * IQR) | (df['number_of_chars'] > Q3 + threshold * IQR)
        ]

    df.drop(outliers.index, inplace=True)
    return df


def _replace_invalid_shortings(text):
    text = text.lower()
    text = re.sub("isn't", 'is not', text)
    text = re.sub("I've", 'i have', text)
    text = re.sub("he's", 'he is', text)
    text = re.sub("wasn't", 'was not', text)
    text = re.sub("there's", 'there is', text)
    text = re.sub("couldn't", 'could not', text)
    text = re.sub("won't", 'will not', text)
    text = re.sub("they're", 'they are', text)
    text = re.sub("she's", 'she is', text)
    text = re.sub("There's", 'there is', text)
    text = re.sub("wouldn't", 'would not', text)
    text = re.sub("haven't", 'have not', text)
    text = re.sub("That's", 'That is', text)
    text = re.sub("you've", 'you have', text)
    text = re.sub("He's", 'He is', text)
    text = re.sub("what's", 'what is', text)
    text = re.sub("weren't", 'were not', text)
    text = re.sub("we're", 'we are', text)
    text = re.sub("hasn't", 'has not', text)
    text = re.sub("you'd", 'you would', text)
    text = re.sub("shouldn't", 'should not', text)
    text = re.sub("let's", 'let us', text)
    text = re.sub("they've", 'they have', text)
    text = re.sub("You'll", 'You will', text)
    text = re.sub("i'm", 'i am', text)
    text = re.sub("we've", 'we have', text)
    text = re.sub("it's", 'it is', text)
    text = re.sub("don't", 'do not', text)
    text = re.sub("that´s", 'that is', text)
    text = re.sub("I´m", 'I am', text)
    text = re.sub("it’s", 'it is', text)
    text = re.sub("she´s", 'she is', text)
    text = re.sub("he’s'", 'he is', text)
    text = re.sub('I’m', 'I am', text)
    text = re.sub('I’d', 'I did', text)
    text = re.sub("he’s'", 'he is', text)
    text = re.sub('there’s', 'there is', text)

    return text


def _clean_review_text(
        df: pd.DataFrame
) -> pd.DataFrame:
    PUNCT_TO_REMOVE = string.punctuation
    STOPWORDS = set(stopwords.words('english'))
    STOPWORDS = STOPWORDS.union(set([w.title() for w in STOPWORDS]))
    STOPWORDS = STOPWORDS.union(set([w.translate(str.maketrans('', '', PUNCT_TO_REMOVE)) for w in STOPWORDS]))

    df['cleaned_review'] = df['review'].apply(lambda x: x.replace('<br />', ' ')) \
        .apply(lambda x: x.translate(str.maketrans('', '', PUNCT_TO_REMOVE))) \
        .apply(lambda x: re.sub(r'[0-9]+', '', x)) \
        .apply(lambda x: ''.join(filter(lambda y: y in string.printable, x))) \
        .apply(lambda x: " ".join([word for word in x.split() if word not in STOPWORDS])) \
        .apply(lambda x: _replace_invalid_shortings(x))

    return df


def _tokenize_review_text(
        df: pd.DataFrame
) -> pd.DataFrame:
    def tokenize_words(
            text: str
    ):
        import nltk
        nltk.download('punkt_tab')

        return nltk.tokenize.word_tokenize(text)

    df['tokenized_review'] = df['cleaned_review'].parallel_apply(tokenize_words)
    return df


def _review_text_lemmatization(
        df: pd.DataFrame
) -> pd.DataFrame:
    def lemmatize_words(text: list):
        import nltk
        from nltk.corpus import wordnet
        from nltk.stem import WordNetLemmatizer

        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger_eng')

        lemmatizer = WordNetLemmatizer()
        wordnet_map = {
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "J": wordnet.ADJ,
            "R": wordnet.ADV
        }
        pos_tagged_text = nltk.pos_tag(text)
        return ' '.join(
            [lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text]
        )

    df['legitimatized_review'] = df['tokenized_review'].parallel_apply(lemmatize_words)
    return df


def _review_text_stemming(
        df: pd.DataFrame
) -> pd.DataFrame:
    def stem_words(text: list):
        from nltk.stem.porter import PorterStemmer
        stemmer = PorterStemmer()
        return ' '.join([stemmer.stem(word) for word in text])

    df['stemmed_review'] = df['tokenized_review'].parallel_apply(stem_words)
    return df
