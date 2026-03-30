import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bertopic import BERTopic
from gensim import corpora
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

import nltk


sns.set_theme(style="whitegrid")

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("vader_lexicon")

DATA_PATH = Path("Data/NLP_data.csv")
TEXT_COLUMN = "content"
SAMPLE_SIZE = 5000
N_TOPICS = 8


def clean_text(text, stop_words):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if len(token) > 2 and token not in stop_words]
    return tokens


df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=[TEXT_COLUMN]).copy()
df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
df = df.dropna(subset=["published_at"]).copy()
df = df[["article_id", "title", "category", "published_at", TEXT_COLUMN]].copy()

if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
    df = df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)

stop_words = set(stopwords.words("english"))
df["tokens"] = df[TEXT_COLUMN].apply(lambda text: clean_text(text, stop_words))
df["clean_text"] = df["tokens"].apply(lambda tokens: " ".join(tokens))
df["published_date"] = df["published_at"].dt.date
df["published_year"] = df["published_at"].dt.year
df["published_month"] = df["published_at"].dt.to_period("M").astype(str)

sia = SentimentIntensityAnalyzer()
df["sentiment_score"] = df[TEXT_COLUMN].apply(
    lambda text: sia.polarity_scores(str(text))["compound"]
)
df["sentiment_label"] = pd.cut(
    df["sentiment_score"],
    bins=[-1.0, -0.05, 0.05, 1.0],
    labels=["Negative", "Neutral", "Positive"],
)

print(df.shape)
print(df["category"].fillna("Unknown").value_counts().head(10))
print(df[["published_at", "published_month", "sentiment_score", "sentiment_label"]].head())

vectorizer = CountVectorizer(max_df=0.95, min_df=10, stop_words="english")
doc_term_matrix = vectorizer.fit_transform(df["clean_text"])

lda_model = LatentDirichletAllocation(
    n_components=N_TOPICS,
    random_state=42,
    learning_method="batch",
)
lda_model.fit(doc_term_matrix)


def print_lda_topics(model, feature_names, top_n=10):
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-top_n - 1 : -1]]
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")


feature_names = vectorizer.get_feature_names_out()
print_lda_topics(lda_model, feature_names)

lda_topic_matrix = lda_model.transform(doc_term_matrix)
df["lda_topic"] = lda_topic_matrix.argmax(axis=1)
print(df[["title", "category", "published_month", "sentiment_label", "lda_topic"]].head(10))

dictionary = corpora.Dictionary(df["tokens"])
corpus = [dictionary.doc2bow(tokens) for tokens in df["tokens"]]
topic_terms = []

for topic in lda_model.components_:
    top_indices = topic.argsort()[:-11:-1]
    topic_terms.append([feature_names[i] for i in top_indices])

coherence_model = CoherenceModel(
    topics=topic_terms,
    texts=df["tokens"],
    dictionary=dictionary,
    coherence="c_v",
)

print({"lda_coherence": coherence_model.get_coherence()})

topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True)
bertopic_topics, bertopic_probs = topic_model.fit_transform(df["clean_text"].tolist())
df["bertopic_topic"] = bertopic_topics

topic_info = topic_model.get_topic_info()
print(topic_info.head(10))
print(topic_model.get_topic(0))

sentiment_by_topic = (
    df.groupby("bertopic_topic")["sentiment_score"]
    .agg(["mean", "count"])
    .sort_values(by="count", ascending=False)
)
print(sentiment_by_topic.head(10))

topic_month_trend = (
    df.groupby(["published_month", "bertopic_topic"])
    .size()
    .reset_index(name="document_count")
    .sort_values(["published_month", "document_count"], ascending=[True, False])
)
print(topic_month_trend.head(20))

fig = topic_model.visualize_barchart(top_n_topics=10)
fig.show()
