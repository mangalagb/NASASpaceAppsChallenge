import json
import pickle
from pathlib import Path
from string import punctuation as punctuation

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


# Read the crawled summary files of CSA data
def read_corpus():
    base_path = Path(__file__).parent
    path = base_path / "../data/raw_data/summaries/summary_file.txt"

    with open(path, 'r') as read_file:
        corpus_dict = json.load(read_file)
        return corpus_dict

#Function to tokenize the text blob
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    lemma = find_lemma(tokens)
    return lemma

# Remove stop words, punctuation, lowercase and accents
def preprocess(corpus_dict):
    for title, summary in corpus_dict.items():
        summary_without_punctuation = convert_lowercase_and_remove_punctuation(summary)
        summary_without_stop_words = remove_stop_words(summary_without_punctuation)
        corpus_dict[title] = summary_without_stop_words
    return corpus_dict


# Remove lowercase and punctuation
def convert_lowercase_and_remove_punctuation(text):
    remove_punctuation_map = dict((ord(char), None) for char in punctuation)
    return text.lower().translate(remove_punctuation_map)

# Remove stop words
def remove_stop_words(data):
    tokens = nltk.word_tokenize(data)
    filtered_text = ' '.join([w for w in tokens if not w in stopwords.words('english')])
    return filtered_text

# Lemmatize words for better matching
def find_lemma(tokens):
    wordnet_lemmatizer = WordNetLemmatizer()
    result = []
    for word in tokens:
        lemma_word = wordnet_lemmatizer.lemmatize(word)
        result.append(lemma_word)
    return result


# Find unigrams and bigrams for the corpus and train the vectorizer on these bag of words
def find_bigrams(corpus_dict):
    # Find both unigrams and bigrams
    vectorizer = TfidfVectorizer(tokenizer=tokenize, ngram_range=(1, 2))
    corpus_tfidf = vectorizer.fit_transform(corpus_dict.values())

    # Dump the file
    base_path = Path(__file__).parent
    vectorizer_path = base_path / "../data/trained_model/corpus_vectorizer.pkl"
    corpus_tfidf_path = base_path / "../data/trained_model/corpus_tfidf.pkl"

    pickle.dump(vectorizer, open(vectorizer_path, "wb"))
    pickle.dump(corpus_tfidf, open(corpus_tfidf_path, "wb"))


def create_and_train_tf_idf_model():
    # Read the corpus
    corpus_dict = read_corpus()

    #Normalize and preprocess the summaries of the data set
    corpus_dict = preprocess(corpus_dict)

    # Find bigrams and create tf-idf matrix
    find_bigrams(corpus_dict)

def main():
    create_and_train_tf_idf_model()

#Call main method
if __name__ == "__main__":
    main()
