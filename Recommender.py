import collections
import json
import pickle
from pathlib import Path

import nltk
from sklearn.metrics.pairwise import cosine_similarity


def process_query(query):
    # Paths to load the trained model
    base_path = Path(__file__).parent
    corpus_dict_path = base_path / "data/raw_data/summaries/summary_file.txt"
    vectorizer_path = base_path / "data/trained_model/corpus_vectorizer.pkl"
    corpus_tfidf_path = base_path / "data/trained_model/corpus_tfidf.pkl"

    corpus_vectorizer = pickle.load(open(vectorizer_path, "rb"))
    corpus_tfidf = pickle.load(open(corpus_tfidf_path, "rb"))
    corpus_dict = collections.OrderedDict()
    with open(corpus_dict_path, 'r') as read_file:
        corpus_dict = json.load(read_file)

    tfidf_matrix_test = corpus_vectorizer.transform([query])
    cosine_similarity_matrix = cosine_similarity(corpus_tfidf, tfidf_matrix_test)
    return corpus_dict, cosine_similarity_matrix


# Map the original corpus to its cosine score
def get_recommendations(corpus_dict, cosine_similarity_matrix):
    items = list(corpus_dict.items())
    recommendation_dict = collections.OrderedDict()

    for i in range(0, len(items)):
        corpus_text = items[i]
        title = corpus_text[0]
        cosine_score = cosine_similarity_matrix[i]
        recommendation_dict[title] = cosine_score

    sorted_recommendation_dict = {k: v for k, v in
                                  sorted(recommendation_dict.items(), reverse=True, key=lambda item: item[1])}
    return sorted_recommendation_dict


# Print the recommendations
def print_recommendations(sorted_recommendation_dict, corpus_dict):
    print("Based on your search query, look at these datasets from CSA :")

    # We limit the search results to 5
    limit = 5
    count = 0
    result_dict = collections.OrderedDict()
    for title, cosine_similarity in sorted_recommendation_dict.items():
        if cosine_similarity == 0.0 or count == limit:
            break
        result_dict[title] = corpus_dict[title]
        count += 1

    for k,v in result_dict.items():
        print(k)
        print(v)
        print("-----------------------------------------------------------------------------------------")

    return result_dict


# Function to tokenize the text blob
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    lemma = find_lemma(tokens)
    return lemma


# Lemmatize words for better matching
def find_lemma(tokens):
    wordnet_lemmatizer = nltk.WordNetLemmatizer()
    result = []
    for word in tokens:
        lemma_word = wordnet_lemmatizer.lemmatize(word)
        result.append(lemma_word)
    return result


def get_user_query(query_dict):
    # Read the input document that needs to be compared
    return query_dict["query"]


def main():
    query_dict = {"query": "How did the universe begin? What are the earliest stars?"}
    #query_dict = {"query": "I want data about mars planet and past missions"}
    #query_dict = {"query": "Ozone gases and particles"}

    user_query = get_user_query(query_dict)
    corpus_dict, cosine_similarity_matrix = process_query(user_query)
    recommendation_dict = get_recommendations(corpus_dict, cosine_similarity_matrix)
    print_recommendations(recommendation_dict, corpus_dict)


# Call main method
if __name__ == "__main__":
    main()
