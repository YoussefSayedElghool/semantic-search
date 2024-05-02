import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from fuzzywuzzy import fuzz
import re

STOP_WORDS_FILE = "StopArabicWords.xlsx"


def clean_and_fuzzy_match(user_input, dataset_text):
    # Clean user input
    cleaned_input = preprocess_input(user_input)

    # Perform fuzzy matching
    best_matches = fuzzy_match_keywords(cleaned_input, dataset_text)

    # Construct corrected input
    corrected_input = ' '.join(best_matches)
    return corrected_input


def preprocess_input(input_text):
    cleaned_input = re.sub(r'[^\u0621-\u064A\s]', ' ', input_text)
    cleaned_input = re.sub(r'\s+', ' ', cleaned_input)
    cleaned_input = cleaned_input.replace("\n", " "). \
        replace("ة", "ه"). \
        replace("ؤ", "و"). \
        replace("ئ", "ي"). \
        replace("ى", "ي"). \
        replace("أ", "ا"). \
        replace("إ", "ا"). \
        replace("آ", "ا"). \
        replace("ء", "ا"). \
        strip()
    return cleaned_input


def fuzzy_match_keywords(user_input, dataset_text):
    # Remove stop Arabic words
    stop_arabic_words = set(pd.read_excel(STOP_WORDS_FILE)["Aword"])
    cleaned_keywords = set(user_input.split()) - stop_arabic_words

    # Calculate fuzzy match scores for each keyword
    best_matches = []
    all_dataset_words = dataset_text.split()
    for keyword in cleaned_keywords:
        scores = [fuzz.ratio(keyword, dataset_word)
                  for dataset_word in all_dataset_words]
        if max(scores) >= 80:
            best_match_index = scores.index(max(scores))
            best_match_word = all_dataset_words[best_match_index]
            best_matches.append(best_match_word)
    return best_matches


def search(labels, description, user_input, SIMILARITY_THRESHOLD=0.04):
    # Text Vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(description)

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    # Train Support Vector Machine (SVM) Classifier
    clf = SVC(kernel='linear')
    clf.fit(X, y)

    # Clean Up User Input and Perform Fuzzy Matching
    corrected_input = clean_and_fuzzy_match(
        user_input, ' '.join(description))
    print("Corrected Input:", corrected_input)

    # Preprocess the prompt text
    prompt_vector = vectorizer.transform([corrected_input])

    # Calculate the similarity between the prompt vector and all data points
    similarities = np.dot(X, prompt_vector.T).toarray().flatten()

    # Get the indices of data points with similarity scores above the threshold
    top_indices = np.where(similarities >= SIMILARITY_THRESHOLD)[0]
    # Sort top_indices based on similarity scores in descending order
    sorted_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

    if len(sorted_indices) > 0:
        # Get the corresponding labels in the sorted order
        predicted_labels_encoded = clf.predict(X[sorted_indices])
        predicted_labels = label_encoder.inverse_transform(
            predicted_labels_encoded)

        # list_result = [{"id": str(i + 1), "herb": label, "illness": label} for i, label in enumerate(top_labels)]
        return predicted_labels
    else:
        return [{"id": "1", "herb": "No Result", "illness": "No Result"}]


data = pd.read_excel("G:\\My Drive\\NLP\\NlP Project\\illAfterAddDiscChatGpt.xlsx")
user_input = "بطن"

results = search( data["ill"] , data["preprocessed_text"], user_input)

print("Search Results:")
for result in results:
    print(result)