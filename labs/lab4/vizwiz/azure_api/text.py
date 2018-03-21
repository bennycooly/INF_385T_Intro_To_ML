
import io
import requests
import json
from collections import OrderedDict

text_analytics_base_url = "https://southcentralus.api.cognitive.microsoft.com/text/analytics/v2.0/"
language_api_url = text_analytics_base_url + "languages"
sentiment_api_url = text_analytics_base_url + "sentiment"
key_phrases_api_url = text_analytics_base_url + "keyPhrases"


class TextApi:
    def __init__(self, subscription_key):
        self.subscription_key = subscription_key

    # Analyze text documents using Azure Text Analytics API
    def analyze_text(self, questions):
        # Convert questions to documents
        questions_documents = []
        for i, question in enumerate(questions):
            questions_documents.append({
                "id": i,
                "text": question
            })
        documents = {
            "documents": questions_documents
        }

        headers = {
            "Ocp-Apim-Subscription-Key": self.subscription_key
        }
        
        # query API
        response = requests.post(language_api_url, headers=headers, json=documents)
        languages = response.json().get("documents")
        languages = list(map(lambda l: l.get("detectedLanguages"), languages))
        
        response = requests.post(sentiment_api_url, headers=headers, json=documents)
        sentiments = response.json().get("documents")
        sentiments = list(map(lambda l: l.get("score"), sentiments))
        
        response = requests.post(key_phrases_api_url, headers=headers, json=documents)
        key_phrases = response.json().get("documents")
        key_phrases = list(map(lambda l: l.get("keyPhrases"), key_phrases))

        # extract features from analysis
        # print(list(zip(languages, sentiments, key_phrases)))
        features = list(map(lambda t: self.extract_features(t[0], t[1], t[2]), zip(languages, sentiments, key_phrases)))

        # print(features)
        return features
    
    def extract_features(self, languages, sentiments, key_phrases):
        features = [
            # language score
            self.extract_languages(languages),

            # sentiment score is already a double
            sentiments,

            # key phrases
            self.extract_key_phrases(key_phrases)
        ]
        return features
    
    # Return the average score
    def extract_languages(self, languages):
        sum = 0
        for language in languages:
            sum += language.get("score")
        return sum / len(languages)
    
    def extract_key_phrases(self, key_phrases):
        return len(key_phrases)


