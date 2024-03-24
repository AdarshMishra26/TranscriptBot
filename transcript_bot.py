import os
import re
import time
import logging
import requests
from requests.exceptions import RequestException
from typing import List, Dict
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from youtube_transcript_api import YouTubeTranscriptApi
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import google.generativeai as genai

# nltk.download('punkt')
# nltk.download('stopwords')

with open('config.json') as f:
    params = json.load(f)['param']

class TranscriptBot:
    def __init__(self):
        self.model_engine = "local-model"  # Set the desired LLM model engine
        self.base_url = "http://localhost:1234/v1"
        self.rake = Rake()
        self.vectorizer = TfidfVectorizer()
        self.transcript = ""

    @staticmethod
    def get_video_id(url: str) -> str:
        """Extract the video ID from a YouTube video URL."""
        video_id = url.replace('https://www.youtube.com/watch?v=', '')
        return video_id

    def fetch_and_fit_transcript(self, video_id: str) -> str:
        """Fetch the transcript of the given YouTube video and fit the vectorizer."""
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            self.transcript = "\n".join([entry["text"] for entry in transcript])
            self.vectorizer.fit([self.transcript])
            return self.transcript
        except Exception as e:
            return f"An error occurred while fetching and fitting the transcript: {str(e)}"

    def save_transcript_to_file(self, video_id: str):
        """Save the transcript of the given YouTube video to a text file."""
        try:
            filename = f"{video_id}_transcript.txt"
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(self.transcript)
            print(f"Transcript has been saved to {filename}")
        except Exception as e:
            print(f"An error occurred while saving the transcript: {str(e)}")

    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess the given text."""
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(str(text))
        words = [word for word in words if word.isalnum()]
        words = [word for word in words if not word in stop_words]
        words = [word.lower() for word in words]
        return words

    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from the given text using Rake (Rapid Automatic Keyword Extraction) algorithm."""
        self.rake.extract_keywords_from_text(text)
        return self.rake.get_ranked_phrases()

    def find_relevant_passage(self, user_input: str, transcript: str) -> str:
        # Split the transcript into sentences
        sentences = sent_tokenize(transcript)

        # Calculate the similarity between the user input and each sentence
        similarities = []
        for sentence in sentences:
            user_input_vector = self.vectorizer.transform([" ".join(self.preprocess_text(user_input))]).toarray().flatten()
            sentence_vector = self.vectorizer.transform([" ".join(self.preprocess_text(sentence))]).toarray().flatten()
            similarity = 1 - cosine(user_input_vector, sentence_vector)
            similarities.append(similarity)

        # Find the index of the sentence with the highest similarity
        max_similarity_index = similarities.index(max(similarities))

        # Return the most relevant passage (e.g., the sentence with the highest similarity and its surrounding sentences)
        context_size = 2
        start_index = max(0, max_similarity_index - context_size)
        end_index = min(len(sentences), max_similarity_index + context_size + 1)
        relevant_passage = "\n".join(sentences[start_index:end_index])

        return relevant_passage
    
    def generate_response(self, user_input: str, transcript: str) -> str:
        # relevant_passage = self.find_relevant_passage(user_input, transcript)

        # Configure the API key and create a GenerativeModel instance
        genai.configure(api_key=params['API_Key'])
        model = genai.GenerativeModel('gemini-pro')

        # Generate a response using the Gemini model
        prompt = f"Please answer the question based on the provided passage:\n\nQuestion: {user_input}\nPassage: {transcript}"
        response = model.generate_content(prompt)

        return response.text

    def run_chatbot(self, video_url: str):
        video_id = self.get_video_id(video_url)
        transcript = self.fetch_and_fit_transcript(video_id)  # Fetch and fit the transcript
        self.save_transcript_to_file(video_id)  # Save transcript to a file
        logging.basicConfig(filename="chatbot.log", level=logging.INFO)
        while True:
            user_input = input("Ask me a question: ")
            if user_input.lower() == "quit":
                break

            start_time = time.time()
            response = self.generate_response(user_input, transcript)
            response_time = time.time() - start_time
            logging.info(f"User Input: {user_input}")
            logging.info(f"Response Time: {response_time}")
            print(f"Chatbot Response: {response}")

            # Delete the generated transcript file
            transcript_file = f"{video_id}_transcript.txt"
            if os.path.exists(transcript_file):
                os.remove(transcript_file)

if __name__ == "__main__":
    bot = TranscriptBot()
    video_url = input("Please enter the YouTube URL: ")
    bot.run_chatbot(video_url)
