import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import argparse
import numpy as np

class SentimentAnalyzer:
    def __init__(self, model_path='sentiment_model.h5'):
        self.model = keras.models.load_model(model_path)
        self.max_length = 200
        self.word_index = keras.datasets.imdb.get_word_index()
        
    def preprocess_text(self, text):
        words = text.lower().split()
        sequence = [self.word_index.get(word, 0) for word in words]
        padded = pad_sequences([sequence], maxlen=self.max_length, padding='post')
        return padded
    
    def analyze(self, text):
        processed = self.preprocess_text(text)
        prediction = self.model.predict(processed, verbose=0)[0][0]
        
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        return sentiment, confidence

def main():
    parser = argparse.ArgumentParser(description='Analyze sentiment of text')
    parser.add_argument('--text', required=True, help='Text to analyze')
    args = parser.parse_args()
    
    analyzer = SentimentAnalyzer()
    sentiment, confidence = analyzer.analyze(args.text)
    
    print(f'\nText: {args.text}')
    print(f'Sentiment: {sentiment}')
    print(f'Confidence: {confidence:.2%}')

if __name__ == '__main__':
    main()
