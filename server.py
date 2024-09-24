import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server
import logging

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        allowed_locations = {
            "Albuquerque, New Mexico",
            "Carlsbad, California",
            "Chula Vista, California",
            "Colorado Springs, Colorado",
            "Denver, Colorado",
            "El Cajon, California",
            "El Paso, Texas",
            "Escondido, California",
            "Fresno, California",
            "La Mesa, California",
            "Las Vegas, Nevada",
            "Los Angeles, California",
            "Oceanside, California",
            "Phoenix, Arizona",
            "Sacramento, California",
            "Salt Lake City, Utah",
            "San Diego, California",
            "Tucson, Arizona"
        }

        if environ["REQUEST_METHOD"] == "GET":
            query_string = environ.get("QUERY_STRING", "")
            params = parse_qs(query_string)
            
            location = params.get("location", [None])[0]
            start_date_str = params.get("start_date", [None])[0]
            end_date_str = params.get("end_date", [None])[0]

            start_date = None
            end_date = None

            if start_date_str:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            if end_date_str:
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
                            
            filtered_reviews = []
            for review in reviews:
                review_location = review.get('Location')   
                if location and review_location not in allowed_locations:
                    continue
                if location and review_location != location:
                    continue

                review_timestamp_str = review.get('Timestamp')
                if review_timestamp_str:
                    try:
                        review_timestamp = datetime.strptime(review_timestamp_str, "%Y-%m-%d %H:%M:%S")  
                        if (start_date and review_timestamp < start_date) or (end_date and review_timestamp > end_date):
                            continue
                    except ValueError:
                        pass

                sentiment_scores = self.analyze_sentiment(review.get('ReviewBody', ''))
                review['sentiment'] = sentiment_scores
                filtered_reviews.append({
                    "ReviewId": review.get('ReviewId', str(uuid.uuid4())),
                    "ReviewBody": review.get('ReviewBody', ''),
                    "Location": review.get('Location', ''),
                    "Timestamp": review.get('Timestamp', ''),
                    "sentiment": sentiment_scores
                })
            filtered_reviews.sort(key=lambda x: x.get('sentiment', {}).get('compound', 0), reverse=True)
            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")

            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            
            return [response_body]

        if environ["REQUEST_METHOD"] == "POST":
            logging.debug("Received POST request")

            content_length = int(environ.get("CONTENT_LENGTH", 0))
            post_data = environ["wsgi.input"].read(content_length).decode("utf-8")
            logging.debug(f"POST data: {post_data}")

            post_params = parse_qs(post_data)
            logging.debug(f"POST params: {post_params}")

            location = post_params.get("Location", [None])[0]
            review_body = post_params.get("ReviewBody", [None])[0]

            if not review_body or not location:
                start_response("400 Bad Request", [("Content-Type", "application/json")])
                response_body = json.dumps({"error": "Missing required parameters 'ReviewBody' or 'Location'"}).encode("utf-8")
                return [response_body]

            if location not in allowed_locations:
                start_response("400 Bad Request", [("Content-Type", "application/json")])
                response_body = json.dumps({"error": "Invalid location"}).encode("utf-8")
                return [response_body]

            sentiment_scores = self.analyze_sentiment(review_body)

            review_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            response_data = {
                "ReviewId": review_id,
                "ReviewBody": review_body,
                "Location": location,
                "Timestamp": timestamp,
                "Sentiment": sentiment_scores
            }

            response_body = json.dumps(response_data, indent=2).encode("utf-8")
            start_response("201 Created", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            logging.debug(f"Response Data: {response_data}")

            return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = int(os.environ.get('PORT', 8000))
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()