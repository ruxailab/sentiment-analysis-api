"""
This module contains the service layer for sentiment analysis.
"""
from collections import Counter
from app.config import Config
from app.utils.logger import logger

# Data Layer for fetching and processing transcripts
from app.data.sentiment_data import SentimentDataLayer

config = Config().config  # Load the configuration

class SentimentService:
    def __init__(self):
        self.debug = config.get('debug')
        self.sentiment_data_layer = SentimentDataLayer(config)

    def analyze(self, text: str) -> tuple:
        """
        Legacy method for single text analysis (No context, No timeline).
        :param text: Input text for sentiment analysis.
        :return: predicted label, and confidence score.
        """
        try:
            result = self.sentiment_data_layer.analyze(text)

            if isinstance(result, dict) and 'error' in result:
                return {'error': result['error']}

            return {
                'label': result['label'],
                'confidence': result['confidence']
            }
        
        except Exception as e:
            logger.error(f"[error] [Service Layer] [SentimentService] [analyze] {str(e)}")
            return {'error': 'An unexpected error occurred while processing the request.'}

    def analyze_timeline(self, chunks: list) -> dict:
        """
        Perform sentiment analysis on a list of transcript chunks using Context (Past + Future).
        :param chunks: List of chunks with text and timestamps from Whisper.
        :return: JSON object with 'average_emotions' and 'timeline'.
        """
        timeline = []
        emotion_counts = Counter()
        
        # Initialize all standard emotions to 0
        all_emotions = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]
        for emo in all_emotions:
            emotion_counts[emo] = 0

        try:
            for i, chunk in enumerate(chunks):
                # --- 1. EXTRACT TIMESTAMP ---
                # Whisper returns 'timestamp': (start, end)
                timestamps = chunk.get('timestamp')
                
                # Robust extraction in case of None or varying formats
                if isinstance(timestamps, (tuple, list)) and len(timestamps) >= 2:
                    start_time = timestamps[0]
                    end_time = timestamps[1]
                else:
                    start_time, end_time = 0.0, 0.0

                # --- 2. BUILD CONTEXT (The Sandwich) ---
                current_text = chunk.get('text', '').strip()
                
                # Past Context (Previous chunk's text)
                past_text = chunks[i-1]['text'].strip() if i > 0 else ""
                
                # Future Context (Next chunk's text)
                future_text = chunks[i+1]['text'].strip() if i < len(chunks) - 1 else ""

                # Combine them. Note: Adjust ' [SEP] ' if your specific model uses a different separator.
                # For many standard models, just space or " [SEP] " works well.
                full_context_input = f"{past_text} [SEP] {current_text} [SEP] {future_text}"

                # --- 3. ANALYZE ---
                # We feed the FULL context to get the emotion for the CURRENT moment
                result = self.sentiment_data_layer.analyze(full_context_input)

                if isinstance(result, dict) and 'error' in result:
                    continue  # Skip chunks that fail analysis

                label = result['label']
                confidence = result['confidence']

                # --- 4. POPULATE TIMELINE ---
                timeline_entry = {
                    "id": i + 1,
                    "starting_time": start_time,
                    "ending_time": end_time,
                    "emotion": label,
                    # Convert 0.98 -> 98 (Int)
                    "value": int(confidence * 100) if confidence <= 1 else int(confidence)
                }
                timeline.append(timeline_entry)

                # --- 5. UPDATE COUNTS ---
                # Normalize label (e.g. "SURPRISED" -> "Surprised") to match keys
                normalized_label = label.capitalize()
                if normalized_label in emotion_counts:
                    emotion_counts[normalized_label] += 1
                elif normalized_label == "Joy": # Handle common mapping difference
                    emotion_counts["Happy"] += 1

            # --- 6. CALCULATE AVERAGES ---
            total_sentences = len(timeline)
            average_emotions = {}

            if total_sentences > 0:
                for emo in all_emotions:
                    percent = (emotion_counts[emo] / total_sentences) * 100
                    average_emotions[emo] = round(percent, 2)
            else:
                average_emotions = {emo: 0.0 for emo in all_emotions}

            return {
                "average_emotions": average_emotions,
                "timeline": timeline
            }

        except Exception as e:
            logger.error(f"[error] [Service Layer] [SentimentService] [analyze_timeline] {str(e)}")
            return {'error': 'An unexpected error occurred while processing the timeline.'}

# if __name__ == "__main__":
#     sentiment_service = SentimentService()

#     result = sentiment_service.analyze("I love this product!")
#     print("result",result)

#     result = sentiment_service.analyze("I hate this product!")
#     print("result",result)

#     result = sentiment_service.analyze("I am neutral about this product.")
#     print("result",result)

# #  Run:
# #  python -m app.services.sentiment_service