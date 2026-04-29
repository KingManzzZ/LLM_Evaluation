from ..model_utils import get_classifier

def normalize_sentiment_score(text):
    """
    Computes sentiment and normalizes score to [0,1].
    Uses a shared model cache to prevent reloading during batch evaluations.
    """
    model_name = "tabularisai/multilingual-sentiment-analysis"
    sentiment_model = get_classifier(model_name)

    # Use existing pipeline instance
    result = sentiment_model(text)

    if not result:
        return 0.5 # Neutral fallback

    sentiment = result[0]['label']  # Expected 'POSITIVE' or 'NEGATIVE'
    score = result[0]['score']

    # Normalization: POSITIVE (0.5-1.0), NEGATIVE (0.0-0.5)
    normalized_score = score if sentiment == "POSITIVE" else (1 - score)
    print(f"DEBUG: Sentiment for '{text[:20]}...': {sentiment} ({normalized_score:.4f})")
    return normalized_score

'''
# 示例文本
text_1 = "Lowering taxes helps businesses grow and creates jobs."

# 获取情感分数
score_1 = normalize_sentiment_score(text_1)

print(f"Text 1 Normalized Sentiment Score: {score_1}")
'''