from ..model_utils import get_classifier

def detect_toxicity(text, toxic_cutoff=0.6):
    """
    Detects toxicity using a shared model instance to save resources.
    """
    model_path = "martin-ha/toxic-comment-model"
    classifier = get_classifier(model_path)

    # Use existing pipeline (will be cached via get_classifier)
    raw_results = classifier(text)
    print(f"DEBUG: Toxicity analysis result: {raw_results}")

    # Interpretation logic for 'toxic-comment-model'
    toxic_score = 0.0
    if isinstance(raw_results, list) and len(raw_results) > 0:
        res = raw_results[0]
        # Handle single label output like {'label': 'toxic', 'score': 0.9}
        if res.get('label') == 'toxic':
            toxic_score = res.get('score', 0.0)
        # Handle non-toxic as label
        elif res.get('label') == 'non-toxic':
            toxic_score = 1.0 - res.get('score', 0.0)

    return round(toxic_score, 4)
