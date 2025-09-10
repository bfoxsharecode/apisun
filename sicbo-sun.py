import time
import requests
from functools import wraps
from flask import Flask, jsonify

# ================================================================
# === Advanced Axios-like Configuration & Error Handling ===
# ================================================================

class APIClient:
    """A simple API client with retry logic."""
    def __init__(self, timeout=15):
        self.timeout = timeout

    def get_with_retry(self, url, max_retries=7):
        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
                return response
            except requests.exceptions.RequestException as e:
                retries += 1
                if retries >= max_retries:
                    raise
                delay = 2**retries * 0.5
                print(f"[APIClient] Connection or server error ({e}). Retrying attempt {retries} after {delay}s...")
                time.sleep(delay)

api_client = APIClient()

# ================================================================
# === Basic Analysis Functions ===
# ================================================================

def get_true_sum_probabilities():
    """Calculates the theoretical probabilities for each sum in Sic Bo."""
    sum_counts = {}
    for d1 in range(1, 7):
        for d2 in range(1, 7):
            for d3 in range(1, 7):
                s = d1 + d2 + d3
                sum_counts[s] = sum_counts.get(s, 0) + 1
    total_outcomes = 216
    probabilities = {s: count / total_outcomes for s, count in sum_counts.items()}
    return probabilities

def get_history_status(history):
    """Maps raw history to a simplified status."""
    status_list = []
    for h in history:
        status_list.append({
            'phien': h.get('phien'),
            'tong': h.get('tong'),
            'taixiu': 'T' if h.get('tong', 0) >= 11 else 'X',
            'chanle': 'C' if h.get('tong', 0) % 2 == 0 else 'L',
        })
    return status_list

# ================================================================
# === Advanced Analysis Algorithms (AI Super-Accurate) ===
# ================================================================

# Model 1: Markov Chain & Streak Analysis
def analyze_markov_chain_and_streak(status_history):
    if len(status_history) < 5:
        return {'prediction': None, 'certainty': 0, 'description': "Not enough data for Markov and Streak Analysis."}

    # Streak detection
    streak_count = 1
    streak_value = status_history[0]['taixiu']
    for i in range(1, min(len(status_history), 15)):
        if status_history[i]['taixiu'] == streak_value:
            streak_count += 1
        else:
            break
    
    if streak_count >= 4:
        if streak_count < 7:
            return {
                'prediction': streak_value,
                'certainty': min(100, 50 + streak_count * 5),
                'description': f"Markov: Detected a streak of {streak_count} sessions. Predicting it will continue."
            }
        else:
            inverse_prediction = 'X' if streak_value == 'T' else 'T'
            return {
                'prediction': inverse_prediction,
                'certainty': min(100, 50 + (streak_count - 7) * 2),
                'description': f"Markov: Very long streak ({streak_count} sessions). High chance of a reversal."
            }

    # Markov chain analysis
    last_two_states = ''.join([s['taixiu'] for s in status_history[0:2][::-1]])
    transitions = {}
    for i in range(2, len(status_history)):
        key = status_history[i-2]['taixiu'] + status_history[i-1]['taixiu']
        next_state = status_history[i]['taixiu']
        transitions.setdefault(key, {'T': 0, 'X': 0})
        transitions[key][next_state] += 1
    
    current_transition = transitions.get(last_two_states)
    if current_transition:
        total = current_transition['T'] + current_transition['X']
        if total > 0:
            predicted = 'T' if current_transition['T'] > current_transition['X'] else 'X'
            certainty = round((max(current_transition['T'], current_transition['X']) / total) * 100)
            return {
                'prediction': predicted,
                'certainty': certainty,
                'description': f'Markov: Based on sequence "{last_two_states}", predicts "{predicted}" with high probability.'
            }

    return {'prediction': None, 'certainty': 0, 'description': "Markov: No clear pattern found."}

# Model 2: Multi-Timeframe Balance Analysis
def analyze_multi_timeframe_balance(status_history):
    if len(status_history) < 30:
        return {'prediction': None, 'certainty': 0, 'description': "Balance: Needs more data to analyze."}

    short_term_counts = {'T': 0, 'X': 0}
    long_term_counts = {'T': 0, 'X': 0}

    for h in status_history[:15]:
        short_term_counts[h['taixiu']] += 1
    for h in status_history[:30]:
        long_term_counts[h['taixiu']] += 1

    short_term_diff = abs(short_term_counts['T'] - short_term_counts['X'])
    long_term_diff = abs(long_term_counts['T'] - long_term_counts['X'])

    if short_term_diff >= 4:
        prediction = 'X' if short_term_counts['T'] > short_term_counts['X'] else 'T'
        certainty = min(short_term_diff * 10, 80)
        return {
            'prediction': prediction,
            'certainty': certainty,
            'description': "Balance: Short-term balance (15s) is skewed. Predicting a rebalancing."
        }

    if long_term_diff >= 7:
        prediction = 'X' if long_term_counts['T'] > long_term_counts['X'] else 'T'
        certainty = min(long_term_diff * 8, 75)
        return {
            'prediction': prediction,
            'certainty': certainty,
            'description': "Balance: Long-term balance (30s) is skewed. High chance of a reversal to rebalance."
        }

    return {'prediction': None, 'certainty': 0, 'description': "Balance: The trend is stable."}

# Model 3: Heuristic & Complex Pattern Learning
def analyze_heuristic_patterns(status_history):
    if len(status_history) < 5:
        return {'prediction': None, 'certainty': 0, 'description': "Heuristic: Needs at least 5 sessions to analyze."}

    recent = [s['taixiu'] for s in status_history[:5]]

    # 3-2-1 pattern
    if recent[0] != recent[1] and recent[1] == recent[2] and recent[2] == recent[3]:
        prediction = recent[0]
        return {'prediction': prediction, 'certainty': 75, 'description': "Heuristic: Identified 3-2-1 pattern. Predicting a reversal."}

    # 1-1-2 pattern
    if recent[0] != recent[1] and recent[1] != recent[2] and recent[2] == recent[3]:
        prediction = recent[1]
        return {'prediction': prediction, 'certainty': 70, 'description': "Heuristic: Identified 1-1-2 pattern. Predicting a short streak."}

    return {'prediction': None, 'certainty': 0, 'description': "Heuristic: No complex patterns found."}

# ================================================================
# === Advanced Prediction Algorithm (Multi-Dimensional Analysis) ===
# ================================================================

DYNAMIC_WEIGHTS = {
    'markov': 1.0,
    'balance': 1.0,
    'heuristic': 1.0
}

def predict_advanced(history):
    print('--------------------------------------------------')
    print('[AI Prediction] Starting prediction process...')
    if len(history) < 15:
        print('[AI Prediction] Not enough data for advanced prediction (requires > 15 sessions).')
        return {
            'du_doan': "Not enough data",
            'doan_vi': [],
            'confidence': 0,
            'giai_thich': "Please wait for more sessions for the AI to analyze."
        }

    status_history = get_history_status(history)
    markov_result = analyze_markov_chain_and_streak(status_history)
    balance_result = analyze_multi_timeframe_balance(status_history)
    heuristic_result = analyze_heuristic_patterns(status_history)

    scores = {'T': 0, 'X': 0}
    explanation = []
    prediction_breakdown = {}

    all_models = [
        {'name': 'markov', 'result': markov_result, 'weight': DYNAMIC_WEIGHTS['markov']},
        {'name': 'balance', 'result': balance_result, 'weight': DYNAMIC_WEIGHTS['balance']},
        {'name': 'heuristic', 'result': heuristic_result, 'weight': DYNAMIC_WEIGHTS['heuristic']},
    ]

    for model in all_models:
        if model['result']['prediction']:
            score = model['result']['certainty'] * model['weight']
            scores[model['result']['prediction']] += score
            explanation.append(model['result']['description'])
            prediction_breakdown[model['name']] = {
                'prediction': model['result']['prediction'],
                'certainty': model['result']['certainty'],
                'score': score
            }

    final_prediction = None
    final_confidence = 0
    total_score = scores['T'] + scores['X']

    if total_score > 0:
        if scores['T'] > scores['X']:
            final_prediction = 'Tài'
            final_confidence = round((scores['T'] / total_score) * 100)
        elif scores['X'] > scores['T']:
            final_prediction = 'Xỉu'
            final_confidence = round((scores['X'] / total_score) * 100)
        else:
            final_prediction = 'Tài'
            final_confidence = 50
    else:
        final_prediction = 'Tài'
        final_confidence = 50
        explanation.append("Models did not find enough basis, predicting randomly based on probability.")

    agreement_count = sum(1 for model_name, breakdown in prediction_breakdown.items()
                          if breakdown['prediction'] == ('T' if final_prediction == 'Tài' else 'X'))

    if agreement_count < 2:
        final_confidence = min(final_confidence, 60)

    predicted_sums = predict_sums(history, final_prediction)

    print('[AI Prediction] Final Result:', {
        'du_doan': final_prediction,
        'doan_vi': predicted_sums,
        'confidence': final_confidence,
        'giai_thich': ' '.join(explanation)
    })
    print('--------------------------------------------------')

    return {
        'du_doan': final_prediction,
        'doan_vi': predicted_sums,
        'confidence': final_confidence,
        'giai_thich': ' '.join(explanation)
    }

def predict_sums(history, taixiu_prediction):
    """Predicts specific sums (3 dice) based on historical data and probabilities."""
    sum_probabilities = get_true_sum_probabilities()
    recent_sums = [h['tong'] for h in history[:70]]
    sums_to_predict = [11, 12, 13, 14, 15, 16, 17] if taixiu_prediction == 'Tài' else [4, 5, 6, 7, 8, 9, 10]

    sum_scores = {}
    for s in sums_to_predict:
        # Score 1: Recent frequency (vs. theoretical probability)
        recent_freq = recent_sums.count(s) / 70
        freq_score = abs(recent_freq - sum_probabilities.get(s, 0)) * 100

        # Score 2: Time not seen
        try:
            last_seen_index = recent_sums.index(s)
            time_not_seen = last_seen_index
        except ValueError:
            time_not_seen = len(recent_sums)
        time_score = min(time_not_seen / 20, 1.0)

        # Score 3: Alignment with Tai/Xiu prediction
        pattern_score = 1.5 if (s >= 11 and taixiu_prediction == 'Tài') or (s < 11 and taixiu_prediction == 'Xỉu') else 1.0

        total_score = (freq_score * 1.5) + (time_score * 2.0) + (pattern_score * 1.0)
        sum_scores[s] = total_score
    
    predicted_sums = sorted(sum_scores.items(), key=lambda item: item[1], reverse=True)[:3]
    return [int(s[0]) for s in predicted_sums]

# ================================================================
# === Flask APP & API Endpoint ===
# ================================================================

app = Flask(__name__)

@app.errorhandler(Exception)
def handle_error(e):
    print(f"[Flask Error] Server error: {e}")
    return jsonify({
        'error': {
            'message': str(e) or 'An unknown server error occurred.'
        }
    }), 500

@app.route('/api/sicbo/khoiancuto', methods=['GET'])
def get_prediction():
    try:
        print("[API Request] Received request from client...")
        
        # Use the custom API client with retry logic
        response = api_client.get_with_retry('https://sicbosun-100.onrender.com/api')
        history = response.json()
        
        print(f"[API Request] Successfully fetched historical data from third-party server. Sessions: {len(history)}")

        if not isinstance(history, list) or len(history) < 15:
            return jsonify({
                'phien_sau': "N/A",
                'du_doan': "Not enough historical data to predict.",
                'doan_vi': [],
                'do_tin_cay': "0%",
                'giai_thich': "Need at least 15 sessions to start advanced analysis.",
                'luu_y': "MUA TOOL THÌ IB @lykhoian NHÉ HÂHHAHAHAHA"
            })

        latest = history[0]
        phien_after = "N/A"
        try:
            phien_before_str = str(latest.get('phien', ''))
            cleaned_phien = ''.join(filter(str.isdigit, phien_before_str))
            if cleaned_phien:
                phien_after = str(int(cleaned_phien) + 1)
        except (ValueError, TypeError):
            pass

        prediction_results = predict_advanced(history)
        
        result = {
            'phien_truoc': latest.get('phien'),
            'xuc_xac': f"{latest.get('xuc_xac_1')} - {latest.get('xuc_xac_2')} - {latest.get('xuc_xac_3')}",
            'tong': latest.get('tong'),
            'ket_qua': latest.get('ket_qua'),
            'phien_sau': phien_after,
            'du_doan': prediction_results['du_doan'],
            'doan_vi': prediction_results['doan_vi'],
            'do_tin_cay': f"{prediction_results['confidence']}%",
            'giai_thich': prediction_results['giai_thich'],
            'luu_y': "Warning: This is a statistical analysis tool, not a guaranteed prediction. Sic Bo is a game of chance.",
            'lien_he': "@lykhoian"
        }
        
        return jsonify(result)
        
    except requests.exceptions.RequestException as e:
        # Catch specific requests errors and return a user-friendly message
        return jsonify({
            'error': {
                'message': f'Failed to retrieve data from the external API: {e}'
            }
        }), 503
    except Exception as e:
        # General catch-all for other errors
        return handle_error(e)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=true, port=8080)
    print("✅ Sic Bo Analysis & Prediction API is running at http://localhost:8080")
    print("⚠️ Note: This is a statistical analysis tool, not a guaranteed prediction. Sic Bo is a game of chance.")
