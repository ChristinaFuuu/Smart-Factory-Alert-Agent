from datetime import datetime


def explain_reason(row):
    # Simple rule-based reason extraction
    if row.get('vibration', 0) > 0.07:
        return 'High vibration detected'
    if row.get('temp', 0) > 52 or row.get('temp', 0) < 43:
        return 'Temperature outside normal range'
    if row.get('pressure', 0) > 1.08 or row.get('pressure', 0) < 0.97:
        return 'Pressure outside normal range'
    return 'Unusual pattern detected'


def generate_alert(timestamp, abnormal_prob, row=None):
    ts = timestamp if isinstance(timestamp, str) else timestamp.strftime('%Y-%m-%d %H:%M:%S')
    level = 'NORMAL'
    action = 'No action required'

    if abnormal_prob >= 0.8:
        level = 'CRITICAL'
        action = 'Stop machine immediately and notify engineer'
    elif abnormal_prob >= 0.6:
        level = 'WARNING'
        action = 'Schedule maintenance inspection'

    reason = explain_reason(row if row is not None else {})

    text = f"[{ts}]\nStatus: {level}\nAbnormal Probability: {abnormal_prob:.2f}\nReason: {reason}\nSuggested Action: {action}\n"
    return {'timestamp': ts, 'level': level, 'prob': float(abnormal_prob), 'reason': reason, 'action': action, 'text': text}


if __name__ == '__main__':
    sample = generate_alert(datetime.now(), 0.86, {'vibration': 0.09})
    print(sample['text'])
