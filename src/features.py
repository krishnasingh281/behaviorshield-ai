import numpy as np

def extract_keystroke_features(events):
    """
    Extracts dwell time and flight time from chronological keystroke logs.
    """
    keystrokes = [e for e in events if e['type'] in ['key_down', 'key_up']]
    dwell_times = []
    flight_times = []
    
    key_down_timestamps = {}
    last_key_up_time = None
    
    for event in keystrokes:
        target = event['target']
        ts = event['timestamp']
        
        if event['type'] == 'key_down':
            key_down_timestamps[target] = ts
            if last_key_up_time is not None:
                # Flight Time: Gap between previous key release and current key press
                flight_times.append(ts - last_key_up_time)
                
        elif event['type'] == 'key_up':
            if target in key_down_timestamps:
                # Dwell Time: Duration a single key was held down
                dwell_times.append(ts - key_down_timestamps[target])
                last_key_up_time = ts
                del key_down_timestamps[target]
                
    return dwell_times, flight_times

def extract_swipe_features(events):
    """
    Calculates touch movement velocities from screen interaction sequences.
    """
    swipes = [e for e in events if e['type'] in ['touch_start', 'touch_end']]
    velocities = []
    
    # Process alternating touch pairs (start -> end)
    for i in range(0, len(swipes) - 1, 2):
        start = swipes[i]
        end = swipes[i+1]
        
        if start['type'] == 'touch_start' and end['type'] == 'touch_end':
            dt = end['timestamp'] - start['timestamp']
            if dt > 0:
                # Distance formula: sqrt((x2-x1)^2 + (y2-y1)^2)
                distance = np.sqrt((end['x'] - start['x'])**2 + (end['y'] - start['y'])**2)
                velocities.append(distance / dt)
                
    return velocities

def build_feature_vector(raw_payload):
    """
    Main entry point to convert raw client payloads into a single flat vector.
    """
    events = raw_payload.get('events', [])
    
    dwells, flights = extract_keystroke_features(events)
    swipes = extract_swipe_features(events)
    
    # Build unified vector with statistical fallback values if lists are empty
    feature_vector = {
        "mean_dwell": float(np.mean(dwells)) if dwells else 0.0,
        "std_dwell": float(np.std(dwells)) if len(dwells) > 1 else 0.0,
        "mean_flight": float(np.mean(flights)) if flights else 0.0,
        "mean_swipe_velocity": float(np.mean(swipes)) if swipes else 0.0
    }
    
    return feature_vector