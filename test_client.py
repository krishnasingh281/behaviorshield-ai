import asyncio
import websockets
import json

async def simulate_browser_client():
    uri = "ws://127.0.0.1:8000/ws/auth/user_94f2a18c"
    
    print(f"Connecting to AI Engine at {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected! Sending behavioral payload...\n")
            
            # Simulated payload: A user typing slightly slower than normal
            # Simulated payload: A Bot attacking via script (0ms flight, 10ms identical dwells)
            payload = {
                "events": [
                    {"type": "key_down", "target": "p", "timestamp": 1000},
                    {"type": "key_up", "target": "p", "timestamp": 1010}, 
                    {"type": "key_down", "target": "a", "timestamp": 1010}, 
                    {"type": "key_up", "target": "a", "timestamp": 1020},
                    {"type": "key_down", "target": "s", "timestamp": 1020}, 
                    {"type": "key_up", "target": "s", "timestamp": 1030}
                ],
                "context": {
                    "km_from_last_login": 0,
                    "hours_since_last_login": 1,
                    "is_trusted_device": True # The bot spoofed your device ID
                }
            }
            
            # Send the payload to the FastAPI server
            await websocket.send(json.dumps(payload))
            
            # Wait for the AI's real-time verdict
            response = await websocket.recv()
            
            print("--- AI Engine Verdict Received ---")
            print(json.dumps(json.loads(response), indent=2))
            
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(simulate_browser_client())