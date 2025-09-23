import asyncio
import websockets
import json

# client side for websocket

connected_clients = set()

async def handler(websocket):
    connected_clients.add(websocket)
    print("🌐 Client connected")
    try:
        async for message in websocket:
            print(f"📥 Received from client: {message}")
            # Echo back to the client
            await websocket.send(json.dumps({"echo": message}))
    except websockets.exceptions.ConnectionClosed:
        print("⚠️ Client disconnected")
    finally:
        connected_clients.remove(websocket)

async def main():
    print("WebSocket server listening on ws://localhost:8765")
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())