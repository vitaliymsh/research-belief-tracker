import asyncio
import threading
from starlette.websockets import WebSocket

class ConnectionManager:
    def __init__(self):
        self.active_connections = []
        self.loop = None
        self._latest_message = None
        self._lock = threading.Lock()
        self._broadcast_task = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Start the background broadcast loop if not already running
        if not self._broadcast_task or self._broadcast_task.done():
            self._broadcast_task = asyncio.get_event_loop().create_task(self._broadcast_loop())

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def _broadcast_loop(self):
        """Dedicated async loop running on FastAPI event loop to consume and send messages."""
        while self.active_connections:
            message = None
            with self._lock:
                if self._latest_message is not None:
                    message = self._latest_message
                    self._latest_message = None
            
            if message is not None:
                # Create send tasks in parallel and execute
                tasks = [self.send_to_connection(conn, message) for conn in list(self.active_connections)]
                if tasks:
                    await asyncio.gather(*tasks)
            
            # Rate-limit WebSocket updates to max ~30 FPS to reduce event loop congestion
            await asyncio.sleep(0.033)

    async def send_to_connection(self, connection: WebSocket, message: str):
        try:
            await connection.send_text(message)
        except Exception:
            self.disconnect(connection)

    def send_broadcast_sync(self, message: str):
        """Non-blocking write to the single-slot buffer. Extremely cheap and GIL-safe."""
        if not self.active_connections:
            return
        with self._lock:
            self._latest_message = message
