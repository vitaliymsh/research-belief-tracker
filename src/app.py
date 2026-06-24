import os
import sys
import asyncio
try:
    from contextlib import asynccontextmanager
except ImportError:
    from async_generator import asynccontextmanager
from fastapi import FastAPI
from starlette.requests import Request
from starlette.websockets import WebSocket, WebSocketDisconnect
from starlette.responses import HTMLResponse

# Ensure current folder and src folder are in sys.path
src_dir = os.path.dirname(__file__)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from config import get_logger
from pipeline import TrackingPipeline
from utils import ConnectionManager

logger = get_logger("App")
manager = ConnectionManager()

# Cache the index template
_TEMPLATE_CACHE = None

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global _TEMPLATE_CACHE
    logger.info("System starting up...")
    manager.loop = asyncio.get_event_loop()
    
    # Pre-load template
    template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates", "index.html")
    try:
        with open(template_path, "r") as f:
            _TEMPLATE_CACHE = f.read()
    except Exception as e:
        logger.error(f"Failed to load template: {e}")
        _TEMPLATE_CACHE = "Error: Template not found"

    # Initialize the pipeline
    pipeline = TrackingPipeline()
    app.state.pipeline = pipeline
    pipeline.on_payload_ready = manager.send_broadcast_sync
    pipeline.start()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("System shutting down...")
    if hasattr(app.state, "pipeline"):
        app.state.pipeline.camera.release()

@app.post("/toggle_tracking")
async def toggle_tracking(request: Request):
    enabled = request.app.state.pipeline.toggle_tracking()
    return {"enabled": enabled}

@app.post("/toggle_pipeline")
async def toggle_pipeline(request: Request):
    enabled = request.app.state.pipeline.toggle_pipeline()
    return {"enabled": enabled}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        manager.disconnect(websocket)

@app.get("/")
async def index(request: Request):
    pipeline = request.app.state.pipeline
    replacements = {
        "__STATUS__": "ON" if pipeline.tracking_enabled else "OFF",
        "__BTN_CLASS__": "" if pipeline.tracking_enabled else "disabled",
        "__PIPELINE_STATUS__": "ON" if pipeline.pipeline_enabled else "OFF",
        "__PIPELINE_BTN_CLASS__": "" if pipeline.pipeline_enabled else "disabled"
    }
    
    content = _TEMPLATE_CACHE or "Template Error"
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)
    return HTMLResponse(content)

if __name__ == "__main__":
    # Workaround for websockets 8.x KeyError in keepalive_ping
    try:
        import websockets.protocol
        async def dummy_keepalive_ping(self):
            try:
                await asyncio.sleep(86400)
            except asyncio.CancelledError:
                pass
        websockets.protocol.WebSocketCommonProtocol.keepalive_ping = dummy_keepalive_ping
    except ImportError:
        pass

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
