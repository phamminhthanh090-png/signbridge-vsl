"""SignBridge - máy chủ web (FastAPI). Lõi nhận diện nằm ở cloud_core.py."""
import os
import json
import queue
import asyncio
import threading

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

import cloud_core as core
from cloud_core import Session, worker, ENGINE, MAX_SESSIONS, HERE

_sessions = 0
_sess_lock = threading.Lock()

app = FastAPI()


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"), headers={"Cache-Control": "no-store"})


@app.get("/health")
def health():
    return JSONResponse({"ready": ENGINE["ready"], "err": ENGINE["err"],
                         "words": len(ENGINE["bundle"]["actions"]) if ENGINE["bundle"] else 0, "sessions": _sessions})


@app.get("/avatar")
def avatar(text: str = ""):
    try:
        import avatar_ky_hieu
        html_, found, missing = avatar_ky_hieu.build_player(text)
        if not html_:
            return HTMLResponse("<body style='font-family:sans-serif;color:#aaa;background:#111'>Chưa có ký hiệu khớp với câu này.</body>")
        return HTMLResponse(html_)
    except Exception as e:
        return HTMLResponse(f"<body style='font-family:sans-serif;color:#f88;background:#111'>Lỗi: {e}</body>")


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    global _sessions
    await websocket.accept()
    with _sess_lock:
        if _sessions >= MAX_SESSIONS:
            await websocket.send_text(json.dumps({"t": "busy"}))
            await websocket.close()
            return
        _sessions += 1
    loop = asyncio.get_running_loop()
    outq = asyncio.Queue()

    def emit(msg):
        loop.call_soon_threadsafe(outq.put_nowait, msg)

    sess = Session(emit)
    frames = queue.Queue(maxsize=2)
    stop = threading.Event()
    threading.Thread(target=worker, args=(sess, frames, stop, emit), daemon=True).start()

    async def sender():
        while True:
            msg = await outq.get()
            await websocket.send_text(json.dumps(msg, ensure_ascii=False))

    send_task = asyncio.create_task(sender())
    try:
        while True:
            m = await websocket.receive()
            if m.get("type") == "websocket.disconnect":
                break
            if m.get("bytes"):
                b = m["bytes"]
                t = float(np.frombuffer(b[:8], dtype="<f8")[0])
                if frames.full():
                    try:
                        frames.get_nowait()         # luôn giữ khung MỚI NHẤT, bỏ khung cũ -> không bị trễ dồn
                    except queue.Empty:
                        pass
                frames.put((t, b[8:]))
            elif m.get("text"):
                try:
                    cmd = json.loads(m["text"]).get("cmd")
                    sess.command(cmd)
                except Exception:
                    pass
    except WebSocketDisconnect:
        pass
    finally:
        stop.set()
        send_task.cancel()
        with _sess_lock:
            _sessions -= 1
