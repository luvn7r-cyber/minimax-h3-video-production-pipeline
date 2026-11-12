# -*- coding: utf-8 -*-
"""
本地 Chrome DevTools 单例守护桥 (默认监听 127.0.0.1:19222)
解决多 Agent / 多脚本并发直连 Chrome 调试端口互踢与重复触发远程调试弹窗问题，全程后台零抢焦点。
"""
import argparse
import asyncio
import json
import os
import pathlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import websockets

sys.stdout.reconfigure(encoding="utf-8")

ws_conn = None
ws_loop = None
msg_counter = 10
pending_futures = {}


def default_devtools_port_file() -> pathlib.Path:
    if sys.platform == "win32":
        local_app = os.environ.get("LOCALAPPDATA", "")
        return pathlib.Path(local_app) / "Google" / "Chrome" / "User Data" / "DevToolsActivePort"
    elif sys.platform == "darwin":
        return pathlib.Path.home() / "Library" / "Application Support" / "Google" / "Chrome" / "DevToolsActivePort"
    return pathlib.Path.home() / ".config" / "google-chrome" / "DevToolsActivePort"


async def ws_reader(ws):
    try:
        async for raw in ws:
            data = json.loads(raw)
            mid = data.get("id")
            if mid and mid in pending_futures:
                fut = pending_futures.pop(mid)
                if not fut.done():
                    fut.set_result(data)
    except Exception as e:
        print(f"[CDP桥] WebSocket 读取结束: {e}", flush=True)


async def cdp_call(method, params=None, session_id=None, timeout=30):
    global msg_counter
    msg_counter += 1
    mid = msg_counter
    payload = {"id": mid, "method": method}
    if params:
        payload["params"] = params
    if session_id:
        payload["sessionId"] = session_id
    fut = asyncio.get_running_loop().create_future()
    pending_futures[mid] = fut
    await ws_conn.send(json.dumps(payload))
    return await asyncio.wait_for(fut, timeout=timeout)


class BridgeHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        fut = asyncio.run_coroutine_threadsafe(
            cdp_call(body.get("method"), body.get("params"), body.get("sessionId"), body.get("timeout", 30)),
            ws_loop,
        )
        try:
            res = fut.result(timeout=body.get("timeout", 30) + 5)
            out = json.dumps(res).encode("utf-8")
            self.send_response(200)
        except Exception as e:
            out = json.dumps({"error": str(e)}).encode("utf-8")
            self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)


async def main():
    global ws_conn, ws_loop
    ap = argparse.ArgumentParser(description="Chrome CDP 单例守护桥")
    ap.add_argument("--port", type=int, default=19222, help="本地 HTTP 转发监听端口")
    ap.add_argument("--devtools-file", type=pathlib.Path, default=default_devtools_port_file())
    args = ap.parse_args()

    ws_loop = asyncio.get_running_loop()
    server_started = False
    while True:
        lines = args.devtools_file.read_text(encoding="utf-8").strip().splitlines()
        uri = f"ws://127.0.0.1:{lines[0]}{lines[1]}"
        print(f"[CDP桥] 正在连接 Chrome 调试端点: {uri} ...", flush=True)
        try:
            async with websockets.connect(uri, open_timeout=600, ping_interval=20, max_size=50_000_000) as ws:
                ws_conn = ws
                if not server_started:
                    server = HTTPServer(("127.0.0.1", args.port), BridgeHandler)
                    threading.Thread(target=server.serve_forever, daemon=True).start()
                    server_started = True
                print(f"[CDP桥就绪] 监听 http://127.0.0.1:{args.port}", flush=True)
                await asyncio.create_task(ws_reader(ws))
        except Exception as e:
            print(f"[CDP桥重连] {e}", flush=True)
            await asyncio.sleep(3)


if __name__ == "__main__":
    asyncio.run(main())
