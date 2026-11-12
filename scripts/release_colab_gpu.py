# -*- coding: utf-8 -*-
"""
断开并释放 Colab GPU 运行时，并二次验证隧道彻底离线（防后台空转烧钱）
注意：必须挂接具体笔记本页面 (Target.attachToTarget) 后执行断开，严禁仅连浏览器级会话。
"""
import argparse
import json
import os
import pathlib
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")


def check_tunnel_alive(url: str) -> bool:
    if not url:
        return False
    try:
        req = urllib.request.Request(f"{url}/system_stats", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as r:
            return r.status == 200
    except Exception:
        return False


def bridge_call(bridge_url: str, method: str, params=None, session_id=None):
    payload = {"method": method, "params": params or {}}
    if session_id:
        payload["sessionId"] = session_id
    req = urllib.request.Request(bridge_url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser(description="挂接 Colab 页面释放 GPU 并核验离线")
    ap.add_argument("--notebook-id", default=os.environ.get("COLAB_NOTEBOOK_ID", ""), help="Colab 笔记本 ID")
    ap.add_argument("--tunnel-file", type=pathlib.Path, default=pathlib.Path("current_tunnel.txt"))
    ap.add_argument("--bridge-url", default="http://127.0.0.1:19222")
    args = ap.parse_args()

    t_url = args.tunnel_file.read_text(encoding="utf-8").strip() if args.tunnel_file.exists() else ""
    targets = bridge_call(args.bridge_url, "Target.getTargets")["result"]["targetInfos"]
    pages = [t for t in targets if t["type"] == "page" and (args.notebook_id in t.get("url", "") if args.notebook_id else "colab.research.google.com" in t.get("url", ""))]
    if pages:
        sid = bridge_call(args.bridge_url, "Target.attachToTarget", {"targetId": pages[0]["targetId"], "flatten": True})["result"]["sessionId"]
        bridge_call(args.bridge_url, "Runtime.evaluate", {"expression": """(() => {
            const nb = window.colab && window.colab.global && window.colab.global.notebook;
            if (!nb || !nb.cells || !nb.cells.length) return 'NO_NB';
            const c = nb.cells[0];
            if (c.isRunning && c.isRunning()) {
                const b = c.runButton;
                const el = b && (b.click ? b : (b.element || (b.getElement && b.getElement())));
                if (el) el.click();
            }
            return 'STOPPED';
        })()""", "returnByValue": True}, sid)
        time.sleep(4.0)
        bridge_call(args.bridge_url, "Runtime.evaluate", {"expression": """(() => {
            const nb = window.colab && window.colab.global && window.colab.global.notebook;
            if (!nb || !nb.cells || !nb.cells.length) return 'NO_NB';
            const c = nb.cells[0];
            c.setText('from google.colab import runtime\\nruntime.unassign()\\n');
            const b = c.runButton;
            const el = b && (b.click ? b : (b.element || (b.getElement && b.getElement())));
            if (el) el.click();
            if (nb.kernel && nb.kernel.unassignCurrentVm) nb.kernel.unassignCurrentVm();
            return 'UNASSIGNED';
        })()""", "returnByValue": True}, sid)
        time.sleep(6.0)

    alive = check_tunnel_alive(t_url)
    print(f"[GPU释放验证] 隧道 {t_url or '(无记录)'} 当前状态: {'STILL_ONLINE(警告!)' if alive else 'OFFLINE (已安全释放)'}", flush=True)
    if alive:
        sys.exit(1)


if __name__ == "__main__":
    main()
