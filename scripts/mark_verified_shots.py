# -*- coding: utf-8 -*-
"""
人工逐帧审核签字归档工具：仅在人眼审完 18 帧抽检图确认 B1–B6 全部达标后调用！
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="人工签发合格镜头并归档至 renders/verified/")
    ap.add_argument("--project-dir", type=Path, required=True)
    ap.add_argument("shot_ids", nargs="+", help="人工审核通过的镜头编号列表")
    args = ap.parse_args()

    proj = args.project_dir.resolve()
    tbl_path = proj / "atomic_shot_table.json"
    tbl = json.loads(tbl_path.read_text(encoding="utf-8"))
    ver_dir = proj / "renders" / "verified"
    ver_dir.mkdir(parents=True, exist_ok=True)

    archived = []
    for s in tbl:
        if s["sub_id"] in args.shot_ids:
            cand = proj / "renders" / "pending" / f"{s['sub_id']}_cand.mp4"
            if not cand.exists():
                raise FileNotFoundError(f"未找到待归档候选视频: {cand}")
            shutil.copy2(cand, ver_dir / f"{s['sub_id']}_verified.mp4")
            s["status"] = "合格"
            archived.append(s["sub_id"])

    tbl_path.write_text(json.dumps(tbl, ensure_ascii=False, indent=2), encoding="utf-8")
    ok_cnt = sum(1 for x in tbl if x.get("status") == "合格")
    print(f"[Gate B 人工归档完成] 本次归档: {archived} | 全片累计合格: {ok_cnt}/{len(tbl)}")


if __name__ == "__main__":
    main()
