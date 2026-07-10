#!/usr/bin/env python3
"""重新导入 exp2 数据（含 atom_proposition_zh），先更新配置再重导。

安全检查：若项目已有标注（total_annotations>0），中止操作，避免误删标注。
"""
import re
import sys
import json
import http.cookiejar
import urllib.request
import urllib.parse
from pathlib import Path

BASE = "https://fc.fenglin.pro"
USER = "admin@annotation.local"
PWD = "annotation2026"
PID = 13
XML = "config/exp2_evidence_map.xml"
DATA = "data/exp2_tasks_zh.jsonl"

ROOT = Path(__file__).resolve().parents[1]


def build_opener():
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cj),
        urllib.request.ProxyHandler({}),
    )
    return opener, cj


def get(opener, url):
    req = urllib.request.Request(url, headers={"Referer": BASE + "/"})
    with opener.open(req, timeout=30) as r:
        return r.status, r.read().decode("utf-8", "replace")


def post_form(opener, url, data, referer):
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Referer": referer, "Content-Type": "application/x-www-form-urlencoded"},
    )
    with opener.open(req, timeout=30) as r:
        return r.status, r.read().decode("utf-8", "replace")


def csrf(cj):
    for c in cj:
        if c.name == "csrftoken":
            return c.value
    return ""


def login(opener, cj):
    _, page = get(opener, BASE + "/user/login/")
    m = re.search(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"', page)
    if not m:
        raise RuntimeError("no csrf")
    st, _ = post_form(opener, BASE + "/user/login/",
                      {"csrfmiddlewaretoken": m.group(1), "email": USER, "password": PWD},
                      referer=BASE + "/user/login/")
    print(f"登录: HTTP {st}")
    return st in (200, 302)


def api(opener, cj, method, path, payload=None):
    url = f"{BASE}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Content-Type": "application/json",
        "X-CSRFToken": csrf(cj),
        "Referer": BASE + "/",
    })
    try:
        with opener.open(req, timeout=60) as r:
            body = r.read().decode()
            return r.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def main():
    opener, cj = build_opener()
    if not login(opener, cj):
        sys.exit("登录失败")

    # 1. 安全检查：项目是否已有标注
    st, proj = api(opener, cj, "GET", f"/api/projects/{PID}/")
    if not isinstance(proj, dict):
        sys.exit(f"取项目失败: {proj}")
    n_ann = proj.get("total_annotations") or 0
    print(f"项目 {PID} 现有标注数: {n_ann}")
    if n_ann > 0:
        sys.exit(f"⚠ 项目已有 {n_ann} 条标注，中止重导以免误删。请先人工处理。")

    # 2. 更新 label_config
    xml_text = (ROOT / XML).read_text(encoding="utf-8")
    st, res = api(opener, cj, "PATCH", f"/api/projects/{PID}/", {"label_config": xml_text})
    print(f"更新配置: HTTP {st}")
    if st >= 400:
        sys.exit(f"配置更新失败: {res}")

    # 3. 删除旧 task（逐条）
    st, tasks = api(opener, cj, "GET", f"/api/projects/{PID}/tasks/?page_size=1000")
    task_list = tasks if isinstance(tasks, list) else tasks.get("tasks", [])
    ids = [t["id"] for t in task_list]
    print(f"待删除旧 task: {len(ids)}")
    for tid in ids:
        api(opener, cj, "DELETE", f"/api/tasks/{tid}/")

    # 4. 导入新数据
    rows = [json.loads(l) for l in (ROOT / DATA).read_text(encoding="utf-8").splitlines() if l.strip()]
    st, res = api(opener, cj, "POST", f"/api/projects/{PID}/import", rows)
    print(f"导入: HTTP {st}")
    if isinstance(res, dict):
        print(f"  导入 task 数: {res.get('task_count', res)}")

    # 5. 验证
    st, proj = api(opener, cj, "GET", f"/api/projects/{PID}/")
    print(f"项目 {PID} 最终 task 数: {proj.get('task_number')}")


if __name__ == "__main__":
    main()
