#!/usr/bin/env python3
"""登录 Label Studio 并把本地 XML 配置同步到线上项目。"""
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

# 项目 ID → 本地配置文件
PROJECTS = {
    14: "config/exp1_atom_quality.xml",
    15: "config/exp1_atom_quality.xml",
    16: "config/exp2_evidence_map.xml",
    17: "config/exp2_evidence_map.xml",
    18: "config/exp1_atom_quality.xml",
    19: "config/exp2_evidence_map.xml",
}

ROOT = Path(__file__).resolve().parents[1]


def build_opener():
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cj),
        # 绕过环境代理，避免 SSL 抖动
        urllib.request.ProxyHandler({}),
    )
    return opener, cj


def get(opener, url):
    req = urllib.request.Request(url, headers={"Referer": BASE + "/"})
    with opener.open(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def post_form(opener, url, data, referer):
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Referer": referer, "Content-Type": "application/x-www-form-urlencoded"},
    )
    with opener.open(req, timeout=30) as r:
        return r.status, r.read().decode("utf-8", "replace")


def get_csrf_from_cookies(cj):
    for c in cj:
        if c.name == "csrftoken":
            return c.value
    return None


def login(opener, cj):
    page = get(opener, BASE + "/user/login/")
    m = re.search(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"', page)
    if not m:
        raise RuntimeError("找不到 csrfmiddlewaretoken")
    token = m.group(1)
    status, _ = post_form(
        opener, BASE + "/user/login/",
        {"csrfmiddlewaretoken": token, "email": USER, "password": PWD},
        referer=BASE + "/user/login/",
    )
    print(f"登录: HTTP {status}")
    return status in (200, 302)


def patch_config(opener, cj, pid, xml_path):
    csrf = get_csrf_from_cookies(cj)
    url = f"{BASE}/api/projects/{pid}/"
    payload = json.dumps({"label_config": xml_path.read_text(encoding="utf-8")}).encode()
    req = urllib.request.Request(url, data=payload, method="PATCH", headers={
        "Content-Type": "application/json",
        "X-CSRFToken": csrf or "",
        "Referer": BASE + "/",
    })
    try:
        with opener.open(req, timeout=30) as r:
            body = json.loads(r.read().decode())
            return True, body
    except urllib.error.HTTPError as e:
        return False, e.read().decode("utf-8", "replace")


def main():
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    opener, cj = build_opener()
    if not login(opener, cj):
        print("登录失败", file=sys.stderr)
        sys.exit(1)
    for pid, rel in PROJECTS.items():
        if only and str(pid) not in only:
            continue
        xml_path = ROOT / rel
        ok, res = patch_config(opener, cj, pid, xml_path)
        if ok:
            print(f"✓ 项目 {pid} 已更新 ({rel})")
        else:
            print(f"✗ 项目 {pid} 更新失败: {res[:300]}")


if __name__ == "__main__":
    main()
