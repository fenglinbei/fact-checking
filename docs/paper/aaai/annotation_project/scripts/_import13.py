import re, json, http.cookiejar, urllib.request, urllib.parse
from pathlib import Path
BASE="https://fc.fenglin.pro"; USER="admin@annotation.local"; PWD="annotation2026"
ROOT=Path(__file__).resolve().parents[1]
cj=http.cookiejar.CookieJar()
op=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj), urllib.request.ProxyHandler({}))
def get(u):
    return op.open(urllib.request.Request(u, headers={"Referer":BASE+"/"}), timeout=30).read().decode("utf-8","replace")
def csrf():
    return next((c.value for c in cj if c.name=="csrftoken"), None)
page=get(BASE+"/user/login/")
tok=re.search(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"',page).group(1)
body=urllib.parse.urlencode({"csrfmiddlewaretoken":tok,"email":USER,"password":PWD}).encode()
r=op.open(urllib.request.Request(BASE+"/user/login/",data=body,headers={"Referer":BASE+"/user/login/","Content-Type":"application/x-www-form-urlencoded"}),timeout=30)
print("login",r.status)
# 读源数据
rows=[json.loads(l) for l in (ROOT/"data/exp2_tasks_zh.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
print("source rows:",len(rows))
payload=json.dumps(rows).encode()
req=urllib.request.Request(f"{BASE}/api/projects/13/import",data=payload,method="POST",
    headers={"Content-Type":"application/json","X-CSRFToken":csrf() or "","Referer":BASE+"/"})
try:
    res=json.loads(op.open(req,timeout=120).read().decode())
    print("import ok task_count=",res.get("task_count"))
except urllib.error.HTTPError as e:
    print("import FAIL",e.code,e.read().decode()[:300])
