"""查询两个项目的标注完成状态。"""
import re, json, http.cookiejar, urllib.request, urllib.parse
from collections import Counter

BASE="https://fc.fenglin.pro"; USER="admin@annotation.local"; PWD="annotation2026"
PROJECTS = [
    (14, "Yulin / Exp1"),
    (15, "Zhiqiang / Exp1"),
    (16, "Yulin / Exp2"),
    (17, "Zhiqiang / Exp2"),
    (18, "Zijie / Exp1"),
    (19, "Zijie / Exp2"),
]
cj=http.cookiejar.CookieJar()
op=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj), urllib.request.ProxyHandler({}))
def get(u):
    return op.open(urllib.request.Request(u, headers={"Referer":BASE+"/"}), timeout=30).read().decode("utf-8","replace")
def csrf():
    return next((c.value for c in cj if c.name=="csrftoken"), "")
page=get(BASE+"/user/login/")
tok=re.search(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"',page).group(1)
body=urllib.parse.urlencode({"csrfmiddlewaretoken":tok,"email":USER,"password":PWD}).encode()
op.open(urllib.request.Request(BASE+"/user/login/",data=body,headers={"Referer":BASE+"/user/login/","Content-Type":"application/x-www-form-urlencoded"}),timeout=30)

# 用户 id -> email
def user_map(cb):
    if isinstance(cb, dict): return cb.get("email","?")
    return users.get(cb, f"user#{cb}")
users={}
try:
    ulist=json.loads(get(f"{BASE}/api/users/"))
    ulist=ulist if isinstance(ulist,list) else ulist.get("results",[])
    users={u.get("id"):u.get("email","?") for u in ulist}
except Exception:
    pass

for pid, name in PROJECTS:
    p = json.loads(get(f"{BASE}/api/projects/{pid}/"))
    ann_total = p.get("total_annotations_number") or 0
    # 分页拉 task（含 annotations）
    allt=[]; pg=1
    while True:
        r=json.loads(get(f"{BASE}/api/projects/{pid}/tasks/?page={pg}&page_size=100"))
        lst = r if isinstance(r,list) else r.get("tasks",[])
        if not lst: break
        allt.extend(lst)
        if len(lst)<100: break
        pg+=1
    # 统计
    tasks_with_ann = sum(1 for t in allt if t.get("annotations"))
    by_annot=Counter()
    by_status=Counter()
    for t in allt:
        for a in t.get("annotations",[]):
            by_annot[user_map(a.get("completed_by"))]+=1
            by_status[a.get("was_cancelled") and "cancelled" or "submitted"]+=1
    print(f"=== {name} (ID={pid}) ===")
    print(f"  task 总数: {len(allt)} | 有标注的 task: {tasks_with_ann} | 标注条数: {ann_total}")
    print(f"  完成率: {tasks_with_ann/len(allt)*100:.1f}%" if allt else "  完成率: -")
    if by_annot:
        print(f"  按标注者: {dict(by_annot)}")
    else:
        print(f"  按标注者: (无)")
    if by_status:
        print(f"  按状态: {dict(by_status)}")
    print()
