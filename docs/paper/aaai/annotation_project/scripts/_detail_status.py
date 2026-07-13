"""按标注者详细统计各项目的标注情况。"""
import re, json, http.cookiejar, urllib.request, urllib.parse
from collections import Counter, defaultdict

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
page=get(BASE+"/user/login/")
tok=re.search(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"',page).group(1)
body=urllib.parse.urlencode({"csrfmiddlewaretoken":tok,"email":USER,"password":PWD}).encode()
op.open(urllib.request.Request(BASE+"/user/login/",data=body,headers={"Referer":BASE+"/user/login/","Content-Type":"application/x-www-form-urlencoded"}),timeout=30)

# 用户映射
ulist=json.loads(get(f"{BASE}/api/users/"))
ulist=ulist if isinstance(ulist,list) else ulist.get("results",[])
users={u.get("id"):u.get("email","?") for u in ulist}

for pid, pname in PROJECTS:
    allt=[]; pg=1
    while True:
        r=json.loads(get(f"{BASE}/api/projects/{pid}/tasks/?page={pg}&page_size=100"))
        lst = r if isinstance(r,list) else r.get("tasks",[])
        if not lst: break
        allt.extend(lst)
        if len(lst)<100: break
        pg+=1

    by_annot=defaultdict(int)
    detail=defaultdict(list)  # email -> [(task_id, cancelled)]
    for t in allt:
        for a in t.get("annotations",[]):
            uid=a.get("completed_by")
            email=users.get(uid, f"user#{uid}") if isinstance(uid,int) else (uid.get("email","?") if isinstance(uid,dict) else str(uid))
            by_annot[email]+=1
            detail[email].append((t.get("id"), a.get("was_cancelled")))

    # 未标注的 task 数
    unlabeled = sum(1 for t in allt if not t.get("annotations"))

    print(f"=== {pname} (ID={pid}) ===")
    print(f"  task 总数: {len(allt)} | 已标注 task: {len(allt)-unlabeled} | 未标注: {unlabeled}")
    if by_annot:
        for email, cnt in sorted(by_annot.items(), key=lambda x:-x[1]):
            cancelled = sum(1 for _,c in detail[email] if c)
            print(f"    {email}: {cnt} 条 (其中 {cancelled} 条 cancelled)")
    else:
        print(f"    (尚无任何标注)")
    print()

# 额外：检查 draft（草稿/未提交）
print("=== 检查是否有草稿(未提交的标注) ===")
for pid, _ in PROJECTS:
    # Label Studio 把 draft 存在 annotation API 里
    try:
        d=json.loads(get(f"{BASE}/api/projects/{pid}/drafts/"))
        print(f"  项目 {pid}: {len(d) if isinstance(d,list) else d}")
    except Exception as e:
        print(f"  项目 {pid}: drafts API 不可用 ({e})")
