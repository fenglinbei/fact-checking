"""查询所有用户账号与活动情况。"""
import re, json, http.cookiejar, urllib.request, urllib.parse
BASE="https://fc.fenglin.pro"; USER="admin@annotation.local"; PWD="annotation2026"
PROJECTS = (14, 15, 16, 17, 18, 19)
cj=http.cookiejar.CookieJar()
op=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj), urllib.request.ProxyHandler({}))
def get(u):
    return op.open(urllib.request.Request(u, headers={"Referer":BASE+"/"}), timeout=30).read().decode("utf-8","replace")
page=get(BASE+"/user/login/")
tok=re.search(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"',page).group(1)
body=urllib.parse.urlencode({"csrfmiddlewaretoken":tok,"email":USER,"password":PWD}).encode()
op.open(urllib.request.Request(BASE+"/user/login/",data=body,headers={"Referer":BASE+"/user/login/","Content-Type":"application/x-www-form-urlencoded"}),timeout=30)

# 列出所有用户
print("=== 所有用户账号 ===")
ulist=json.loads(get(f"{BASE}/api/users/"))
ulist=ulist if isinstance(ulist,list) else ulist.get("results",[])
for u in ulist:
    print(f"  id={u.get('id')}  email={u.get('email','?'):<32}  active={u.get('active',u.get('is_active','?'))}  first_name={u.get('first_name','')}")

# 各项目的成员/分配情况
print()
for pid in PROJECTS:
    p=json.loads(get(f"{BASE}/api/projects/{pid}/"))
    print(f"=== 项目 {pid} ({p.get('title')}) 配置 ===")
    print(f"  maximum_annotations: {p.get('maximum_annotations')}")
    print(f"  members: {[m.get('email','?') for m in p.get('members',[])] if p.get('members') else '(未单独分配，全组织可见)'}")
