# Evidence Map Selector Comparison Web UI 运维 runbook

## 当前生产入口

截至 2026-06-12，线上访问入口是：

- 外部 URL：`https://fc.fenglin.pro/evidence-map/?token=<EVIDENCE_MAP_TOKEN>`
- 便捷入口：`https://fc.fenglin.pro/` 会由 Nginx `302` 到 `/evidence-map/`
- 数据机器 app：`127.0.0.1:8765`，`BASE_PATH=/evidence-map`
- 公网服务器 tunnel 入口：`165.22.48.237` 上的 `127.0.0.1:18765`
- 公网服务器 Nginx 配置：`/etc/nginx/conf.d/fc-fenglin.conf`
- 证书：`/etc/letsencrypt/live/fc.fenglin.pro/fullchain.pem`

`EVIDENCE_MAP_TOKEN` 是访问凭据，不要写进 Git、Nginx 配置或本文档。当前 launcher 会把 token 作为 `--token` 传给 Python 进程，因此同机 `ps` 可能看到 token；排查时可以看，但不要复制到日志或提交记录。若功能改动后重启服务，可以复用旧 token，也可以生成新 token 并同步给使用者。

## 架构与目标

这个 Web UI 是私有、只读服务。Python app 只在数据机器上运行，默认监听 `127.0.0.1:8765`，读取 `/data/liaozijie/fact-checking` 本地数据，不把数据搬到公网服务器。

公网服务器 `165.22.48.237` 只作为反向代理/转发入口：数据机器通过 SSH reverse tunnel 在公网服务器本机暴露 `127.0.0.1:18765`，Nginx 或 Caddy 再把现有 HTTPS 域名下的 `/evidence-map/` 路径代理到这个本机端口。这样不会抢占公网服务器已有的 80/443 服务，也不会让 Python app 直接公网监听。

如果现有服务已经占用了 `/evidence-map/`，不要覆盖它；改用另一个 base path，并在 launcher 的 `BASE_PATH`、反向代理 route、健康检查 URL 和外部访问 URL 中保持一致。

## 功能修改后的重启流程

代码或 HTML 功能改完后，通常只需要重启数据机器上的 Python app。不要改公网服务器 Nginx，也不要重新申请证书，除非域名、端口、base path 或代理路径发生变化。

先确认当前进程：

```bash
cd /data/liaozijie/fact-checking
ps -eo pid,ppid,stat,etime,cmd | grep -E 'serve_evidence_map_selector_comparison|run_evidence_map_selector_comparison_web|ssh -f -N .*18765' | grep -v grep
```

如果只改了 Python/HTML 渲染代码，停止旧 app 后重新启动即可；tunnel 可以保留，因为它仍然转发到同一个本地端口 `127.0.0.1:8765`：

```bash
kill <run_evidence_map_selector_comparison_web.sh_pid> <serve_evidence_map_selector_comparison.py_pid>

cd /data/liaozijie/fact-checking
export EVIDENCE_MAP_TOKEN="<reuse-or-generate-token>"
HOST=127.0.0.1 PORT=8765 BASE_PATH=/evidence-map SPLITS=val ENABLE_LIVE_TRANSLATION=1 \
  bash scripts/phase5_selectors/run/run_evidence_map_selector_comparison_web.sh
```

如果 tunnel 也断了，重新建立 reverse tunnel。推荐加上 keepalive 和 `ExitOnForwardFailure`，当前线上就是这个形态：

```bash
ssh -f -N \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -R 127.0.0.1:18765:127.0.0.1:8765 \
  165.22.48.237
```

完整重启后按顺序验证：

```bash
# 数据机器本地 app
curl -fsS "http://127.0.0.1:8765/evidence-map/healthz?token=${EVIDENCE_MAP_TOKEN}"

# 公网服务器本机 tunnel 入口
ssh 165.22.48.237 \
  "curl -fsS 'http://127.0.0.1:18765/evidence-map/healthz?token=${EVIDENCE_MAP_TOKEN}'"

# 外部 HTTPS 入口；不带 token 返回 401 是预期保护行为
curl -fsS "https://fc.fenglin.pro/evidence-map/healthz?token=${EVIDENCE_MAP_TOKEN}"
curl -sS -o /dev/null -w "%{http_code}\n" "https://fc.fenglin.pro/evidence-map/"
```

预期结果：前三个健康检查成功；最后一个无 token 请求返回 `401`。

## 首次或手动启动 app

```bash
cd /data/liaozijie/fact-checking
export EVIDENCE_MAP_TOKEN="$(openssl rand -hex 24)"
HOST=127.0.0.1 PORT=8765 BASE_PATH=/evidence-map SPLITS=val \
  bash scripts/phase5_selectors/run/run_evidence_map_selector_comparison_web.sh
```

这个命令会在前台运行 app。保留或复制生成的 `EVIDENCE_MAP_TOKEN` 值；健康检查需要从数据机器的另一个 shell 执行，并使用同一个 token 值：

```bash
curl -fsS "http://127.0.0.1:8765/evidence-map/healthz?token=${EVIDENCE_MAP_TOKEN}"
```

## 建立到 `165.22.48.237` 的 reverse tunnel

从数据机器执行。这个命令只在公网服务器本机暴露 `127.0.0.1:18765`：

```bash
ssh -f -N \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -R 127.0.0.1:18765:127.0.0.1:8765 \
  165.22.48.237
```

登录 `165.22.48.237` 后检查 tunnel。公网服务器仍然不能存放数据，也不能运行这个 app；这里只是为了检查临时粘贴或 export 同一个 token 值，不要把 token 持久写入 service config：

```bash
curl -fsS "http://127.0.0.1:18765/evidence-map/healthz?token=${EVIDENCE_MAP_TOKEN}"
```

## 当前 Nginx 配置方式

`fc.fenglin.pro` 已在 `165.22.48.237` 上由 Nginx 托管，配置文件是 `/etc/nginx/conf.d/fc-fenglin.conf`。核心代理逻辑如下，证书相关行由 certbot 管理：

```nginx
server {
    server_name fc.fenglin.pro;

    location = / {
        return 302 /evidence-map/;
    }

    location /evidence-map/ {
        proxy_pass http://127.0.0.1:18765;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }

    listen 443 ssl; # managed by Certbot
    ssl_certificate /etc/letsencrypt/live/fc.fenglin.pro/fullchain.pem; # managed by Certbot
    ssl_certificate_key /etc/letsencrypt/live/fc.fenglin.pro/privkey.pem; # managed by Certbot
    include /etc/letsencrypt/options-ssl-nginx.conf; # managed by Certbot
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem; # managed by Certbot
}
```

如果未来要在另一个已有 HTTPS 域名下挂路径，而不是使用 `fc.fenglin.pro` 独立域名，则在对应 `server {}` block 中使用同样的 location：

```nginx
location /evidence-map/ {
    proxy_pass http://127.0.0.1:18765;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_read_timeout 120s;
}
```

验证通过后再 reload：

```bash
sudo nginx -t
sudo systemctl reload nginx
```

外部访问应通过已有 HTTPS 域名的 `/evidence-map/` 路径，并带上 token；除非代理层已经有更强认证，否则 token 必须保留。

## Caddy 备选配置方式

如果 `165.22.48.237` 使用 Caddy，在已有 site block 里新增：

```caddyfile
handle /evidence-map/* {
    reverse_proxy 127.0.0.1:18765
}
```

验证通过后再 reload：

```bash
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

## 非干扰 checklist

- Python service 监听 `127.0.0.1:8765`，不是 `0.0.0.0:8765`。
- SSH reverse tunnel 在 `165.22.48.237` 上监听 `127.0.0.1:18765`，不是 `0.0.0.0:18765`。
- 公网服务器只做反向代理/转发，不移动数据，不在公网服务器运行 Python app。
- 现有 80/443 的 `server {}` 或 Caddy site block 继续保留。
- 当前线上域名是 `fc.fenglin.pro`，并只暴露 `/evidence-map/` 路径；如果路径已被占用，换一个 base path 并全链路一致修改。
- reload 前必须先跑 `sudo nginx -t` 或 `sudo caddy validate --config /etc/caddy/Caddyfile`。
- 外部访问走已有 HTTPS 域名，不新增裸端口公开访问。
- 除非代理层已有更强认证，否则 app token 必须作为访问条件。
- `BASE_PATH=/evidence-map` 要和 Nginx `location /evidence-map/` 保持一致。当前 launcher 使用 `BASE_PATH="${BASE_PATH:-/evidence-map}"`，传空值会回退到 `/evidence-map`，不要误以为能用它切到根路径。
- 代码改动后的常规重启不需要动 `/etc/nginx/conf.d/fc-fenglin.conf`，也不需要重跑 certbot。

## 快速排查

```bash
# 数据机器：app 和 tunnel 进程
ps -eo pid,ppid,stat,etime,cmd | grep -E 'serve_evidence_map_selector_comparison|run_evidence_map_selector_comparison_web|ssh -f -N .*18765' | grep -v grep

# 公网服务器：确认 Nginx、tunnel 监听和证书
ssh 165.22.48.237 "systemctl is-active nginx"
ssh 165.22.48.237 "ss -ltnp | grep -E '(:80|:443|:18765)'"
ssh 165.22.48.237 "nginx -t"
ssh 165.22.48.237 "certbot certificates | grep -A8 'fc.fenglin.pro'"

# 外部入口
curl -sS -o /dev/null -w '%{http_code} %{redirect_url} %{ssl_verify_result}\n' https://fc.fenglin.pro/
curl -sS -o /dev/null -w '%{http_code} %{ssl_verify_result}\n' https://fc.fenglin.pro/evidence-map/
```

健康状态判断：

- `https://fc.fenglin.pro/` 返回 `302` 到 `https://fc.fenglin.pro/evidence-map/`。
- `https://fc.fenglin.pro/evidence-map/` 不带 token 返回 `401`。
- 带 token 的 `/healthz` 成功，说明 app、tunnel、Nginx 三段都通。
- 如果公网 `502`，优先查 tunnel 和数据机器 app；如果 `404`，优先查 Nginx `server_name` 或 path；如果 `401`，说明路由通了但 token 缺失或错误。

## 回滚

从 Nginx 或 Caddy 配置中删除本次设置新增的 route，也就是 `/evidence-map/` 或实际选择的 base path，验证后 reload。当前 `fc.fenglin.pro` 独立配置可以通过移除 `/etc/nginx/conf.d/fc-fenglin.conf` 回滚，但先保留文件备份：

```bash
sudo cp -a /etc/nginx/conf.d/fc-fenglin.conf /etc/nginx/conf.d/fc-fenglin.conf.bak-$(date -u +%Y%m%dT%H%M%SZ)
sudo rm /etc/nginx/conf.d/fc-fenglin.conf
sudo nginx -t
sudo systemctl reload nginx
```

或：

```bash
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

然后停止 SSH reverse tunnel 和数据机器上的本地 Python process。因为没有改动已有 location、upstream、证书或公网端口绑定，现有服务应保持不受影响。
