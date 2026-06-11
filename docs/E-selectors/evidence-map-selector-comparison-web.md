# Evidence Map Selector Comparison Web UI 运维 runbook

## 架构与目标

这个 Web UI 是私有、只读服务。Python app 只在数据机器上运行，默认监听 `127.0.0.1:8765`，读取 `/data/liaozijie/fact-checking` 本地数据，不把数据搬到公网服务器。

公网服务器 `165.22.48.237` 只作为反向代理/转发入口：数据机器通过 SSH reverse tunnel 在公网服务器本机暴露 `127.0.0.1:18765`，Nginx 或 Caddy 再把现有 HTTPS 域名下的 `/evidence-map/` 路径代理到这个本机端口。这样不会抢占公网服务器已有的 80/443 服务，也不会让 Python app 直接公网监听。

如果现有服务已经占用了 `/evidence-map/`，不要覆盖它；改用另一个 base path，并在 launcher 的 `BASE_PATH`、反向代理 route、健康检查 URL 和外部访问 URL 中保持一致。

## 在数据机器启动 app

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
ssh -N -R 127.0.0.1:18765:127.0.0.1:8765 165.22.48.237
```

登录 `165.22.48.237` 后检查 tunnel。公网服务器仍然不能存放数据，也不能运行这个 app；这里只是为了检查临时粘贴或 export 同一个 token 值，不要把 token 持久写入 service config：

```bash
curl -fsS "http://127.0.0.1:18765/evidence-map/healthz?token=${EVIDENCE_MAP_TOKEN}"
```

## Nginx 配置方式

在 `165.22.48.237` 已有 HTTPS `server {}` block 里只新增下面的 location，不改已有域名、证书、root、upstream 或其它 location：

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
- 只新增 `/evidence-map/` 路径；如果路径已被占用，换一个 base path 并全链路一致修改。
- reload 前必须先跑 `sudo nginx -t` 或 `sudo caddy validate --config /etc/caddy/Caddyfile`。
- 外部访问走已有 HTTPS 域名，不新增裸端口公开访问。
- 除非代理层已有更强认证，否则 app token 必须作为访问条件。

## 回滚

从 Nginx 或 Caddy 配置中删除本次设置新增的 route，也就是 `/evidence-map/` 或实际选择的 base path，验证后 reload：

```bash
sudo nginx -t
sudo systemctl reload nginx
```

或：

```bash
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

然后停止 SSH reverse tunnel 和数据机器上的本地 Python process。因为没有改动已有 location、upstream、证书或公网端口绑定，现有服务应保持不受影响。
