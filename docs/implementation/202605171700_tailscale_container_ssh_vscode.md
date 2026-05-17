# Tailscale 连接公网服务器、容器服务器与 VS Code

生成日期: 2026-05-17

## 目标

在不暴露公网 SSH 端口的前提下，把三端接入同一个 Tailscale tailnet:

```text
Windows 本地机
public-server  100.95.66.96
inner-server   100.116.7.93  # 容器内 userspace Tailscale
```

最终能力:

- `public-server` 可以直接 `ssh root@100.116.7.93` 进入 `inner-server`
- `inner-server` 可以通过 Tailscale SOCKS5 代理把文件传到 `public-server`
- Windows 本地 VS Code 可以通过 Remote SSH 连接 `inner-server`

本文中的 `tskey-xxxx` 是占位符。不要把真实 auth key 写入文档或提交到仓库；如果 key 泄露，应在 Tailscale 控制台立即 revoke 并重新生成。

## public-server 配置

### 1. 安装 Tailscale

优先使用官方安装脚本:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
```

如果出现 TLS EOF 类错误:

```text
curl: (35) error:0A000126:SSL routines::unexpected eof while reading
```

先改用 IPv4 和重试:

```bash
curl -4fsSL --retry 5 --connect-timeout 15 \
  https://tailscale.com/install.sh -o /tmp/tailscale-install.sh

sh /tmp/tailscale-install.sh
```

### 2. 为 systemd 版 tailscaled 配置代理

如果 `tailscale up` 卡住，日志出现:

```text
Post "https://controlplane.tailscale.com/machine/register": context deadline exceeded
```

说明注册请求到 Tailscale control plane 不稳定。`tailscaled` 是 systemd 服务，不能只在当前 shell 里 `export HTTPS_PROXY`，需要写 systemd override。

```bash
sudo mkdir -p /etc/systemd/system/tailscaled.service.d

printf '%s\n' \
'[Service]' \
'Environment="HTTP_PROXY=http://127.0.0.1:7890"' \
'Environment="HTTPS_PROXY=http://127.0.0.1:7890"' \
'Environment="NO_PROXY=127.0.0.1,localhost"' \
| sudo tee /etc/systemd/system/tailscaled.service.d/override.conf

sudo systemctl daemon-reload
sudo systemctl restart tailscaled
systemctl show tailscaled -p Environment
```

`127.0.0.1:7890` 必须是 public-server 本机可访问的代理。如果代理在其他机器上，替换成 public-server 能访问到的地址。

### 3. 登录并启用 Tailscale SSH

推荐使用 Tailscale auth key，适合 WebSSH/WebIDE 环境:

```bash
sudo tailscale up \
  --auth-key=tskey-xxxx \
  --hostname=public-server \
  --ssh \
  --timeout=2m
```

如果已经登录，只需要开启 Tailscale SSH:

```bash
sudo tailscale set --ssh
```

检查:

```bash
tailscale status
tailscale ip -4
```

本次 public-server 的 Tailscale IP:

```text
100.95.66.96
```

## inner-server 容器配置

### 1. 环境特征

目标服务器运行在容器里，不能使用 systemd:

```text
System has not been booted with systemd as init system
```

并且普通 TUN 模式不可用:

```text
/dev/net/tun does not exist
Permission denied for iptables/netlink
```

因此不能启动普通 `tailscale0` 网卡模式，需要使用 userspace networking。

### 2. 启动 userspace tailscaled

在 inner-server 容器里执行。这个命令以前台方式运行，保持该 shell 不要关闭；长期使用建议放到 `tmux` / `screen` / 容器启动脚本中。

```bash
mkdir -p /data/liaozijie/tailscale/state
rm -f /data/liaozijie/tailscale/tailscaled.sock

HTTPS_PROXY=http://127.0.0.1:7890 \
HTTP_PROXY=http://127.0.0.1:7890 \
tailscaled \
  --state=/data/liaozijie/tailscale/state/tailscaled.state \
  --socket=/data/liaozijie/tailscale/tailscaled.sock \
  --tun=userspace-networking \
  --socks5-server=127.0.0.1:1055 \
  --outbound-http-proxy-listen=127.0.0.1:1055
```

注意:

- `--netfilter-mode=off` 不是 `tailscaled` 参数，不要放在这条命令里
- `127.0.0.1:7890` 必须是 inner-server 容器内部可访问的代理
- userspace 模式不会创建普通可路由的 `tailscale0` 网卡
- userspace 模式会在容器内提供 SOCKS5/HTTP 代理 `127.0.0.1:1055`

### 3. 登录 inner-server

另开 inner-server 容器的第二个 shell:

```bash
tailscale --socket=/data/liaozijie/tailscale/tailscaled.sock up \
  --auth-key=tskey-xxxx \
  --hostname=inner-server \
  --timeout=2m
```

启用 Tailscale SSH:

```bash
tailscale --socket=/data/liaozijie/tailscale/tailscaled.sock set --ssh=true
```

如果 `set --ssh=true` 不支持，可改用:

```bash
tailscale --socket=/data/liaozijie/tailscale/tailscaled.sock up \
  --ssh \
  --hostname=inner-server
```

检查:

```bash
tailscale --socket=/data/liaozijie/tailscale/tailscaled.sock status
tailscale --socket=/data/liaozijie/tailscale/tailscaled.sock ip -4
```

本次 inner-server 的 Tailscale IP:

```text
100.116.7.93
```

## Linux 端互连

### public-server 连接 inner-server

因为 inner-server 已启用 Tailscale SSH，public-server 可以直接连接:

```bash
ssh root@100.116.7.93
```

如果出现 Tailscale SSH 二次认证链接，复制到浏览器中完成确认。

### inner-server 连接 public-server

inner-server 是 userspace 模式，普通 `ssh 100.95.66.96` 可能不会直接走 Tailscale，需要通过 userspace `tailscaled` 提供的 SOCKS5 代理。

先确保容器里有 OpenBSD netcat:

```bash
apt-get update
apt-get install -y netcat-openbsd
```

连接 public-server:

```bash
ssh -o ProxyCommand='nc -X 5 -x 127.0.0.1:1055 %h %p' \
  fenglin@100.95.66.96
```

第一次连接会提示 SSH host key，确认 fingerprint 后输入 `yes`。

## 文件传输

### 从 public-server 拉取 inner-server 文件

public-server 是普通 Tailscale 模式，最方便:

```bash
rsync -avP root@100.116.7.93:/path/to/source/ /path/to/local/target/
```

### 从 inner-server 推送到 public-server

inner-server 需要通过 SOCKS5 ProxyCommand:

```bash
rsync -avP \
  -e "ssh -o ProxyCommand='nc -X 5 -x 127.0.0.1:1055 %h %p'" \
  /path/to/source/ fenglin@100.95.66.96:/path/to/target/
```

如果目录里小文件很多，建议先打包再传:

```bash
tar -I 'zstd -T0 -3' -cf data.tar.zst /path/to/source

rsync -avP \
  -e "ssh -o ProxyCommand='nc -X 5 -x 127.0.0.1:1055 %h %p'" \
  data.tar.zst fenglin@100.95.66.96:/path/to/target/
```

在 public-server 解包:

```bash
tar -I zstd -xf /path/to/target/data.tar.zst -C /path/to/target/
```

### 简化 SSH 配置

可在 inner-server 的 `~/.ssh/config` 中加入:

```sshconfig
Host public-ts
  HostName 100.95.66.96
  User fenglin
  ProxyCommand nc -X 5 -x 127.0.0.1:1055 %h %p
```

之后可直接:

```bash
ssh public-ts
rsync -avP /path/to/source/ public-ts:/path/to/target/
```

## Windows VS Code 连接 inner-server

### 1. Windows 安装并登录 Tailscale

在 Windows 本地安装 Tailscale，登录同一个账号。PowerShell 检查:

```powershell
tailscale status
```

应能看到:

```text
100.116.7.93  inner-server
100.95.66.96  public-server
```

### 2. 先用 Windows 终端验证 SSH

PowerShell:

```powershell
ssh root@100.116.7.93
```

如果出现 Tailscale SSH 认证链接，打开链接完成确认。确认后应能进入 inner-server 容器 shell。

### 3. 配置 VS Code Remote SSH

安装 VS Code 插件:

```text
Remote - SSH
```

编辑 Windows SSH 配置:

```text
C:\Users\<你的用户名>\.ssh\config
```

加入:

```sshconfig
Host inner-server
  HostName 100.116.7.93
  User root
  Port 22
```

VS Code 中执行:

```text
Ctrl+Shift+P
Remote-SSH: Connect to Host...
inner-server
```

如果 VS Code 连接时卡在认证，先回 PowerShell 手动执行一次:

```powershell
ssh root@100.116.7.93
```

完成 Tailscale SSH 的浏览器确认后，再重试 VS Code。

## 常见问题

### `sudo tailscale up` 没有输出

先检查实际状态:

```bash
tailscale status
tailscale ip -4
```

如果显示 `Logged out`，需要重新认证:

```bash
sudo tailscale up --force-reauth --timeout=30s
```

WebSSH/WebIDE 环境里更推荐 auth key:

```bash
sudo tailscale up --auth-key=tskey-xxxx --hostname=public-server
```

### `systemctl edit tailscaled` 显示 temporary file is empty

表示打开编辑器后没有保存任何内容。可以用 `tee` 直接写入 override，见 public-server 配置章节。

### `failed to connect to local tailscaled`

分两种情况:

1. 在 public-server 上:

```bash
sudo systemctl start tailscaled
```

2. 在 inner-server 容器里:

容器没有 systemd，需要先手动启动 userspace `tailscaled`，并且后续 `tailscale` 命令都要带同一个 socket:

```bash
tailscale --socket=/data/liaozijie/tailscale/tailscaled.sock status
```

如果在 public-server 上误用了 inner-server 的 socket 路径，会报:

```text
dial unix /data/liaozijie/tailscale/tailscaled.sock: connect: no such file or directory
```

### `/bin/sh: 1: exec: nc: not found`

容器里没有 `nc`。安装 OpenBSD netcat:

```bash
apt-get update
apt-get install -y netcat-openbsd
```

然后使用:

```bash
ssh -o ProxyCommand='nc -X 5 -x 127.0.0.1:1055 %h %p' \
  fenglin@100.95.66.96
```

### inner-server 重启后无法连接

inner-server 的 `tailscaled` 是手动前台启动的。容器重启或 shell 关闭后，需要重新启动 userspace `tailscaled`。建议用 `tmux` 保持:

```bash
tmux new -s tailscaled
```

然后在 tmux 中运行 userspace `tailscaled` 启动命令。

## 参考

- Tailscale Linux 安装: <https://tailscale.com/docs/install/linux>
- Tailscale SSH: <https://tailscale.com/docs/features/tailscale-ssh>
- Tailscale userspace networking: <https://tailscale.com/docs/concepts/userspace-networking>
- VS Code Tailscale 扩展说明: <https://tailscale.com/kb/1265/vscode-extension>
