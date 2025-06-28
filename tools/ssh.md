
---

# SSH 使用详解文档

> 本文系统讲解 SSH（Secure Shell）在 Linux 系统中的使用方法，包括基础连接、密钥配置、文件传输、端口转发、安全配置等内容，适合初学者入门与进阶用户参考。

---

## 📌 目录导航

- [1. 什么是 SSH？](#1-什么是-ssh)
- [2. SSH 安装与启动](#2-ssh-安装与启动)
- [3. SSH 基础用法](#3-ssh-基础用法)
- [4. SSH 密钥登录配置](#4-ssh-密钥登录配置)
- [5. SSH 文件传输](#5-ssh-文件传输)
- [6. SSH 端口转发](#6-ssh-端口转发)
- [7. SSH 配置文件说明](#7-ssh-配置文件说明)
- [8. SSH 安全加固](#8-ssh-安全加固)
- [9. 常见问题与排查](#9-常见问题与排查)
- [10. 参考资源](#10-参考资源)

---

## 1. 什么是 SSH？

**SSH（Secure Shell）** 是一种加密的网络协议，用于在不安全的网络中安全访问远程计算机，通常用于远程登录服务器、传输文件、执行命令等操作。

SSH 具有以下特点：

- 提供强加密的认证和数据传输
- 支持基于用户名密码或密钥的身份验证
- 支持端口转发与代理隧道
- 可用于远程命令执行与图形界面转发（X11）

---

## 2. SSH 安装与启动

### 2.1 Linux 安装客户端（通常预装）

```bash
sudo apt update
sudo apt install openssh-client
````

### 2.2 安装并启动服务端（服务器）

```bash
sudo apt install openssh-server
sudo systemctl enable ssh
sudo systemctl start ssh
sudo systemctl status ssh
```

### 2.3 查看 SSH 服务端口

```bash
sudo netstat -tnlp | grep ssh
# 默认端口为 22
```

---

## 3. SSH 基础用法

### 3.1 登录远程服务器

```bash
ssh username@remote_ip
# 示例
ssh student@192.168.1.100
```

### 3.2 指定端口登录

```bash
ssh -p 2222 username@remote_ip
```

### 3.3 执行远程命令

```bash
ssh username@remote_ip "ls -l /home/username"
```

### 3.4 保持连接后台运行

```bash
ssh -fN username@remote_ip
```

---

## 4. SSH 密钥登录配置

### 4.1 生成 SSH 密钥对

```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

默认生成文件：

* 私钥：`~/.ssh/id_rsa`
* 公钥：`~/.ssh/id_rsa.pub`

### 4.2 将公钥上传至远程主机

```bash
ssh-copy-id username@remote_ip
```

或手动追加至远程主机的 `~/.ssh/authorized_keys`

### 4.3 测试免密登录

```bash
ssh username@remote_ip
# 不再需要输入密码
```

---

## 5. SSH 文件传输

### 5.1 使用 scp（Secure Copy）

```bash
# 上传本地文件到远程
scp localfile.txt user@remote_ip:/path/to/destination

# 下载远程文件到本地
scp user@remote_ip:/path/to/file.txt ./localdir
```

### 5.2 使用 rsync（增量传输）

```bash
rsync -avz ./localdir/ user@remote_ip:/path/to/remote/dir/
```

---

## 6. SSH 端口转发

### 6.1 本地端口转发（Local Forwarding）

```bash
ssh -L 8080:localhost:80 user@remote_ip
# 将本地8080端口映射到远程服务器的80端口
```

### 6.2 远程端口转发（Remote Forwarding）

```bash
ssh -R 9090:localhost:3000 user@remote_ip
# 将远程9090端口映射到本地3000端口
```

### 6.3 动态端口转发（SOCKS 代理）

```bash
ssh -D 1080 user@remote_ip
# 本地开启一个 SOCKS5 代理
```

---

## 7. SSH 配置文件说明

配置路径：`~/.ssh/config`

```ini
# 示例：多主机自动配置
Host lab
    HostName 192.168.1.100
    User student
    Port 22
    IdentityFile ~/.ssh/id_rsa

Host aws
    HostName ec2-3-123-456.compute.amazonaws.com
    User ubuntu
    IdentityFile ~/.ssh/aws.pem
```

连接方式：

```bash
ssh lab
ssh aws
```

---

## 8. SSH 安全加固

### 8.1 修改默认端口（如 2222）

```bash
sudo vim /etc/ssh/sshd_config
# Port 2222
sudo systemctl restart ssh
```

### 8.2 禁用 root 登录

```bash
# 在 sshd_config 中设置：
PermitRootLogin no
```

### 8.3 禁用密码登录（只允许密钥）

```bash
PasswordAuthentication no
```

### 8.4 配置防火墙规则（如 UFW）

```bash
sudo ufw allow 2222/tcp
sudo ufw enable
```

---

## 9. 常见问题与排查

| 问题                               | 可能原因       | 解决方案                           |
| -------------------------------- | ---------- | ------------------------------ |
| Connection refused               | SSH 服务未启动  | `sudo systemctl start ssh`     |
| Permission denied                | 权限错误/密钥无效  | 检查密钥权限、authorized\_keys 是否正确   |
| 超时                               | 网络不通或防火墙   | 检查网络、防火墙端口是否开放                 |
| Too many authentication failures | SSH 尝试次数过多 | 使用 `IdentitiesOnly=yes` 限制密钥尝试 |

---

## 10. 参考资源

* [OpenSSH 官网](https://www.openssh.com/)
* [SSH Config 文档](https://man.openbsd.org/ssh_config)
* [Linux ssh 命令详解 - Linuxize](https://linuxize.com/post/how-to-use-ssh/)

---

> 📌 提示：建议为重要服务器配置 Fail2ban、设置登录警报、定期检查 authorized\_keys 文件等。

```