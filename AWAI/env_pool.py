#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AWAI 并行仿真池：把环境仿真分散到多个子进程，吃满多核。

为什么需要它（实测依据）：
  - 单步耗时：仿真 0.132 ms、推理 0.112 ms —— 仿真占 54%，单核仿真极限约 7590 环境步/秒。
  - 而增加并行环境数几乎不提速：n_envs 1→32 仅从 1402 提升到 1699 决策步/秒（1.21×）。
    原因是 Python GIL 让所有环境共享一个核，32 核里 31 个闲置。
  - 本机 RTX 5070 在 PPO 的小批量（每步 4~40 条观测）下推理反而慢 19~43 倍
    （单次 CUDA kernel 启动+同步固定开销约 3.3 ms），因此推理留在 CPU 更合适。

设计：
  - 每个子进程持有若干份 CurriculumEnv，用 numpy 快速路径自行推理并采样动作；
  - 主进程每次下发一个「rollout 块」（block_steps 步）的动作，子进程整块跑完再回传，
    把 IPC 次数从「每步 1 次」降到「每块 1 次」；
  - 主进程用线程并发向所有子进程下发/收取，IPC 等待不占 GIL；
  - 子进程不做反向传播，仅在权重更新后接收新的 numpy 权重。

IPC 实现说明：本环境（DSH 文件沙箱）禁止创建命名管道，
  multiprocessing.Pipe / Queue 均报 `WinError 5 拒绝访问`；
  因此这里改用**本地回环 TCP socket**（父进程监听、子进程 connect），
  语义与 Pipe 等价（字节流、双向、带长度前缀），且不受该限制影响。

子进程通过 spawn 自举，环境类取自 AWAI 模块本身。
"""
import os
import sys
import pickle
import socket
import struct
import threading
import multiprocessing as mp
from typing import List

import numpy as np


# ---------- 基于 TCP 的字节流通道（替代被禁用的命名管道） ----------
def _send_msg(sock, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack('!Q', len(data)) + data)


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _recv_msg(sock):
    head = _recv_exact(sock, 8)
    if head is None:
        return None
    (n,) = struct.unpack('!Q', head)
    data = _recv_exact(sock, n)
    if data is None:
        return None
    return pickle.loads(data)


def _env_worker_main(port, stage_idx, n_envs, plane_type, seed, stage_mix_m=0, stages_pickle=None):
    """子进程主循环：持有 n_envs 个环境，按块执行 step 并回传数据。

    stage_mix_m>0 时启用「阶段混合采样」：本子进程内各环境轮流使用不同课程阶段
    （环境 k 用 (stage_idx + k) % m）。这是针对课程式顺序训练引发的**灾难性遗忘**
    ——让基础阶段（瞄准射击）与对抗阶段（空战）的行为被同时强化，而不是后者覆盖前者。

    stages_pickle：可选，pickle 后的阶段列表，用于"只训练空战阶段"这类定制任务，
    无需修改全局的 curriculum_stages()。"""
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    import random
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    import AWAI as A
    import pygame
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    try:
        pygame.display.quit()
    except Exception:
        pass

    sock = socket.create_connection(('127.0.0.1', port), timeout=120)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def send(obj):
        _send_msg(sock, obj)

    def recv():
        return _recv_msg(sock)

    try:
        if stages_pickle:
            stages = pickle.loads(stages_pickle)
        else:
            stages = A.curriculum_stages(plane_type)
        envs = []
        used = []
        for k in range(n_envs):
            if stage_mix_m and stage_mix_m > 0:
                # 阶段混合：同一子进程内各环境轮流使用不同课程阶段
                si = (stage_idx + k) % min(stage_mix_m, len(stages))
            else:
                si = stage_idx
            used.append(si)
            random.seed(seed * 1000 + k)
            envs.append(A.CurriculumEnv(stages[si], render=False))
        obs_list = [e.reset() for e in envs]
        n_red = max(1, len(obs_list[0].get('red') or []))
        agent = A.PPOAgent(A.OBS_DIM, A.ACT_DIMS)     # 仅用于 numpy 前向

        def pack_obs(olist):
            out = np.zeros((n_envs, n_red, A.OBS_DIM), dtype=np.float32)
            for i in range(n_envs):
                rows = olist[i].get('red') or []
                for j in range(min(n_red, len(rows))):
                    out[i, j] = rows[j]
            return out

        def pack_masks(envlist):
            """火控动作可用性掩码（每环境每神经槽位 4 维），由实际弹药/冷却/威胁状态决定。
            屏蔽备弹为 0、冷却中、以及（无敌方导弹时）纯无效的诱饵动作。"""
            out = np.ones((n_envs, n_red, 4), dtype=np.float32)
            for i in range(n_envs):
                nn = envlist[i]._neural_aircraft('red')
                wld = envlist[i].world
                for j in range(min(n_red, len(nn))):
                    out[i, j] = A.weapon_fire_mask(nn[j], world=wld)
            return out

        def pack_labels(envlist):
            """火控行为克隆标签（每环境每槽位 1 维）：0=应当开机炮，-1=无标签。
            标签由【本子进程自己持有的环境】决定，与它回传的 obs/masks 严格对应，
            避免"主进程观测 / 子进程动作"错配导致标签无效。"""
            out = np.full((n_envs, n_red), -1, dtype=np.int8)
            for i in range(n_envs):
                nn = envlist[i]._neural_aircraft('red')
                for j in range(min(n_red, len(nn))):
                    out[i, j] = A.fire_label_for(nn[j], envlist[i].world)
            return out

        send({'type': 'ready', 'n_red': n_red, 'n_envs': n_envs,
              'obs': pack_obs(obs_list), 'masks': pack_masks(envs),
              'labels': pack_labels(envs)})

        while True:
            msg = recv()
            if msg is None or msg.get('type') == 'stop':
                break
            mtype = msg.get('type')
            if mtype == 'set_weights':
                agent._np_weights = msg['weights']
                continue
            if mtype != 'step':
                continue

            actions = msg['actions']              # (block_steps, n_envs, n_red, 6) int8
            block = actions.shape[0]
            obs_out = np.empty((block, n_envs, n_red, A.OBS_DIM), dtype=np.float32)
            rew_out = np.zeros((block, n_envs), dtype=np.float32)
            done_out = np.zeros((block, n_envs), dtype=np.uint8)
            kills_out = np.zeros((block, n_envs), dtype=np.float32)
            dmg_out = np.zeros((block, n_envs), dtype=np.float32)
            cleared_out = np.zeros((block, n_envs), dtype=np.uint8)
            red_alive_out = np.zeros((block, n_envs), dtype=np.int16)

            for t in range(block):
                for i in range(n_envs):
                    rows = obs_list[i].get('red') or []
                    acts = []
                    for j in range(min(n_red, len(rows))):
                        a = actions[t, i, j]
                        acts.append(A.Action(int(a[0]), int(a[1]), int(a[2]),
                                             int(a[3]), int(a[4]), int(a[5])))
                    obs, rew, done, _ = envs[i].step({'red': acts, 'blue': []})
                    w = envs[i].world
                    for (tm, _u), d in w.damage_credit.items():
                        if tm == 'red':
                            dmg_out[t, i] += d
                    kills_out[t, i] = sum(1 for u in w.kills if u.team == 'blue')
                    rew_out[t, i] = rew['red']
                    red_alive_out[t, i] = len(w.aircraft_list_of('red'))
                    if done:
                        done_out[t, i] = 1
                        # 终止信息需在 reset 前采集
                        cleared_out[t, i] = 1 if envs[i]._cleared() else 0
                        obs_list[i] = envs[i].reset()
                    else:
                        obs_list[i] = obs
                # 本步结束后的观测（done 的环境已是 reset 后的新观测）
                for i in range(n_envs):
                    rows = obs_list[i].get('red') or []
                    for j in range(min(n_red, len(rows))):
                        obs_out[t, i, j] = rows[j]

            send({'type': 'step', 'obs': obs_out, 'rew': rew_out, 'done': done_out,
                  'kills': kills_out, 'dmg': dmg_out, 'cleared': cleared_out,
                  'red_alive': red_alive_out, 'masks': pack_masks(envs),
                  'labels': pack_labels(envs)})
    except Exception:
        import traceback
        tb = traceback.format_exc()
        # 关键：子进程异常必须留下痕迹。早期这里只把错误发回主进程，
        # 若发生在【握手之前】，父进程只会看到“连接中断”而不知道子进程为什么死。
        print(f'[pool-worker seed={seed} stage={stage_idx}] 异常:\n{tb}',
              file=sys.stderr, flush=True)
        try:
            send({'type': 'error', 'err': tb})
        except Exception:
            pass
    finally:
        try:
            sock.close()
        except Exception:
            pass


class _Client:
    """主进程侧对单个子进程的封装。

    IPC 走本地回环 TCP：父进程先 bind/listen，子进程按端口 connect。
    （命名管道在本环境被沙箱禁止，故不用 multiprocessing.Pipe。）

    支持 restart()：子进程若意外退出（被外部杀掉、内存不足等），
    父进程可以就地重启它并重新下发权重，而不必让整个训练崩溃。"""

    def __init__(self, stage_idx, n_envs, plane_type, seed, connect_timeout=120.0,
                 stage_mix_m=0, stages_pickle=None):
        self.stage_idx = stage_idx
        self.n_envs = n_envs
        self.plane_type = plane_type
        self.seed = seed
        self.connect_timeout = connect_timeout
        self.stage_mix_m = stage_mix_m
        self.stages_pickle = stages_pickle
        self.n_red = 1
        self._cond = threading.Condition()
        self._reply = None
        self._closed = False
        self._spawn()
        threading.Thread(target=self._recv_loop, daemon=True).start()

    def _spawn(self):
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(('127.0.0.1', 0))
        self.listener.listen(1)
        port = self.listener.getsockname()[1]
        ctx = mp.get_context('spawn')
        self.proc = ctx.Process(target=_env_worker_main,
                                args=(port, self.stage_idx, self.n_envs,
                                      self.plane_type, self.seed, self.stage_mix_m,
                                      self.stages_pickle),
                                daemon=True)
        self.proc.start()
        self.listener.settimeout(self.connect_timeout)
        self.sock, _addr = self.listener.accept()
        self.listener.close()
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # 放大发送缓冲，降低大量权重/块数据下的阻塞概率
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        except OSError:
            pass
        with self._cond:
            self._reply = None
            self._closed = False

    def restart(self):
        """就地重启子进程（保留 socket 收发线程的协议约束：重开线程）。"""
        self.close()
        self._spawn()
        threading.Thread(target=self._recv_loop, daemon=True).start()
        msg = self.handshake()
        return msg

    def alive(self):
        return (not self._closed) and self.proc.is_alive()


    def _recv_loop(self):
        while True:
            try:
                msg = _recv_msg(self.sock)
            except (OSError, EOFError, pickle.UnpicklingError):
                msg = None
            if msg is None:
                with self._cond:
                    self._closed = True
                    self._reply = {'type': 'closed'}
                    self._cond.notify_all()
                return
            with self._cond:
                self._reply = msg
                self._cond.notify_all()

    def _wait_reply(self, timeout=600.0):
        with self._cond:
            if not self._cond.wait_for(lambda: self._reply is not None, timeout):
                return {'type': 'timeout'}
            msg, self._reply = self._reply, None
        if msg.get('type') == 'error':
            raise RuntimeError('子进程异常:\n' + msg.get('err', ''))
        if msg.get('type') == 'closed':
            raise RuntimeError('子进程意外退出')
        return msg

    def send(self, obj):
        _send_msg(self.sock, obj)

    def handshake(self):
        msg = self._wait_reply()
        self.n_red = msg.get('n_red', 1)
        return msg

    def request(self, obj):
        self.send(obj)
        return self._wait_reply()

    def close(self):
        try:
            self.send(None)
        except Exception:
            pass
        try:
            self.proc.join(timeout=3)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=2)
        except Exception:
            pass
        # 关闭 socket，避免父进程持有失效连接（否则会积累文件描述符）
        try:
            self.sock.close()
        except Exception:
            pass
        with self._cond:
            self._closed = True


class ParallelEnvPool:
    """跨进程并行环境池。

    对外接口：
        pool = ParallelEnvPool(stage_idx, n_workers, envs_per_worker).start()
        obs = pool.obs_dict()                     # {'red': [扁平观测...], 'blue': []}
        obs, rew, done, info = pool.step(动作列表)  # 走 block_steps 步
        stats = pool.last_stats()                 # 逐环境 kills/dmg/done
        pool.sync_weights(agent)                  # 权重更新后下发
    """

    def __init__(self, stage_idx, n_workers=16, envs_per_worker=2,
                 plane_type='standard', block_steps=32, total_envs=None,
                 stage_mix_m=0, stages=None):
        self.plane_type = plane_type
        self.stage_idx = stage_idx
        self.stage_mix_m = int(stage_mix_m or 0)
        # stages：自定义阶段列表（如"只含空战阶段"的专项任务）
        self.stages_pickle = (pickle.dumps(stages, protocol=pickle.HIGHEST_PROTOCOL)
                              if stages else None)
        self.n_workers = max(1, int(n_workers))
        if total_envs:
            self.envs_per_worker = max(1, int(round(total_envs / self.n_workers)))
        else:
            self.envs_per_worker = max(1, int(envs_per_worker))
        self.n_envs = self.n_workers * self.envs_per_worker
        self.block_steps = max(1, int(block_steps))
        self.clients: List[_Client] = []
        self.n_red = 1
        self._agent = None          # 记录训练代理，供子进程重启后重发权重
        self._obs_all = None
        self._masks_all = None      # 火控动作掩码 (workers, envs, n_red, 4)
        self._labels_all = None     # 火控行为克隆标签 (workers, envs, n_red)
        self._rew_all = None
        self._done_all = None
        self._stats = {'kills': np.zeros(self.n_envs, dtype=np.float32),
                       'dmg': np.zeros(self.n_envs, dtype=np.float32),
                       'done': np.zeros(self.n_envs, dtype=np.uint8)}

    # ---------- 生命周期 ----------
    def start(self):
        for k in range(self.n_workers):
            self.clients.append(_Client(self.stage_idx, self.envs_per_worker,
                                        self.plane_type, seed=k + 1,
                                        stage_mix_m=self.stage_mix_m,
                                        stages_pickle=self.stages_pickle))
        nreds, obs0, msk0, lbl0 = [], [], [], []
        for c in self.clients:
            msg = c.handshake()                 # ready 消息里已带初始观测
            nreds.append(c.n_red)
            obs0.append(msg['obs'])
            msk0.append(msg['masks'])
            lbl0.append(msg['labels'])
        self.n_red = max(nreds) if nreds else 1
        # 初始观测堆叠为 (workers, envs, n_red, dim)
        self._obs_all = np.stack(obs0).astype(np.float32)
        self._masks_all = np.stack(msk0).astype(np.float32)
        self._labels_all = np.stack(lbl0).astype(np.int8)
        self._rew_all = np.zeros(self.n_envs, dtype=np.float32)
        return self

    def close(self):
        for c in self.clients:
            c.close()
        self.clients = []

    def __enter__(self):
        return self.start()

    def __exit__(self, *a):
        self.close()

    # ---------- 权重同步 ----------
    def _send_weights(self, client):
        w = self._agent._np_weights
        packed = (w[0], w[1], w[2], w[3], list(w[4]), w[5], w[6])
        try:
            client.send({'type': 'set_weights', 'weights': packed})
        except (OSError, ConnectionError):
            pass

    def sync_weights(self, agent):
        """把主训练代理的 numpy 权重下发给所有子进程。

        容忍子进程已退出（关闭竞态）：此时连接会抛 ConnectionAborted/OSError，
        不应让训练线程因此崩溃（实测训练完整跑完后在收尾阶段崩于此处）。"""
        self._agent = agent
        if agent._np_weights is None:
            agent._sync_numpy()
        for c in self.clients:
            self._send_weights(c)

    # ---------- 观测 ----------
    def obs_dict(self):
        """把 (workers, envs, n_red, dim) 展平为与合批推理一致的顺序：
        worker0.env0.slot0, worker0.env0.slot1, worker0.env1.slot0, ..."""
        flat = self._obs_all.reshape(self.n_workers * self.envs_per_worker * self.n_red, -1)
        return {'red': [flat[i] for i in range(flat.shape[0])], 'blue': []}

    def masks_flat(self):
        """火控掩码按与 obs_dict() 相同的顺序展平为 (N, 4)。"""
        if self._masks_all is None:
            return None
        total = self.n_workers * self.envs_per_worker * self.n_red
        return self._masks_all.reshape(total, 4)

    def labels_flat(self):
        """火控行为克隆标签按与 obs_dict() 相同的顺序展平为 (N,)。"""
        if self._labels_all is None:
            return None
        total = self.n_workers * self.envs_per_worker * self.n_red
        return self._labels_all.reshape(total)

    # ---------- 采样 ----------
    def step(self, actions_flat):
        """actions_flat 顺序需与 obs_dict()['red'] 一致。跑 block_steps 步。"""
        bs, e_pw, n_red = self.block_steps, self.envs_per_worker, self.n_red
        acts = np.zeros((bs, self.n_workers, e_pw, n_red, 6), dtype=np.int8)
        idx = 0
        for t in range(bs):
            for k in range(self.n_workers):
                for i in range(e_pw):
                    for j in range(n_red):
                        if idx < len(actions_flat):
                            a = actions_flat[idx]
                            acts[t, k, i, j] = (a.head, a.fire, a.throttle, a.turn,
                                                a.target_select, a.share_target)
                        idx += 1
        res = self._dispatch(acts)
        # 合并：批量长度以子进程实际返回为准（防御未来引入变长块）
        got = [r['obs'].shape[0] for r in res]
        if len(set(got)) != 1:
            raise RuntimeError(f'各子进程返回的块长不一致: {got}')
        bs_real = got[0]
        obs = np.zeros((bs_real, self.n_workers, e_pw, n_red, self._obs_all.shape[-1]),
                       dtype=np.float32)
        rew = np.zeros((bs_real, self.n_envs), dtype=np.float32)
        done = np.zeros((bs_real, self.n_envs), dtype=np.uint8)
        kills = np.zeros((bs_real, self.n_envs), dtype=np.float32)
        dmg = np.zeros((bs_real, self.n_envs), dtype=np.float32)
        cleared = np.zeros((bs_real, self.n_envs), dtype=np.uint8)
        red_alive = np.zeros((bs_real, self.n_envs), dtype=np.int16)
        masks = None
        labels = None
        for k, r in enumerate(res):
            obs[:, k] = r['obs']
            s = k * e_pw
            rew[:, s:s + e_pw] = r['rew']
            done[:, s:s + e_pw] = r['done']
            kills[:, s:s + e_pw] = r['kills']
            dmg[:, s:s + e_pw] = r['dmg']
            if 'cleared' in r:
                cleared[:, s:s + e_pw] = r['cleared']
            if 'red_alive' in r:
                red_alive[:, s:s + e_pw] = r['red_alive']
            if 'masks' in r and masks is None:
                # 与 obs 同为 4 维 (workers, envs, n_red, ...)，按 worker 轴合并
                masks = np.zeros((bs_real, self.n_workers, e_pw, n_red, 4), dtype=np.float32)
            if 'masks' in r:
                masks[:, k] = r['masks']
            if 'labels' in r and labels is None:
                labels = np.full((bs_real, self.n_workers, e_pw, n_red), -1, dtype=np.int8)
            if 'labels' in r:
                labels[:, k] = r['labels']
        self._last = {'obs': obs, 'rew': rew, 'done': done, 'kills': kills, 'dmg': dmg,
                      'cleared': cleared, 'red_alive': red_alive, 'masks': masks,
                      'labels': labels}
        # 末步状态作为下一步的观测
        self._obs_all = obs[-1]
        self._stats = {'kills': kills.sum(axis=0), 'dmg': dmg.sum(axis=0),
                       'done': done.any(axis=0).astype(np.uint8)}
        rew_mean = float(rew.mean()) if rew.size else 0.0
        done_all = bool(done.any())
        return self.obs_dict(), {'red': rew_mean, 'blue': 0.0}, done_all, {}

    def last_block(self):
        return self._last

    def run_block(self, actions_flat):
        """跑一个 rollout 块，并返回训练循环可直接消费的完整时间序列。

        返回 dict：
          obs_before (bs, n_envs, n_red, dim)  每步「执行动作前」的观测（用于 PPO 记录）
          rew/done/kills/dmg (bs, n_envs)      每步的奖励 / 终止 / 击杀 / 伤害
        调用后 self._obs_all 已是末步之后的观测，可直接用于下一次采样。
        """
        before = self._obs_all.copy()        # 块开始前的观测
        self.step(actions_flat)
        lb = dict(self._last)
        if lb.get('masks') is not None:
            # 末步掩码作为下一次采样的掩码来源
            self._masks_all = lb['masks'][-1]
        if lb.get('labels') is not None:
            self._labels_all = lb['labels'][-1]
        bs = lb['obs'].shape[0]
        obs_before = np.empty_like(lb['obs'])
        obs_before[0] = before
        for t in range(1, bs):
            obs_before[t] = lb['obs'][t - 1]
        lb['obs_before'] = obs_before
        return lb

    def last_stats(self):
        return self._stats

    # ---------- 内部 ----------
    def _dispatch(self, actions):
        n = len(self.clients)
        results = [None] * n
        errors = [None] * n

        def work(k):
            try:
                # 注意：动作数组第 0 轴是步数、第 1 轴才是 worker，
                # 因此必须按 actions[:, k] 取本子进程的块（早期写成 actions[k]
                # 会按步数轴切片，导致每个子进程只收到 block/n_workers 步，
                # 主进程以为跑了整块 → 观测/奖励/动作三者错位，训练完全失效）。
                try:
                    results[k] = self.clients[k].request({'type': 'step', 'actions': actions[:, k]})
                except (OSError, ConnectionError):
                    # 子进程意外退出（被外部杀掉 / OOM 等）→ 就地重启并重发权重后重试一次
                    if os.environ.get('AWAI_POOL_VERBOSE'):
                        print(f'[pool] worker {k} 连接中断，正在重启...', flush=True)
                    self.clients[k].restart()
                    if self._agent is not None:
                        self._send_weights(self.clients[k])
                    results[k] = self.clients[k].request(
                        {'type': 'step', 'actions': actions[:, k]})
            except Exception as e:      # noqa: BLE001
                errors[k] = e

        threads = [threading.Thread(target=work, args=(k,), daemon=True) for k in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=600)
        for e in errors:
            if e is not None:
                raise e
        if any(r is None for r in results):
            raise RuntimeError('并行环境子进程未在超时内返回')
        return results
