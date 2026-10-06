# AWAI v3 — 空战 AI 训练平台

红蓝对抗 2D 空战模拟器 + PPO 自博弈强化学习训练平台。**纯 pygame 手绘，不依赖 gym**，
从零实现地形/物理/感知/武器/PPO/课程环境/UI。

- 8 阶段课程训练（S0 索敌 → S7 多环境对抗评估），带断点续训与阶段混合采样
- 跨进程并行仿真池：环境吞吐 5,874 → **49,104 环境步/秒**（8.36×，实测）
- 推理走 numpy 快速路径：单次推理 5.3 ms → **0.11 ms**
- 8 种现实向地图，山体/城区会**阻断视线与雷达**
- 115 维观测：自身状态 + 战术态势 + 记忆池 + 建筑 + 雷达数据链共享
- 图形主菜单 + 命令行双入口，附带战/诊断/评估/报告生成工具

> 本版本相对 v2 的完整优化过程与实测数据（含被证伪的假设）见
> [`AWAI_分析报告.md`](AWAI_分析报告.md)。下面只讲**怎么用**。

---

## 1. 环境要求

| 项目 | 要求 | 本机实测 |
|---|---|---|
| Python | **3.8+** | 3.9.13 |
| pygame | >= 2.1 | 2.6.1 |
| numpy | >= 1.20 | 2.0.2 |
| torch | >= 1.10（CPU 版即可） | 2.8.0+cu128 |

> **Windows 用户注意**：本机的 `python` 是 Microsoft Store 的占位符，直接跑会跳到商店。
> 请统一使用 **`py -3`** 启动。

### 安装

```bash
py -3 -m pip install -r requirements.txt

# 确认装好了
py -3 -c "import pygame, numpy, torch; print('ok')"
```

缺依赖时程序不会甩一大段 traceback，而是直接告诉你缺什么、怎么装。

---

## 2. 快速开始

```bash
# 1) 看看有哪些现成模型（仓库自带 24 个兼容最终模型）与它们是否可用
py AWAI\AWAI.py --list-models

# 2) 看全部命令行参数
py AWAI\AWAI.py --help

# 3) 带界面启动 —— 下面三条"管线"都从这里进
py AWAI\AWAI.py
```

### 三条使用管线

主菜单提供三种用法，对应三类使用者。

#### ① 游玩者：部署机队，看 AI 交战

> 主菜单 → **部署对战（自编机队）**

**你不开飞机**——飞机由 AI 驾驶，你负责**部署**：编成红蓝双方的机队，然后看结果。
`↑/↓` 选行，`←/→` 改数值，`Enter` 开战：

| 可配置项 | 取值 |
|---|---|
| 机型（红 / 蓝各一） | 全部 10 种 |
| 数量（红 / 蓝各一） | 1 / 2 / 3 / 4 / 6 / 8 架 |
| 驾驶员（红 / 蓝各一） | **硬编码规则 AI（专家系统）** 或 **任意一个神经网络模型** |

底部会实时显示编成摘要（如"红 4× 标准战斗机（standard_v9_final_red） vs 蓝 2× 重型战斗机（硬编码规则AI）"），
并会**提前警告**所选的神经网络模型与当前代码不兼容。开战后每局结束自动重开下一局，
方便连续试不同编成；`ESC` 返回、`F` 切换跟随、`R` 立即重开。

想自己开飞机的话，主菜单另有 **手动驾驶（自己开一架）**：`↑/↓` 挑座机机型、
`←/→` 挑僚机数量（同机型，规则 AI 驾驶），`Enter` 起飞。键位：

| 键 | 作用 | 键 | 作用 |
|---|---|---|---|
| `W` / `S` | 加力 / 减速板 | `空格` | 机炮 |
| `A` / `D` | 左转 / 右转 | `M` / `B` / `F` | 导弹 / 炸弹 / 诱饵 |
| `Q` / `E` | 机头左 / 右偏一个扇区 | `R` | 重开一局 |
| `Tab` | **锁定最近的敌机** | `Y` | 锁定最近的敌方地面目标 |
| `U` | 切换锁定目标 | `C` | 解除锁定 |
| `V` | 相机跟随开关 | `ESC` | 返回主菜单 |

> **先按 `Tab` 锁定目标**：机炮的对准辅助（按提前点修正机头）只在有锁定目标时生效，
> 导弹也需要锁定才能指定目标。左下角面板会显示「锁定 · 敌机 · 距离」。
> 鼠标滚轮缩放、中键拖动平移。

#### ② 观赏者：看两个模型对战

> 主菜单 → **观战模式（模型对战）**

`↑/↓` 选红方模型、`Q`/`W` 选蓝方模型（`PgUp`/`PgDn` 翻页），`Enter` 开打。
两列各自滚动、选中项高亮，底部会写明当前选中的红蓝模型。战斗中 `ESC` 退出、
`F` 切换跟随、`R` 重开，结束后按任意键返回。

命令行等价写法（1v1，神经网络 vs 规则 AI）：

```bash
py AWAI\watch_show.py standard_v9_final_red mountain_corridor
```

#### ③ 训练者：训练新模型

> 主菜单 → **课程训练（8 阶段渐进）** 或 **AI 训练（PPO 自博弈）**

先选机型（10 种全可选），再输入模型名前缀。

命令行等价写法：

```bash
# 8 阶段课程训练（跨进程并行，无头约 10~20 分钟）
py AWAI\AWAI.py --curriculum standard --episodes 100 --norender --prefix standard_v3

# 只看某个阶段学得怎么样：伤害/击杀是否随回合上升
py AWAI\train_probe.py S1_定点目标 100

# 10v10 团战压力测试（10 种阵型，结果写 report/team_battle_10v10.json）
py AWAI\team_battle_test.py 1 standard_v9_final_red mountain_corridor

# 生成 HTML 训练检查页（report/AWAI_training_report.html）
py AWAI\report_gen.py
```

### 其它常用命令

```bash
# 1v1 对抗评估：神经网络 vs 规则 AI（对称竞技场，胜负只由空战决定）
py AWAI\bench_duel.py standard_v9_final_red -n 30

# 列出可用地图 / 课程阶段
py AWAI\AWAI.py --list-maps
py AWAI\AWAI.py --list-stages
```

---

## 3. 命令行

所有入口都支持 `--help`，且**在依赖没装全时也能打印帮助**（帮助逻辑早于
`import pygame/torch`）。同目录的辅助脚本还可直接复用主程序的列表：
`--list-models` / `--list-maps` / `--list-stages`。

### `AWAI\AWAI.py` — 主程序

```bash
py AWAI\AWAI.py                                     # 无参数 → 图形主菜单
py AWAI\AWAI.py --curriculum <机型> [选项]           # 课程训练
py AWAI\AWAI.py --evaluate <机型> [模型前缀]         # 评估各阶段表现
py AWAI\AWAI.py --bench [--episodes N] [--stage N]  # 并行吞吐基准
py AWAI\AWAI.py --list-models | --list-maps | --list-stages
```

`--curriculum` 选项：`--episodes N`（默认 100）、`--n_envs N`（默认 2）、
`--start S`（断点续训，默认 0）、`--norender`、`--prefix NAME`（默认 `<机型>_cur`）、
`--noparallel`、`--stage-mix`、`--workers N`（默认 8）、`--envs_per_worker N`（默认 8）、
`--block N`（默认 256）、`--fire_aux C`（默认 0.0）。

可用机型：`standard` `speedy` `heavy` `attacker` `advanced_attacker` `bomber`
`dogfighter` `jet` `missile_plane` `attacker_rear`

### 辅助脚本

| 脚本 | 用途 | 示例 |
|---|---|---|
| `watch_show.py` | 实时观战 | `py AWAI\watch_show.py [模型名] [地图]` |
| `bench_duel.py` | 1v1 对称空战评估 | `py AWAI\bench_duel.py <模型...> -n 30` |
| `train_probe.py` | 单阶段学习诊断 | `py AWAI\train_probe.py <阶段名> [回合数] [--load 模型]` |
| `train_aircombat.py` | 纯空战专项训练 | `py AWAI\train_aircombat.py [回合数] [前缀]` |
| `team_battle_test.py` | 10v10 团战测试 | `py AWAI\team_battle_test.py [回合数] [模型] [地图]` |
| `report_gen.py` | 生成 HTML 检查页 | `py AWAI\report_gen.py [--eval N] [--out 路径]` |
| `env_pool.py` | 跨进程并行仿真池（库，由主程序调用） | — |
| `awai_cli.py` | 命令行公共支持（依赖自检 / 中文报错 / 帮助） | — |

不用记参数顺序：这几个脚本的位置参数会**自动识别**——纯数字当回合数、
已知地图名当地图、其余当模型名。

---

## 4. 文件结构

| 文件 | 说明 |
|---|---|
| `AWAI/AWAI.py` | 全部核心：地形、物理、感知、单位、弹药、PPO、课程环境、UI、命令行入口 |
| `AWAI/env_pool.py` | **跨进程并行仿真池**（把环境仿真分散到多核，绕开 GIL） |
| `AWAI/awai_cli.py` | 命令行公共支持：依赖自检、中文报错、`--help`、通用 argparse 包装 |
| `AWAI/train_probe.py` | 单阶段训练诊断，打印伤害/击杀随回合的变化并给出判定 |
| `AWAI/bench_duel.py` | 1v1 对称空战评估（胜率/击落率/均伤害） |
| `AWAI/train_aircombat.py` | 纯空战专项训练（只练与考核同分布的环境） |
| `AWAI/watch_show.py` | 实时观战（神经网络 vs 规则 AI），支持选地图 |
| `AWAI/team_battle_test.py` | 10v10 团战互殴测试，10 种阵型，输出 JSON |
| `AWAI/report_gen.py` | 生成纯 SVG 的 HTML 训练检查页（自动发现可评估模型） |
| `AWAI/legacy/AWAI_legacy.py` | v1 旧版（模块级全局列表、O(n²) 扫描），保留对照 |
| `AWAI/models/*.pt` | 模型检查点（含 `obs_dim` / `obs_version` 版本信息） |
| `AWAI/modelset.txt` | 模型列表（主菜单「模型管理」读写，决定观战默认模型） |

---

## 5. 性能设计（重要）

### 为什么用 CPU 做推理，而不是 GPU

**因为在这个负载下 GPU 更慢。** 实测 RTX 5070 Laptop（批量 = 每步观测条数）：

| 每步批量 | CPU numpy | GPU torch | GPU 每样本 |
|---|---|---|---|
| 1 | **0.068 ms** | 3.43 ms | 3.5 ms（慢 51×） |
| 4 | **0.076 ms** | 3.30 ms | 0.83 ms（慢 43×） |
| 16 | **0.17 ms** | 3.29 ms | 0.21 ms（慢 19×） |
| 1024 | 5.73 ms | **3.55 ms** | 3.3 µs（GPU 才反超） |

单次 CUDA kernel 启动+同步的固定开销约 **3.3 ms**，与批量无关；PPO 采样每步只有
几条到几十条观测，因此 GPU 完全不划算。**训练（反向传播）仍由 torch 负责**，
推理走 numpy 快速路径（与 torch 数值一致，偏差 <1e-8）。

### 真正的瓶颈：GIL 让仿真只能用一个核

| n_envs | 决策步/秒 | 说明 |
|---|---|---|
| 1 | 1402 | |
| 16 | 1678 | 加 16 倍环境仅提升 1.20× |
| 32 | 1699 | 加 32 倍环境仅提升 1.21× |

原因是 Python GIL：同进程内所有环境共享一个核，32 核里 31 个闲置。
`env_pool.py` 用**跨进程**方案解决（子进程各跑若干环境，自己用 numpy 推理采样，
主进程只做 PPO 更新与权重下发）。

| 配置 | 环境步/秒 | 加速 |
|---|---|---|
| 串行 n_envs=4（基线） | 5,874 | 1.00× |
| 并行 8 workers × 8 envs，块长 256 | **49,104** | **8.36×** |

**关键经验**：块长（`--block`）比 worker 数更重要 —— block 32→128 让环境吞吐
从 8.6 万涨到 33 万，说明瓶颈曾是 **IPC 往返次数**而非仿真本身。

> 注：本环境的 DSH 文件沙箱禁止创建命名管道（`multiprocessing.Pipe`/`Queue` 报
> `WinError 5 拒绝访问`），因此 `env_pool.py` 的 IPC 走**本地回环 TCP**。

### 课程训练的回合数语义

`--episodes N` 是**默认值**，各阶段的 `StageSpec.episodes` 优先
（S0=200、S1/S2/S3=300，对抗阶段用该默认值）。并行模式下每个环境各跑 N 回合，
因此总回合数 = N × 环境数。

---

## 6. 模拟机制要点

- **飞控**：推力 − 阻力(v²) 演化速度；加力/减速板带持续时间与冷却门控；
  转向速率随速度升高而下降（低速灵活、高速笨重）。
- **三头离散动作**：共 8,640 种组合
  （头控制 8 扇区 × 火控 4 × 油门 3 × 转向 3 × 目标选择 5 × 目标共享 6）。
- **感知**：概率可见度（目标大小/类型/距离/机头夹角）+ 统一记忆池（8 条）+
  敌方建筑记忆（4 条）+ 雷达数据链共享 + **地形视线遮挡**。
- **观测 115 维**：自身状态 20 + 战术态势 16（最近 2 敌机 + 最近 2 友机）+ 记忆池 56 +
  建筑 16 + 机场 3 + 锁定 3 + 激进度 1。

---

## 7. 现实向地图（8 种，可选变体）

在菜单「选择地图」里挑选，常规模式与观战模式生效，也可用 `py AWAI\AWAI.py --list-maps`
查看。山体/丘陵/城区会**阻断视线与雷达**（低空可借地形遮蔽突防），
地面单位藏身地形内还会降低被发现概率。

| 布局名 | 名称 | 战术特点 |
|---|---|---|
| `range` | 空旷靶场 | 无地形，课程基础阶段专用 |
| `duel` | 经典战场 | 开阔交战区 + 稀疏丘陵掩护 |
| `mountain_corridor` | 山脊走廊 | 两道山脊夹出中央通道与山口 |
| `river_valley` | 河谷分割 | 水域纵贯，两处缺口为通行要点 |
| `island_chain` | 岛链海域 | 岛体遮蔽雷达，适合低空突防 |
| `urban` | 城区攻防 | 建筑群严重遮蔽雷达与视线 |
| `coastal` | 海陆交界 | 一侧近海一侧丘陵，攻防纵深大 |
| `desert_ridge` | 荒漠台地 | 高耸台地，高地雷达视野广阔 |

---

## 8. 训练机制

**8 阶段课程**（S0 索敌 → S7 多环境对抗评估）：前一阶段检查点自动作为下一阶段起点
（断点续训）。阶段达标可提前晋级，否则跑满回合。检查点保存为
`models/{prefix}_S{n}_red.pt`，最终复制为 `{prefix}_final_red.pt`。

**奖励设计（v3 重构，战斗输出主导）**

| 项 | 设计 | 说明 |
|---|---|---|
| 伤害 | 每点伤害给分（`damage_coef`） | 主战斗信号 |
| 击杀 | 价值 × `kill_mult`，击落敌机额外 +50 | 高价值目标 |
| 索敌 | **每个目标仅首次发现给一次** | 防止"盘旋刷索敌分"（旧版此项占总奖励 97~100%，导致 0 击杀） |
| 锁定 | 保持锁定小额；**每目标首次锁定一次性** | 防止反复切换锁定刷分（旧版此项占 92.7%） |
| 接近 | 势能差分（靠近得分、远离扣分，对称有界） | 提供靠近梯度，避免盘旋停滞或送死 |
| 清场 | `cleared_bonus` | 关卡通关 |

**PPO 超参（经实测调优）**：`lr=1e-4`、`epochs=6`、`batch=128`、`clip=0.2`、
`ent_coef=0.03`、`gamma=0.99`、`lam=0.95`，长回合每 **500 步**做一次中途更新。
推理走 numpy 快速路径（训练仍由 torch 负责）。

**已知的重要局限（务必先读）**

课程式顺序训练存在**灾难性遗忘**：后段对抗阶段会覆盖前段学到的基础瞄准行为。
实测 v4 完整课程后，基础阶段（S0–S3）确定性评估全部归零，而对抗阶段（S4–S7）正常。
因此：

- **"单一最终模型"本身不适合这个课程**——请按用途选对应阶段的检查点；
- 对抗评估请用 `bench_duel.py`（对称 1v1 竞技场），它才是"空战谁更强"的判据；
- 仓库自带的 `standard_v9_final_red` 在对称 1v1 评估中胜率 **0%**（v7 约 60%），
  完整数据与失败原因见 [`AWAI_分析报告.md`](AWAI_分析报告.md) 第五节。

---

## 9. 模型兼容性

观测布局变更会让旧检查点失效。检查点内记录了 `obs_dim` / `obs_version`：

| obs_dim | 版本 | 对应代码 | 状态 |
|---|---|---|---|
| 95 | — | cur4 | 不兼容 |
| 98 | — | cur6 | 不兼容 |
| 99 | 2 | cur7 | 不兼容 |
| **115** | **3** | 当前 | 可用 |

加载不兼容模型时 `PPOAgent.load()` / `try_load_agent()` 会给出**中文诊断**而不是崩溃。
想知道手上哪个模型能用，直接跑：

```bash
py AWAI\AWAI.py --list-models
```

**本仓库只分发当前观测版本的最终模型（24 个）**：

- `standard_v{3..9}_final_{red,blue}.pt`
- `standard_v{5..9}_mixed_{red,blue}.pt`

其它检查点（旧观测维度 95/98/99 的 cur4/cur6/cur7 系列、各阶段中间检查点 S1~S8、
临时测试模型）留在开发机本地，不入库——它们不是加载不了，就是会被后段训练覆盖，
分发出去只会让人踩坑。

---

## 10. 排错

| 现象 | 原因与处理 |
|---|---|
| `python` 打开了 Microsoft Store | 用 `py -3`，不要用 `python` |
| 缺少依赖 / 一堆 `ImportError` | `py -3 -m pip install -r requirements.txt` |
| `--help` 没反应、直接弹窗 | 已修复；旧版本会这样。请更新到当前代码 |
| 模型加载失败并提示 `obs_dim` 不符 | 旧观测维度的检查点，用 `--list-models` 挑可用模型 |
| 观战提示"没有可用模型" | `models/` 为空或 `modelset.txt` 里的条目都不存在；先训练或用 `--list-models` 挑 |
| 多进程训练报 `WinError 5` | IPC 已改用本地回环 TCP，正常不该再出现；若仍报请附完整堆栈提 issue |
| 中文日志写进文件后乱码 | 重定向输出时程序会自动切 UTF-8；若自行重定向请加 `PYTHONIOENCODING=utf-8` |

---

## 11. 许可

本项目以 **GNU General Public License v3.0** 发布，全文见 [`LICENSE`](LICENSE)。

```
AWAI — 空战 AI 训练平台
Copyright (C) 2026 LLYlab

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
```
