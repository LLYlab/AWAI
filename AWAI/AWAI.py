#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AWAI v3 - 空战 AI 训练平台
================================================================================
红蓝对抗 2D 空战模拟器 + PPO 自博弈强化学习训练平台。纯 pygame 手绘，不依赖 gym。

**不带任何参数**运行会进入图形主菜单：

    py AWAI\\AWAI.py

命令行用法：

    py AWAI\\AWAI.py --curriculum <机型> [选项]      按 8 阶段课程训练
    py AWAI\\AWAI.py --evaluate <机型> [模型前缀]    评估模型在各课程阶段的表现
    py AWAI\\AWAI.py --bench [--episodes N] [--stage N]
                                                     训练吞吐基准（串行 vs 跨进程并行）
    py AWAI\\AWAI.py --list-models                   列出 models/ 里的模型及其兼容性
    py AWAI\\AWAI.py --list-maps                     列出可用地图布局
    py AWAI\\AWAI.py --list-stages                   列出课程阶段
    py AWAI\\AWAI.py --version                      显示版本

常用示例：

    py AWAI\\AWAI.py --curriculum standard --episodes 100 --norender --prefix standard_v3
    py AWAI\\AWAI.py --evaluate standard standard_v3_final

可用机型（--curriculum 与 --evaluate 的第一个参数）：

    standard  speedy  heavy  attacker  advanced_attacker  bomber
    dogfighter  jet  missile_plane  attacker_rear

同目录辅助脚本：

    watch_show.py        实时观战：神经网络 vs 规则 AI
    bench_duel.py        1v1 对称空战评估（胜率 / 击落率）
    train_probe.py       单阶段学习诊断（伤害 / 击杀是否随回合上升）
    train_aircombat.py   纯空战专项训练
    team_battle_test.py  10v10 团战压力测试
    report_gen.py        生成 HTML 训练报告
    env_pool.py          跨进程并行仿真池（由 AWAI.py 调用，一般无需直接运行）

环境要求：Python 3.8+、pygame、numpy、torch（CPU 版即可运行）。见 requirements.txt。
详细文档见项目根目录 README.md。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import awai_cli

# 下面两步必须发生在导入 pygame / numpy / torch 之前：
# 让 `--help` 在没有依赖的机器上也能立即返回，缺依赖时给中文安装指引而不是 traceback。
if __name__ == '__main__':
    awai_cli.handle_early_exit(__doc__)
awai_cli.check_requirements()

import argparse
import math
import random
import time
import threading
import collections
import types as _types
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import pygame
from pygame.locals import *

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ======================================================================
# 0. 全局设置
# ======================================================================
WORLD_W, WORLD_H = 3000, 2000
SCREEN_W, SCREEN_H = 1200, 800
FPS = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODELSET_FILE = os.path.join(BASE_DIR, "modelset.txt")

pygame.init()
try:
    FONT = pygame.font.SysFont('simhei', 18)
    BIG_FONT = pygame.font.SysFont('simhei', 26)
    MENU_FONT = pygame.font.SysFont('simhei', 40)
    SMALL_FONT = pygame.font.SysFont('simhei', 14)
except Exception:
    FONT = pygame.font.SysFont('Arial', 18)
    BIG_FONT = pygame.font.SysFont('Arial', 26)
    MENU_FONT = pygame.font.SysFont('Arial', 40)
    SMALL_FONT = pygame.font.SysFont('Arial', 14)

WHITE = (255, 255, 255); BLACK = (0, 0, 0); RED = (220, 50, 50); BLUE = (50, 50, 220)
GREEN = (0, 200, 0); GRAY = (150, 150, 150); YELLOW = (255, 255, 0); ORANGE = (255, 150, 0)
DARK_RED = (200, 0, 0); LIGHT_BLUE = (100, 200, 255); CYAN = (0, 200, 200)
DARK_GREEN = (0, 100, 0)

DETECT_RANGE = 1500.0      # 探测距离（硬上限）
MEMORY_FORGET_MS = 5000    # 记忆超时
MS_PER_STEP = 1000.0 / 60.0  # 每模拟步对应的毫秒数（模拟时间基准，替代墙钟）


# ======================================================================
# 1. 配置区（dataclass 化）
# ======================================================================
@dataclass
class GunSpec:
    damage: float = 10
    range: float = 100
    cooldown: int = 5
    ammo: int = 500
    angle: float = 5          # 火控锥角（度，相对机头/炮塔）：进入该锥角才算取得射击条件
    bullet_speed: float = 8.0   # 弹丸绝对速度（帧步长）。真实机炮 ~1000m/s，
                                # 远快于本模拟原来的"机速+6"；速度过慢会让
                                # "飞行时间 × 转向速率"的漂移远超锥角，导致永远打不中。
    hit_radius: float = 2.0     # 额外命中半径（与目标 size 相加），代表弹散布/命中容差

@dataclass
class BombSpec:
    count: int = 1
    damage: float = 150
    radius: float = 40
    guided: bool = False
    speed: float = 3.0

@dataclass
class MissileSpec:
    mtype: str = "A-IR"
    count: int = 0

@dataclass
class FlightSpec:
    cruise_speed: float = 2.0
    turn_rate: float = 2.0    # 巡航速度下的最大转向速率（度/帧）
    drag_k: float = 0.002
    ab_max_ticks: int = 120   # 加力最长持续帧数（2 秒 @60fps）
    ab_cd_ticks: int = 240    # 加力冷却帧数（4 秒）
    brake_max_ticks: int = 90    # 减速板最长持续帧数（1.5 秒）
    brake_cd_ticks: int = 180    # 减速板冷却帧数（3 秒）
    # 由 cruise_speed 派生：推力 = drag_k * v^2 使巡航时匀速
    @property
    def cruise_thrust(self): return self.drag_k * self.cruise_speed ** 2
    @property
    def afterburner_thrust(self): return 3.0 * self.cruise_thrust
    @property
    def airbrake_decel(self): return 0.25
    @property
    def max_speed(self): return self.cruise_speed * 1.7
    @property
    def min_speed(self): return max(0.3, self.cruise_speed * 0.3)

@dataclass
class AircraftSpec:
    name: str
    max_hp: float
    size: int
    shape: str
    flight: FlightSpec
    gun: GunSpec
    bomb: BombSpec
    missile: MissileSpec
    flares: int = 2
    visibility: float = 5.0
    color: Tuple[int, int, int] = (200, 50, 50)

AIRCRAFT_SPECS: Dict[str, AircraftSpec] = {
    # 机炮手感调参（v3）：原 5° 锥角 + 10 伤害使标准机在移动/机动目标上几乎无法命中。
    # v9：补上空对空导弹。原先 standard 被配成 MissileSpec('A-IR', 0)（**零导弹**），
    #     只能靠机炮，而受控实测机炮打机动目标命中率仅约 3%（锥角 9°~30° 都一样，
    #     说明瓶颈是弹丸提前量而非角度）→ 1v1 空战双方几乎都零伤害，
    #     胜负只能靠时间到的血量比较，根本分不出高下。
    #     真实战斗机的主力是导弹，这里给标准机 4 发 A-IR（追踪弹，命中率高）。
    'standard': AircraftSpec('标准战斗机', 100, 7, 'triangle',
        FlightSpec(cruise_speed=2.0, turn_rate=2.0),
        GunSpec(14, 260, 4, 600, 12, bullet_speed=26.0, hit_radius=14.0),
        BombSpec(1, 150, 40), MissileSpec('A-IR', 4),
        visibility=5),
    'speedy': AircraftSpec('高速战斗机', 60, 5, 'narrow',
        FlightSpec(cruise_speed=4.0, turn_rate=1.0),
        GunSpec(8, 80, 5, 400, 5), BombSpec(1, 120, 35), MissileSpec('A-IR', 0), visibility=3),
    'heavy': AircraftSpec('重型战斗机', 200, 10, 'triangle',
        FlightSpec(cruise_speed=2.5, turn_rate=1.0),
        GunSpec(30, 150, 20, 300, 5), BombSpec(2, 200, 50), MissileSpec('A-IR', 0), visibility=8),
    'attacker': AircraftSpec('攻击机', 150, 11, 'triangle',
        FlightSpec(cruise_speed=1.5, turn_rate=1.0),
        GunSpec(40, 120, 20, 400, 5), BombSpec(2, 250, 60), MissileSpec('A-IR', 0), visibility=10),
    'advanced_attacker': AircraftSpec('先进攻击机', 160, 11, 'triangle',
        FlightSpec(cruise_speed=1.6, turn_rate=1.0),
        GunSpec(45, 140, 18, 350, 10), BombSpec(2, 350, 80, guided=True, speed=4.0),
        MissileSpec('A-IR', 0), visibility=12),
    'bomber': AircraftSpec('轰炸机', 300, 9, 'wide_triangle',
        FlightSpec(cruise_speed=1.5, turn_rate=0.5),
        GunSpec(10, 60, 20, 200, 15), BombSpec(8, 400, 100), MissileSpec('A-IR', 0), visibility=20),
    'dogfighter': AircraftSpec('狗斗机', 80, 6, 'small_triangle',
        FlightSpec(cruise_speed=1.8, turn_rate=5.0),
        GunSpec(8, 90, 5, 600, 5), BombSpec(1, 100, 30), MissileSpec('A-IR', 0), visibility=4),
    'jet': AircraftSpec('德国喷气', 80, 6, 'narrow',
        FlightSpec(cruise_speed=5.5, turn_rate=0.8),
        GunSpec(10, 100, 5, 400, 5), BombSpec(1, 130, 35), MissileSpec('A-IR', 0), visibility=3),
    'missile_plane': AircraftSpec('导弹飞机', 120, 9, 'triangle',
        FlightSpec(cruise_speed=2.5, turn_rate=2.0),
        GunSpec(0, 0, 50, 0, 30), BombSpec(0, 0, 0),
        MissileSpec('A-IR', 8), visibility=6, color=(0, 0, 255)),
    'attacker_rear': AircraftSpec('攻击机改', 150, 11, 'triangle',
        FlightSpec(cruise_speed=1.5, turn_rate=1.0),
        GunSpec(40, 120, 10, 500, 180), BombSpec(2, 220, 55), MissileSpec('A-IR', 0), visibility=10),
}

MISSILE_TYPES = {
    'A-IR':  {'speed': 7, 'turn': 10, 'range': 400, 'type': 'ir', 'view': 30, 'color': (255, 100, 100)},
    'A-IRB': {'speed': 7, 'turn': 10, 'range': 400, 'type': 'ir', 'view': 6,  'color': (255, 50, 50)},
    'G-IR':  {'speed': 8, 'turn': 12, 'range': 600, 'type': 'ir', 'view': 30, 'color': (255, 80, 80)},
    'G-IRA': {'speed': 5, 'turn': 8,  'range': 1500,'type': 'ir', 'view': 30, 'color': (200, 60, 60)},
    'G-RA':  {'speed': 9, 'turn': 15, 'range': 600, 'type': 'radar', 'color': (200, 200, 255)},
    'GC-IR': {'speed': 7, 'turn': 10, 'range': 700, 'type': 'ir', 'view': 30, 'color': (255, 120, 120)},
    'GC-R':  {'speed': 8, 'turn': 5,  'range': 500, 'type': 'radar', 'search_angle': 20, 'beam': 1, 'color': (150, 150, 255)},
}

@dataclass
class BuildingSpec:
    name: str
    max_hp: float
    size: int
    color: Tuple[int, int, int]
    attack_range: float = 0
    cooldown: int = 0
    damage: float = 0
    src_range: float = 0          # 火控/搜索范围
    missile_main: str = ''        # 主导弹类型
    missile_special: str = ''     # 特殊导弹类型
    scan_range: float = 0         # 雷达扫描范围
    value: float = 20

BUILDING_SPECS: Dict[str, BuildingSpec] = {
    'airbase': BuildingSpec('基地机场', 6000, 40, (200, 50, 50), value=200),
    'forward_airbase': BuildingSpec('前线机场', 3000, 30, (200, 80, 80), value=100),
    'resource_base': BuildingSpec('资源基地', 1000, 25, (0, 200, 200), value=150),
    'missile_aa': BuildingSpec('导弹防空塔', 500, 20, (200, 50, 50),
                               attack_range=300, cooldown=120, src_range=300,
                               missile_main='G-IR', missile_special='G-RA', value=50),
    'long_range_aa': BuildingSpec('远程导弹防空塔', 400, 22, (200, 50, 50),
                                  attack_range=400, cooldown=180, src_range=400,
                                  missile_main='G-IRA', missile_special='G-RA', value=50),
    'gun_aa': BuildingSpec('机炮防空塔', 300, 18, (200, 50, 50),
                           attack_range=200, cooldown=8, damage=15, src_range=260, value=50),
    'radar': BuildingSpec('预警雷达', 200, 20, (0, 100, 0), scan_range=2500, value=20),
    'rally': BuildingSpec('部队集结点', 400, 15, (150, 150, 150), value=30),
}

@dataclass
class GroundSpec:
    name: str
    max_hp: float
    size: int
    speed: float
    color: Tuple[int, int, int]
    attack_range: float
    damage: float
    cooldown: int
    bullet_speed: float = 6
    src_range: float = 0
    missile_main: str = ''
    missile_special: str = ''
    value: float = 15

GROUND_SPECS: Dict[str, GroundSpec] = {
    # 关卡可学性调参（v3）：原本坦克 300 血 / 机炮 10 伤 = 需 30 发连续命中，
    # 配合移动目标与 2000 帧回合根本不可达成（实测规则AI 2000 帧零伤害）。
    # 现按“静止靶 ~8 发、坦克 ~10 发可毁”重新配平，保留相对强弱关系。
    'tank': GroundSpec('坦克', 120, 8, 0.8, (200, 50, 50), 200, 40, 40, value=40),
    'gun_aa_vehicle': GroundSpec('机炮防空车', 90, 10, 1.2, (200, 50, 50),
                                 250, 15, 8, bullet_speed=8, src_range=250, value=30),
    'missile_aa_vehicle': GroundSpec('导弹防空车', 80, 10, 1.2, (200, 50, 50),
                                     350, 30, 120, src_range=350,
                                     missile_main='GC-IR', missile_special='GC-R', value=30),
    'infantry': GroundSpec('机动步兵', 50, 6, 0.6, (150, 150, 150), 150, 5, 5,
                           bullet_speed=10, value=15),
}

# ======================================================================
# 1.5 地形系统（更贴近现实的地图）
# ======================================================================
# 设计动机：原来的战场是一张空地（'range'）或带几个建筑的平地（'duel'），
# 没有地形概念 —— 侦察没有遮挡、没有走廊、没有高地优势，战术维度基本为零。
# 这里引入圆形地形特征：山体/丘陵/城区，具有“高度”，会遮挡视线与雷达，
# 从而产生雷达盲区、山口走廊与高地雷达站等真实战术要素。
# 注意：地形只遮挡【探测与火控】，不阻挡飞行（保持飞行手感不变）。
TERRAIN_CLEAR, TERRAIN_HILL, TERRAIN_MOUNTAIN, TERRAIN_CITY, TERRAIN_WATER = 0, 1, 2, 3, 4
TERRAIN_NAMES = {TERRAIN_CLEAR: '平原', TERRAIN_HILL: '丘陵', TERRAIN_MOUNTAIN: '山体',
                 TERRAIN_CITY: '城区', TERRAIN_WATER: '水域'}
TERRAIN_COLORS = {TERRAIN_CLEAR: (245, 245, 240), TERRAIN_HILL: (222, 214, 190),
                  TERRAIN_MOUNTAIN: (196, 184, 168), TERRAIN_CITY: (214, 214, 218),
                  TERRAIN_WATER: (186, 214, 232)}
# 各类型基础高度（用于视线遮挡判定）
TERRAIN_BASE_HEIGHT = {TERRAIN_CLEAR: 0.0, TERRAIN_HILL: 45.0, TERRAIN_MOUNTAIN: 110.0,
                       TERRAIN_CITY: 35.0, TERRAIN_WATER: 0.0}


@dataclass
class TerrainFeature:
    """圆形地形特征：位置 + 半径 + 高度。"""
    x: float
    y: float
    r: float
    kind: int = TERRAIN_HILL
    height: Optional[float] = None
    name: str = ''

    def __post_init__(self):
        if self.height is None:
            self.height = TERRAIN_BASE_HEIGHT.get(self.kind, 0.0)

    @property
    def blocks_los(self) -> bool:
        return self.kind in (TERRAIN_HILL, TERRAIN_MOUNTAIN, TERRAIN_CITY) and self.height > 0

    def contains(self, x, y) -> bool:
        return (x - self.x) ** 2 + (y - self.y) ** 2 <= self.r * self.r

    def edge_distance(self, x, y) -> float:
        """点到地形外缘的距离（内部为负）。"""
        return math.hypot(x - self.x, y - self.y) - self.r


class TerrainMap:
    """地形集合：提供视线遮挡查询与绘制。

    性能说明：LOS 查询是每帧最高频的几何运算（实测 10v10 下约 9000 次/步）。
    因此这里做两项优化：
      1) 逐【地形】计算其圆心到线段的最近距离，而不是逐【采样点】遍历全部地形
         （原实现 = O(采样点 × 地形数)，在 15 个地形的山脊图上让每步从 12.8ms 涨到 40ms）；
      2) 用 squared-distance + 包围盒早退，避免 sqrt 与无关地形的判定。
    """

    def __init__(self, features=None, name='', desc=''):
        self.features: List[TerrainFeature] = list(features or [])
        self.name = name
        self.desc = desc
        self.sight_height = 60.0     # 视线的标称高度（用于“能否越过高地看到”）

    def add(self, *features):
        self.features.extend(features)
        return self

    def height_at(self, x, y) -> float:
        """该点地形高度（取覆盖该点的最高地形）。"""
        h = 0.0
        for f in self.features:
            if f.height > h and f.contains(x, y):
                h = f.height
        return h

    def cover_bonus(self, x, y) -> float:
        """地形掩护加成：处于地形内部时降低被发现概率。"""
        for f in self.features:
            if f.blocks_los and f.contains(x, y):
                return 0.45 if f.kind == TERRAIN_MOUNTAIN else 0.3
        return 0.0

    def blocks_sight(self, x0, y0, x1, y1) -> bool:
        """视线是否被地形阻断。

        判定：对每个地形求其圆心到线段的最近距离，若线段擦过地形内部，则比较
        “地形高度”与“视线在该处的高度”（视线高度随航程线性抬升，远处可见性更好）。
        高度更高的地形才能遮挡 —— 这就是雷达盲区与山口走廊的物理来源。"""
        feats = self.features
        if not feats:
            return False
        dx, dy = x1 - x0, y1 - y0
        seg2 = dx * dx + dy * dy
        if seg2 < 1e-9:
            return False
        sh = self.sight_height
        for f in feats:
            if not f.blocks_los:
                continue
            # 线段到圆心的最近参数 t
            if seg2 > 1e-9:
                t = ((f.x - x0) * dx + (f.y - y0) * dy) / seg2
            else:
                t = 0.0
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            cxp, cyp = x0 + dx * t, y0 + dy * t
            ddx, ddy = f.x - cxp, f.y - cyp
            if ddx * ddx + ddy * ddy >= f.r * f.r:
                continue                      # 线段不经过该地形
            sight_h = sh * (t + 0.15)         # 视线高度：随航程抬升
            if f.height > sight_h:
                return True
        return False

    def draw(self, screen, camera):
        for f in self.features:
            sx, sy = camera.world_to_screen(f.x, f.y)
            sr = max(2, int(f.r * camera.zoom))
            if sx + sr < 0 or sx - sr > camera.sw or sy + sr < 0 or sy - sr > camera.sh:
                continue
            color = TERRAIN_COLORS.get(f.kind, (230, 230, 230))
            pygame.draw.circle(screen, color, (sx, sy), sr)
            # 山体轮廓线：用高度决定描边深浅，制造比高感
            edge = tuple(max(0, c - 45) for c in color)
            pygame.draw.circle(screen, edge, (sx, sy), sr, 1)
            if f.kind == TERRAIN_MOUNTAIN and sr > 12:
                pygame.draw.circle(screen, tuple(max(0, c - 30) for c in color), (sx, sy), int(sr * 0.55))
            if f.kind == TERRAIN_CITY and sr > 8:
                for k in range(3):
                    cx = sx - sr // 2 + k * sr // 2
                    pygame.draw.rect(screen, edge, (cx - 2, sy - 3, 5, 6))


# ---------- 现实向地图布局（多变体随机轮换） ----------
def build_terrain_map(layout, variant=0) -> TerrainMap:
    """按布局名与变体号生成地形。所有布局都以 (WORLD_W, WORLD_H) 为界。"""
    W, H = WORLD_W, WORLD_H
    rnd = random.Random(20260815 + variant * 977)
    tm = TerrainMap(name=layout)

    if layout in ('range', 'duel'):
        # 经典布局：保留少量丘陵作掩护，但保证交战线开阔（不破坏既有课程可学性）
        if layout == 'duel':
            for _ in range(3 + variant % 2):
                x = rnd.uniform(W * 0.30, W * 0.70)
                y = rnd.uniform(H * 0.18, H * 0.82)
                if abs(y - H * 0.5) < 260 and abs(x - W * 0.5) < 300:
                    continue          # 让出中央交战区
                tm.add(TerrainFeature(x, y, rnd.uniform(120, 200), TERRAIN_HILL))
        return tm

    if layout == 'mountain_corridor':
        # 山脊走廊：两道南北向山脊夹出一条中央通道，逼出“沿走廊交战 / 绕山迂回”
        tm.desc = '山脊夹峙，仅中央通道与两处山口可通'
        for i in range(7):
            y = H * 0.08 + i * H * 0.14
            tm.add(TerrainFeature(W * 0.30 + rnd.uniform(-70, 70), y, rnd.uniform(150, 210),
                                  TERRAIN_MOUNTAIN, name='西山脊'))
            tm.add(TerrainFeature(W * 0.70 + rnd.uniform(-70, 70), y, rnd.uniform(150, 210),
                                  TERRAIN_MOUNTAIN, name='东山脊'))
        tm.add(TerrainFeature(W * 0.5, H * 0.5, 150, TERRAIN_HILL, name='中央高地'))
        return tm

    if layout == 'river_valley':
        # 河谷：一条南北向水域把战场一分为二，仅两处丘陵缺口可低空穿越
        tm.desc = '河谷纵贯，两处丘陵缺口为通行要点'
        for i in range(9):
            tm.add(TerrainFeature(W * 0.5 + rnd.uniform(-60, 60), H * 0.06 + i * H * 0.11,
                                  rnd.uniform(110, 160), TERRAIN_WATER, name='主河道'))
        tm.add(TerrainFeature(W * 0.5, H * 0.22, 130, TERRAIN_HILL, height=20, name='北缺口'))
        tm.add(TerrainFeature(W * 0.5, H * 0.78, 130, TERRAIN_HILL, height=20, name='南缺口'))
        for _ in range(4):
            tm.add(TerrainFeature(rnd.uniform(W * 0.10, W * 0.38), rnd.uniform(H * 0.15, H * 0.85),
                                  rnd.uniform(90, 150), TERRAIN_HILL))
            tm.add(TerrainFeature(rnd.uniform(W * 0.62, W * 0.90), rnd.uniform(H * 0.15, H * 0.85),
                                  rnd.uniform(90, 150), TERRAIN_HILL))
        return tm

    if layout == 'island_chain':
        # 岛链：多座岛屿与礁石，低空可借岛体遮蔽雷达
        tm.desc = '岛链散布，可借岛体遮蔽雷达'
        for i in range(6):
            tm.add(TerrainFeature(rnd.uniform(W * 0.15, W * 0.85), rnd.uniform(H * 0.12, H * 0.88),
                                  rnd.uniform(130, 230), TERRAIN_MOUNTAIN, name=f'岛屿{i+1}'))
        for i in range(5):
            tm.add(TerrainFeature(rnd.uniform(W * 0.1, W * 0.9), rnd.uniform(H * 0.1, H * 0.9),
                                  rnd.uniform(60, 100), TERRAIN_HILL))
        return tm

    if layout == 'urban':
        # 城区：密集城区地形形成巷战式掩护，雷达受建筑遮蔽严重
        tm.desc = '密集城区，建筑群严重遮蔽雷达'
        for gx in range(4):
            for gy in range(3):
                tm.add(TerrainFeature(W * 0.30 + gx * W * 0.13 + rnd.uniform(-40, 40),
                                      H * 0.20 + gy * H * 0.30 + rnd.uniform(-40, 40),
                                      rnd.uniform(90, 140), TERRAIN_CITY, name='城区'))
        tm.add(TerrainFeature(W * 0.15, H * 0.5, 150, TERRAIN_HILL))
        tm.add(TerrainFeature(W * 0.85, H * 0.5, 150, TERRAIN_HILL))
        return tm

    if layout == 'coastal':
        # 海岸线：一侧为海（无地形），另一侧为陆地丘陵，适合“海陆交界”攻防
        tm.desc = '海陆交界，陆侧丘陵提供掩护'
        for i in range(8):
            tm.add(TerrainFeature(rnd.uniform(W * 0.02, W * 0.35), H * 0.06 + i * H * 0.12,
                                  rnd.uniform(100, 160), TERRAIN_WATER, name='近海'))
        for i in range(7):
            tm.add(TerrainFeature(rnd.uniform(W * 0.55, W * 0.96), rnd.uniform(H * 0.08, H * 0.92),
                                  rnd.uniform(110, 190), TERRAIN_MOUNTAIN if i % 3 == 0 else TERRAIN_HILL))
        return tm

    if layout == 'desert_ridge':
        # 荒漠高地：稀疏但高耸的台地，高处雷达站视野覆盖大
        tm.desc = '荒漠台地，高地视野覆盖广阔'
        for i in range(5):
            tm.add(TerrainFeature(rnd.uniform(W * 0.2, W * 0.8), rnd.uniform(H * 0.15, H * 0.85),
                                  rnd.uniform(170, 240), TERRAIN_MOUNTAIN, name='台地'))
        for i in range(6):
            tm.add(TerrainFeature(rnd.uniform(W * 0.05, W * 0.95), rnd.uniform(H * 0.05, H * 0.95),
                                  rnd.uniform(70, 120), TERRAIN_HILL))
        return tm

    return tm


# 现实向地图注册表：布局名 → 中文名 / 描述 / 建议地图变体数
REALISTIC_MAPS = {
    'range':            ('空旷靶场', '无地形，用于课程基础阶段', 1),
    'duel':             ('经典战场', '开阔交战区 + 稀疏丘陵掩护', 4),
    'mountain_corridor': ('山脊走廊', '两道山脊夹出中央通道与山口', 4),
    'river_valley':     ('河谷分割', '水域纵贯，两处缺口为通行要点', 3),
    'island_chain':     ('岛链海域', '岛体遮蔽雷达，适合低空突防', 3),
    'urban':            ('城区攻防', '建筑群严重遮蔽雷达与视线', 3),
    'coastal':          ('海陆交界', '一侧近海一侧丘陵，攻防纵深大', 3),
    'desert_ridge':     ('荒漠台地', '高耸台地，高地雷达视野广阔', 3),
}

FIRE_GUN, FIRE_MISSILE, FIRE_BOMB, FIRE_FLARE = 0, 1, 2, 3
HEAD_SECTORS = 8          # 头控制：8 个 45° 扇区（相对机头）
# 目标选择（火控锁定）：对导弹机=导弹锁定目标（雷达弹可改目标）；对机炮机=锁定目标（接近时飞控辅助对准）
LOCK_HOLD, LOCK_NEAREST_AIR, LOCK_NEAREST_GROUND, LOCK_NEXT, LOCK_CLEAR = 0, 1, 2, 3, 4
# 目标共享：把记忆/锁定/建筑目标共享给本队（写入世界共享板，友军可感知）
SHARE_NONE, SHARE_LOCK, SHARE_NEAR_AC, SHARE_BUILDING, SHARE_AIRBASE, SHARE_ALL = 0, 1, 2, 3, 4, 5
ACT_DIMS = [HEAD_SECTORS, 4, 3, 3, 5, 6]     # head, fire, throttle, turn, target_select, share_target
FIRE_HEAD_INDEX = 1                          # 火控头在 ACT_DIMS 中的下标（熵权重需单独调小）
THR_AFTERBURN, THR_CRUISE, THR_BRAKE = 0, 1, 2
TURN_LEFT, TURN_STRAIGHT, TURN_RIGHT = 0, 1, 2

# 单位摧毁价值（用于资源与奖励）
KILL_VALUE = {
    'aircraft': 30, 'tank': 40, 'gun_aa_vehicle': 30, 'missile_aa_vehicle': 30,
    'infantry': 15, 'airbase': 200, 'forward_airbase': 100, 'resource_base': 150,
    'missile_aa': 50, 'long_range_aa': 50, 'gun_aa': 50, 'radar': 20, 'rally': 30,
}


@dataclass
class Action:
    """多头离散动作。"""
    head: int = 0           # 0..7 目标头扇区（相对机头）
    fire: int = 0           # 0 机炮 1 导弹 2 炸弹 3 诱饵
    throttle: int = 1       # 0 加力 1 匀速 2 减速板
    turn: int = 1           # 0 最大左转 1 直飞 2 最大右转
    target_select: int = 0  # 火控目标选择：0保持 1锁定最近敌机 2锁定最近地面 3切换下一目标 4清除锁定
    share_target: int = 0   # 目标共享：0无 1锁定 2最近敌机记忆 3最近建筑 4本方机场 5全部


def head_sector_center(sector: int) -> float:
    """扇区中心对应的相对角度（度，0=机头正前，顺时针为正）。扇区 4 正前。"""
    return (sector * 45 - 180 + 22.5 + 360) % 360 - 180


def angle_to_sector(rel_angle_deg: float) -> int:
    return int(((rel_angle_deg + 180 - 22.5) % 360) // 45) % HEAD_SECTORS


def norm_deg(a: float) -> float:
    return (a + 540) % 360 - 180


def _point_seg_dist(px, py, x0, y0, x1, y1) -> float:
    """点 (px,py) 到线段 (x0,y0)-(x1,y1) 的最短距离。

    用于弹丸的**扫掠命中检测**：高速弹丸一帧位移可能远大于目标半径，
    用移动后的单点做圆判定会漏检（实测弹速 26 时 268 发 0 命中）。"""
    dx, dy = x1 - x0, y1 - y0
    seg2 = dx * dx + dy * dy
    if seg2 < 1e-9:
        return math.hypot(px - x0, py - y0)
    t = ((px - x0) * dx + (py - y0) * dy) / seg2
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return math.hypot(px - (x0 + dx * t), py - (y0 + dy * t))


def unit_value(unit) -> float:
    t = unit.type
    if isinstance(unit, Aircraft):
        return KILL_VALUE['aircraft']
    if isinstance(unit, Building):
        return BUILDING_SPECS[t].value
    return GROUND_SPECS[t].value


# ======================================================================
# 2. 核心基础设施：对象池 / 空间网格 / 世界
# ======================================================================
class EntityPool:
    """带 uid 的对象池：spawn 复用死槽位，update_all 只遍历存活实体。"""

    def __init__(self, kind, preallocate=0, on_spawn=None):
        self.kind = kind
        self._items: List = []
        self._next_uid = 1
        self.on_spawn = on_spawn      # 生成钩子（用于把新实体登记进空间网格等）
        for _ in range(preallocate):
            self._items.append(None)
        self._alive_cache: Optional[List] = None   # 存活缓存（生成/显式失效时重建）

    def invalidate(self):
        """显式失效存活缓存（世界 step 每帧调用一次）。"""
        self._alive_cache = None

    def spawn(self, *args, **kwargs):
        # 复用已死亡实体（同类型弹药池）
        for item in self._items:
            if item is not None and not item.alive and type(item) is self.kind:
                item.__init__(*args, **kwargs)
                item.uid = self._next_uid
                self._next_uid += 1
                item.alive = True
                self._alive_cache = None
                if self.on_spawn:
                    self.on_spawn(item)
                return item
        obj = self.kind(*args, **kwargs)
        obj.uid = self._next_uid
        self._next_uid += 1
        obj.alive = True
        self._items.append(obj)
        self._alive_cache = None
        if self.on_spawn:
            self.on_spawn(obj)
        return obj

    def alive_list(self) -> List:
        c = self._alive_cache
        if c is None:
            c = [it for it in self._items if it is not None and it.alive]
            self._alive_cache = c
        return c

    def count_alive(self) -> int:
        """存活数量（不构造列表，供高频计数使用）。"""
        return sum(1 for it in self._items if it is not None and it.alive)

    def all(self) -> List:
        return [it for it in self._items if it is not None]

    def update_all(self, world, dt=1):
        for it in self.alive_list():
            it.update(world, dt)

    def __len__(self):
        return self.count_alive()


class SpatialGrid:
    """空间哈希网格：邻域查询 O(1)~O(邻居数)，替代 O(n^2) 全量扫描。"""

    def __init__(self, cell=64):
        self.cell = cell
        self.buckets: Dict[Tuple[int, int], List] = {}

    def clear(self):
        self.buckets.clear()

    def insert(self, entity):
        cx, cy = int(entity.x // self.cell), int(entity.y // self.cell)
        self.buckets.setdefault((cx, cy), []).append(entity)

    def query_circle(self, x, y, r) -> List:
        out = []
        x0, x1 = int((x - r) // self.cell), int((x + r) // self.cell)
        y0, y1 = int((y - r) // self.cell), int((y + r) // self.cell)
        r2 = r * r
        for cx in range(x0, x1 + 1):
            for cy in range(y0, y1 + 1):
                for e in self.buckets.get((cx, cy), ()):
                    if not getattr(e, 'alive', False):
                        continue
                    dx, dy = e.x - x, e.y - y
                    if dx * dx + dy * dy <= r2:
                        out.append(e)
        return out

    def nearest(self, x, y, r, predicate=None):
        best, best_d = None, float('inf')
        for e in self.query_circle(x, y, r):
            if predicate and not predicate(e):
                continue
            d = math.hypot(e.x - x, e.y - y)
            if d < best_d:
                best, best_d = e, d
        return best


class World:
    """模拟世界：持有所有实体池与空间网格，每帧 step()。"""

    def __init__(self, plane_type='standard', red_ai_factory=None, blue_ai_factory=None,
                 red_slots=None, blue_slots=None, layout='duel',
                 red_offsets=None, blue_offsets=None,
                 red_spawn=(0.2, 0.5), blue_spawn=(0.8, 0.5),
                 map_variants=1, scale=1.0):
        """red_slots/blue_slots: [(plane_type, controller_factory|None), ...] 多机编队；
        不传则回退为 1v1（red_ai_factory/blue_ai_factory 或规则 AI）。
        layout: 'duel'=标准战场（基地/防空/雷达），'range'=空旷靶场（无建筑）。
        red_offsets/blue_offsets: 出生阵型偏移 [(dx,dy), ...]，长度需与槽位一致；None=默认横排。
        red_spawn/blue_spawn: 阵营中心（世界坐标比例）。
        map_variants: 同类型地图变体数（每回合随机一个布局）。
        scale: 交战规模（地面部队/目标数量倍率）。"""
        self.plane_type = plane_type
        self.red_slots = red_slots if red_slots is not None else [(plane_type, red_ai_factory)]
        self.blue_slots = blue_slots if blue_slots is not None else [(plane_type, blue_ai_factory)]
        self.layout = layout
        self.red_offsets = red_offsets
        self.blue_offsets = blue_offsets
        self.red_spawn = red_spawn
        self.blue_spawn = blue_spawn
        self.map_variants = max(1, map_variants)
        self.scale = max(0.25, scale)
        self._variant = 0
        self.terrain = TerrainMap()          # 地形（每回合按变体重建）
        self.aircraft = EntityPool(Aircraft)
        self.ground = EntityPool(GroundUnit)
        self.buildings = EntityPool(Building)
        self.bullets = EntityPool(Bullet)
        self.bombs = EntityPool(Bomb)
        self.missiles = EntityPool(Missile)
        self.spatial = SpatialGrid(256)
        self.step_count = 0
        self.resources = {'red': 500.0, 'blue': 500.0}
        # 每帧事件（供奖励计算）
        self.damage_credit = {}      # (team, uid) -> damage dealt this step
        self.kills = []              # 本帧被摧毁的单位
        self.flare_success = []      # 本帧诱饵成功欺骗的导弹（记录目标 team）
        self.gun_fired = []          # 本帧"有效开火"（机炮命中条件已成立）的队伍列表
        self._enemy_cache = {}       # 每帧敌方列表缓存 (unit.uid, r) -> list
        self._friend_cache = {}      # 每帧友军列表缓存 (unit.uid, r) -> list
        self.shared = {'red': [], 'blue': []}   # 目标共享板：{kind,x,y,t}，友军可感知
        self._bind_pools()
        self.reset()

    def _bind_pools(self):
        """为所有实体池绑定生成钩子：新实体立即登记进空间网格。

        这样就不必每帧全量 _rebuild_spatial（原先每步重建 6 个池的全部实体，
        实战中每帧新生成实体通常只有个位数）。"""
        for pool in (self.aircraft, self.ground, self.buildings,
                     self.bullets, self.bombs, self.missiles):
            pool.on_spawn = self.spatial.insert

    # ---------- 构造 ----------
    def reset(self):
        self.aircraft = EntityPool(Aircraft)
        self.ground = EntityPool(GroundUnit)
        self.buildings = EntityPool(Building)
        self.bullets = EntityPool(Bullet)
        self.bombs = EntityPool(Bomb)
        self.missiles = EntityPool(Missile)
        self._bind_pools()
        self.spatial.clear()
        self.step_count = 0
        self.resources = {'red': 500.0, 'blue': 500.0}
        self.damage_credit = {}
        self.kills = []
        self.flare_success = []
        self.gun_fired = []
        self._enemy_cache = {}
        self._friend_cache = {}
        self.shared = {'red': [], 'blue': []}
        # 同类型地图随机变体（训练多样化）
        self._variant = random.randrange(self.map_variants) if self.map_variants > 1 else 0
        self._init_map(self._variant, self.scale)
        self._spawn_team('red', self.red_slots, WORLD_W * self.red_spawn[0],
                         WORLD_H * self.red_spawn[1], self.red_offsets)
        self._spawn_team('blue', self.blue_slots, WORLD_W * self.blue_spawn[0],
                         WORLD_H * self.blue_spawn[1], self.blue_offsets)
        self._rebuild_spatial()
        for pool in (self.aircraft, self.ground, self.buildings,
                     self.bullets, self.bombs, self.missiles):
            pool.invalidate()
        return self

    def now_ms(self):
        """模拟时间基准（毫秒），由步数派生而非墙钟。

        原先用 pygame.time.get_ticks()：暂停、倍速演示、--norender 离线训练都会让
        “记忆老化”与“模拟步数”脱钩（4 倍速下记忆相对步数衰减快 4 倍）。
        改为步数派生后，训练与观战完全一致、可复现。"""
        return self.step_count * MS_PER_STEP


    def _spawn_team(self, team, slots, bx, by, offsets=None):
        n = len(slots)
        for i, (ptype, ctrl_factory) in enumerate(slots):
            if offsets and len(offsets) == n:
                dx, dy = offsets[i]
                if team == 'blue':
                    dx = -dx   # 蓝方镜像
                x, y = bx + dx, by + dy
            elif n == 1:
                x, y = bx, by + (200 if team == 'red' else -200)
            else:
                spread = (i - (n - 1) / 2) * 140
                x, y = bx + spread * (1 if team == 'red' else -1), by + (i - (n - 1) / 2) * 60
            angle = 0 if team == 'red' else 180
            ctrl = ctrl_factory() if ctrl_factory else None
            self.aircraft.spawn(max(50, min(WORLD_W - 50, x)), max(50, min(WORLD_H - 50, y)),
                                angle, ptype, team, ctrl)

    def _init_map(self, variant=0, scale=1.0):
        # 地形对所有布局生效（'range' 靶场为无地形空白，保持课程基础阶段可学性）
        self.terrain = build_terrain_map(self.layout, variant)
        if self.layout == 'range':
            return  # 空旷靶场：无建筑
        rnd = random.Random(1000 + variant * 7919)   # 变体固定种子 → 同类型地图随机布局
        def jitter():
            return rnd.uniform(-180, 180)
        def b(x, y, t, team):
            self.buildings.spawn(max(60, min(WORLD_W - 60, x + jitter())),
                                 max(60, min(WORLD_H - 60, y + jitter())), t, team)
        b(300, WORLD_H // 2, 'airbase', 'red')
        b(WORLD_W - 300, WORLD_H // 2, 'airbase', 'blue')
        b(800, WORLD_H // 2 - 200, 'gun_aa', 'red')
        b(WORLD_W - 800, WORLD_H // 2 + 200, 'gun_aa', 'blue')
        b(1000, WORLD_H // 2 + 250, 'radar', 'red')
        b(WORLD_W - 1000, WORLD_H // 2 - 250, 'radar', 'blue')
        # 少量集结点（会刷坦克/步兵）
        b(1500, WORLD_H // 2 - 300, 'rally', 'red')
        b(WORLD_W - 1500, WORLD_H // 2 + 300, 'rally', 'blue')
        # 初始地面部队（数量随 scale 增长）
        n_tank = max(2, int(4 * scale))
        n_aa = max(1, int(2 * scale))
        for i in range(n_tank):
            self.ground.spawn(600 + i * 60, WORLD_H // 2 + 120, 'tank', 'red')
            self.ground.spawn(WORLD_W - 600 - i * 60, WORLD_H // 2 - 120, 'tank', 'blue')
        for i in range(n_aa):
            self.ground.spawn(700 + i * 50, WORLD_H // 2 + 180, 'gun_aa_vehicle', 'red')
            self.ground.spawn(WORLD_W - 700 - i * 50, WORLD_H // 2 - 180, 'gun_aa_vehicle', 'blue')

    def _rebuild_spatial(self):
        self.spatial.clear()
        for pool in (self.aircraft, self.ground, self.buildings, self.bullets, self.bombs, self.missiles):
            for e in pool._items:
                if e is not None and e.alive:
                    self.spatial.insert(e)

    # ---------- 查询 ----------
    def query_circle(self, x, y, r):
        return self.spatial.query_circle(x, y, r)

    def nearest(self, x, y, r, predicate=None):
        return self.spatial.nearest(x, y, r, predicate)

    def _enemy_pred(self, unit):
        return lambda e: e.alive and getattr(e, 'team', None) != unit.team

    def nearest_enemy_aircraft(self, unit):
        return self.nearest(unit.x, unit.y, DETECT_RANGE, lambda e: isinstance(e, Aircraft) and e.team != unit.team)

    def nearest_enemy_ground(self, unit):
        return self.nearest(unit.x, unit.y, DETECT_RANGE,
                            lambda e: (isinstance(e, GroundUnit) or isinstance(e, Building)) and e.team != unit.team)

    def nearest_enemy_missile(self, unit):
        return self.nearest(unit.x, unit.y, 2000,
                            lambda e: isinstance(e, Missile) and e.team != unit.team)

    def enemies_of(self, unit, r=DETECT_RANGE):
        """敌方单位列表（每帧缓存，避免同帧内被观测/索敌/规则AI重复大半径查询）。"""
        key = (unit.uid, r)
        cached = self._enemy_cache.get(key)
        if cached is not None:
            return cached
        lst = [e for e in self.spatial.query_circle(unit.x, unit.y, r)
               if e.alive and getattr(e, 'team', None) != unit.team]
        self._enemy_cache[key] = lst
        return lst

    def friends_of(self, unit, r=DETECT_RANGE):
        """友方飞机列表（每帧缓存，供编队协作观测使用）。"""
        key = (unit.uid, r)
        cached = self._friend_cache.get(key)
        if cached is not None:
            return cached
        lst = [e for e in self.spatial.query_circle(unit.x, unit.y, r)
               if e.alive and isinstance(e, Aircraft) and e is not unit and e.team == unit.team]
        self._friend_cache[key] = lst
        return lst

    def friend_aircraft(self, unit):
        return self.nearest(unit.x, unit.y, DETECT_RANGE,
                            lambda e: isinstance(e, Aircraft) and e.team == unit.team and e is not unit)

    # ---------- 事件 ----------
    def record_damage(self, attacker_team, attacker_uid, dmg):
        if attacker_team:
            key = (attacker_team, attacker_uid)
            self.damage_credit[key] = self.damage_credit.get(key, 0.0) + dmg

    def on_unit_destroyed(self, unit, attacker_team):
        self.kills.append(unit)
        # 资源：击毁方获得对方单位的价值
        if attacker_team and attacker_team != unit.team:
            self.resources[attacker_team] += unit_value(unit)

    # ---------- 主循环 ----------
    def step(self, dt=1):
        self.damage_credit = {}
        self.kills = []
        self.flare_success = []
        self.gun_fired = []
        self._enemy_cache = {}
        self._friend_cache = {}
        now = self.now_ms()
        for team in ('red', 'blue'):
            self.shared[team] = [s for s in self.shared[team] if now - s['t'] < 10000]
        self.buildings.update_all(self, dt)
        self.ground.update_all(self, dt)
        self.aircraft.update_all(self, dt)
        self.bullets.update_all(self, dt)
        self.bombs.update_all(self, dt)
        self.missiles.update_all(self, dt)
        self.step_count += dt
        # 新生成实体由 on_spawn 钩子即时登记进空间网格，无需每帧全量重建；
        # 只需清空各池的存活缓存，让下一帧重新统计（含本帧内死亡的单位）。
        for pool in (self.aircraft, self.ground, self.buildings,
                     self.bullets, self.bombs, self.missiles):
            pool.invalidate()
        return self

    def base_destroyed(self):
        """返回被摧毁基地的 team（该队不存在基地或基地未失守时返回 None）。"""
        red_total = red_alive = blue_total = blue_alive = 0
        for b in self.buildings.all():
            if b.type != 'airbase':
                continue
            if b.team == 'red':
                red_total += 1
                if b.alive:
                    red_alive += 1
            elif b.team == 'blue':
                blue_total += 1
                if b.alive:
                    blue_alive += 1
        if red_total and not red_alive:
            return 'red'
        if blue_total and not blue_alive:
            return 'blue'
        return None

    def aircraft_of(self, team):
        for a in self.aircraft.alive_list():
            if a.team == team:
                return a
        return None

    def aircraft_list_of(self, team):
        return [a for a in self.aircraft.alive_list() if a.team == team]

    # ---------- 目标共享 ----------
    def add_shared(self, team, kind, x, y):
        """把目标信息写入本队共享板（限 8 条，按时间替换最旧）。"""
        now = self.now_ms()
        lst = self.shared[team]
        lst.append({'kind': kind, 'x': float(x), 'y': float(y), 't': now})
        if len(lst) > 8:
            lst.sort(key=lambda s: s['t'])
            lst.pop(0)

    def nearest_friendly_airbase(self, unit):
        """本方最近机场（基地/前线机场）。"""
        best, best_d = None, 1e9
        for b in self.buildings.alive_list():
            if b.team == unit.team and b.type in ('airbase', 'forward_airbase'):
                d = math.hypot(b.x - unit.x, b.y - unit.y)
                if d < best_d:
                    best, best_d = b, d
        return best


# ======================================================================
# 3. 组件：飞控 / 武器 / 感知 / 控制器
# ======================================================================
class FlightModel:
    """空气动力学飞控：推力 - 阻力(v^2) 演化速度；低速更灵活、高速更笨重。"""

    def __init__(self, spec: FlightSpec):
        self.spec = spec
        self.speed = spec.cruise_speed
        self.throttle_state = THR_CRUISE
        self.ab_ticks = 0       # 本次加力已用帧数
        self.ab_cd = 0          # 加力剩余冷却帧数
        self.br_ticks = 0       # 本次减速板已用帧数
        self.br_cd = 0          # 减速板剩余冷却帧数

    def thrust_for(self, throttle):
        if throttle == THR_AFTERBURN:
            return self.spec.afterburner_thrust
        if throttle == THR_BRAKE:
            return -self.spec.airbrake_decel
        return self.spec.cruise_thrust

    def max_turn_rate(self):
        v = self.speed
        f = 2.0 / (1.0 + (v / self.spec.cruise_speed) ** 2)
        return self.spec.turn_rate * max(0.35, min(1.5, f))

    def update(self, throttle, turn_dir, dt=1):
        """加力/减速板带冷却：最长持续 ab_max_ticks 后强制巡航并进入冷却；
        冷却期间请求加力/减速被忽略（以巡航代替）。"""
        # 冷却递减
        if self.ab_cd > 0:
            self.ab_cd -= dt
        if self.br_cd > 0:
            self.br_cd -= dt
        # 加力门控
        if throttle == THR_AFTERBURN:
            if self.ab_cd > 0:
                throttle = THR_CRUISE          # 冷却中：加力无效
            else:
                self.ab_ticks += dt
                if self.ab_ticks >= self.spec.ab_max_ticks:
                    self.ab_ticks = 0
                    self.ab_cd = self.spec.ab_cd_ticks
                    throttle = THR_CRUISE      # 用满上限：强制巡航 + 冷却
        else:
            if self.ab_ticks > 0:              # 提前退出加力 → 少量冷却惩罚
                self.ab_cd = max(self.ab_cd, self.spec.ab_cd_ticks // 2)
                self.ab_ticks = 0
        # 减速板门控
        if throttle == THR_BRAKE:
            if self.br_cd > 0:
                throttle = THR_CRUISE
            else:
                self.br_ticks += dt
                if self.br_ticks >= self.spec.brake_max_ticks:
                    self.br_ticks = 0
                    self.br_cd = self.spec.brake_cd_ticks
                    throttle = THR_CRUISE
        else:
            if self.br_ticks > 0:
                self.br_cd = max(self.br_cd, self.spec.brake_cd_ticks // 2)
                self.br_ticks = 0
        self.throttle_state = throttle
        accel = self.thrust_for(throttle) - self.spec.drag_k * self.speed * self.speed
        self.speed = max(self.spec.min_speed, min(self.spec.max_speed, self.speed + accel * dt))
        return turn_dir * self.max_turn_rate()   # 返回角速度（度/帧）


class WeaponSystem:
    """武器系统：机炮/导弹/炸弹/诱饵弹，请求式执行（弹药/冷却/目标可行性）。"""

    def __init__(self, spec: AircraftSpec, team):
        self.team = team
        self.gun = spec.gun
        self.bomb = spec.bomb
        self.missile = spec.missile
        self.ammo = spec.gun.ammo
        self.bombs = spec.bomb.count
        self.missiles = spec.missile.count
        self.flares = spec.flares
        self.max_ammo = spec.gun.ammo
        self.max_bombs = spec.bomb.count
        self.max_missiles = spec.missile.count
        self.max_flares = spec.flares
        self.gun_cd = 0
        self.missile_cd = 0

    def update(self, dt=1):
        if self.gun_cd > 0: self.gun_cd -= dt
        if self.missile_cd > 0: self.missile_cd -= dt

    def rearm(self):
        self.ammo = self.max_ammo
        self.bombs = self.max_bombs
        self.missiles = self.max_missiles

    def try_fire(self, world, unit, fire_action, sensor):
        """请求式开火：按可行条件执行，返回 True 表示发射成功。"""
        if fire_action == FIRE_GUN:
            return self._fire_gun(world, unit, sensor)
        if fire_action == FIRE_MISSILE:
            return self._fire_missile(world, unit, sensor)
        if fire_action == FIRE_BOMB:
            return self._drop_bomb(world, unit, sensor)
        if fire_action == FIRE_FLARE:
            return self._drop_flare(unit)
        return False

    def _fire_gun(self, world, unit, sensor):
        if self.gun.damage <= 0 or self.ammo <= 0 or self.gun_cd > 0:
            return False
        # 目标选择：优先“机身轴线锥内”的目标（锁定目标只有在也位于锥内时才优先）。
        # 原实现只要存在 lock_target 就无条件改用锁定目标，而锁定目标常常不在机炮锥内，
        # 导致大量子弹朝空处射出——实测 S2/S3 成功开火 900 次、造成伤害为 0。
        target = None
        if (unit.lock_target is not None and unit.lock_target.alive
                and isinstance(unit.lock_target, Aircraft)):
            lt = unit.lock_target
            d = math.hypot(lt.x - unit.x, lt.y - unit.y)
            if d <= self.gun.range:
                rel = norm_deg(math.degrees(math.atan2(lt.y - unit.y, lt.x - unit.x)) - unit.angle)
                if abs(rel) <= self.gun.angle:
                    target = lt
        if target is None:
            target = sensor.gun_target(world, unit)
        if target is None:
            return False
        # 机炮固定于机身轴线：子弹沿机头（机身方向 angle）射出，与绘制一致
        rad = math.radians(unit.angle)
        bx, by = unit.x + 12 * math.cos(rad), unit.y + 12 * math.sin(rad)
        world.bullets.spawn(bx, by, unit.angle, self.gun.bullet_speed,
                            self.gun.damage, unit.team, owner_uid=unit.uid,
                            is_aa=False, target=target, max_range=self.gun.range,
                            hit_radius=self.gun.hit_radius)
        # 记录"有效开火"事件（命中条件已成立）：供奖励函数给火控头一条稠密信号。
        # 动机：火控头唯一的梯度来源是稀疏的伤害奖励，被其余动作头共享的 advantage
        # 噪声淹没，实测其策略分布长期停在近均匀（与观测无关）。这里补上即时信号。
        world.gun_fired.append(unit.team)
        self.ammo -= 1
        self.gun_cd = self.gun.cooldown
        return True

    def _fire_missile(self, world, unit, sensor):
        if self.missiles <= 0 or self.missile_cd > 0:
            return False
        # 优先使用锁定目标发射；无锁定则自动选择射程内敌机
        target = unit.lock_target if (unit.lock_target is not None and
                                      unit.lock_target.alive and isinstance(unit.lock_target, Aircraft)) else None
        if target is None:
            target = sensor.missile_target(world, unit)
        if target is None:
            return False
        world.missiles.spawn(unit.x, unit.y, unit.angle + unit.head_angle, target,
                             unit.team, self.missile.mtype, owner_uid=unit.uid)
        self.missiles -= 1
        self.missile_cd = 40
        return True

    def _drop_bomb(self, world, unit, sensor):
        if self.bombs <= 0:
            return False
        target = sensor.bomb_target(world, unit)
        if target is None:
            return False
        if self.bomb.guided:
            world.bombs.spawn(unit.x, unit.y, unit.angle, unit.flight.speed,
                              self.bomb.damage, self.bomb.radius, unit.team,
                              owner_uid=unit.uid, target=target, turn=3, max_range=3000, guided=True)
        else:
            b = world.bombs.spawn(unit.x, unit.y, unit.angle, unit.flight.speed,
                                  self.bomb.damage, self.bomb.radius, unit.team, owner_uid=unit.uid)
            b.set_timer(30)
        self.bombs -= 1
        return True

    def _drop_flare(self, unit):
        if self.flares <= 0:
            return False
        self.flares -= 1
        unit.flare_timer = 60
        return True


class MemoryEntry:
    """统一目标记忆：敌机/地面/导弹共用一条池，记录位置+相对速度+目击时间。"""
    __slots__ = ('entity_id', 'kind', 'team', 'x', 'y', 'vx', 'vy',
                 'last_seen', 'status', 'certainty')

    def __init__(self, entity, pos, vel, time, kind=None, status='tracked'):
        self.entity_id = id(entity)
        self.kind = kind or ('aircraft' if isinstance(entity, Aircraft) else
                             ('ground' if isinstance(entity, GroundUnit) else
                              ('building' if isinstance(entity, Building) else 'missile')))
        self.team = entity.team
        self.x, self.y = pos
        self.vx, self.vy = vel
        self.last_seen = time
        self.status = status
        self.certainty = 1.0


class Sensor:
    """感知系统：可见度概率探测 + 统一目标记忆池(8条) + 建筑目标记忆(4条) + 共享目标读取。

    可见度由目标大小、类型、距离、观察者与机头夹角决定：机头朝向目标、目标越大越近越易发现。
    """

    def __init__(self, spec: AircraftSpec):
        self.visibility = spec.visibility
        self.visibility_buff = 0
        self.full_aware_timer = 0
        self.last_hit_time = 0
        self.memory: List[MemoryEntry] = []       # 统一目标记忆池（敌机/地面/导弹），容量 8
        self.building_mem: List[tuple] = []       # 敌方建筑记忆 [dist, x, y, type_code, last_seen]，容量 4
        self.flare_timer = 0
        self._approach_best = {}                  # id(目标) -> 历史最近距离（接近势能塑形用）
        self._seen_ids = set()                    # 已首次发现过的目标 id（索敌一次性奖励去重）
        self._lock_ids = set()                    # 已首次锁定过的目标 id（锁敌一次性奖励去重）

    # ---------- 可见度模型 ----------
    def visibility_prob(self, world, unit, other):
        """观察概率 = f(目标大小、类型、距离、机头夹角) × 地形遮挡。

        地形（山体/丘陵/城区）会阻断视线：被地形隔断时完全看不到，从而产生
        雷达盲区、山口走廊等战术要素；地面单位藏身地形内还会获得掩护降概率。"""
        dx, dy = other.x - unit.x, other.y - unit.y
        dist = math.hypot(dx, dy)
        if dist > DETECT_RANGE:
            return 0.0
        if self.visibility_buff > 0:
            return 1.0
        terrain = getattr(world, 'terrain', None)
        los_open = True
        cover = 0.0
        if terrain is not None and terrain.features:
            if terrain.blocks_sight(unit.x, unit.y, other.x, other.y):
                los_open = False
            if not isinstance(other, Aircraft):
                cover = terrain.cover_bonus(other.x, other.y)
        size = getattr(other, 'size', 10)
        size_f = min(1.0, size / 30.0)
        kind_f = 0.9 if isinstance(other, Aircraft) else (1.1 if isinstance(other, GroundUnit) else 1.3)
        dist_f = 1.0 - dist / DETECT_RANGE
        angle_to = math.degrees(math.atan2(dy, dx))
        rel = norm_deg(angle_to - (unit.angle + unit.head_angle))   # 机头朝向目标可见度高
        angle_f = max(0.0, math.cos(math.radians(rel)))
        prob = 0.10 + 0.30 * size_f * kind_f + 0.30 * dist_f + 0.30 * angle_f
        # 地形遮挡：视线被山体阻断 → 概率大幅下降（低空突防有效）
        if not los_open:
            prob = min(prob, 0.05)
        if cover > 0:
            prob *= (1.0 - cover)
        # 视野加成：地形较低时（默认）不影响；此处留给后续高度/预警机扩展
        return max(0.0, min(1.0, prob))

    def can_detect(self, world, unit, other):
        if not other.alive:
            return False
        return random.random() < self.visibility_prob(world, unit, other)

    # ---------- 记忆池管理 ----------
    def _rel_vel(self, unit, other):
        """目标相对本机的速度矢量。"""
        if hasattr(other, 'flight'):
            tvx = other.flight.speed * math.cos(math.radians(other.angle))
            tvy = other.flight.speed * math.sin(math.radians(other.angle))
        else:
            tvx = tvy = 0.0
        uvx = unit.flight.speed * math.cos(math.radians(unit.angle))
        uvy = unit.flight.speed * math.sin(math.radians(unit.angle))
        return tvx - uvx, tvy - uvy

    def _upsert_memory(self, other, now, unit, kind=None):
        """探测到目标：更新已有记忆或新增（容量 8，按最旧淘汰）。"""
        for mem in self.memory:
            if mem.entity_id == id(other):
                mem.x, mem.y = other.x, other.y
                mem.vx, mem.vy = self._rel_vel(unit, other)
                mem.last_seen = now
                mem.certainty = 1.0
                mem.status = 'tracked'
                return
        if len(self.memory) >= 8:
            self.memory.sort(key=lambda m: m.last_seen)   # 最旧优先淘汰
            self.memory.pop(0)
        vx, vy = self._rel_vel(unit, other)
        self.memory.append(MemoryEntry(other, (other.x, other.y), (vx, vy), now, kind=kind))

    def _upsert_shared(self, s, now):
        """把本队共享目标并入记忆池（共享信息无实体 id，用坐标近似匹配）。"""
        kind = 'aircraft' if s['kind'] in ('lock', 'air') else 'building'
        # 就近更新同类型记忆
        for mem in self.memory:
            if mem.kind == kind and abs(mem.x - s['x']) < 300 and abs(mem.y - s['y']) < 300:
                mem.x, mem.y = s['x'], s['y']
                mem.last_seen = now
                mem.certainty = 0.85
                mem.status = 'shared'
                return
        if len(self.memory) >= 8:
            self.memory.sort(key=lambda m: m.last_seen)
            self.memory.pop(0)
        pseudo = _types.SimpleNamespace(team='', alive=True)
        ent = MemoryEntry(pseudo, (s['x'], s['y']), (0, 0), now, kind=kind, status='shared')
        self.memory.append(ent)

    def _update_building_mem(self, building, now, unit):
        """敌方建筑记忆：见过即记录，保留最近 4 个。"""
        d = math.hypot(building.x - unit.x, building.y - unit.y)
        tcode = {'airbase': 0.2, 'forward_airbase': 0.4, 'resource_base': 0.6,
                 'missile_aa': 0.8, 'long_range_aa': 0.8, 'gun_aa': 0.8,
                 'radar': 1.0, 'rally': 0.7}.get(building.type, 0.5)
        self.building_mem = [b for b in self.building_mem if b[1] != building.x or b[2] != building.y]
        self.building_mem.append((d, building.x, building.y, tcode, now))
        self.building_mem.sort(key=lambda b: b[0])
        self.building_mem = self.building_mem[:4]

    def update(self, world, unit):
        now = world.now_ms()
        # 位置在敌人名单确定后再取，避免 spawn 期间顺序问题
        if self.visibility_buff > 0:
            self.visibility_buff -= 1
        # 统一探测：敌机/地面/建筑/来袭导弹，按可见度概率
        # 导弹已包含在 enemies_of 的邻域查询结果中（原实现额外全量遍历全世界导弹，
        # 是 O(机数 × 弹数) 的热点），这里直接在邻域结果内分派。
        for other in world.enemies_of(unit):
            if isinstance(other, Projectile):
                if isinstance(other, Missile) and self.can_detect(world, unit, other):
                    self._upsert_memory(other, now, unit, kind='missile')
                continue
            if isinstance(other, Building):
                if self.can_detect(world, unit, other):
                    self._update_building_mem(other, now, unit)
                continue
            if self.can_detect(world, unit, other):
                self._upsert_memory(other, now, unit)
        # 本队共享目标并入记忆池
        for s in world.shared.get(unit.team, []):
            if now - s['t'] < 10000:
                self._upsert_shared(s, now)
        # 遗忘与老化
        keep = []
        for mem in self.memory:
            if mem.status == 'tracked' and now - mem.last_seen > MEMORY_FORGET_MS:
                mem.status = 'estimated'
                mem.certainty = 0.8
            if mem.status in ('estimated', 'shared'):
                mem.certainty *= 0.99
                if mem.certainty < 0.2:
                    mem.status = 'forgotten'
            if mem.status != 'forgotten':
                keep.append(mem)
        self.memory = keep[-8:]
        # 建筑记忆老化（>30s 移除）
        self.building_mem = [b for b in self.building_mem if now - b[4] < 30000]

    # ---- 目标选择（用实时探测 + 记忆池） ----
    def nearest_memory_air(self, unit=None):
        """记忆池中最近的敌机记忆（相对本机）。"""
        best, best_d = None, 1e9
        for mem in self.memory:
            if mem.kind == 'aircraft':
                if unit is not None:
                    d = math.hypot(mem.x - unit.x, mem.y - unit.y)
                else:
                    d = math.hypot(mem.x, mem.y)
                if d < best_d:
                    best, best_d = mem, d
        return best

    def nearest_building_mem(self):
        return self.building_mem[0] if self.building_mem else None

    def _alive_enemy_aircraft(self, world, unit):
        return [e for e in world.enemies_of(unit) if isinstance(e, Aircraft)]

    def _alive_enemy_ground(self, world, unit):
        return [e for e in world.enemies_of(unit)
                if isinstance(e, GroundUnit) or isinstance(e, Building)]

    def gun_target(self, world, unit):
        """机炮目标：机身轴线锥角内、射程内最近的目标（机炮固定机身轴线）。

        ⚠ 重要修正（v6）：原实现只遍历 `Aircraft`，**完全无视地面单位**。
        实测：坦克摆在红机正前方 120 处、夹角 0°、已锁定，`gun_target` 仍返回 None，
        锥内帧恒为 0 —— 也就是说机炮从来打不到地面目标，所有"定点/动点/多目标"
        阶段的目标（坦克、防空车）在机制上就不可能被机炮命中，导致这些阶段
        必然零伤害。现在改为：优先敌机，其次地面单位与建筑。"""
        best, best_score = None, float('inf')
        for e in self._alive_enemy_aircraft(world, unit):
            d = math.hypot(e.x - unit.x, e.y - unit.y)
            if d > unit.weapons.gun.range:
                continue
            ang = math.degrees(math.atan2(e.y - unit.y, e.x - unit.x))
            rel = norm_deg(ang - unit.angle)
            if abs(rel) <= unit.weapons.gun.angle and d < best_score:
                best, best_score = e, d
        if best is not None:
            return best
        # 无空中目标时考虑对地扫射
        for e in self._alive_enemy_ground(world, unit):
            d = math.hypot(e.x - unit.x, e.y - unit.y)
            if d > unit.weapons.gun.range:
                continue
            ang = math.degrees(math.atan2(e.y - unit.y, e.x - unit.x))
            rel = norm_deg(ang - unit.angle)
            if abs(rel) <= unit.weapons.gun.angle and d < best_score:
                best, best_score = e, d
        return best

    def missile_target(self, world, unit):
        """导弹：当前探测范围内最近敌机（必须是真实实体，避免指向记忆条目）。"""
        mtype = unit.weapons.missile.mtype
        rng = MISSILE_TYPES[mtype]['range']
        best, best_d = None, float('inf')
        for e in self._alive_enemy_aircraft(world, unit):
            d = math.hypot(e.x - unit.x, e.y - unit.y)
            if d <= rng and d < best_d:
                best, best_d = e, d
        return best

    def bomb_target(self, world, unit):
        """炸弹：下方/前方航迹上的敌方地面目标。"""
        best, best_d = None, float('inf')
        for e in self._alive_enemy_ground(world, unit):
            d = math.hypot(e.x - unit.x, e.y - unit.y)
            if d > 260:
                continue
            # 目标应大致在飞行方向前方
            ang = math.degrees(math.atan2(e.y - unit.y, e.x - unit.x))
            rel = norm_deg(ang - unit.angle)
            if abs(rel) < 45 and d < best_d:
                best, best_d = e, d
        return best

    # ---- 观测 ----
    def build_observation(self, world, unit) -> np.ndarray:
        obs = []
        w = unit.weapons
        obs += [unit.hp / unit.max_hp,
                w.ammo / max(1, w.max_ammo),
                w.bombs / max(1, w.max_bombs),
                w.missiles / max(1, w.max_missiles),
                w.flares / max(1, w.max_flares),
                unit.flight.speed / unit.spec.flight.cruise_speed,
                math.sin(math.radians(unit.head_angle)),
                math.cos(math.radians(unit.head_angle)),
                math.sin(math.radians(unit.angle)),
                math.cos(math.radians(unit.angle))]
        obs += [1.0 if unit.flight.throttle_state == i else 0.0 for i in range(3)]
        # 最近来袭导弹（紧急威胁通道）
        tm = world.nearest_enemy_missile(unit)
        if tm:
            dx, dy = tm.x - unit.x, tm.y - unit.y
            d = math.hypot(dx, dy)
            closing = 1.0 if d > 40 else 0.0
            obs += [dx / 2000, dy / 2000, d / 2000, closing]
        else:
            obs += [0, 0, 1.0, 0]
        # 状态
        obs += [self.visibility_buff / 500.0,
                1.0 if self.flare_timer > 0 else 0.0,
                1.0 if unit.returning_to_base else 0.0]
        # ---------------- 战术态势（TACTICAL_DIM = 16，编队协作 + 多目标分配）----------------
        # 此前观测里完全没有友军，10v10 时各机互不知情、无法协作；这里补上最近 2 架友机
        # 的相对位置与血量，并补上第二近敌机，使模型能做目标分配与僚机配合。
        enemies = []
        for e in world.enemies_of(unit):
            if isinstance(e, Aircraft) and e.alive:
                d = math.hypot(e.x - unit.x, e.y - unit.y)
                enemies.append((d, e))
        enemies.sort(key=lambda t: t[0])
        for i in range(2):
            if i < len(enemies):
                d, e = enemies[i]
                obs += [d / 2000, norm_deg(math.degrees(math.atan2(e.y - unit.y, e.x - unit.x))
                                           - unit.angle) / 180.0,
                        e.hp / max(1.0, e.max_hp),
                        math.cos(math.radians(norm_deg(e.angle - unit.angle)))]
            else:
                obs += [1.0, 0.0, 0.0, 0.0]
        friends = []
        for f in world.friends_of(unit):
            friends.append((math.hypot(f.x - unit.x, f.y - unit.y), f))
        friends.sort(key=lambda t: t[0])
        for i in range(2):
            if i < len(friends):
                d, f = friends[i]
                obs += [d / 2000, norm_deg(math.degrees(math.atan2(f.y - unit.y, f.x - unit.x))
                                           - unit.angle) / 180.0,
                        f.hp / max(1.0, f.max_hp), 0.0]
            else:
                obs += [1.0, 0.0, 0.0, 0.0]
        now = world.now_ms()
        # 统一目标记忆池（8 条 × 7 维）：类型 / 相对位置 dx,dy,dist / 相对速度 vx,vy / 目击年龄
        for mem in self.memory[:8]:
            kind_code = {'aircraft': 0.2, 'ground': 0.5, 'building': 0.35, 'missile': 0.8}.get(mem.kind, 0.35)
            dx, dy = mem.x - unit.x, mem.y - unit.y
            age = min(1.0, (now - mem.last_seen) / 10000.0)
            obs += [kind_code, dx / 2000, dy / 2000, math.hypot(dx, dy) / 2000,
                    mem.vx / 6.0, mem.vy / 6.0, age]
        while len(obs) < TACTICAL_START + 56:
            obs.append(0.0)
        # 敌方建筑目标记忆（4 条 × 4 维：dx, dy, dist, 类型码）
        for (d0, bx, by, tcode, bt) in self.building_mem[:4]:
            dx, dy = bx - unit.x, by - unit.y
            obs += [dx / 2000, dy / 2000, math.hypot(dx, dy) / 2000, tcode]
        while len(obs) < TACTICAL_START + 56 + 16:
            obs.append(0.0)
        # 本方最近机场（3 维：dx, dy, dist）
        base = world.nearest_friendly_airbase(unit)
        if base is not None:
            dx, dy = base.x - unit.x, base.y - unit.y
            obs += [dx / 2000, dy / 2000, math.hypot(dx, dy) / 2000]
        else:
            obs += [0, 0, 1.0]
        # 火控锁定状态（3 维）：是否锁定、锁定类型（敌机/地面）、锁定距离（教模型学会锁敌）
        lock = unit.lock_target
        if lock is not None and lock.alive:
            has_lock = 1.0
            lock_type = 1.0 if isinstance(lock, Aircraft) else 2.0
            lock_dist = math.hypot(lock.x - unit.x, lock.y - unit.y) / 2000
        else:
            has_lock = lock_type = lock_dist = 0.0
        obs += [has_lock, lock_type, lock_dist]
        # 激进度（1 维）：0=保命型 0.5=均衡 1=激进型（模型据此调整战术）
        obs += [unit.aggressiveness]
        while len(obs) < OBS_DIM:
            obs.append(0.0)
        return np.array(obs[:OBS_DIM], dtype=np.float32)


class Controller:
    """控制器接口：每帧由单位调用，产出动作。"""
    def decide(self, world, unit) -> Action:
        raise NotImplementedError


class CommandedController(Controller):
    """外部命令控制器：动作由环境（训练循环）注入。"""
    def __init__(self):
        self.next_action = Action()

    def decide(self, world, unit):
        return self.next_action


class ManualController(Controller):
    """键盘手动控制（常规模式红方）。"""
    def decide(self, world, unit):
        keys = pygame.key.get_pressed()
        act = Action()
        if keys[K_w]: act.throttle = THR_AFTERBURN
        elif keys[K_s]: act.throttle = THR_BRAKE
        else: act.throttle = THR_CRUISE
        if keys[K_a]: act.turn = TURN_LEFT
        elif keys[K_d]: act.turn = TURN_RIGHT
        else: act.turn = TURN_STRAIGHT
        if keys[K_q]: act.head = (act.head - 1) % HEAD_SECTORS
        elif keys[K_e]: act.head = (act.head + 1) % HEAD_SECTORS
        if keys[K_SPACE]: act.fire = FIRE_GUN
        elif keys[K_m]: act.fire = FIRE_MISSILE
        elif keys[K_b]: act.fire = FIRE_BOMB
        elif keys[K_f]: act.fire = FIRE_FLARE
        return act


class StationaryAI(Controller):
    """静止靶机：不做任何机动/开火（用于定点目标阶段）。"""
    def decide(self, world, unit):
        return Action(head=0, fire=FIRE_GUN, throttle=THR_CRUISE, turn=TURN_STRAIGHT)


class DummyAI(Controller):
    """直线匀速靶机：不转向、不攻击（用于动点/多目标阶段）。"""
    def decide(self, world, unit):
        return Action(head=0, fire=FIRE_GUN, throttle=THR_CRUISE, turn=TURN_STRAIGHT)


# ======================================================================
# 4. 单位：飞机 / 建筑 / 地面单位 / 弹药
# ======================================================================
class Unit:
    kind = 'unit'

    def __init__(self, x, y, team):
        self.x, self.y = float(x), float(y)
        self.team = team
        self.alive = True
        self.uid = 0

    def take_damage(self, world, dmg, attacker_team=None, attacker_uid=None):
        if not self.alive:
            return
        self.hp -= dmg
        world.record_damage(attacker_team, attacker_uid, dmg)
        if self.hp <= 0:
            self.alive = False
            world.on_unit_destroyed(self, attacker_team)

    def update(self, world, dt=1):
        raise NotImplementedError

    def draw(self, screen, camera):
        raise NotImplementedError


class Aircraft(Unit):
    kind = 'aircraft'

    def __init__(self, x, y, angle, type_key, team, controller=None):
        super().__init__(x, y, team)
        self.type = type_key
        self.spec = AIRCRAFT_SPECS[type_key]
        self.angle = angle
        self.max_hp = self.spec.max_hp
        self.hp = self.spec.max_hp
        self.size = self.spec.size
        self.shape = self.spec.shape
        self.flight = FlightModel(self.spec.flight)
        self.weapons = WeaponSystem(self.spec, team)
        self.sensor = Sensor(self.spec)
        self.controller = controller if controller else RuleAI(team)
        self.controller.unit = self
        self.head_angle = 0.0
        self.flare_timer = 0
        self.returning_to_base = False
        self.stationary = False     # 静止靶机标志（不机动）
        self.no_fire = False        # 无还手标志（靶机不开火）
        self.lock_target = None     # 火控锁定目标（导弹目标 / 机炮辅助瞄准目标）
        self.aggressiveness = 0.5   # 激进度：0=保命型 0.5=均衡 1=激进型（观测维度，奖励按此塑形）
        self.color = self.spec.color
        if team == 'blue' and self.color == (200, 50, 50):
            self.color = (50, 50, 220)
        if team == 'red' and self.color == (200, 50, 50):
            self.color = (220, 50, 50)

    def update(self, world, dt=1):
        if not self.alive:
            return
        # 补给（靠近本方机场）
        for b in world.buildings.alive_list():
            if b.team == self.team and b.type in ('airbase', 'forward_airbase'):
                if math.hypot(b.x - self.x, b.y - self.y) < b.size + 20:
                    self.weapons.rearm()
                    self.returning_to_base = False
                    break
        self.sensor.update(world, self)
        self.weapons.update(dt)
        if self.flare_timer > 0:
            self.flare_timer -= dt
        # 低弹药/低血量 → 标记返航（供规则AI与观测使用）
        if (self.weapons.ammo < self.weapons.max_ammo * 0.2 or self.hp < self.max_hp * 0.3):
            self.returning_to_base = True
        # 锁定目标失效清理
        if self.lock_target is not None and (not getattr(self.lock_target, 'alive', False)):
            self.lock_target = None
        # 决策并应用动作
        act = self.controller.decide(world, self)
        self._apply(world, act)

    def _apply_target_select(self, world, sel):
        """火控目标选择：更新锁定目标（导弹机=导弹目标，机炮机=辅助瞄准目标）。"""
        if sel == LOCK_HOLD:
            return
        if sel == LOCK_CLEAR:
            self.lock_target = None
            return
        if sel == LOCK_NEAREST_AIR:
            new_lock = world.nearest_enemy_aircraft(self)
        elif sel == LOCK_NEAREST_GROUND:
            new_lock = world.nearest_enemy_ground(self)
        elif sel == LOCK_NEXT:
            enemies = world.enemies_of(self)
            if not enemies:
                new_lock = None
            elif self.lock_target in enemies:
                i = enemies.index(self.lock_target)
                new_lock = enemies[(i + 1) % len(enemies)]
            else:
                new_lock = enemies[0]
        else:
            return
        if new_lock is not None and new_lock != self.lock_target:
            self.lock_target = new_lock
            # 雷达弹重定向：本机在飞的中途可改目标的雷达制导导弹 → 新锁定目标
            mtype = self.weapons.missile.mtype
            if mtype and MISSILE_TYPES.get(mtype, {}).get('type') == 'radar':
                for m in world.missiles.alive_list():
                    if m.owner_uid == self.uid and not m.is_ir:
                        m.target = new_lock

    def aim_assist(self, world, max_nudge=2.0, gain=0.35):
        """火控对准辅助：锁定目标且机头接近（±30°）时，自动微调机头对准**提前点**。

        ⚠ 关键修正（v9）：原实现只对准目标【当前位置】，对移动目标等于没有提前量。
        受控实测：机炮打静止目标 8 发 7 中并击毁，打机动目标（速度 4）
        **120 发仅 2 中** —— 9° 锥角下需要近乎完美的提前量，这成了整个项目
        命中率低下的根因，且与策略无关（规则 AI 同样打不中）。
        现在按相对速度估算弹丸飞行时间并对**提前点**取角，即真实火控的前置量计算。

        返回本次微调的角度（度）。"""
        tgt = self.lock_target
        if tgt is None or not getattr(tgt, 'alive', False) or tgt is self:
            return 0.0
        # 目标速度矢量
        if hasattr(tgt, 'flight'):
            tvx = tgt.flight.speed * math.cos(math.radians(tgt.angle))
            tvy = tgt.flight.speed * math.sin(math.radians(tgt.angle))
        else:
            tvx = tvy = 0.0
        # 弹丸相对速度：机炮初速 = 机速 + 6（与 WeaponSystem._fire_gun 一致），
        # 沿瞄准方向相对目标的速度 = 弹速 - 目标在该方向的投影速度
        bullet = self.flight.speed + 6.0
        dirx = math.cos(math.radians(self.angle))
        diry = math.sin(math.radians(self.angle))
        closing = bullet - (tvx * dirx + tvy * diry)
        rel_speed = max(0.5, closing)
        dx, dy = tgt.x - self.x, tgt.y - self.y
        t_go = math.hypot(dx, dy) / rel_speed
        # 提前点（限制外推距离，避免高速目标导致过分超前）
        t_go = min(t_go, 200.0 / rel_speed)
        px, py = tgt.x + tvx * t_go, tgt.y + tvy * t_go
        lead = math.degrees(math.atan2(py - self.y, px - self.x))
        diff = norm_deg(lead - self.angle)
        if abs(diff) >= 30.0:
            return 0.0
        nudge = max(-max_nudge, min(max_nudge, diff * gain))
        self.angle = norm_deg(self.angle + nudge)
        return nudge

    def _apply(self, world, act: Action):
        # 头控制：向目标扇区有限角速度转动
        target = head_sector_center(act.head)
        diff = norm_deg(target - self.head_angle)
        rate = 4.0   # 度/帧（240°/s）
        self.head_angle = norm_deg(self.head_angle + max(-rate, min(rate, diff)))
        if not self.stationary:
            # 飞控
            turn_dir = -1 if act.turn == TURN_LEFT else (1 if act.turn == TURN_RIGHT else 0)
            turn_rate = self.flight.update(act.throttle, turn_dir, 1)
            self.angle = norm_deg(self.angle + turn_rate)
            # 火控对准辅助（统一生效，规则 AI 与神经网络对等）
            self.aim_assist(world)
            # 边界飞控接管：接近边界时强行把机头转回场内（不碰撞，接管优先于一切）
            self._boundary_override()
            # 移动
            rad = math.radians(self.angle)
            self.x += self.flight.speed * math.cos(rad)
            self.y += self.flight.speed * math.sin(rad)
            self.x = max(20, min(WORLD_W - 20, self.x))
            self.y = max(20, min(WORLD_H - 20, self.y))
        # 火控目标选择
        self._apply_target_select(world, act.target_select)
        # 目标共享（把记忆/锁定/建筑/机场位置分享给本队）
        self._apply_share(world, act.share_target)
        # 火控（请求式；靶机不开火）
        if not self.no_fire:
            self.weapons.try_fire(world, self, act.fire, self.sensor)

    def _boundary_override(self):
        """边界飞控接管：接近地图边界（M 距离内）时强行把机头转回场内，不碰撞。"""
        M = 260.0
        tx, ty = self.x, self.y
        if tx < M:
            tx = M
        if tx > WORLD_W - M:
            tx = WORLD_W - M
        if ty < M:
            ty = M
        if ty > WORLD_H - M:
            ty = WORLD_H - M
        if abs(tx - self.x) < 1 and abs(ty - self.y) < 1:
            return
        target_angle = math.degrees(math.atan2(ty - self.y, tx - self.x))
        diff = norm_deg(target_angle - self.angle)
        rate = 7.0   # 接管转角速度（度/帧），比正常转向更快
        self.angle = norm_deg(self.angle + max(-rate, min(rate, diff)))

    def _apply_share(self, world, sel):
        """目标共享：把选定目标写入本队共享板（友军通过记忆池感知）。"""
        if sel == SHARE_NONE:
            return
        items = []
        if sel in (SHARE_LOCK, SHARE_ALL) and self.lock_target is not None and self.lock_target.alive:
            items.append(('lock', self.lock_target.x, self.lock_target.y))
        if sel in (SHARE_NEAR_AC, SHARE_ALL):
            m = self.sensor.nearest_memory_air(self)
            if m is not None:
                items.append(('air', m.x, m.y))
        if sel in (SHARE_BUILDING, SHARE_ALL):
            b = self.sensor.nearest_building_mem()
            if b is not None:
                items.append(('building', b[1], b[2]))
        if sel in (SHARE_AIRBASE, SHARE_ALL):
            base = world.nearest_friendly_airbase(self)
            if base is not None:
                items.append(('airbase', base.x, base.y))
        for kind, x, y in items:
            world.add_shared(self.team, kind, x, y)

    def draw(self, screen, camera):
        if not self.alive:
            return
        s = self.size
        if self.shape == 'narrow':
            local = [(s * 1.2, 0), (-s * 0.6, -s * 0.6), (-s * 0.3, 0), (-s * 0.6, s * 0.6)]
        elif self.shape == 'wide_triangle':
            local = [(s * 1.0, 0), (-s * 0.8, -s * 1.2), (-s * 0.8, s * 1.2)]
        elif self.shape == 'small_triangle':
            local = [(s * 0.6, 0), (-s * 0.6, -s * 0.6), (-s * 0.6, s * 0.6)]
        else:
            local = [(s * 0.7, 0), (-s * 0.7, -s * 0.7), (-s * 0.7, s * 0.7)]
        rad = math.radians(self.angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        world_pts = [(self.x + lx * cos_a - ly * sin_a, self.y + lx * sin_a + ly * cos_a) for lx, ly in local]
        sw = [camera.world_to_screen(px, py) for px, py in world_pts]
        pygame.draw.polygon(screen, self.color, sw)
        # 机头方向（头控制指示）
        hx, hy = camera.world_to_screen(
            self.x + 18 * math.cos(math.radians(self.angle + self.head_angle)),
            self.y + 18 * math.sin(math.radians(self.angle + self.head_angle)))
        sx, sy = camera.world_to_screen(self.x, self.y)
        pygame.draw.line(screen, YELLOW, (sx, sy), (hx, hy), 1)
        # 血条
        pygame.draw.rect(screen, BLACK, (sx - 10, sy - 14, 20, 3))
        pygame.draw.rect(screen, GREEN, (sx - 10, sy - 14, int(20 * self.hp / self.max_hp), 3))


class Building(Unit):
    kind = 'building'

    def __init__(self, x, y, btype, team):
        super().__init__(x, y, team)
        self.type = btype
        self.spec = BUILDING_SPECS[btype]
        self.max_hp = self.spec.max_hp
        self.hp = self.max_hp
        self.size = self.spec.size
        self.color = self.spec.color
        if team == 'blue' and self.color in ((200, 50, 50),):
            self.color = (50, 50, 220)
        self.angle = 0.0
        self.reload = 0
        self.src_angle = random.uniform(0, 360)
        self.fire_locked = None
        self.missile_clip = []
        self.clip_reload = 0
        self.spawn_timer = 0
        self.tracked_targets = []
        self.fill_clip()

    def fill_clip(self):
        if self.spec.missile_main:
            self.missile_clip = [self.spec.missile_main] * 3 + [self.spec.missile_special]

    def update(self, world, dt=1):
        if not self.alive:
            return
        if self.type == 'rally':
            self.spawn_timer += dt
            if self.spawn_timer >= 3600:
                self.spawn_timer = 0
                for _ in range(3):
                    world.ground.spawn(self.x + random.randint(-20, 20), self.y + random.randint(-20, 20), 'tank', self.team)
                for _ in range(5):
                    world.ground.spawn(self.x + random.randint(-20, 20), self.y + random.randint(-20, 20), 'infantry', self.team)
            return
        if self.type == 'radar':
            self.src_angle = (self.src_angle + 0.5 * dt) % 360
            self.scan_radar(world)
            return
        if self.spec.src_range > 0:
            self.src_angle = (self.src_angle + 1 * dt) % 360
            self.update_fire_control(world, dt)

    def scan_radar(self, world):
        self.tracked_targets = [t for t in self.tracked_targets
                                if t.alive and math.hypot(t.x - self.x, t.y - self.y) <= self.spec.scan_range]
        for a in world.aircraft.alive_list():
            if a.team == self.team or len(self.tracked_targets) >= 5:
                continue
            dx, dy = a.x - self.x, a.y - self.y
            if math.hypot(dx, dy) <= self.spec.scan_range:
                angle_to = math.degrees(math.atan2(dy, dx))
                if abs(norm_deg(angle_to - self.src_angle)) <= 90 and a not in self.tracked_targets:
                    self.tracked_targets.append(a)
        # 数据链：把雷达探测到的敌机位置/速度写入本队所有飞机的记忆池（无需本机目视）
        now = pygame.time.get_ticks()
        for t in self.tracked_targets:
            t.sensor.visibility_buff = max(t.sensor.visibility_buff, 500)
            for a in world.aircraft_list_of(self.team):
                if a.alive:
                    a.sensor._upsert_memory(t, now, a)

    def update_fire_control(self, world, dt):
        best_target, best_priority = None, -1
        for bomb in world.bombs.alive_list():
            if bomb.team == self.team:
                continue
            d = math.hypot(bomb.x - self.x, bomb.y - self.y)
            if d > self.spec.src_range:
                continue
            priority = 12 if getattr(bomb, 'timer', 999) < 30 else 5
            priority -= d / 100
            if priority > best_priority:
                best_priority, best_target = priority, bomb
        for a in world.aircraft.alive_list():
            if a.team == self.team:
                continue
            d = math.hypot(a.x - self.x, a.y - self.y)
            if d > self.spec.src_range:
                continue
            priority = 10 if a.sensor.visibility_buff > 0 else (5 if a in self.tracked_targets else 0)
            priority -= d / 100
            if priority > best_priority:
                best_priority, best_target = priority, a
        if best_target:
            self.fire_locked = best_target
            target_angle = math.degrees(math.atan2(best_target.y - self.y, best_target.x - self.x))
            diff = norm_deg(target_angle - self.angle)
            turn_rate = 3
            self.angle = norm_deg(self.angle + max(-turn_rate, min(turn_rate, diff)))
            if abs(norm_deg(target_angle - self.angle)) < 5 and \
                    math.hypot(best_target.x - self.x, best_target.y - self.y) <= self.spec.attack_range:
                self.attack_target(world)
        else:
            self.angle = (self.angle + 1) % 360
        if self.reload > 0:
            self.reload -= dt
        if self.clip_reload > 0:
            self.clip_reload -= dt
            if self.clip_reload <= 0:
                self.fill_clip()

    def attack_target(self, world):
        if self.type == 'gun_aa':
            if self.fire_locked and self.reload <= 0:
                rad = math.radians(self.angle)
                world.bullets.spawn(self.x + 10 * math.cos(rad), self.y + 10 * math.sin(rad),
                                    self.angle, 8, self.spec.damage, self.team, owner_uid=self.uid,
                                    is_aa=True, target=self.fire_locked, max_range=self.spec.attack_range)
                self.reload = self.spec.cooldown
        elif self.type in ('missile_aa', 'long_range_aa'):
            if self.fire_locked and self.missile_clip and self.reload <= 0:
                mtype = self.missile_clip.pop(0)
                world.missiles.spawn(self.x, self.y, self.angle, self.fire_locked, self.team, mtype, owner_uid=self.uid)
                self.reload = self.spec.cooldown
                if not self.missile_clip:
                    self.clip_reload = self.spec.cooldown * 4

    def draw(self, screen, camera):
        if not self.alive:
            return
        sx, sy = camera.world_to_screen(self.x, self.y)
        r = max(4, int(self.size * camera.zoom))
        pygame.draw.rect(screen, self.color, (sx - r // 2, sy - r // 2, r, r))
        pygame.draw.rect(screen, BLACK, (sx - r // 2, sy - r // 2 - 5, r, 3))
        pygame.draw.rect(screen, GREEN, (sx - r // 2, sy - r // 2 - 5, int(r * self.hp / self.max_hp), 3))


class GroundUnit(Unit):
    kind = 'ground'

    def __init__(self, x, y, utype, team):
        super().__init__(x, y, team)
        self.type = utype
        self.spec = GROUND_SPECS[utype]
        self.max_hp = self.spec.max_hp
        self.hp = self.max_hp
        self.size = self.spec.size
        self.color = self.spec.color
        if team == 'blue' and self.color == (200, 50, 50):
            self.color = (50, 50, 220)
        self.angle = random.uniform(0, 360)
        self.turret_angle = self.angle
        self.reload = 0
        self.src_angle = random.uniform(0, 360) if 'aa' in utype else 0
        self.fire_locked = None
        self.missile_clip = []
        self.clip_reload = 0
        self.evasive_timer = 0
        self.passive = False    # 被动靶标：不攻击
        self.patrol = False     # 巡逻靶标：绕圈移动
        self._patrol_angle = random.uniform(0, 360)
        self.fill_clip()

    def fill_clip(self):
        if self.type == 'missile_aa_vehicle':
            self.missile_clip = [self.spec.missile_main] * 3 + [self.spec.missile_special]

    def update(self, world, dt=1):
        if not self.alive:
            return
        if self.passive:
            # 被动靶标：静止或绕小圈巡逻
            if self.patrol:
                self._patrol_angle += 0.6 * dt
                rad = math.radians(self._patrol_angle)
                self.x += self.spec.speed * 0.6 * math.cos(rad) * dt
                self.y += self.spec.speed * 0.6 * math.sin(rad) * dt
            return
        # 规避来袭炸弹
        if 'aa' in self.type and self.evasive_timer == 0:
            for bomb in world.bombs.alive_list():
                if bomb.team == self.team:
                    continue
                d = math.hypot(bomb.x - self.x, bomb.y - self.y)
                if d < 200 and getattr(bomb, 'timer', 999) < 20:
                    away = math.degrees(math.atan2(self.y - bomb.y, self.x - bomb.x))
                    rad = math.radians(away)
                    self.x += self.spec.speed * 4 * math.cos(rad)
                    self.y += self.spec.speed * 4 * math.sin(rad)
                    self.evasive_timer = 30
                    break
        if self.evasive_timer > 0:
            self.evasive_timer -= dt
            return
        if self.type in ('gun_aa_vehicle', 'missile_aa_vehicle'):
            self.src_angle = (self.src_angle + 2 * dt) % 360
            self.update_fire_control(world, dt)
        else:
            self.combat_ground(world)

    def update_fire_control(self, world, dt):
        best_target, best_priority = None, -1
        for bomb in world.bombs.alive_list():
            if bomb.team == self.team:
                continue
            d = math.hypot(bomb.x - self.x, bomb.y - self.y)
            if d > self.spec.src_range:
                continue
            priority = 14 if getattr(bomb, 'timer', 999) < 30 else 6
            priority -= d / 100
            if priority > best_priority:
                best_priority, best_target = priority, bomb
        for a in world.aircraft.alive_list():
            if a.team == self.team:
                continue
            d = math.hypot(a.x - self.x, a.y - self.y)
            if d > self.spec.src_range:
                continue
            priority = 10 if a.sensor.visibility_buff > 0 else 0
            priority -= d / 100
            if priority > best_priority:
                best_priority, best_target = priority, a
        if best_target:
            self.fire_locked = best_target
            target_angle = math.degrees(math.atan2(best_target.y - self.y, best_target.x - self.x))
            diff = norm_deg(target_angle - self.angle)
            self.angle = norm_deg(self.angle + max(-4, min(4, diff)))
            tdiff = norm_deg(target_angle - self.turret_angle)
            self.turret_angle = norm_deg(self.turret_angle + max(-6, min(6, tdiff)))
            if abs(norm_deg(target_angle - self.turret_angle)) < 8 and \
                    math.hypot(best_target.x - self.x, best_target.y - self.y) <= self.spec.attack_range:
                self.attack_target(world)
        else:
            self.angle = (self.angle + 2) % 360
            self.follow_friends(world)
        if self.reload > 0:
            self.reload -= dt
        if self.clip_reload > 0:
            self.clip_reload -= dt
            if self.clip_reload <= 0:
                self.fill_clip()

    def attack_target(self, world):
        if self.type == 'gun_aa_vehicle' and self.reload <= 0:
            target = self.fire_locked
            shot = math.degrees(math.atan2(target.y - self.y, target.x - self.x)) if target else self.turret_angle
            rad = math.radians(shot)
            world.bullets.spawn(self.x + 10 * math.cos(rad), self.y + 10 * math.sin(rad),
                                shot, self.spec.bullet_speed, self.spec.damage, self.team, owner_uid=self.uid,
                                is_aa=True, target=target, max_range=self.spec.attack_range)
            self.reload = self.spec.cooldown
        elif self.type == 'missile_aa_vehicle' and self.missile_clip and self.reload <= 0:
            mtype = self.missile_clip.pop(0)
            world.missiles.spawn(self.x, self.y, self.turret_angle, self.fire_locked, self.team, mtype, owner_uid=self.uid)
            self.reload = self.spec.cooldown
            if not self.missile_clip:
                self.clip_reload = self.spec.cooldown * 4

    def follow_friends(self, world):
        friends = [g for g in world.ground.alive_list()
                   if g.team == self.team and g is not self and 'aa' not in g.type]
        if friends:
            friend = min(friends, key=lambda g: math.hypot(g.x - self.x, g.y - self.y))
            target_angle = math.degrees(math.atan2(friend.y - self.y, friend.x - self.x))
            diff = norm_deg(target_angle - self.angle)
            self.angle = norm_deg(self.angle + max(-2, min(2, diff)))
            rad = math.radians(self.angle)
            self.x += self.spec.speed * 0.5 * math.cos(rad)
            self.y += self.spec.speed * 0.5 * math.sin(rad)

    def combat_ground(self, world):
        enemies = [g for g in world.ground.alive_list() if g.team != self.team] + \
                  [b for b in world.buildings.alive_list() if b.team != self.team]
        if not enemies:
            return
        closest = min(enemies, key=lambda e: math.hypot(e.x - self.x, e.y - self.y))
        dist = math.hypot(closest.x - self.x, closest.y - self.y)
        target_angle = math.degrees(math.atan2(closest.y - self.y, closest.x - self.x))
        diff = norm_deg(target_angle - self.angle)
        self.angle = norm_deg(self.angle + max(-2, min(2, diff)))
        self.turret_angle = target_angle
        if dist > self.spec.attack_range:
            rad = math.radians(self.angle)
            self.x += self.spec.speed * math.cos(rad)
            self.y += self.spec.speed * math.sin(rad)
        else:
            if self.reload <= 0:
                rad = math.radians(self.turret_angle)
                world.bullets.spawn(self.x + 10 * math.cos(rad), self.y + 10 * math.sin(rad),
                                    self.turret_angle, self.spec.bullet_speed, self.spec.damage,
                                    self.team, owner_uid=self.uid)
                self.reload = self.spec.cooldown
            else:
                self.reload -= 1

    def draw(self, screen, camera):
        if not self.alive:
            return
        sx, sy = camera.world_to_screen(self.x, self.y)
        r = max(3, int(self.size * camera.zoom))
        pygame.draw.circle(screen, self.color, (sx, sy), r)


# ---------------- 弹药 ----------------
class Projectile(Unit):
    kind = 'projectile'

    def __init__(self, x, y, angle, team, owner_uid=None):
        super().__init__(x, y, team)
        self.angle = angle
        self.owner_uid = owner_uid


class Bullet(Projectile):
    def __init__(self, x, y, angle, speed, damage, team, owner_uid=None,
                 is_aa=False, target=None, max_range=None, hit_radius=2.0):
        super().__init__(x, y, angle, team, owner_uid)
        self.speed = speed
        self.damage = damage
        self.is_aa = is_aa
        self.target = target
        self.traveled = 0.0
        self.max_range = max_range
        self.hit_radius = hit_radius

    def update(self, world, dt=1):
        if not self.alive:
            return
        rad = math.radians(self.angle)
        dx, dy = self.speed * math.cos(rad) * dt, self.speed * math.sin(rad) * dt
        self.x += dx
        self.y += dy
        self.traveled += math.hypot(dx, dy)
        if self.max_range and self.traveled >= self.max_range:
            self.alive = False
            return
        if self.x < 0 or self.x > WORLD_W or self.y < 0 or self.y > WORLD_H:
            self.alive = False
            return
        # 防空弹拦截炸弹
        if self.is_aa:
            for bomb in world.bombs.alive_list():
                if math.hypot(bomb.x - self.x, bomb.y - self.y) < 10:
                    prob = 0.35 if getattr(bomb, 'timer', 999) < 15 else 0.08
                    if random.random() < prob:
                        bomb.alive = False
                        self.alive = False
                        return
        # 命中敌方单位：扫掠检测（对本帧位移线段求与目标的最近距离）。
        # 实测依据：原先用"移动后的点"做圆判定，弹速 26 时会直接跨过目标，
        # 规则AI 打出 268 发、夹角平均仅 2.1°、距离均值 45，却 0 命中；
        # 而同一门炮打静止目标却能正常击毁 —— 典型的点采样漏检。
        sx0, sy0 = self.x - dx, self.y - dy
        mid_r = 14 + self.hit_radius + math.hypot(dx, dy) * 0.5 + 8
        for e in world.spatial.query_circle((self.x + sx0) / 2.0, (self.y + sy0) / 2.0, mid_r):
            if not e.alive or e.team == self.team or isinstance(e, Projectile):
                continue
            if _point_seg_dist(e.x, e.y, sx0, sy0, self.x, self.y) < getattr(e, 'size', 10) + self.hit_radius:
                e.take_damage(world, self.damage, self.team, self.owner_uid)
                self.alive = False
                return

    def draw(self, screen, camera):
        if not self.alive:
            return
        sx, sy = camera.world_to_screen(self.x, self.y)
        pygame.draw.circle(screen, YELLOW, (sx, sy), max(2, int(3 * camera.zoom)))


class Bomb(Projectile):
    def __init__(self, x, y, angle, speed, damage, radius, team, owner_uid=None, target=None,
                 turn=3, max_range=2000, powered_dist=500, guided=False):
        super().__init__(x, y, angle, team, owner_uid)
        self.damage = damage
        self.radius = radius
        self.timer = 30
        self.guided = guided
        self.target = target
        if guided:
            self.turn = turn
            self.max_range = max_range
            self.powered_dist = powered_dist
            self.traveled = 0
            self.powered_traveled = 0
            self.cur_speed = speed
            self.init_speed = speed
            self.wing_turn = max(0.6, turn * 0.35)
            self.drag_k = 0.005
        else:
            self.speed = speed

    def set_timer(self, t):
        self.timer = max(1, int(t))

    def update(self, world, dt=1):
        if not self.alive:
            return
        if self.guided:
            self._update_guided(world, dt)
            return
        rad = math.radians(self.angle)
        v = max(0.0, self.speed)
        v = max(0.0, v - 0.005 * v * v * dt)
        self.speed = v
        self.x += v * math.cos(rad) * dt
        self.y += v * math.sin(rad) * dt
        self.timer -= dt
        if self.timer <= 0:
            self._explode(world)

    def _update_guided(self, world, dt):
        if not self.target or not self.target.alive:
            self._explode(world)
            return
        target_angle = math.degrees(math.atan2(self.target.y - self.y, self.target.x - self.x))
        diff = norm_deg(target_angle - self.angle)
        if self.powered_traveled < self.powered_dist:
            turn = self.turn
            max_speed = self.init_speed * 1.2
            if self.cur_speed < max_speed:
                self.cur_speed = min(max_speed, self.cur_speed + 0.02 * self.init_speed * dt)
        else:
            turn = self.wing_turn
            self.cur_speed = max(0.0, self.cur_speed - self.drag_k * self.cur_speed ** 2 * dt)
        self.angle = norm_deg(self.angle + max(-turn, min(turn, diff)))
        rad = math.radians(self.angle)
        self.x += self.cur_speed * math.cos(rad) * dt
        self.y += self.cur_speed * math.sin(rad) * dt
        self.powered_traveled += self.cur_speed * dt
        self.traveled += self.cur_speed * dt
        self.timer -= dt
        if self.traveled >= self.max_range or self.timer <= 0 or \
                math.hypot(self.target.x - self.x, self.target.y - self.y) <= 15:
            self._explode(world)

    def _explode(self, world):
        for e in world.spatial.query_circle(self.x, self.y, self.radius):
            if not e.alive or e.team == self.team:
                continue
            if isinstance(e, (GroundUnit, Building)):
                d = math.hypot(e.x - self.x, e.y - self.y)
                if d <= self.radius:
                    e.take_damage(world, self.damage * (1 - d / self.radius), self.team, self.owner_uid)
        self.alive = False

    def draw(self, screen, camera):
        if not self.alive:
            return
        sx, sy = camera.world_to_screen(self.x, self.y)
        color = DARK_RED if self.guided else ORANGE
        pygame.draw.circle(screen, color, (sx, sy), max(3, int(5 * camera.zoom)))


class Missile(Projectile):
    def __init__(self, x, y, angle, target, team, mtype='A-IR', owner_uid=None):
        super().__init__(x, y, angle, team, owner_uid)
        cfg = MISSILE_TYPES[mtype]
        self.type = mtype
        self.target = target
        self.speed = cfg['speed']
        self.turn = cfg['turn']
        self.max_range = cfg['range']
        self.is_ir = cfg['type'] == 'ir'
        self.damage = 100
        self.traveled = 0
        self.color = cfg.get('color', YELLOW)

    def update(self, world, dt=1):
        if not self.alive:
            return
        if not self.target or not self.target.alive:
            self.alive = False
            return
        # 诱饵弹：红外弹接近正在放诱饵的目标时丢失
        if self.is_ir and isinstance(self.target, Aircraft) and self.target.flare_timer > 0:
            if math.hypot(self.target.x - self.x, self.target.y - self.y) < 200:
                self.alive = False
                world.flare_success.append(self.target.team)
                return
        target_angle = math.degrees(math.atan2(self.target.y - self.y, self.target.x - self.x))
        diff = norm_deg(target_angle - self.angle)
        self.angle = norm_deg(self.angle + max(-self.turn, min(self.turn, diff)))
        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad) * dt
        self.y += self.speed * math.sin(rad) * dt
        self.traveled += self.speed * dt
        if math.hypot(self.target.x - self.x, self.target.y - self.y) < 15:
            self.target.take_damage(world, self.damage, self.team, self.owner_uid)
            self.alive = False
        elif self.traveled >= self.max_range:
            self.alive = False

    def draw(self, screen, camera):
        if not self.alive:
            return
        s = 4
        local = [(s * 1.5, 0), (-s * 1.5, -s), (-s * 1.5, s)]
        rad = math.radians(self.angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        world_pts = [(self.x + lx * cos_a - ly * sin_a, self.y + lx * sin_a + ly * cos_a) for lx, ly in local]
        sw = [camera.world_to_screen(px, py) for px, py in world_pts]
        pygame.draw.polygon(screen, self.color, sw)


# ======================================================================
# 5. 观测维度（Sensor.build_observation 使用）
# ======================================================================
BASE_OBS_DIM = 20                 # 自身状态 + 威胁导弹 + 状态位
TACTICAL_DIM = 16                 # 最近 2 敌机 + 最近 2 友机（编队协作 / 目标分配）
MEMORY_DIM = 8 * 7                # 统一目标记忆池 8 条 × 7 维
BUILDING_DIM = 4 * 4              # 敌方建筑记忆 4 条 × 4 维
TACTICAL_START = BASE_OBS_DIM + TACTICAL_DIM
OBS_DIM = BASE_OBS_DIM + TACTICAL_DIM + MEMORY_DIM + BUILDING_DIM + 3 + 3 + 1
# 20 基础 + 16 战术 + 56 记忆 + 16 建筑 + 3 机场 + 3 锁定 + 1 激进度 = 115
OBS_VERSION = 3                   # 观测布局版本：任何维度/语义变更必须递增（检查点兼容校验用）


# ======================================================================
# 6. AI：规则专家（FSM）/ 神经网络 / PPO / 环境
# ======================================================================
class RuleAI(Controller):
    """规则专家 AI（FSM）：巡逻 → 接敌 → 规避 → 返航补给。"""

    PATROL, ENGAGE, EVADE, RETURN = 0, 1, 2, 3

    def __init__(self, team):
        self.team = team
        self.state = self.PATROL
        self.waypoint = None
        self.state_timer = 0
        self.scan_timer = 0     # 头部扫描计时
        self.scan_dir = 1       # 扫描方向

    def decide(self, world, unit):
        act = Action()
        sensor = unit.sensor
        threat = world.nearest_enemy_missile(unit)
        enemy_ac = sensor._alive_enemy_aircraft(world, unit)
        enemy_ground = sensor._alive_enemy_ground(world, unit)

        # ---- 状态判定 ----
        if threat is not None and math.hypot(threat.x - unit.x, threat.y - unit.y) < 400:
            self.state = self.EVADE
        elif unit.returning_to_base or (unit.hp < unit.max_hp * 0.3):
            self.state = self.RETURN
        elif enemy_ac and math.hypot(enemy_ac[0].x - unit.x, enemy_ac[0].y - unit.y) < 1200:
            # 规则 AI 具备 1200 内接敌感知（作为强对手），探测到即主动求战
            self.state = self.ENGAGE
        elif enemy_ground and math.hypot(enemy_ground[0].x - unit.x, enemy_ground[0].y - unit.y) < 400:
            self.state = self.ENGAGE
        else:
            self.state = self.PATROL

        if self.state == self.EVADE:
            # 背向导弹逃离 + 加力 + 放诱饵
            away = math.degrees(math.atan2(unit.y - threat.y, unit.x - threat.x))
            diff = norm_deg(away - unit.angle)
            act.turn = TURN_LEFT if diff < 0 else (TURN_RIGHT if diff > 0 else TURN_STRAIGHT)
            act.throttle = THR_AFTERBURN
            if unit.sensor.flare_timer <= 0:
                act.fire = FIRE_FLARE
            return act

        if self.state == self.RETURN:
            base = None
            for b in world.buildings.alive_list():
                if b.team == self.team and b.type in ('airbase', 'forward_airbase'):
                    base = b
                    break
            if base:
                ang = math.degrees(math.atan2(base.y - unit.y, base.x - unit.x))
                diff = norm_deg(ang - unit.angle)
                act.turn = TURN_LEFT if diff < 0 else (TURN_RIGHT if diff > 0 else TURN_STRAIGHT)
                act.head = angle_to_sector(diff)
                act.throttle = THR_AFTERBURN if math.hypot(base.x - unit.x, base.y - unit.y) > 600 else THR_CRUISE
                if math.hypot(base.x - unit.x, base.y - unit.y) < 120:
                    unit.returning_to_base = False
            return act

        if self.state == self.ENGAGE:
            # 地面目标：优先用机炮扫射（可连续射击、命中率高），仅在带弹且未对准时投弹。
            # 原实现只投弹，而单枚炸弹从投放到引爆前会随载机前冲 ~60 单位，命中率极低，
            # 实测规则AI 在纯地面目标阶段零伤害。
            gtarget = sensor.bomb_target(world, unit)
            gwide = world.nearest_enemy_ground(unit)
            if gwide is not None and unit.weapons.gun.damage > 0:
                ang = math.degrees(math.atan2(gwide.y - unit.y, gwide.x - unit.x))
                diff = norm_deg(ang - unit.angle)
                if abs(diff) <= unit.weapons.gun.angle and \
                        math.hypot(gwide.x - unit.x, gwide.y - unit.y) <= unit.weapons.gun.range:
                    act.fire = FIRE_GUN
                    act.target_select = LOCK_NEAREST_GROUND
                    act.head = angle_to_sector(0)
                    act.turn = TURN_STRAIGHT
                    act.throttle = THR_CRUISE
                    return act
            if gtarget is not None and unit.weapons.bombs > 0 and unit.weapons.gun.damage <= 0:
                act.fire = FIRE_BOMB
                act.target_select = LOCK_NEAREST_GROUND
                ang = math.degrees(math.atan2(gtarget.y - unit.y, gtarget.x - unit.x))
                diff = norm_deg(ang - unit.angle)
                act.turn = TURN_LEFT if diff < 0 else (TURN_RIGHT if diff > 0 else TURN_STRAIGHT)
                act.head = angle_to_sector(0)
                act.throttle = THR_CRUISE
                return act
            # 无空中目标时，朝最近地面目标逼近以取得射击条件
            if not enemy_ac and gwide is not None:
                act.target_select = LOCK_NEAREST_GROUND
                act.share_target = SHARE_BUILDING
                ang = math.degrees(math.atan2(gwide.y - unit.y, gwide.x - unit.x))
                diff = norm_deg(ang - unit.angle)
                act.turn = TURN_LEFT if diff < 0 else (TURN_RIGHT if diff > 0 else TURN_STRAIGHT)
                act.head = angle_to_sector(0)
                act.throttle = THR_CRUISE
                if unit.weapons.gun.damage > 0 and \
                        math.hypot(gwide.x - unit.x, gwide.y - unit.y) <= unit.weapons.gun.range \
                        and abs(diff) <= unit.weapons.gun.angle:
                    act.fire = FIRE_GUN
                return act
            if enemy_ac:
                e = min(enemy_ac, key=lambda x: math.hypot(x.x - unit.x, x.y - unit.y))
                act.target_select = LOCK_NEAREST_AIR
                act.share_target = SHARE_LOCK          # 交战即共享锁定目标给僚机
                ang = math.degrees(math.atan2(e.y - unit.y, e.x - unit.x))
                diff = norm_deg(ang - unit.angle)
                act.turn = TURN_LEFT if diff < 0 else (TURN_RIGHT if diff > 0 else TURN_STRAIGHT)
                act.head = angle_to_sector(diff)
                # 距离远用导弹，近用机炮
                d = math.hypot(e.x - unit.x, e.y - unit.y)
                if unit.weapons.missiles > 0 and d > 150:
                    act.fire = FIRE_MISSILE
                else:
                    act.fire = FIRE_GUN
                act.throttle = THR_AFTERBURN if d > 300 else THR_CRUISE
                return act

        # 巡逻：飞向随机航点；头部持续左右扫描（搜索敌机）
        if self.waypoint is None or self.state_timer <= 0 or \
                math.hypot(self.waypoint[0] - unit.x, self.waypoint[1] - unit.y) < 100:
            self.waypoint = (random.uniform(200, WORLD_W - 200), random.uniform(200, WORLD_H - 200))
            self.state_timer = 300
        self.state_timer -= 1
        self.scan_timer += 1
        if self.scan_timer >= 45:            # 每 0.75 秒换向扫描
            self.scan_timer = 0
            self.scan_dir = -self.scan_dir
        act.head = angle_to_sector(self.scan_dir * 50)   # 左右 50° 内往复扫描
        ang = math.degrees(math.atan2(self.waypoint[1] - unit.y, self.waypoint[0] - unit.x))
        diff = norm_deg(ang - unit.angle)
        act.turn = TURN_LEFT if diff < 0 else (TURN_RIGHT if diff > 0 else TURN_STRAIGHT)
        act.throttle = THR_CRUISE
        return act


class ActorCritic(nn.Module):
    """多头离散 Actor-Critic。"""

    def __init__(self, obs_dim, act_dims):
        super().__init__()
        self.act_dims = act_dims
        self.fc = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(),
                                nn.Linear(256, 256), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(256, d) for d in act_dims])
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        h = self.fc(x)
        return [hd(h) for hd in self.heads], self.critic(h)


def weapon_fire_mask(unit, n_options=4, world=None) -> np.ndarray:
    """火控动作可用性掩码：屏蔽【本帧确实无法执行或纯无效】的开火选项。

    实测依据：模型（含未加掩码的版本）会把火控头 100% 收敛到「诱饵」——
    在 1v1 对规则 AI、没有来袭导弹的场景里放诱饵没有任何收益，是个纯无效动作，
    却因为没有即时惩罚而成为稳定的局部最优：模型抵近到距离 2、锁定帧 100%、
    射程内帧 25.9%，却始终不开炮（零伤害）。

    规则：
      1. 备弹为 0 / 冷却中的选项一律屏蔽；
      2. **无来袭导弹（且记忆中无敌方导弹）时屏蔽诱饵** —— 有导弹威胁时才允许放出。
    注意**不屏蔽"目前没有瞄准目标"**：那是策略需要自己学会的取舍。
    """
    m = np.zeros(n_options, dtype=np.float32)
    w = unit.weapons
    m[FIRE_GUN] = 1.0 if (w.gun.damage > 0 and w.ammo > 0 and w.gun_cd <= 0) else 0.0
    m[FIRE_MISSILE] = 1.0 if (w.missiles > 0 and w.missile_cd <= 0) else 0.0
    m[FIRE_BOMB] = 1.0 if w.bombs > 0 else 0.0
    if w.flares > 0:
        threatened = False
        if world is not None:
            try:
                threatened = world.nearest_enemy_missile(unit) is not None
            except Exception:
                threatened = False
            if not threatened:
                threatened = any(mem.kind == 'missile' for mem in unit.sensor.memory)
        m[FIRE_FLARE] = 1.0 if threatened else 0.0
    if m.sum() <= 0:
        # 全部不可用时允许"机炮"（空动作），避免整头被屏蔽导致策略无法表达
        m[FIRE_GUN] = 1.0
    return m


def fire_label_for(unit, world):
    """火控头的监督标签：当前是否"应该开机炮"。

    返回 0 表示应当选机炮（命中条件已成立：锥内+射程内确有目标），
    返回 -1 表示无标签（不施加监督）。

    用途：给信号最稀疏的火控头一条稠密通道。实测火控头的策略分布在 S1 与 S4 上
    几乎相同（25.7/24.1/16.0/34.2 vs 26.2/25.6/14.2/33.9），说明它从未学到
    "依观测选武器"；单靠伤害奖励 + 6 头共享 backbone，其梯度被淹没。
    这里用环境状态直接生成标签，无需外部示范数据。"""
    try:
        if unit.weapons.gun.damage <= 0 or unit.weapons.ammo <= 0 or unit.weapons.gun_cd > 0:
            return -1
        return 0 if unit.sensor.gun_target(world, unit) is not None else -1
    except Exception:
        return -1


class PPOAgent:
    """正确实现的 PPO（GAE + 裁剪 + 熵正则），多头离散策略。

    推理路径说明（性能关键）：单条观测经 torch+CUDA 前向的固定开销极大。
    实测 S1 回合（2000 步）中仿真仅 1.7s，而 act_batch 推理累计 53.3s（约 30 倍），
    整个课程训练被推理开销拖垮。因此这里为【采样推理】增加 numpy 快速路径：
    权重从 torch 同步到 numpy，在 CPU 上做批量矩阵乘，推理开销降到可忽略。
    训练（前向+反向+优化器）仍完全由 torch 负责，两者结果一致（有等价性校验）。
    """

    def __init__(self, obs_dim, act_dims, lr=1e-4, gamma=0.99, lam=0.95,
                 clip=0.2, ent_coef=0.03, vf_coef=0.5, epochs=6, batch_size=128,
                 max_grad_norm=0.5, ent_coef_fire=0.002, fire_aux_coef=0.0):
        self.obs_dim = obs_dim
        self.act_dims = act_dims
        self.gamma, self.lam = gamma, lam
        self.clip, self.ent_coef, self.vf_coef = clip, ent_coef, vf_coef
        self.epochs, self.batch_size = epochs, batch_size
        self.max_grad_norm = max_grad_norm
        # 逐动作头熵权重：火控头（索引 1）单独用很小的系数。
        # 实测依据：统一 ent_coef=0.03 时，火控头的策略分布长期停在
        # 机炮27%/导弹24%/炸弹14%/诱饵33% —— 与"什么都没学到"的均匀分布几乎一致，
        # 所有伤害都只是"25% 概率随机撞上机炮"的结果；训练有伤害、确定性评估恒为 0。
        # 原因是火控头的奖励信号最稀疏（要命中才有效），却被同样的熵奖励拉平。
        self.head_ent = [self.ent_coef] * len(act_dims)
        if len(self.head_ent) > 1:
            self.head_ent[FIRE_HEAD_INDEX] = ent_coef_fire
        # 火控头行为克隆辅助损失系数。
        # 实测依据（关键）：火控头的策略分布在 S1 与 S4 上几乎相同
        # （25.7/24.1/16.0/34.2 vs 26.2/25.6/14.2/33.9）→ 与观测无关，等于没学到；
        # 单靠稀疏的伤害奖励 + 6 头共享 backbone，它的梯度一直被淹没。
        # 这里给它一条【稠密且带标签】的通道：只要"机炮锥内射程内确实有目标"，
        # 就用交叉熵直接教它选机炮。标签完全来自环境状态，无需外部示范数据。
        self.fire_aux_coef = float(fire_aux_coef)
        self.model = ActorCritic(obs_dim, act_dims).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self._np_weights = None      # numpy 推理权重缓存
        self.clear_buffers()
        self._sync_numpy()

    # ---------- numpy 快速推理 ----------
    def _sync_numpy(self):
        """把 torch 权重同步到 numpy（每步前向约 0.05ms，远快于 torch 单样本）。"""
        m = self.model
        w0 = m.fc[0].weight.detach().cpu().numpy().T.astype(np.float32)
        b0 = m.fc[0].bias.detach().cpu().numpy().astype(np.float32)
        w1 = m.fc[2].weight.detach().cpu().numpy().T.astype(np.float32)
        b1 = m.fc[2].bias.detach().cpu().numpy().astype(np.float32)
        heads = [(hd.weight.detach().cpu().numpy().T.astype(np.float32),
                  hd.bias.detach().cpu().numpy().astype(np.float32)) for hd in m.heads]
        vw = m.critic.weight.detach().cpu().numpy().T.astype(np.float32)
        vb = m.critic.bias.detach().cpu().numpy().astype(np.float32)
        self._np_weights = (w0, b0, w1, b1, heads, vw, vb)

    def _np_forward(self, X):
        if self._np_weights is None:
            self._sync_numpy()
        w0, b0, w1, b1, heads, vw, vb = self._np_weights
        h = np.maximum(X @ w0 + b0, 0.0)
        h = np.maximum(h @ w1 + b1, 0.0)
        logits = [h @ w + b for (w, b) in heads]
        val = (h @ vw + vb).reshape(-1)
        return logits, val

    def clear_buffers(self):
        self.b_obs = []
        self.b_acts = []       # list of (head_idx list)
        self.b_logp = []
        self.b_val = []
        self.b_rew = []
        self.b_done = []
        self.b_fire_lbl = []   # 火控行为克隆标签（-1=无标签，0机炮=有效开火，1..=其他）

    @staticmethod
    def _sanitize(a):
        """把非有限值（NaN/Inf）替换为 0。

        实测踩坑：训练中出现 NaN 发散时，nan 会先污染观测缓冲，再经前向扩散到
        所有 logits，最终 Categorical 抛 "found invalid values"。在入口处净化
        可以把单点污染挡在门外，避免整轮训练报废。"""
        return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    def act(self, obs, deterministic=False):
        acts, logps, vals = self.act_batch([obs], deterministic=deterministic)
        if not acts:
            return Action(), 0.0, 0.0
        return acts[0], logps[0], vals[0]

    def act_batch(self, obs_list, deterministic=False, masks=None):
        """对 N 个观测批量采样（numpy 快速路径），返回 (actions, logps, values)。

        masks: 可选，形状 (N, 6) 的动作掩码数组（1=可选，0=屏蔽）。目前只使用
       第 1 列（火控）——屏蔽备弹为 0 / 冷却中的开火选项。被屏蔽项在 softmax 前
       置为 -1e9，因此永远不会被采样到；logp 同样基于掩码后的分布计算，保证一致。
        """
        n = len(obs_list)
        if n == 0:
            return [], [], []
        X = np.asarray(obs_list, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        logits_all, vals = self._np_forward(X)
        acts = [[] for _ in range(n)]
        logps = np.zeros(n, dtype=np.float32)
        for h_idx, logits in enumerate(logits_all):
            lg = np.array(logits, dtype=np.float32, copy=True)
            if masks is not None:
                mk = np.asarray(masks, dtype=np.float32)
                if mk.ndim == 1:
                    mk = mk[None, :]
                if mk.shape[0] == n and h_idx < mk.shape[1]:
                    col = mk[:, h_idx]
                    # 用 -1e4 而非 -1e9：既保证几乎不可能采样到，又避免
                    # 极端 logits 在后续（如辅助损失的梯度）产生数值问题
                    lg[col <= 0] = -1e4
            # 数值稳定的 softmax
            z = lg - lg.max(axis=1, keepdims=True)
            e = np.exp(z)
            p = e / e.sum(axis=1, keepdims=True)
            if deterministic:
                a = p.argmax(axis=1)
            else:
                c = np.cumsum(p, axis=1)
                u = np.random.rand(n, 1).astype(np.float32)
                a = (c < u).sum(axis=1)
                a = np.minimum(a, p.shape[1] - 1)
            logps += np.log(np.maximum(p[np.arange(n), a], 1e-12))
            for i in range(n):
                acts[i].append(int(a[i]))
        return [Action(*x) for x in acts], [float(v) for v in logps], [float(v) for v in vals]

    def remember(self, obs, action, logp, val, rew, done, fire_label=-1):
        self.b_obs.append(np.asarray(obs, dtype=np.float32))
        self.b_acts.append([action.head, action.fire, action.throttle, action.turn,
                            action.target_select, action.share_target])
        self.b_logp.append(logp)
        self.b_val.append(val)
        self.b_rew.append(rew)
        self.b_done.append(1.0 if done else 0.0)
        self.b_fire_lbl.append(int(fire_label))

    def compute_returns(self, last_val):
        """GAE(lambda) 优势估计。"""
        n = len(self.b_rew)
        adv = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            next_val = last_val if t == n - 1 else self.b_val[t + 1]
            delta = self.b_rew[t] + self.gamma * next_val * (1 - self.b_done[t]) - self.b_val[t]
            gae = delta + self.gamma * self.lam * (1 - self.b_done[t]) * gae
            adv[t] = gae
        returns = np.asarray(self.b_val, dtype=np.float32) + adv
        return returns, adv

    def update(self, last_val):
        if len(self.b_obs) < 4:
            return 0.0
        returns, adv = self.compute_returns(last_val)
        adv = np.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        obs_t = torch.FloatTensor(self._sanitize(np.stack(self.b_obs))).to(DEVICE)
        acts_t = [torch.LongTensor([a[i] for a in self.b_acts]).to(DEVICE) for i in range(len(self.act_dims))]
        logp_old = torch.FloatTensor(np.nan_to_num(np.asarray(self.b_logp, dtype=np.float32))).to(DEVICE)
        ret_t = torch.FloatTensor(np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)).to(DEVICE)
        adv_t = torch.FloatTensor(adv).to(DEVICE)

        n = len(self.b_obs)
        idx = np.arange(n)
        total_loss = 0.0
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.batch_size):
                mb = idx[start:start + self.batch_size]
                mb = torch.LongTensor(mb.tolist()).to(DEVICE)
                logits_list, values = self.model(obs_t[mb])
                # 前向输出健康检查：若本批出现非有限值，说明网络已被污染，
                # 直接跳过这一批（不更新参数），避免训练整体崩掉
                if not all(torch.isfinite(l).all() for l in logits_list) or \
                        not torch.isfinite(values).all():
                    self.optimizer.zero_grad()
                    continue

                logp_new = torch.zeros(len(mb), device=DEVICE)
                entropy_term = torch.zeros(len(mb), device=DEVICE)
                for i, (logits, acts) in enumerate(zip(logits_list, acts_t)):
                    dist = Categorical(logits=logits)
                    logp_new = logp_new + dist.log_prob(acts[mb])
                    # 逐头熵权重（每个系数就是该头原来的 ent_coef；
                    #  火控头单独取小值，避免其策略被熵奖励拉平成"什么都没学到"）
                    entropy_term = entropy_term + self.head_ent[i] * dist.entropy()
                ratio = torch.exp(logp_new - logp_old[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t[mb]
                # 熵是"奖励"项，因此从损失中减去（等价于原来 -ent_coef*entropy 的逐头版本）
                policy_loss = -torch.min(surr1, surr2).mean() - entropy_term.mean()
                value_loss = 0.5 * (ret_t[mb] - values.squeeze(-1)).pow(2).mean()
                loss = policy_loss + self.vf_coef * value_loss
                # ---- 火控头行为克隆辅助损失 ----
                # 只对"确实有有效开火机会"的样本施加监督（标签由环境给出），
                # 直接教火控头在能打的时候选机炮。这是给最稀疏的那个头补一条稠密通道。
                if self.fire_aux_coef > 0 and self.b_fire_lbl:
                    lbl = torch.LongTensor([self.b_fire_lbl[i] for i in mb.tolist()]).to(DEVICE)
                    mask = lbl >= 0
                    if bool(mask.any()):
                        logits_fire = logits_list[FIRE_HEAD_INDEX]
                        # 夹紧 logits，避免极端值经 cross_entropy 产生 NaN（实测踩过）
                        lf = torch.clamp(logits_fire[mask], -20.0, 20.0)
                        aux = torch.nn.functional.cross_entropy(lf, lbl[mask], reduction='mean')
                        if torch.isfinite(aux):
                            loss = loss + self.fire_aux_coef * aux
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                # 数值健康检查：出现 NaN/Inf 时跳过本步优化器更新，
                # 避免参数被污染后整轮训练发散（实测出现过 loss NaN 导致全部 logits NaN）
                bad = False
                for p in self.model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        bad = True
                        break
                if not bad and torch.isfinite(loss):
                    self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += float(loss.item()) if torch.isfinite(loss) else 0.0
        self.clear_buffers()
        self._sync_numpy()          # 训练后同步 numpy 推理权重
        return total_loss / (self.epochs * max(1, n // self.batch_size))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'model': self.model.state_dict(),
                    'obs_dim': self.obs_dim, 'act_dims': self.act_dims,
                    'obs_version': OBS_VERSION}, path)

    def load(self, path, strict_dims=True):
        """加载检查点。维度/版本不匹配时给出中文可读诊断，而不是裸抛 state_dict 异常。"""
        data = torch.load(path, map_location=DEVICE, weights_only=False)
        ck_obs = data.get('obs_dim')
        ck_acts = data.get('act_dims')
        ck_ver = data.get('obs_version')
        if strict_dims and (ck_obs != self.obs_dim or (ck_acts and list(ck_acts) != list(self.act_dims))):
            raise ValueError(
                f"检查点与当前代码不兼容：{os.path.basename(path)}\n"
                f"  检查点 obs_dim={ck_obs} act_dims={ck_acts} obs_version={ck_ver}\n"
                f"  当前   obs_dim={self.obs_dim} act_dims={self.act_dims} obs_version={OBS_VERSION}\n"
                f"  观测布局变更后旧模型无法直接加载，请用对应版本的代码或重新训练。")
        self.model.load_state_dict(data['model'])
        self.model.eval()
        self._np_weights = None     # 使 numpy 推理缓存失效，下次推理时重建
        self._sync_numpy()
        return data


class AirCombatEnv:
    """训练环境：封装 World，红蓝各一个受命令控制器，负责奖励塑形与回合管理。"""

    def __init__(self, plane_type='standard', render=False, max_steps=2400,
                 red_ai_factory=None, blue_ai_factory=None,
                 layout='duel', map_variants=4, scale=1.0):
        self.plane_type = plane_type
        self.render = render
        self.max_steps = max_steps
        self.world = World(plane_type, red_ai_factory, blue_ai_factory,
                           layout=layout, map_variants=map_variants, scale=scale)
        self.red_controller = None
        self.blue_controller = None
        if render:
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            self.camera = Camera(SCREEN_W, SCREEN_H)
            self.hud = HUD()
            self.minimap = Minimap()
        else:
            self.screen = None
            self.camera = None
        self.steps = 0

    def reset(self):
        self.world.reset()
        self.steps = 0
        return self._obs()

    def step(self, actions: Dict[str, Action]):
        # 注入命令
        for a in self.world.aircraft.alive_list():
            if isinstance(a.controller, CommandedController):
                a.controller.next_action = actions.get(a.team, Action())
        self.world.step()
        self.steps += 1
        rewards = self._rewards()
        done = self._done()
        if self.render:
            self._render()
        return self._obs(), rewards, done, {}

    def _obs(self):
        obs = {}
        for a in self.world.aircraft.alive_list():
            obs[a.team] = a.sensor.build_observation(self.world, a)
        return obs

    def _rewards(self):
        rew = {'red': 0.0, 'blue': 0.0}
        # 存活
        for a in self.world.aircraft.alive_list():
            rew[a.team] += 0.01
        # 伤害
        for (team, uid), dmg in self.world.damage_credit.items():
            rew[team] += 0.03 * dmg
        # 摧毁：击毁者由本帧伤害积分近似（通常即最后一击者）
        for unit in self.world.kills:
            val = unit_value(unit)
            best_team, best_dmg = None, 0.0
            for (team, uid), dmg in self.world.damage_credit.items():
                if dmg > best_dmg:
                    best_team, best_dmg = team, dmg
            if best_team:
                if isinstance(unit, Aircraft):
                    rew[best_team] += 30.0
                else:
                    rew[best_team] += val * 0.5
                rew['red' if best_team == 'blue' else 'blue'] -= val * 0.1
        # 己方飞机被击毁
        for unit in self.world.kills:
            if isinstance(unit, Aircraft):
                rew[unit.team] -= 50.0
        # 诱饵成功
        for team in self.world.flare_success:
            rew[team] += 10.0
        # 基地被摧毁
        lost = self.world.base_destroyed()
        if lost:
            rew[lost] -= 300.0
            rew['red' if lost == 'blue' else 'blue'] += 300.0
        return rew

    def _done(self):
        if self.world.base_destroyed() is not None:
            return True
        if len(self.world.aircraft.alive_list()) == 0:
            return True
        if self.steps >= self.max_steps:
            return True
        return False

    def _render(self):
        self.screen.fill(WHITE)
        self.world_objs()
        self.camera.draw_grid(self.screen)
        self.camera.draw_terrain(self.screen, self.world)
        for pool in (self.world.buildings, self.world.ground, self.world.aircraft,
                     self.world.bullets, self.world.bombs, self.world.missiles):
            for e in pool.alive_list():
                e.draw(self.screen, self.camera)
        self.hud.draw(self.screen, self.world)
        self.minimap.draw(self.screen, self.world)
        pygame.display.flip()

    def world_objs(self):
        pass

    def close(self):
        if self.render:
            pygame.quit()


# ======================================================================
# 6.5 课程学习（Curriculum）：分阶段训练环境
#      S1 定点 → S2 动点 → S3 多目标 → S4 复杂攻防 → S5 规则AI对抗
#      → S6 规则+神经网络混合对抗 → S7 多环境评估
# ======================================================================
CTRL_NEURAL = 'neural'   # 由 PPO 控制（CommandedController）
CTRL_RULE = 'rule'       # 规则专家 AI
CTRL_DUMMY = 'dummy'     # 直线匀速靶机（不还手）
CTRL_STATIC = 'static'   # 静止靶机（不还手）
CTRL_MANUAL = 'manual'   # 手动


@dataclass
class PassCriteria:
    """阶段达成条件：滑动窗口内达标即提前晋级下一阶段（None 字段=不要求）。"""
    window: int = 20                 # 统计最近 N 回合
    clear_rate: Optional[float] = None   # 清场率阈值（0~1）
    win_rate: Optional[float] = None     # 胜率阈值（清场或红奖励>蓝奖励）
    min_avg_reward: Optional[float] = None  # 平均奖励下限
    min_avg_kills: Optional[float] = None   # 平均击杀下限
    min_avg_hunt: Optional[float] = None    # 平均索敌数下限（每回合首次发现目标数）
    min_avg_damage: Optional[float] = None  # 平均造成伤害下限（防止“只搜索不输出”晋级）


@dataclass
class StageSpec:
    name: str
    red_team: List[Tuple[str, str]]        # [(机型, 控制类型), ...]
    blue_team: List[Tuple[str, str]]       # [(机型, 控制类型), ...]
    layout: str = 'range'                  # 'range'=靶场 / 'duel'=标准战场
    extra_ground: List[Tuple[str, str, str, bool]] = field(default_factory=list)
    #                                        [(机型, team, 行为, 数量), ...] 行为: 'static'|'patrol'|'active'
    extra_air: List[Tuple[str, str, str, bool]] = field(default_factory=list)
    #                                        [(机型, team, 控制类型, 数量)] 控制类型: CTRL_STATIC/CTRL_DUMMY/CTRL_RULE
    max_steps: int = 2400
    cleared_bonus: float = 0.0             # 清空所有目标/全歼奖励
    desc: str = ''
    formation: List[Tuple[float, float]] = field(default_factory=list)
    #                                      红方出生偏移 [(dx,dy),...]；空=默认横排，蓝方自动镜像
    red_spawn: Tuple[float, float] = (0.2, 0.5)    # 阵营中心（世界比例）
    blue_spawn: Tuple[float, float] = (0.8, 0.5)
    pass_criteria: Optional[PassCriteria] = None   # 阶段达成条件（达标提前晋级；None=跑满）
    kill_mult: float = 2.0                 # 击杀奖励倍率（高击杀回报）
    hunt_bonus: float = 0.0                # 索敌奖励：首次发现一个目标的一次性奖励
    lock_bonus: float = 0.0                # 锁敌奖励：每帧保持锁定 +lock_bonus
    lock_acquire_bonus: float = 0.0        # 新锁定一次性奖励
    damage_coef: float = 0.0               # 伤害奖励系数（每点伤害）
    approach_coef: float = 0.0             # 接近引导：向最近目标推进时按改进量给奖励（势能塑形）
    combat_approach_coef: float = 0.0      # 战斗抵近：距最近目标越近每帧奖励越高（稠密）
    combat_range: float = 300.0            # 抵近奖励的门控距离（进入该距离后不再给分）
    aim_fire_bonus: float = 0.0            # 有效开火奖励：机炮命中条件成立时开火即得分
    close_bonus: float = 0.0               # 线性抵近奖励（无门控上限），用于空对空逼近
    map_variants: int = 1                  # 同类型地图变体数（每回合随机一个）
    scale: float = 1.0                     # 交战规模（单位/目标数量倍率）
    aggression_modes: List[float] = field(default_factory=lambda: [0.5])
    #                                      激进度预设档位（每回合随机一个给神经飞机）
    episodes: Optional[int] = None         # 本阶段回合数上限（None=用全局 episodes_per_stage）


def _ctrl_factory(kind, team):
    """控制类型 → 控制器工厂。"""
    if kind == CTRL_NEURAL:
        return lambda: CommandedController()
    if kind == CTRL_RULE:
        return lambda: RuleAI(team)
    if kind == CTRL_DUMMY:
        return lambda: DummyAI()
    if kind == CTRL_STATIC:
        return lambda: StationaryAI()
    if kind == CTRL_MANUAL:
        return lambda: ManualController()
    raise ValueError(f"未知控制类型: {kind}")


def curriculum_stages(plane_type):
    """按机型生成 7 阶段课程。机型武器差异自适应：
    - 无导弹（missiles==0）→ 空中目标仍可机炮/炸弹处理，阶段不变
    - 无机炮且无炸弹（missile_plane）→ 地面阶段改用空中靶机（导弹只打空中）"""
    spec = AIRCRAFT_SPECS[plane_type]
    has_gun = spec.gun.damage > 0 and spec.gun.ammo > 0
    has_bomb = spec.bomb.count > 0
    has_missile = spec.missile.count > 0

    # 地面目标适配：无机炮无炸弹的机型（纯导弹机）打不了地面 → 用空中靶机
    def ground_targets(team='blue', n_static=2, n_patrol=1):
        if has_gun or has_bomb:
            out = []
            if n_static:
                out.append(('tank', team, 'static', n_static))
            if n_patrol:
                out.append(('tank', team, 'patrol', n_patrol))
            return out
        return []   # 纯导弹机：全部换成空中靶机

    def air_targets(team='blue', n_static=0, n_dummy=1):
        out = []
        if n_static:
            out.append(('standard', team, CTRL_STATIC, n_static))
        if n_dummy:
            # 用 standard（巡航 2.0）而非 speedy（巡航 4.0）做直线靶机：
            # 目标速度高于本机时，在 9°机炮锥内几乎无法取得提前量，目标阶段不可学。
            out.append(('standard', team, CTRL_DUMMY, n_dummy))
        return out

    stages = [
        StageSpec(
            name='S0_索敌', desc='远处多架靶机：学会主动转头搜索/雷达共享追踪（先发现再交战）',
            red_team=[(plane_type, CTRL_NEURAL)], blue_team=[],
            layout='range',
            extra_ground=ground_targets('blue', 1, 0),
            extra_air=air_targets('blue', 1, 2),
            max_steps=1500, cleared_bonus=50, episodes=200,
            hunt_bonus=25, lock_bonus=0.05, lock_acquire_bonus=25,
            damage_coef=1.0, approach_coef=2.0, combat_approach_coef=2.0, aim_fire_bonus=0.3, close_bonus=1.5,
            pass_criteria=PassCriteria(min_avg_hunt=4, min_avg_damage=15, clear_rate=0.4, window=20),
        ),
        StageSpec(
            name='S1_定点目标', desc='静止地面目标 + 悬停靶机：学会瞄准与射击',
            red_team=[(plane_type, CTRL_NEURAL)], blue_team=[],
            layout='range',
            extra_ground=ground_targets('blue', 2, 0) + [('gun_aa_vehicle', 'blue', 'static', 1)],
            extra_air=air_targets('blue', 1, 0) if not (has_gun or has_bomb) else air_targets('blue', 1, 0),
            max_steps=2000, cleared_bonus=150, episodes=300,
            hunt_bonus=15, lock_bonus=0.05, lock_acquire_bonus=15,
            damage_coef=1.0, approach_coef=2.0, combat_approach_coef=2.0, aim_fire_bonus=0.3, close_bonus=1.5,
            pass_criteria=PassCriteria(clear_rate=0.6, min_avg_kills=2, min_avg_damage=40, window=20),
        ),
        StageSpec(
            name='S2_动点目标', desc='巡游坦克 + 直线匀速靶机：练习提前量',
            red_team=[(plane_type, CTRL_NEURAL)], blue_team=[],
            layout='range',
            extra_ground=ground_targets('blue', 0, 2),
            extra_air=air_targets('blue', 0, 1),
            max_steps=2000, cleared_bonus=150, episodes=300,
            hunt_bonus=15, lock_bonus=0.05, lock_acquire_bonus=15,
            damage_coef=1.0, approach_coef=2.0, combat_approach_coef=2.0, aim_fire_bonus=0.3, close_bonus=1.5,
            pass_criteria=PassCriteria(clear_rate=0.5, min_avg_damage=30, window=20),
        ),
        StageSpec(
            name='S3_多目标', desc='5 个混合目标：学会目标分配与连续接战',
            red_team=[(plane_type, CTRL_NEURAL)], blue_team=[],
            layout='range',
            extra_ground=ground_targets('blue', 3, 1),
            extra_air=air_targets('blue', 0, 2),
            max_steps=2600, cleared_bonus=200, episodes=300,
            hunt_bonus=15, lock_bonus=0.05, lock_acquire_bonus=15,
            damage_coef=1.0, approach_coef=2.0, combat_approach_coef=2.0, aim_fire_bonus=0.3, close_bonus=1.5,
            pass_criteria=PassCriteria(clear_rate=0.4, min_avg_damage=40, window=20),
        ),
        StageSpec(
            name='S4_复杂攻防', desc='敌方 2 机还手 + 防空/雷达：攻防一体',
            red_team=[(plane_type, CTRL_NEURAL)], blue_team=[('standard', CTRL_RULE), ('speedy', CTRL_RULE)],
            layout='duel', max_steps=2400, cleared_bonus=300,
            damage_coef=2.0, hunt_bonus=10, lock_bonus=0.05, lock_acquire_bonus=10, combat_approach_coef=2.0, aim_fire_bonus=0.3, close_bonus=1.5,
            pass_criteria=PassCriteria(win_rate=0.5, min_avg_damage=60, window=20),
            map_variants=4,
        ),
        StageSpec(
            name='S5_规则AI对抗', desc='1v1 对决规则专家 AI',
            red_team=[(plane_type, CTRL_NEURAL)], blue_team=[('standard', CTRL_RULE)],
            layout='duel', max_steps=2400, cleared_bonus=200,
            damage_coef=2.0, hunt_bonus=10, lock_bonus=0.05, lock_acquire_bonus=10, combat_approach_coef=2.0, aim_fire_bonus=0.3, close_bonus=1.5,
            pass_criteria=PassCriteria(win_rate=0.6, min_avg_damage=60, window=20),
            map_variants=4,
        ),
        StageSpec(
            name='S6_规则+神经网络混合', desc='红蓝各 1 神经 + 1 规则僚机（自博弈）',
            red_team=[(plane_type, CTRL_NEURAL), ('dogfighter', CTRL_RULE)],
            blue_team=[('standard', CTRL_NEURAL), ('dogfighter', CTRL_RULE)],
            layout='duel', max_steps=2400, cleared_bonus=300, scale=1.5,
            damage_coef=2.0, hunt_bonus=10, lock_bonus=0.05, lock_acquire_bonus=10, combat_approach_coef=2.0, aim_fire_bonus=0.3, close_bonus=1.5,
            pass_criteria=PassCriteria(win_rate=0.5, window=20),
            map_variants=4,
        ),
        StageSpec(
            name='S7_多环境对抗评估', desc='在所有阶段环境轮转评估',
            red_team=[(plane_type, CTRL_NEURAL)], blue_team=[('standard', CTRL_RULE)],
            layout='duel', max_steps=2400, cleared_bonus=0,
            damage_coef=2.0, hunt_bonus=10, lock_bonus=0.05, lock_acquire_bonus=10, combat_approach_coef=2.0, aim_fire_bonus=0.3, close_bonus=1.5,
            map_variants=4,
        ),
    ]
    # 全部阶段启用三档激进度混合训练（每回合随机：0=保命 / 0.5=均衡 / 1=激进）
    for st in stages:
        st.aggression_modes = [0.0, 0.5, 1.0]
    return stages


class CurriculumEnv:
    """课程环境：支持多机编队、混合控制器、额外目标、阶段化奖励。"""

    def __init__(self, stage: StageSpec, render=False, is_eval=False):
        self.stage = stage
        self.is_curriculum = True
        self.is_eval = is_eval
        self.render = render
        self.max_steps = stage.max_steps
        self.steps = 0
        self._objectives = []
        self.hunt_count = {'red': 0, 'blue': 0}   # 本回合索敌计数（发现/跟踪目标次数）
        self._prev_lock = {}                      # 上帧锁定状态（检测新锁定）
        self.world = self._build_world()
        if render:
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            self.camera = Camera(SCREEN_W, SCREEN_H)
        else:
            self.screen = None

    def _build_world(self):
        s = self.stage
        red_slots = [(pt, _ctrl_factory(k, 'red')) for pt, k in s.red_team]
        blue_slots = [(pt, _ctrl_factory(k, 'blue')) for pt, k in s.blue_team]
        world = World(s.red_team[0][0] if s.red_team else 'standard',
                      red_slots=red_slots, blue_slots=blue_slots, layout=s.layout,
                      red_offsets=s.formation or None,
                      red_spawn=s.red_spawn, blue_spawn=s.blue_spawn,
                      map_variants=s.map_variants, scale=s.scale)
        self._spawn_objectives(world)
        world._rebuild_spatial()
        return world

    def _spawn_objectives(self, world):
        """生成额外目标（靶机/地面靶标）。reset 时也会调用以重建目标。"""
        s = self.stage
        self._objectives = []
        # 额外地面目标（靠近红方一侧，降低早期探索难度；数量随 scale 增长）
        for utype, team, behav, cnt in s.extra_ground:
            for i in range(max(1, int(cnt * s.scale))):
                gx = random.uniform(WORLD_W * 0.40, WORLD_W * 0.70)
                gy = random.uniform(WORLD_H * 0.35, WORLD_H * 0.65)
                g = world.ground.spawn(gx, gy, utype, team)
                g.passive = (behav != 'active')
                g.patrol = (behav == 'patrol')
                self._objectives.append(g)
        # 额外空中目标（数量随 scale 增长）
        for ptype, team, ctrl_kind, cnt in s.extra_air:
            for i in range(max(1, int(cnt * s.scale))):
                ax = random.uniform(WORLD_W * 0.45, WORLD_W * 0.75)
                ay = random.uniform(WORLD_H * 0.35, WORLD_H * 0.65)
                ctrl = _ctrl_factory(ctrl_kind, team)()
                a = world.aircraft.spawn(ax, ay, random.uniform(0, 360), ptype, team, ctrl)
                a.no_fire = True
                a.stationary = (ctrl_kind == CTRL_STATIC)
                self._objectives.append(a)
        world._rebuild_spatial()
        return world

    # ---------- 观测/动作（每个神经槽位一个观测/动作） ----------
    def _neural_aircraft(self, team):
        kinds = {k for _, k in self.stage.red_team if k == CTRL_NEURAL} if team == 'red' else \
                {k for _, k in self.stage.blue_team if k == CTRL_NEURAL}
        if not kinds:
            return []
        return [a for a in self.world.aircraft_list_of(team)
                if isinstance(a.controller, CommandedController)]

    def _obs_team(self, team):
        out = []
        for a in self._neural_aircraft(team):
            if a.alive:
                out.append(a.sensor.build_observation(self.world, a))
            else:
                out.append(np.zeros(OBS_DIM, dtype=np.float32))
        return out

    def reset(self):
        self.world.reset()
        self._spawn_objectives(self.world)   # 重建额外目标（reset 会清空实体池）
        self.world._rebuild_spatial()
        self.steps = 0
        self.hunt_count = {'red': 0, 'blue': 0}
        self._prev_lock = {}
        # 每回合随机激进度档位（预设模式：低/中/高）
        for team in ('red', 'blue'):
            for a in self._neural_aircraft(team):
                a.aggressiveness = random.choice(self.stage.aggression_modes)
        return {'red': self._obs_team('red'), 'blue': self._obs_team('blue')}

    def step(self, actions):
        """actions: {'red': [Action...], 'blue': [Action...]} 按神经槽位顺序注入。"""
        for team in ('red', 'blue'):
            acts = actions.get(team) or []
            nns = self._neural_aircraft(team)
            for a, act in zip(nns, acts):
                if a.alive:
                    a.controller.next_action = act
        self.world.step()
        self.steps += 1
        rewards = self._rewards()
        done = self._done()
        if self.render:
            self._render()
        return {'red': self._obs_team('red'), 'blue': self._obs_team('blue')}, rewards, done, {}

    # ---------- 奖励（队伍级，阶段自适应） ----------
    # 设计原则（v3 重构）：
    #   1) 战斗输出必须压倒“看见目标”。原实现 hunt_bonus 每帧重复发放（记忆池每帧刷新），
    #      实测索敌奖励占目标阶段总奖励的 97%~100%，而伤害/击杀恒为 0 —— 盘旋搜索成为
    #      全局最优，模型学会“看见”却没学会“击落”。现在索敌改为【首次发现一次性】奖励，
    #      并显式加入 damage_coef 与提高的相对权重，让开火成为唯一的高收益路径。
    #   2) 用势能塑形（approach）替代持续索敌奖励做探索引导：只有“比历史最近距离更近”
    #      才给分，盘旋/远离不再得分，从而给出朝目标推进的梯度而非停留的梯度。
    def _rewards(self):
        rew = {'red': 0.0, 'blue': 0.0}
        w = self.world
        uid_agg = {a.uid: a.aggressiveness for a in w.aircraft.all()}
        # ---- 战斗抵近奖励与势能（对"目标 + 敌方飞机"统一生效）----
        # ⚠ 实测教训：原先 approach_coef / combat_approach_coef 只考虑
        # `_objectives_alive()`（地面靶标），**空中目标完全没有抵近奖励**。
        # 结果在对称空战竞技场里神经网络伤害恒为 0、双方满血 —— 它根本没被教过
        # "如何追上空中的敌机"。这里把敌方飞机也纳入同一套抵近/势能信号。
        if self.stage.approach_coef > 0 or self.stage.combat_approach_coef > 0 \
                or self.stage.close_bonus > 0:
            ac = self.stage.approach_coef
            cc = self.stage.combat_approach_coef
            cb = self.stage.close_bonus
            gate = self.stage.combat_range
            span = max(1.0, DETECT_RANGE - gate)
            objs = self._objectives_alive()
            for r in w.aircraft_list_of('red'):
                targets = list(objs) + list(w.aircraft_list_of('blue'))
                if not targets:
                    break
                best = r.sensor._approach_best
                live_ids = set()
                delta = 0.0
                cur_best = None
                for o in targets:
                    oid = id(o)
                    live_ids.add(oid)
                    d = math.hypot(o.x - r.x, o.y - r.y)
                    prev = best.get(oid)
                    if prev is not None:
                        delta += prev - d
                    best[oid] = d
                    if cur_best is None or d < cur_best:
                        cur_best = d
                if len(best) > len(live_ids):
                    for kk in [kk for kk in best if kk not in live_ids]:
                        del best[kk]
                # 势能差分（对称有界）
                if ac > 0:
                    rew['red'] += ac * delta / 100.0
                if cur_best is None:
                    continue
                # 门控抵近：射程外才给分，进入射程后收益只能来自开火命中
                if cc > 0 and gate < cur_best < DETECT_RANGE:
                    rew['red'] += cc * (1.0 - (cur_best - gate) / span)
                # 线性抵近（无门控上限），用于空对空逼近
                if cb > 0 and cur_best < DETECT_RANGE:
                    rew['red'] += cb * (1.0 - cur_best / DETECT_RANGE)
        # 存活：极小的生存维持项，不因“本帧未开火”惩罚（原实现对激进档每帧扣分，
        # 会与“勇敢接战”相互抵消，造成高激进度智能体长期负回报）
        for team in ('red', 'blue'):
            for a in w.aircraft_list_of(team):
                rew[team] += 0.04 if uid_agg.get(a.uid, 0.5) < 0.3 else 0.01
        # 伤害：主要战斗信号（每点伤害）
        for (team, uid), dmg in w.damage_credit.items():
            ag = uid_agg.get(uid, 0.5)
            mult = 0.7 if ag < 0.3 else (1.0 if ag < 0.7 else 1.3)
            rew[team] += self.stage.damage_coef * mult * dmg
        # 摧毁：按价值 × 激进度倍率，击杀敌机额外重奖
        for unit in w.kills:
            val = unit_value(unit)
            best_team, best_dmg, best_uid = None, 0.0, None
            for (team, uid), dmg in w.damage_credit.items():
                if dmg > best_dmg:
                    best_team, best_dmg, best_uid = team, dmg, uid
            if best_team and best_team != unit.team:
                ag = uid_agg.get(best_uid, 0.5)
                km = 1.0 if ag < 0.3 else (self.stage.kill_mult if ag < 0.7 else self.stage.kill_mult * 1.5)
                rew[best_team] += val * km
                if isinstance(unit, Aircraft):
                    rew[best_team] += 50.0
                rew[unit.team] -= val * 0.4
        # 己方飞机被击毁
        for unit in w.kills:
            if isinstance(unit, Aircraft):
                rew[unit.team] -= 50.0
        # 诱饵成功
        for team in w.flare_success:
            rew[team] += 10.0
        # 基地被摧毁
        lost = w.base_destroyed()
        if lost:
            rew[lost] -= 300.0
            rew['red' if lost == 'blue' else 'blue'] += 300.0
        # 清场奖励
        if self._cleared():
            rew['red'] += self.stage.cleared_bonus
        # ---- 有效开火奖励（火控头的稠密信号）----
        # 实测依据：火控头唯一的梯度来自稀疏的伤害奖励，被其余动作头共享的 advantage
        # 噪声淹没 —— 其策略分布在 S1 与 S4 上几乎相同（25.7/24.1/16.0/34.2 vs
        # 26.2/25.6/14.2/33.9），即与观测无关，等于什么都没学到；所有伤害都只是
        # "约 25% 概率随机撞上机炮"。这里对"机炮命中条件已成立时开火"给即时奖励，
        # 让火控头能直接把"开火"这一动作与收益联系起来。
        if self.stage.aim_fire_bonus > 0:
            for team in w.gun_fired:
                rew[team] += self.stage.aim_fire_bonus
        # ---- 索敌：首次发现某目标的一次性奖励（按目标去重，不再每帧重复计分） ----
        if self.stage.hunt_bonus > 0:
            hb = self.stage.hunt_bonus
            for team in ('red', 'blue'):
                for a in w.aircraft_list_of(team):
                    seen = a.sensor._seen_ids
                    for m in a.sensor.memory:
                        if m.entity_id not in seen:
                            seen.add(m.entity_id)
                            rew[team] += hb
                            self.hunt_count[team] += 1
        # ---- 锁敌：保持锁定每帧小额奖励 + 【每个目标一次】的首次锁定奖励 ----
        # 注意：acquire 奖励必须按目标去重。否则只要反复切换锁定目标（LOCK_NEXT）就能
        # 无限刷分——实测 4 个目标在 400 帧内可触发 62 次“新锁定”，单这一项就占到总奖励 93%。
        if self.stage.lock_bonus > 0 or self.stage.lock_acquire_bonus > 0:
            for team in ('red', 'blue'):
                for a in w.aircraft_list_of(team):
                    tgt = a.lock_target
                    if tgt is not None and tgt.alive:
                        rew[team] += self.stage.lock_bonus
                        if self.stage.lock_acquire_bonus > 0:
                            key = id(tgt)
                            if key not in a.sensor._lock_ids:
                                a.sensor._lock_ids.add(key)
                                rew[team] += self.stage.lock_acquire_bonus
        return rew

    def _objectives_alive(self):
        return [o for o in self._objectives if o.alive]

    def _cleared(self):
        """目标阶段：所有目标被摧毁；对抗阶段：敌方编队全灭。

        修正：原实现 `if obs: return len(obs)==0` 是死分支，且当阶段既没有额外目标、
        蓝队也为空时，reset 后立刻返回 True，回合瞬间结束。现在无目标定义时不做清场判定。"""
        if self._objectives:
            return len(self._objectives_alive()) == 0
        if not self.stage.blue_team:
            return False          # 无蓝队且无目标 → 不存在“清场”概念
        return len(self.world.aircraft_list_of('blue')) == 0

    def _done(self):
        if self.world.base_destroyed() is not None:
            return True
        if self._cleared():
            return True
        red_alive = len(self.world.aircraft_list_of('red'))
        if red_alive == 0:
            return True
        if self.steps >= self.max_steps:
            return True
        return False

    def _render(self):
        if not self.render:
            return
        self.screen.fill(WHITE)
        for pool in (self.world.buildings, self.world.ground, self.world.aircraft,
                     self.world.bullets, self.world.bombs, self.world.missiles):
            for e in pool.alive_list():
                e.draw(self.screen, self.camera)
        pygame.display.flip()

    def close(self):
        if self.render:
            pygame.quit()


# ======================================================================
# 7. 渲染与 UI：相机 / HUD / 小地图 / 菜单 / 训练视图
# ======================================================================
class Camera:
    def __init__(self, sw, sh):
        self.sw, self.sh = sw, sh
        self.x, self.y = WORLD_W / 2, WORLD_H / 2
        self.zoom = 0.8
        self.dragging = False
        self.last_mouse = (0, 0)
        self.follow = None

    def world_to_screen(self, wx, wy):
        sx = (wx - self.x) * self.zoom + self.sw // 2
        sy = (wy - self.y) * self.zoom + self.sh // 2
        return int(sx), int(sy)

    def screen_to_world(self, sx, sy):
        return (sx - self.sw // 2) / self.zoom + self.x, (sy - self.sh // 2) / self.zoom + self.y

    def handle(self, event):
        if event.type == MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            wx, wy = self.screen_to_world(mx, my)
            self.zoom *= 1.1 if event.y > 0 else 0.9
            self.zoom = max(0.3, min(3.0, self.zoom))
            self.x = wx - (mx - self.sw // 2) / self.zoom
            self.y = wy - (my - self.sh // 2) / self.zoom
        elif event.type == MOUSEBUTTONDOWN and event.button == 2:
            self.dragging, self.last_mouse = True, event.pos
        elif event.type == MOUSEBUTTONUP and event.button == 2:
            self.dragging = False
        elif event.type == MOUSEMOTION and self.dragging:
            dx = event.pos[0] - self.last_mouse[0]
            dy = event.pos[1] - self.last_mouse[1]
            self.x -= dx / self.zoom
            self.y -= dy / self.zoom
            self.last_mouse = event.pos
        elif event.type == KEYDOWN:
            if event.key == K_LEFT: self.x -= 80 / self.zoom
            elif event.key == K_RIGHT: self.x += 80 / self.zoom
            elif event.key == K_UP: self.y -= 80 / self.zoom
            elif event.key == K_DOWN: self.y += 80 / self.zoom

    def update_follow(self, world, team='red'):
        if self.follow:
            a = world.aircraft_of(self.follow)
            if a:
                self.x, self.y = a.x, a.y

    def draw_terrain(self, screen, world):
        """先画地形（在网格之上、实体之下）。"""
        t = getattr(world, 'terrain', None)
        if t is not None:
            t.draw(screen, self)

    def draw_grid(self, screen):
        step = 200
        if self.zoom * step < 10:
            step = 1000
        x0 = int((self.x - self.sw / 2 / self.zoom) // step) * step
        y0 = int((self.y - self.sh / 2 / self.zoom) // step) * step
        color = (230, 230, 230)
        for gx in range(int(x0), int(x0 + self.sw / self.zoom) + step, step):
            sx, _ = self.world_to_screen(gx, 0)
            pygame.draw.line(screen, color, (sx, 0), (sx, self.sh))
        for gy in range(int(y0), int(y0 + self.sh / self.zoom) + step, step):
            _, sy = self.world_to_screen(0, gy)
            pygame.draw.line(screen, color, (0, sy), (self.sw, sy))


class HUD:
    def __init__(self):
        self.selected = None
        self.follow_team = 'red'

    def draw(self, screen, world):
        # ---- 顶部：双方资源与兵力 ----
        reds = world.aircraft_list_of('red')
        blues = world.aircraft_list_of('blue')
        txt = FONT.render(f"红方资源 {int(world.resources['red'])}   蓝方资源 {int(world.resources['blue'])}", True, BLACK)
        screen.blit(txt, (10, 8))
        hitxt = FONT.render(f"红机 {len(reds)}  蓝机 {len(blues)}", True, RED)
        screen.blit(hitxt, (300, 8))
        txt2 = SMALL_FONT.render(
            f"步数 {world.step_count}   地面 {len(world.ground.alive_list())}   "
            f"建筑 {len(world.buildings.alive_list())}   弹药 "
            f"{len(world.bullets.alive_list()) + len(world.bombs.alive_list()) + len(world.missiles.alive_list())}",
            True, GRAY)
        screen.blit(txt2, (10, 30))
        # ---- 地形信息 ----
        t = getattr(world, 'terrain', None)
        if t is not None and t.features:
            cn = REALISTIC_MAPS.get(t.name, (t.name, '', 1))[0]
            kinds = {}
            for f in t.features:
                k = TERRAIN_NAMES.get(f.kind, '?')
                kinds[k] = kinds.get(k, 0) + 1
            kinfo = ' '.join(f'{k}×{v}' for k, v in kinds.items())
            screen.blit(SMALL_FONT.render(f"地图 {cn}   地形 {kinfo}", True, (90, 110, 90)), (10, 48))

        # ---- 焦点单位面板（优先显示红方长机）----
        focus = reds[0] if reds else (blues[0] if blues else None)
        if focus is not None:
            self._draw_unit_panel(screen, world, focus, 8, SCREEN_H - 118)

    def _draw_unit_panel(self, screen, world, a, x, y):
        w, h = 300, 108
        panel = pygame.Surface((w, h))
        panel.set_alpha(215)
        panel.fill((250, 250, 250))
        screen.blit(panel, (x, y))
        pygame.draw.rect(screen, GRAY, (x, y, w, h), 1)
        team_cn = '红方' if a.team == 'red' else '蓝方'
        col = RED if a.team == 'red' else BLUE
        screen.blit(SMALL_FONT.render(f"{team_cn} 长机 · {a.spec.name}", True, col), (x + 6, y + 4))
        # 血条
        pygame.draw.rect(screen, (220, 220, 220), (x + 6, y + 24, w - 12, 8))
        hp_w = int((w - 12) * max(0.0, a.hp / a.max_hp))
        pygame.draw.rect(screen, GREEN if a.hp > a.max_hp * 0.5 else ORANGE, (x + 6, y + 24, hp_w, 8))
        screen.blit(SMALL_FONT.render(f"HP {a.hp:.0f}/{a.max_hp:.0f}", True, BLACK), (x + 6, y + 34))
        # 武器
        wp = a.weapons
        screen.blit(SMALL_FONT.render(
            f"机炮 {wp.ammo}/{wp.max_ammo}   炸弹 {wp.bombs}/{wp.max_bombs}   "
            f"导弹 {wp.missiles}/{wp.max_missiles}   诱饵 {wp.flares}/{wp.max_flares}",
            True, BLACK), (x + 6, y + 50))
        # 飞行状态
        thr_cn = {THR_AFTERBURN: '加力', THR_CRUISE: '巡航', THR_BRAKE: '减速板'}.get(a.flight.throttle_state, '?')
        screen.blit(SMALL_FONT.render(
            f"速度 {a.flight.speed:.1f}  {thr_cn}   航向 {a.angle:.0f}°   机头偏转 {a.head_angle:+.0f}°",
            True, BLACK), (x + 6, y + 66))
        # 锁定/激进度
        lock = a.lock_target
        if lock is not None and lock.alive:
            lt = '敌机' if isinstance(lock, Aircraft) else '地面'
            ld = math.hypot(lock.x - a.x, lock.y - a.y)
            ltxt = f"锁定 {lt}  距离 {ld:.0f}"
            lcol = (200, 0, 0)
        else:
            ltxt, lcol = "未锁定", GRAY
        screen.blit(SMALL_FONT.render(ltxt, True, lcol), (x + 6, y + 82))
        ag = {0.0: '保命', 1.0: '激进'}.get(a.aggressiveness, '均衡')
        screen.blit(SMALL_FONT.render(f"激进度 {ag}   记忆 {len(a.sensor.memory)}/8", True, GRAY), (x + 150, y + 82))


class Minimap:
    def __init__(self, w=200, h=136):
        self.w, self.h = w, h
        self.x = SCREEN_W - w - 10
        self.y = SCREEN_H - h - 10

    def draw(self, screen, world):
        pygame.draw.rect(screen, (240, 240, 240), (self.x, self.y, self.w, self.h))
        sx = self.x + self.w * 0.08
        sw = self.w * 0.84
        sy = self.y + self.h * 0.08
        sh = self.h * 0.84

        def to_px(x, y):
            return int(sx + (x / WORLD_W) * sw), int(sy + (y / WORLD_H) * sh)

        # 地形底图
        t = getattr(world, 'terrain', None)
        if t is not None:
            for f in t.features:
                px, py = to_px(f.x, f.y)
                r = max(1, int(f.r / WORLD_W * sw))
                pygame.draw.circle(screen, TERRAIN_COLORS.get(f.kind, (230, 230, 230)), (px, py), r)
        # 建筑 / 地面
        for b in world.buildings.alive_list():
            px, py = to_px(b.x, b.y)
            pygame.draw.rect(screen, RED if b.team == 'red' else BLUE, (px - 2, py - 2, 4, 4))
        for g in world.ground.alive_list():
            px, py = to_px(g.x, g.y)
            pygame.draw.circle(screen, (200, 120, 120) if g.team == 'red' else (120, 120, 210), (px, py), 1)
        # 弹药（只画导弹，避免糊成一片）
        for m in world.missiles.alive_list():
            px, py = to_px(m.x, m.y)
            pygame.draw.circle(screen, (255, 130, 130) if m.team == 'red' else (130, 130, 255), (px, py), 1)
        # 飞机（上方画血量点）
        for a in world.aircraft.alive_list():
            px, py = to_px(a.x, a.y)
            col = RED if a.team == 'red' else BLUE
            pygame.draw.circle(screen, col, (px, py), 3)
            frac = max(0.0, min(1.0, a.hp / a.max_hp))
            pygame.draw.line(screen, GREEN if frac > 0.5 else ORANGE, (px - 3, py - 5), (px - 3 + int(6 * frac), py - 5), 2)
        pygame.draw.rect(screen, GRAY, (self.x, self.y, self.w, self.h), 1)
        screen.blit(SMALL_FONT.render("小地图", True, GRAY), (self.x + 4, self.y + 2))


class Button:
    def __init__(self, x, y, w, h, text, color=GREEN, font=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.font = font or BIG_FONT

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        surf = self.font.render(self.text, True, WHITE)
        screen.blit(surf, (self.rect.x + (self.rect.w - surf.get_width()) // 2,
                           self.rect.y + (self.rect.h - surf.get_height()) // 2))

    def hit(self, pos):
        return self.rect.collidepoint(pos)


def main_menu(screen):
    """主菜单：返回选择的模式字符串。"""
    buttons = [
        Button(SCREEN_W // 2 - 150, 210, 300, 42, "课程训练（8 阶段渐进）", (0, 150, 0)),
        Button(SCREEN_W // 2 - 150, 262, 300, 42, "常规模式（手动 vs 规则AI）", GREEN),
        Button(SCREEN_W // 2 - 150, 314, 300, 42, "AI 训练（PPO 自博弈）", BLUE),
        Button(SCREEN_W // 2 - 150, 366, 300, 42, "观战模式（模型对战）", ORANGE),
        Button(SCREEN_W // 2 - 150, 418, 300, 42, "选择地图（8 种现实地形）", (120, 90, 160)),
        Button(SCREEN_W // 2 - 150, 470, 300, 42, "模型管理", GRAY),
        Button(SCREEN_W // 2 - 150, 522, 300, 42, "退出", DARK_RED),
    ]
    while True:
        screen.fill(WHITE)
        title = MENU_FONT.render("AWAI 空战 AI 平台 v3", True, BLACK)
        screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, 110))
        sub = FONT.render("红蓝对抗 · 地形遮挡感知 · 编队协作观测 · PPO 课程自博弈", True, GRAY)
        screen.blit(sub, (SCREEN_W // 2 - sub.get_width() // 2, 168))
        for b in buttons:
            b.draw(screen)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); sys.exit()
            if event.type == MOUSEBUTTONDOWN:
                for b in buttons:
                    if b.hit(event.pos):
                        return b.text.split('（')[0]
        pygame.display.flip()
        pygame.time.Clock().tick(30)


def _text_input(screen, prompt, x, y, font=FONT):
    """简单文本输入框，返回输入的字符串。"""
    value = ''
    active = True
    rect = pygame.Rect(x, y, 240, 30)
    while active:
        pygame.draw.rect(screen, WHITE, (x - 200, y - 30, 700, 90))
        p = font.render(prompt, True, BLACK)
        screen.blit(p, (x - 200, y - 24))
        pygame.draw.rect(screen, BLACK, rect, 2)
        surf = font.render(value, True, BLACK)
        screen.blit(surf, (rect.x + 5, rect.y + 5))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_RETURN:
                    return value
                if event.key == K_ESCAPE:
                    return None
                if event.key == K_BACKSPACE:
                    value = value[:-1]
                else:
                    value += event.unicode


def training_setup(screen):
    """训练设置：机型选择 + 模型名。返回 (plane_type, model_name, episodes)。"""
    plane_types = list(AIRCRAFT_SPECS.keys())[:8]
    selected = 0
    while True:
        screen.fill(WHITE)
        title = BIG_FONT.render("AI 训练 - 机型与模型", True, BLACK)
        screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, 50))
        for i, pt in enumerate(plane_types):
            color = BLUE if i == selected else BLACK
            txt = FONT.render(f"{AIRCRAFT_SPECS[pt].name} ({pt})", True, color)
            screen.blit(txt, (200, 150 + i * 30))
        hint = SMALL_FONT.render("↑/↓ 选择机型    Enter 确认并输入模型名", True, GRAY)
        screen.blit(hint, (200, 420))
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_UP: selected = (selected - 1) % len(plane_types)
                elif event.key == K_DOWN: selected = (selected + 1) % len(plane_types)
                elif event.key == K_RETURN:
                    name = _text_input(screen, "模型文件名（留空则新建 red.pt/blue.pt）:", 400, 500)
                    if name is None:
                        return None
                    return plane_types[selected], name.strip(), 200
                elif event.key == K_ESCAPE:
                    return None
        pygame.display.flip()
        pygame.time.Clock().tick(30)


class TrainView:
    """训练界面：世界渲染 + 实时奖励曲线 + 倍速/暂停/保存。"""

    def __init__(self, screen):
        self.screen = screen
        self.camera = Camera(SCREEN_W, SCREEN_H)
        self.camera.follow = 'red'
        self.hud = HUD()
        self.minimap = Minimap()
        self.speed = 1
        self.paused = False
        self.red_curve, self.blue_curve = [], []
        self.ep_red, self.ep_blue = 0.0, 0.0
        self.ep = 0
        self.steps = 0

    def draw_curve(self, screen):
        x0, y0, w, h = SCREEN_W - 380, 60, 360, 150
        pygame.draw.rect(screen, (245, 245, 245), (x0, y0, w, h))
        pygame.draw.rect(screen, GRAY, (x0, y0, w, h), 1)
        surf = SMALL_FONT.render(f"训练进度  Episode {self.ep}", True, BLACK)
        screen.blit(surf, (x0 + 6, y0 + 4))
        # 曲线
        def draw_line(pts, color):
            if len(pts) < 2:
                return
            maxv = max(abs(min(pts)), abs(max(pts)), 1.0)
            px = [x0 + 6 + i * (w - 12) / (len(pts) - 1) for i in range(len(pts))]
            py = [y0 + h - 20 - ((p + maxv) / (2 * maxv)) * (h - 30) for p in pts]
            pygame.draw.lines(screen, color, False, list(zip(px, py)), 2)
        draw_line(self.red_curve, RED)
        draw_line(self.blue_curve, BLUE)
        leg1 = SMALL_FONT.render("红方", True, RED)
        leg2 = SMALL_FONT.render("蓝方", True, BLUE)
        screen.blit(leg1, (x0 + 6, y0 + h - 18))
        screen.blit(leg2, (x0 + 50, y0 + h - 18))
        info = SMALL_FONT.render(
            f"步数 {self.steps}  倍速 x{self.speed}  {'暂停' if self.paused else '运行'}",
            True, BLACK)
        screen.blit(info, (x0 + 6, y0 + 26))
        keys = SMALL_FONT.render("1/2/3 倍速  P 暂停  S 保存模型  ESC 退出", True, GRAY)
        screen.blit(keys, (x0 + 6, y0 + 44))

    def draw(self, world):
        self.screen.fill(WHITE)
        self.camera.update_follow(world, 'red')
        self.camera.draw_grid(self.screen)
        self.camera.draw_terrain(self.screen, world)
        for pool in (world.buildings, world.ground, world.aircraft,
                     world.bullets, world.bombs, world.missiles):
            for e in pool.alive_list():
                e.draw(self.screen, self.camera)
        self.hud.draw(self.screen, world)
        self.minimap.draw(self.screen, world)
        self.draw_curve(self.screen)
        pygame.display.flip()


def map_setup(screen, title='选择地图'):
    """现实向地图选择。返回 (layout, map_variants) 或 None。"""
    names = list(REALISTIC_MAPS.keys())
    sel = 0
    while True:
        screen.fill(WHITE)
        t = BIG_FONT.render(title, True, BLACK)
        screen.blit(t, (SCREEN_W // 2 - t.get_width() // 2, 40))
        for i, name in enumerate(names):
            cn, desc, nv = REALISTIC_MAPS[name]
            y = 110 + i * 62
            active = (i == sel)
            if active:
                pygame.draw.rect(screen, (235, 245, 235), (100, y - 6, SCREEN_W - 200, 54))
            screen.blit(FONT.render(f"{cn}  ({name})", True, (0, 120, 0) if active else BLACK), (120, y))
            screen.blit(SMALL_FONT.render(f"{desc}    变体数 {nv}", True, GRAY), (120, y + 22))
        # 右侧预览地形缩略图
        cn, desc, nv = REALISTIC_MAPS[names[sel]]
        tm = build_terrain_map(names[sel], 0)
        bx, by, bw, bh = SCREEN_W - 340, 120, 300, 200
        pygame.draw.rect(screen, (248, 248, 248), (bx, by, bw, bh))
        for f in tm.features:
            px = int(bx + f.x / WORLD_W * bw)
            py = int(by + f.y / WORLD_H * bh)
            r = max(1, int(f.r / WORLD_W * bw))
            pygame.draw.circle(screen, TERRAIN_COLORS.get(f.kind, (230, 230, 230)), (px, py), r)
            pygame.draw.circle(screen, tuple(max(0, c - 40) for c in TERRAIN_COLORS.get(f.kind, (230, 230, 230))),
                               (px, py), r, 1)
        pygame.draw.rect(screen, GRAY, (bx, by, bw, bh), 1)
        screen.blit(SMALL_FONT.render(f"预览：{cn}（变体 0）", True, GRAY), (bx, by - 20))
        screen.blit(SMALL_FONT.render("↑/↓ 选择地图    Enter 确认    ESC 返回", True, GRAY), (120, SCREEN_H - 50))
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_UP:
                    sel = (sel - 1) % len(names)
                elif event.key == K_DOWN:
                    sel = (sel + 1) % len(names)
                elif event.key == K_RETURN:
                    return names[sel], REALISTIC_MAPS[names[sel]][2]
                elif event.key == K_ESCAPE:
                    return None
        pygame.display.flip()
        pygame.time.Clock().tick(30)


def watch_setup(screen):
    """观战设置：选择红/蓝模型。返回 (red_model, blue_model)。"""
    models = list_models()
    if not models:
        return None
    sel = [0, 0]
    while True:
        screen.fill(WHITE)
        title = BIG_FONT.render("观战模式 - 选择模型", True, BLACK)
        screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, 50))
        for row, label in ((0, '红方:'), (1, '蓝方:')):
            y = 150 + row * 220
            screen.blit(FONT.render(label, True, BLACK), (150, y))
            for i, m in enumerate(models):
                color = BLUE if i == sel[row] else BLACK
                screen.blit(FONT.render(m, True, color), (260, y + i * 24))
        hint = SMALL_FONT.render("↑/↓ 选择红方  Q/W 选择蓝方  Enter 开始  ESC 返回", True, GRAY)
        screen.blit(hint, (150, 620))
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_UP: sel[0] = (sel[0] - 1) % len(models)
                elif event.key == K_DOWN: sel[0] = (sel[0] + 1) % len(models)
                elif event.key == K_q: sel[1] = (sel[1] - 1) % len(models)
                elif event.key == K_w: sel[1] = (sel[1] + 1) % len(models)
                elif event.key == K_RETURN:
                    return models[sel[0]], models[sel[1]]
                elif event.key == K_ESCAPE:
                    return None
        pygame.display.flip()
        pygame.time.Clock().tick(30)


def model_manager(screen):
    """模型管理：列出 models 目录与 modelset.txt 中的模型，可删除。"""
    while True:
        models = list_models()
        screen.fill(WHITE)
        title = BIG_FONT.render("模型管理", True, BLACK)
        screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, 40))
        if not models:
            screen.blit(FONT.render("（暂无模型，去 AI 训练里生成）", True, GRAY), (200, 150))
        for i, m in enumerate(models):
            y = 120 + i * 34
            screen.blit(FONT.render(m, True, BLACK), (200, y))
            del_btn = Button(SCREEN_W - 260, y - 4, 80, 26, "删除", DARK_RED, FONT)
            del_btn.draw(screen)
        back = Button(SCREEN_W // 2 - 60, SCREEN_H - 90, 120, 40, "返回", GRAY)
        back.draw(screen)
        hint = SMALL_FONT.render("modelset.txt 内容会被同步到 models 目录", True, GRAY)
        screen.blit(hint, (200, SCREEN_H - 60))
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); sys.exit()
            if event.type == MOUSEBUTTONDOWN:
                if back.hit(event.pos):
                    return
                for i, m in enumerate(models):
                    del_btn = pygame.Rect(SCREEN_W - 260, 120 + i * 34 - 4, 80, 26)
                    if del_btn.collidepoint(event.pos):
                        delete_model(m)
        pygame.display.flip()
        pygame.time.Clock().tick(30)


# ---------- 模型管理 ----------
def _read_modelset():
    """读取 modelset.txt（UTF-8，失败则按 GBK 读）。返回模型名列表。"""
    if not os.path.exists(MODELSET_FILE):
        return []
    try:
        with open(MODELSET_FILE, 'r', encoding='utf-8') as f:
            return [l.strip() for l in f if l.strip() and not l.startswith('#')]
    except UnicodeDecodeError:
        with open(MODELSET_FILE, 'r', encoding='gbk') as f:
            return [l.strip() for l in f if l.strip() and not l.startswith('#')]


def _write_modelset(names):
    os.makedirs(os.path.dirname(MODELSET_FILE), exist_ok=True)
    with open(MODELSET_FILE, 'w', encoding='utf-8') as f:
        f.write("# 模型文件列表，每行一个文件名\n")
        for n in names:
            f.write(n + "\n")


def list_models():
    """合并 models 目录文件与 modelset.txt 条目，去重。"""
    names = []
    if os.path.isdir(MODEL_DIR):
        for fn in sorted(os.listdir(MODEL_DIR)):
            if fn.endswith('.pt'):
                names.append(fn[:-3])
    for n in _read_modelset():
        if n and n not in names and os.path.exists(model_path(n)):
            names.append(n)
    return names


def model_path(name):
    if name.endswith('.pt'):
        name = name[:-3]
    return os.path.join(MODEL_DIR, name + '.pt')


def delete_model(name):
    p = model_path(name)
    if os.path.exists(p):
        os.remove(p)


def default_model_name(prefer='_final_red'):
    """挑一个**真实存在**的默认模型，供无参运行的观战/评估脚本使用。

    修掉了旧行为：直接取 modelset.txt 首条，而该条可能指向根本不存在的文件
    （随附的 modelset.txt 里 `_iso_g_lr1e-5_epochs1_red` 就不存在），
    导致 `py AWAI\\watch_show.py` 不带参数必然失败。

    优先级：modelset.txt 中真实存在的条目 → 名字含 prefer 的模型（取版本号最大的）
    → 其它任意可用模型。models/ 为空时返回 None。
    """
    available = list_models()
    if not available:
        return None
    for name in _read_modelset():
        if name in available:
            return name
    preferred = sorted(n for n in available if prefer in n)
    if preferred:
        return preferred[-1]
    return sorted(available)[0]


def model_status(name):
    """读检查点里的 obs_dim / obs_version，判断能否被当前代码加载。

    返回 (ok, 中文说明)。文件不存在或元信息读不出时 ok=False。"""
    p = model_path(name)
    if not os.path.exists(p):
        return False, '文件不存在'
    try:
        data = torch.load(p, map_location='cpu')
    except Exception as ex:            # noqa: BLE001
        return False, f'无法读取（{type(ex).__name__}）'
    obs_dim, obs_ver = data.get('obs_dim'), data.get('obs_version')
    if obs_dim != OBS_DIM or obs_ver != OBS_VERSION:
        return False, f'不兼容（obs_dim={obs_dim} obs_version={obs_ver}）'
    return True, f'兼容（obs_dim={obs_dim} obs_version={obs_ver}）'


def suggested_models(limit=6):
    """给「模型找不到 / 不兼容」类错误用的候选清单文本。"""
    names = list_models()
    if not names:
        return ('    （models/ 目录为空）请先训练：\n'
                '      py AWAI\\AWAI.py --curriculum standard --episodes 100 '
                '--norender --prefix standard_v3')
    lines = []
    for n in names[:limit]:
        ok, why = model_status(n)
        lines.append(f'    {"[可用]" if ok else "[跳过]"} {n}  {why}')
    if len(names) > limit:
        lines.append(f'    ... 其余 {len(names) - limit} 个用 --list-models 查看')
    return '\n'.join(lines)


# ======================================================================
# 8. 训练循环
# ======================================================================
def _obs_for(obs, team):
    """观测缺失（该队飞机已阵亡）时返回零向量。"""
    v = obs.get(team)
    if v is None:
        return np.zeros(OBS_DIM, dtype=np.float32)
    return v


def train_ppo(plane_type, model_name, episodes=200, render=True):
    """红蓝各一个 PPOAgent 自博弈训练。"""
    red_agent = PPOAgent(OBS_DIM, ACT_DIMS)
    blue_agent = PPOAgent(OBS_DIM, ACT_DIMS)
    red_path = model_path(model_name + '_red')
    blue_path = model_path(model_name + '_blue')
    if os.path.exists(red_path):
        red_agent.load(red_path)
    if os.path.exists(blue_path):
        blue_agent.load(blue_path)

    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H)) if render else None
    view = TrainView(screen) if render else None

    env = AirCombatEnv(plane_type, render=False, red_ai_factory=lambda: CommandedController(),
                       blue_ai_factory=lambda: CommandedController())

    clock = pygame.time.Clock()
    obs = env.reset()
    red_controller = env.world.aircraft_of('red').controller
    blue_controller = env.world.aircraft_of('blue').controller

    for ep in range(episodes):
        obs = env.reset()
        red_controller = env.world.aircraft_of('red').controller
        blue_controller = env.world.aircraft_of('blue').controller
        done = False
        ep_red = ep_blue = 0.0
        steps = 0
        while not done:
            # 倍速与暂停
            if render:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit(); sys.exit()
                    if event.type == KEYDOWN:
                        if event.key == K_1: view.speed = 1
                        elif event.key == K_2: view.speed = 2
                        elif event.key == K_3: view.speed = 4
                        elif event.key == K_p: view.paused = not view.paused
                        elif event.key == K_s:
                            red_agent.save(red_path)
                            blue_agent.save(blue_path)
                            _write_modelset([model_name + '_red', model_name + '_blue'])
                            print(f"[save] 已保存 {red_path} / {blue_path}")
                        elif event.key == K_ESCAPE:
                            pygame.quit(); sys.exit()
                if view.paused:
                    clock.tick(FPS)
                    continue
                for _ in range(view.speed):
                    o_r, o_b = _obs_for(obs, 'red'), _obs_for(obs, 'blue')
                    act_r, lp_r, v_r = red_agent.act(o_r)
                    act_b, lp_b, v_b = blue_agent.act(o_b)
                    actions = {'red': act_r, 'blue': act_b}
                    n_obs, rewards, done, _ = env.step(actions)
                    red_agent.remember(o_r, act_r, lp_r, v_r, rewards['red'], done)
                    blue_agent.remember(o_b, act_b, lp_b, v_b, rewards['blue'], done)
                    obs = n_obs
                    ep_red += rewards['red']; ep_blue += rewards['blue']
                    steps += 1
                    if done:
                        break
            else:
                o_r, o_b = _obs_for(obs, 'red'), _obs_for(obs, 'blue')
                act_r, lp_r, v_r = red_agent.act(o_r)
                act_b, lp_b, v_b = blue_agent.act(o_b)
                actions = {'red': act_r, 'blue': act_b}
                n_obs, rewards, done, _ = env.step(actions)
                red_agent.remember(o_r, act_r, lp_r, v_r, rewards['red'], done)
                blue_agent.remember(o_b, act_b, lp_b, v_b, rewards['blue'], done)
                obs = n_obs
                ep_red += rewards['red']; ep_blue += rewards['blue']
                steps += 1
            if render:
                view.ep = ep
                view.steps = env.steps
                view.ep_red, view.ep_blue = ep_red, ep_blue
                view.red_curve.append(ep_red)
                view.blue_curve.append(ep_blue)
                if len(view.red_curve) > 200:
                    view.red_curve = view.red_curve[-200:]
                    view.blue_curve = view.blue_curve[-200:]
                view.draw(env.world)
                clock.tick(FPS)
            if steps > env.max_steps + 10:
                done = True
        # 回合结束：若为超时而非终止，用末态价值引导（bootstrap）
        terminal = env.world.base_destroyed() is not None or len(env.world.aircraft.alive_list()) == 0
        last_val_r = 0.0 if terminal else red_agent.act(_obs_for(obs, 'red'))[2]
        last_val_b = 0.0 if terminal else blue_agent.act(_obs_for(obs, 'blue'))[2]
        red_agent.update(last_val_r)
        blue_agent.update(last_val_b)
        print(f"Ep {ep} 红 {ep_red:.1f} / 蓝 {ep_blue:.1f}  步数 {steps}")
        if ep % 10 == 0:
            red_agent.save(red_path)
            blue_agent.save(blue_path)
            _write_modelset([model_name + '_red', model_name + '_blue'])
    env.close()
    if render:
        pygame.quit()


# ======================================================================
# 8.5 后台并行训练：训练线程全速演算，前端线程实时抓取最新快照演示
# ======================================================================
class Snapshot:
    """后台训练线程发布给前端的轻量世界快照（线程安全，只读）。"""
    __slots__ = ('step', 'ep', 'total_episodes', 'red_curve', 'blue_curve',
                 'entities', 'resources', 'red_reward', 'blue_reward',
                 'train_fps', 'finished')

    def __init__(self):
        self.step = 0
        self.ep = 0
        self.total_episodes = 0
        self.red_curve = []
        self.blue_curve = []
        self.entities = []          # (kind, team, x, y, angle, hp, maxhp, size, color, subtype)
        self.resources = {'red': 0.0, 'blue': 0.0}
        self.red_reward = 0.0
        self.blue_reward = 0.0
        self.train_fps = 0.0
        self.finished = False


def _snapshot_from_world(world, ep, total_episodes, red_curve, blue_curve,
                         ep_red, ep_blue, train_fps):
    snap = Snapshot()
    snap.step = world.step_count
    snap.ep = ep
    snap.total_episodes = total_episodes
    snap.red_curve = list(red_curve)
    snap.blue_curve = list(blue_curve)
    snap.red_reward = ep_red
    snap.blue_reward = ep_blue
    snap.resources = dict(world.resources)
    snap.train_fps = train_fps
    ents = []
    for pool, kind in ((world.aircraft, 'aircraft'), (world.ground, 'ground'),
                       (world.buildings, 'building'), (world.bullets, 'bullet'),
                       (world.bombs, 'bomb'), (world.missiles, 'missile')):
        for e in pool.alive_list():
            ents.append((kind, e.team, e.x, e.y, getattr(e, 'angle', 0.0),
                         getattr(e, 'hp', 0), getattr(e, 'max_hp', 1),
                         getattr(e, 'size', 3), getattr(e, 'color', GRAY),
                         getattr(e, 'type', '')))
    snap.entities = ents
    return snap


class TrainWorker(threading.Thread):
    """后台训练线程：多环境并行演算（无 tick 限制），定期发布快照。

    两种并行方式：
      - env_factory 为课程环境工厂且 parallel=True → 使用跨进程 ParallelEnvPool
        （吃满多核；实测 S1 环境吞吐 5.1 万 → 64 万环境步/秒）；
      - 否则回退为同进程多环境（受 GIL 限制，仅作兼容）。
    前端通过 latest_snapshot() 抓取最新结果；训练循环不受渲染帧率影响。
    """

    def __init__(self, plane_type, model_name, episodes=200, n_envs=2,
                 snapshot_every=8, update_buffer=500, env_factory=None,
                 parallel=True, n_workers=8, envs_per_worker=8, block_steps=256,
                 stage_mix=False, use_fire_mask=True, fire_aux_coef=0.0, custom_stages=None):
        super().__init__(daemon=True, name='TrainWorker')
        self.plane_type = plane_type
        self.model_name = model_name
        self.episodes = episodes
        self.n_envs = max(1, n_envs)
        self.snapshot_every = max(1, snapshot_every)
        self.update_buffer = update_buffer
        self.env_factory = env_factory          # None → 1v1 AirCombatEnv；否则课程环境工厂
        self.use_curriculum = env_factory is not None
        self.parallel = bool(parallel and self.use_curriculum)
        self.n_workers = max(1, int(n_workers))
        self.envs_per_worker = max(1, int(envs_per_worker))
        self.block_steps = max(1, int(block_steps))
        # stage_mix：并行环境各自采样不同课程阶段（防灾难性遗忘）
        self.stage_mix = bool(stage_mix)
        # use_fire_mask：是否屏蔽备弹为 0 / 冷却中的开火选项
        self.use_fire_mask = bool(use_fire_mask)
        # fire_aux_coef：火控头行为克隆辅助损失系数（>0 时启用）
        self.fire_aux_coef = float(fire_aux_coef)
        # custom_stages：自定义阶段列表（用于"只训练空战阶段"这类专项任务）
        self.custom_stages = list(custom_stages) if custom_stages else None
        if self.custom_stages and self.stage_mix:
            self.stage_idx = 0
            self._stage_criteria = self.custom_stages[0].pass_criteria
        # 从 env_factory 的闭包中取出阶段索引（run_curriculum 用 st=stage 捕获）
        self.stage_idx = 0
        if self.parallel and env_factory is not None:
            try:
                cells = [c.cell_contents for c in (env_factory.__closure__ or ())]
                found = [c for c in cells if isinstance(c, StageSpec)]
                if found:
                    all_stages = curriculum_stages(plane_type)
                    self.stage_idx = next(
                        (i for i, s in enumerate(all_stages) if s is found[0]), 0)
            except Exception:
                self.stage_idx = 0
        # 阶段达成条件（并行路径无法直接访问 env.stage，故在此缓存）
        try:
            self._stage_criteria = curriculum_stages(plane_type)[self.stage_idx].pass_criteria
        except Exception:
            self._stage_criteria = None
        self.agents = {'red': PPOAgent(OBS_DIM, ACT_DIMS, fire_aux_coef=self.fire_aux_coef),
                       'blue': PPOAgent(OBS_DIM, ACT_DIMS)}
        for team in ('red', 'blue'):
            p = model_path(f"{model_name}_{team}")
            if os.path.exists(p):
                try:
                    self.agents[team].load(p)
                    print(f"[bg] 已加载模型 {p}")
                except Exception as ex:
                    print(f"[bg] 加载 {p} 失败: {ex}")
        self._lock = threading.Lock()
        self._latest = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._save_event = threading.Event()
        self._step_counter = 0
        self._last_pub_time = time.time()
        self._stage_passed = False     # 阶段达成标志（达标提前晋级）
        self._recent_results = collections.deque(maxlen=20)   # 最近回合表现

    # ---------- 前端接口 ----------
    def latest_snapshot(self):
        with self._lock:
            return self._latest

    def request_save(self):
        self._save_event.set()

    def set_pause(self, paused):
        if paused:
            self._pause_event.set()
        else:
            self._pause_event.clear()

    def stop(self):
        self._stop_event.set()

    def stage_passed(self):
        return self._stage_passed

    def stage_summary(self):
        """本阶段训练结果汇总（用于挑选最佳检查点、输出指标）。"""
        q = list(self._recent_results)
        n = len(q) or 1
        return {
            'episodes': len(q),
            'clears': sum(1 for r in q if r['cleared']),
            'clear_rate': sum(1 for r in q if r['cleared']) / n,
            'win_rate': sum(1 for r in q if r['red_reward'] > r['blue_reward']) / n,
            'avg_reward': sum(r['red_reward'] for r in q) / n,
            'avg_kills': sum(r['red_kills'] for r in q) / n,
            'avg_damage': sum(r['red_damage'] for r in q) / n,
        }

    def _pass_check(self, criteria: 'PassCriteria'):
        """滑动窗口内是否达成阶段条件（所有非 None 指标同时满足）。"""
        q = self._recent_results
        if len(q) < min(criteria.window, 5):
            return False
        win = sum(1 for r in q if r['cleared'] or r['red_reward'] > r['blue_reward'])
        clear = sum(1 for r in q if r['cleared'])
        avg_r = sum(r['red_reward'] for r in q) / len(q)
        avg_k = sum(r['red_kills'] for r in q) / len(q)
        avg_h = sum(r.get('red_hunt', 0) for r in q) / len(q)
        avg_d = sum(r.get('red_damage', 0.0) for r in q) / len(q)
        n = len(q)
        if criteria.clear_rate is not None and clear / n < criteria.clear_rate:
            return False
        if criteria.win_rate is not None and win / n < criteria.win_rate:
            return False
        if criteria.min_avg_reward is not None and avg_r < criteria.min_avg_reward:
            return False
        if criteria.min_avg_kills is not None and avg_k < criteria.min_avg_kills:
            return False
        if criteria.min_avg_hunt is not None and avg_h < criteria.min_avg_hunt:
            return False
        if criteria.min_avg_damage is not None and avg_d < criteria.min_avg_damage:
            return False
        return True

    def _publish(self, world, ep, total, red_curve, blue_curve, ep_red, ep_blue, finished=False):
        now = time.time()
        dt = now - self._last_pub_time
        self._last_pub_time = now
        fps = (self.snapshot_every / dt) if dt > 0 else 0.0
        snap = _snapshot_from_world(world, ep, total, red_curve, blue_curve,
                                    ep_red, ep_blue, fps)
        snap.finished = finished
        with self._lock:
            self._latest = snap

    # ---------- 训练主体 ----------
    def run(self):
        if self.parallel:
            return self._run_parallel()
        return self._run_serial()

    def _run_parallel(self):
        """跨进程并行训练：仿真在 n_workers 个子进程里跑，主线程只做推理与 PPO 更新。

        数据流：主线程采样动作 → 下发一个 block → 子进程整块推进并回传完整时间序列
        → 主线程把整块写入 PPO 缓冲并（按需）更新 → 更新后把新权重下发给子进程。
        """
        from env_pool import ParallelEnvPool
        n_stages = len(curriculum_stages(self.plane_type))
        use_custom = self.custom_stages is not None
        if use_custom:
            n_stages = len(self.custom_stages)
        pool = ParallelEnvPool(self.stage_idx, n_workers=self.n_workers,
                               envs_per_worker=self.envs_per_worker,
                               plane_type=self.plane_type,
                               block_steps=self.block_steps,
                               stage_mix_m=(n_stages if self.stage_mix else 0),
                               stages=(self.custom_stages if use_custom else None)).start()
        n_envs, n_red = pool.n_envs, pool.n_red
        # 更新缓冲按环境数等比放大：串行 4 环境时 500 样本 ≈ 每环境 125 步一次更新。
        # 若固定 500，64 环境下每 8 个 block 就更新一次（频率高 16 倍），
        # 实测导致策略退化（前半程伤害 37 → 后半程 18）。
        upd_buf = max(self.update_buffer,
                      int(self.update_buffer * n_envs / max(1, self.n_envs)))
        print(f"[bg] 并行训练：workers={self.n_workers} 环境数={n_envs} "
              f"块长={self.block_steps} 更新缓冲={upd_buf} "
              f"阶段混合={'开（各环境轮流采样 '+str(n_stages)+' 个阶段）' if self.stage_mix else '关'}",
              flush=True)
        ep_rewards = [0.0] * n_envs
        ep_kills = [0] * n_envs
        ep_damage = [0.0] * n_envs
        red_curve, blue_curve = [], []
        ep_count = 0

        def do_save():
            for team in ('red', 'blue'):
                self.agents[team].save(model_path(f"{self.model_name}_{team}"))
            _write_modelset([f"{self.model_name}_red", f"{self.model_name}_blue"])
            print(f"[bg] 已保存模型 → models/{self.model_name}_red.pt / _blue.pt", flush=True)

        try:
            # 关键：必须在任何采样之前把权重下发到子进程。
            # 早期版本只在首次 PPO 更新后才 sync_weights，导致子进程在整个首段
            # 训练里都用【随机初始化的网络】采样动作，主进程记录的 (观测, 动作) 对
            # 完全错位 —— 实测表现为目标阶段伤害恒为 0、只有后续阶段偶然学到东西。
            pool.sync_weights(self.agents['red'])
            flat = pool.obs_dict()['red']
            while (ep_count < self.episodes and not self._stop_event.is_set()
                   and not self._stage_passed):
                if self._pause_event.is_set():
                    time.sleep(0.05)
                    continue
                if self._save_event.is_set():
                    self._save_event.clear()
                    do_save()
                # 采样 → 下发整块（可选火控动作掩码：屏蔽备弹为 0 / 冷却中的开火选项）
                acts, lps, vals = self.agents['red'].act_batch(
                    flat, masks=(pool.masks_flat() if self.use_fire_mask else None))
                blk = pool.run_block(acts)
                obs_b, rew_b, done_b = blk['obs_before'], blk['rew'], blk['done']
                cleared_b = blk.get('cleared')
                alive_b = blk.get('red_alive')
                bs = obs_b.shape[0]
                # 把整块写入 PPO 缓冲（红方；蓝方为靶机/规则AI，无训练缓冲）
                lbl_b = blk.get('labels')
                for t in range(bs):
                    ob_t = obs_b[t].reshape(n_envs * n_red, -1)
                    # 标签原本是 (workers, envs, n_red)，先展平成与 obs/actions 同一顺序
                    lbl_t = (lbl_b[t].reshape(n_envs * n_red) if lbl_b is not None else None)
                    for i in range(n_envs):
                        for j in range(n_red):
                            k = i * n_red + j
                            fl = int(lbl_t[k]) if lbl_t is not None else -1
                            self.agents['red'].remember(ob_t[k], acts[k], lps[k], vals[k],
                                                        float(rew_b[t, i]), bool(done_b[t, i]),
                                                        fire_label=fl)
                        ep_rewards[i] += float(rew_b[t, i])
                        ep_kills[i] += int(blk['kills'][t, i])
                        ep_damage[i] += float(blk['dmg'][t, i])
                    # 结束环境：只记账，不做 PPO 更新
                    # （更新统一由外层「缓冲满 500」触发。早期版本在这里逐环境调用
                    #  update()，而 64 个环境常在同一步一起结束，会瞬间更新 64 次，
                    #  策略在训练初期就被压死 → 实测伤害恒为 0。）
                    for i in range(n_envs):
                        if not done_b[t, i]:
                            continue
                        red_curve.append(ep_rewards[i])
                        blue_curve.append(0.0)
                        if len(red_curve) > 500:
                            red_curve = red_curve[-500:]
                            blue_curve = blue_curve[-500:]
                        ep_count += 1
                        self._recent_results.append({
                            'cleared': bool(cleared_b is not None and cleared_b[t, i]),
                            'red_reward': ep_rewards[i],
                            'blue_reward': 0.0,
                            'red_kills': ep_kills[i],
                            'red_damage': ep_damage[i],
                            'red_hunt': 0})
                        criteria = self._stage_criteria
                        if criteria and self._pass_check(criteria):
                            self._stage_passed = True
                            n_res = len(self._recent_results)
                            print(f"[bg] ★ 阶段达成（近{n_res}回合："
                                  f"清场率{sum(1 for r in self._recent_results if r['cleared'])/n_res:.0%} "
                                  f"均击杀{sum(r['red_kills'] for r in self._recent_results)/n_res:.1f} "
                                  f"均伤害{sum(r['red_damage'] for r in self._recent_results)/n_res:.0f}），提前晋级",
                                  flush=True)
                        ep_rewards[i] = 0.0
                        ep_kills[i] = 0
                        ep_damage[i] = 0.0
                        if ep_count % 5 == 0:
                            print(f"[bg] Ep {ep_count}/{self.episodes} "
                                  f"红 {red_curve[-1]:.1f} "
                                  f"击杀{self._recent_results[-1]['red_kills']} "
                                  f"伤害{self._recent_results[-1]['red_damage']:.0f}", flush=True)
                        if ep_count % 10 == 0:
                            do_save()
                # 缓冲够大就更新并把新权重下发
                if len(self.agents['red'].b_obs) >= upd_buf:
                    self.agents['red'].update(0.0)
                    pool.sync_weights(self.agents['red'])
                self._step_counter += 1
                if self._step_counter % self.snapshot_every == 0:
                    self._publish_parallel(pool, ep_count, red_curve)
        finally:
            pool.close()
        if not self._stop_event.is_set():
            do_save()
        print(f"[bg] 训练结束：共 {ep_count} 回合（并行 workers={self.n_workers} "
              f"环境数={n_envs} 块长={self.block_steps}）", flush=True)

    def _publish_parallel(self, pool, ep_count, red_curve):
        """并行模式下发布快照：用一个独立的单进程环境渲染演示画面。"""
        snap = Snapshot()
        snap.ep = ep_count
        snap.total_episodes = self.episodes
        snap.red_curve = list(red_curve)
        snap.blue_curve = []
        snap.step = self._step_counter * self.block_steps
        snap.finished = False
        with self._lock:
            self._latest = snap

    def _run_serial(self):
        if self.use_curriculum:
            envs = [self.env_factory() for _ in range(self.n_envs)]
        else:
            envs = [AirCombatEnv(self.plane_type, render=False,
                                 red_ai_factory=lambda: CommandedController(),
                                 blue_ai_factory=lambda: CommandedController())
                    for _ in range(self.n_envs)]
        obs_list = [e.reset() for e in envs]
        ep_rewards = [{'red': 0.0, 'blue': 0.0} for _ in envs]
        ep_kills = [0] * self.n_envs
        ep_damage = [0.0] * self.n_envs
        red_curve, blue_curve = [], []
        ep_count = 0

        def do_save():
            for team in ('red', 'blue'):
                self.agents[team].save(model_path(f"{self.model_name}_{team}"))
            _write_modelset([f"{self.model_name}_red", f"{self.model_name}_blue"])
            print(f"[bg] 已保存模型 → models/{self.model_name}_red.pt / _blue.pt", flush=True)

        def obs_rows(obs, team):
            """返回该队神经槽位的观测列表（legacy 单机兼容为单元素列表）。"""
            v = obs.get(team)
            if v is None:
                return []
            if isinstance(v, np.ndarray):
                return [v]
            return list(v)

        def bootstrap_val(agent, rows):
            if not rows:
                return 0.0
            return agent.act(rows[0])[2]

        while ep_count < self.episodes and not self._stop_event.is_set() and not self._stage_passed:
            if self._pause_event.is_set():
                time.sleep(0.05)
                continue
            if self._save_event.is_set():
                self._save_event.clear()
                do_save()
            # 批量前向：N 个并行环境合批一次推理（GPU 高效）
            all_o_reds = [obs_rows(obs_list[i], 'red') for i in range(self.n_envs)]
            all_o_blues = [obs_rows(obs_list[i], 'blue') for i in range(self.n_envs)]
            flat_red = [o for lst in all_o_reds for o in lst]
            flat_blue = [o for lst in all_o_blues for o in lst]
            acts_r, lps_r, vals_r = self.agents['red'].act_batch(flat_red)
            acts_b, lps_b, vals_b = self.agents['blue'].act_batch(flat_blue)
            # 按环境切回
            pos_r, pos_b = 0, 0
            for i in range(self.n_envs):
                env = envs[i]
                nr = len(all_o_reds[i]); nb = len(all_o_blues[i])
                ar = acts_r[pos_r:pos_r + nr]; lr = lps_r[pos_r:pos_r + nr]; vr = vals_r[pos_r:pos_r + nr]
                pos_r += nr
                ab = acts_b[pos_b:pos_b + nb]; lb = lps_b[pos_b:pos_b + nb]; vb = vals_b[pos_b:pos_b + nb]
                pos_b += nb
                if self.use_curriculum:
                    actions = {'red': ar, 'blue': ab}
                else:
                    actions = {'red': ar[0] if ar else Action(), 'blue': ab[0] if ab else Action()}
                n_obs, rew, done, _ = env.step(actions)
                for k in range(nr):
                    self.agents['red'].remember(all_o_reds[i][k], ar[k], lr[k], vr[k], rew['red'], done)
                for k in range(nb):
                    self.agents['blue'].remember(all_o_blues[i][k], ab[k], lb[k], vb[k], rew['blue'], done)
                obs_list[i] = n_obs
                ep_rewards[i]['red'] += rew['red']
                ep_rewards[i]['blue'] += rew['blue']
                ep_kills[i] += sum(1 for u in env.world.kills if u.team == 'blue')
                ep_damage[i] += sum(d for (t, _u), d in env.world.damage_credit.items() if t == 'red')
                if done:
                    terminal = env.world.base_destroyed() is not None or \
                        len(env.world.aircraft.alive_list()) == 0
                    rows_r = obs_rows(obs_list[i], 'red')
                    rows_b = obs_rows(obs_list[i], 'blue')
                    lv_r = 0.0 if terminal else bootstrap_val(self.agents['red'], rows_r)
                    lv_b = 0.0 if terminal else bootstrap_val(self.agents['blue'], rows_b)
                    self.agents['red'].update(lv_r)
                    self.agents['blue'].update(lv_b)
                    red_curve.append(ep_rewards[i]['red'])
                    blue_curve.append(ep_rewards[i]['blue'])
                    if len(red_curve) > 500:
                        red_curve = red_curve[-500:]
                        blue_curve = blue_curve[-500:]
                    ep_count += 1
                    # 阶段达成检测：达标则提前晋级下一阶段
                    cleared = env._cleared() if hasattr(env, '_cleared') else False
                    hunt = env.hunt_count.get('red', 0) if hasattr(env, 'hunt_count') else 0
                    self._recent_results.append({'cleared': cleared,
                                                 'red_reward': ep_rewards[i]['red'],
                                                 'blue_reward': ep_rewards[i]['blue'],
                                                 'red_kills': ep_kills[i],
                                                 'red_damage': ep_damage[i],
                                                 'red_hunt': hunt})
                    criteria = getattr(getattr(env, 'stage', None), 'pass_criteria', None)
                    if criteria and self._pass_check(criteria):
                        self._stage_passed = True
                        nr = len(self._recent_results)
                        print(f"[bg] ★ 阶段达成（近{nr}回合："
                              f"清场率{sum(1 for r in self._recent_results if r['cleared'])/nr:.0%} "
                              f"胜率{sum(1 for r in self._recent_results if r['red_reward']>r['blue_reward'])/nr:.0%} "
                              f"均击杀{sum(r['red_kills'] for r in self._recent_results)/nr:.1f} "
                              f"均伤害{sum(r['red_damage'] for r in self._recent_results)/nr:.0f}），提前晋级", flush=True)
                    ep_rewards[i] = {'red': 0.0, 'blue': 0.0}
                    ep_kills[i] = 0
                    ep_damage[i] = 0.0
                    obs_list[i] = env.reset()
                    if ep_count % 5 == 0:
                        print(f"[bg] Ep {ep_count}/{self.episodes} "
                              f"红 {red_curve[-1]:.1f} / 蓝 {blue_curve[-1]:.1f} "
                              f"击杀{self._recent_results[-1]['red_kills']} "
                              f"伤害{self._recent_results[-1]['red_damage']:.0f}", flush=True)
                    if ep_count % 10 == 0:
                        do_save()
            # 长回合防缓冲无限增长：用末态价值 bootstrap 做中途更新。
            # 缓冲上限 500 是实测调参结果：回合长 2000 步时，只在回合结束更新一次
            # 会让 PPO 每回合仅做一轮小批量梯度（KL 剧烈震荡、价值函数发散、策略退化），
            # 改为每 500 步更新一次后，S1 阶段伤害 70→109、击杀 0.20→0.80。
            if len(self.agents['red'].b_obs) >= self.update_buffer:
                lv_r = bootstrap_val(self.agents['red'], obs_rows(obs_list[0], 'red'))
                lv_b = bootstrap_val(self.agents['blue'], obs_rows(obs_list[0], 'blue'))
                self.agents['red'].update(lv_r)
                self.agents['blue'].update(lv_b)
            # 发布快照（演示环境取 envs[0]）
            self._step_counter += 1
            if self._step_counter % self.snapshot_every == 0:
                self._publish(envs[0].world, ep_count, self.episodes,
                              red_curve, blue_curve,
                              ep_rewards[0]['red'], ep_rewards[0]['blue'])
        # 结束
        if not self._stop_event.is_set():
            do_save()
        self._publish(envs[0].world, ep_count, self.episodes,
                      red_curve, blue_curve,
                      ep_rewards[0]['red'], ep_rewards[0]['blue'], finished=True)
        print(f"[bg] 训练结束：共 {ep_count} 回合")


# ---------- 快照渲染（前端只依赖 Snapshot，不接触世界对象） ----------
def _draw_snapshot_entities(screen, camera, snap):
    for (kind, team, x, y, angle, hp, maxhp, size, color, subtype) in snap.entities:
        if kind == 'aircraft':
            s = size
            local = [(s * 0.7, 0), (-s * 0.7, -s * 0.7), (-s * 0.7, s * 0.7)]
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            pts = [camera.world_to_screen(x + lx * cos_a - ly * sin_a,
                                          y + lx * sin_a + ly * cos_a) for lx, ly in local]
            pygame.draw.polygon(screen, color, pts)
            sx, sy = camera.world_to_screen(x, y)
            pygame.draw.rect(screen, BLACK, (sx - 10, sy - 14, 20, 3))
            pygame.draw.rect(screen, GREEN, (sx - 10, sy - 14, int(20 * hp / maxhp), 3))
        elif kind == 'building':
            sx, sy = camera.world_to_screen(x, y)
            r = max(4, int(size * camera.zoom))
            pygame.draw.rect(screen, color, (sx - r // 2, sy - r // 2, r, r))
        elif kind == 'ground':
            sx, sy = camera.world_to_screen(x, y)
            r = max(3, int(size * camera.zoom))
            pygame.draw.circle(screen, color, (sx, sy), r)
        elif kind == 'bullet':
            sx, sy = camera.world_to_screen(x, y)
            pygame.draw.circle(screen, YELLOW, (sx, sy), max(2, int(3 * camera.zoom)))
        elif kind == 'bomb':
            sx, sy = camera.world_to_screen(x, y)
            pygame.draw.circle(screen, ORANGE, (sx, sy), max(3, int(5 * camera.zoom)))
        elif kind == 'missile':
            sx, sy = camera.world_to_screen(x, y)
            pygame.draw.circle(screen, color, (sx, sy), max(3, int(4 * camera.zoom)))


def _draw_snap_hud(screen, snap):
    txt = FONT.render(f"红方资源: {int(snap.resources['red'])}    蓝方资源: {int(snap.resources['blue'])}",
                      True, BLACK)
    screen.blit(txt, (10, 10))
    txt2 = SMALL_FONT.render(
        f"步数 {snap.step}  训练FPS {snap.train_fps:.0f}  回合 {snap.ep}/{snap.total_episodes}"
        f"  本回合 红 {snap.red_reward:.0f} / 蓝 {snap.blue_reward:.0f}", True, GRAY)
    screen.blit(txt2, (10, 32))


def _draw_snap_minimap(screen, snap):
    w, h = 180, 120
    x, y = SCREEN_W - w - 10, SCREEN_H - h - 10
    pygame.draw.rect(screen, (240, 240, 240), (x, y, w, h))
    pygame.draw.rect(screen, GRAY, (x, y, w, h), 1)
    sx, sw = x + w * 0.1, w * 0.8
    sy, sh = y + h * 0.1, h * 0.8
    for (kind, team, ex, ey, *_rest) in snap.entities:
        px = int(sx + (ex / WORLD_W) * sw)
        py = int(sy + (ey / WORLD_H) * sh)
        col = RED if team == 'red' else (BLUE if team == 'blue' else GRAY)
        if kind == 'aircraft':
            pygame.draw.circle(screen, col, (px, py), 3)
        elif kind == 'building':
            pygame.draw.rect(screen, col, (px - 2, py - 2, 4, 4))
        elif kind == 'ground':
            pygame.draw.circle(screen, col, (px, py), 1)


def _draw_train_curve(screen, red_curve, blue_curve, ep, steps, speed, paused, total_episodes=0):
    x0, y0, w, h = SCREEN_W - 380, 60, 360, 150
    pygame.draw.rect(screen, (245, 245, 245), (x0, y0, w, h))
    pygame.draw.rect(screen, GRAY, (x0, y0, w, h), 1)
    title = SMALL_FONT.render(
        f"后台全速训练  Episode {ep}" + (f"/{total_episodes}" if total_episodes else ""), True, BLACK)
    screen.blit(title, (x0 + 6, y0 + 4))

    def draw_line(pts, color):
        if len(pts) < 2:
            return
        maxv = max(abs(min(pts)), abs(max(pts)), 1.0)
        px = [x0 + 6 + i * (w - 12) / (len(pts) - 1) for i in range(len(pts))]
        py = [y0 + h - 20 - ((p + maxv) / (2 * maxv)) * (h - 30) for p in pts]
        pygame.draw.lines(screen, color, False, list(zip(px, py)), 2)

    draw_line(red_curve, RED)
    draw_line(blue_curve, BLUE)
    screen.blit(SMALL_FONT.render("红方", True, RED), (x0 + 6, y0 + h - 18))
    screen.blit(SMALL_FONT.render("蓝方", True, BLUE), (x0 + 50, y0 + h - 18))
    info = SMALL_FONT.render(f"步数 {steps}  演示倍速 x{speed}  {'暂停' if paused else '运行'}", True, BLACK)
    screen.blit(info, (x0 + 6, y0 + 26))
    keys = SMALL_FONT.render("P 暂停  S 保存模型  1/2/4 演示倍速  ESC 退出", True, GRAY)
    screen.blit(keys, (x0 + 6, y0 + 44))


def run_train_bg(screen, plane_type, model_name, episodes=200, n_envs=2):
    """训练入口：后台线程全速并行演算（无频率限制），前端实时抓取最新快照演示。"""
    worker = TrainWorker(plane_type, model_name, episodes=episodes, n_envs=n_envs)
    worker.start()
    camera = Camera(SCREEN_W, SCREEN_H)
    camera.follow = 'red'
    clock = pygame.time.Clock()
    paused = False
    speed = 1          # 演示倍速：每 speed 帧抓取一次最新快照
    frame = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                worker.stop()
                pygame.quit(); sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    worker.stop()
                    running = False
                elif event.key == K_p:
                    paused = not paused
                    worker.set_pause(paused)
                elif event.key == K_s:
                    worker.request_save()
                    print("[front] 已请求保存模型（后台线程执行中）")
                elif event.key == K_1:
                    speed = 1
                elif event.key == K_2:
                    speed = 2
                elif event.key == K_4:
                    speed = 4
                elif event.key == K_f:
                    camera.follow = None if camera.follow else 'red'
            camera.handle(event)
        frame += 1
        if frame % speed != 0:
            clock.tick(FPS)
            continue
        snap = worker.latest_snapshot()
        if snap is None:
            clock.tick(FPS)
            continue
        # 相机跟随红机
        if camera.follow:
            for (kind, team, x, y, *_rest) in snap.entities:
                if kind == 'aircraft' and team == camera.follow:
                    camera.x, camera.y = x, y
                    break
        screen.fill(WHITE)
        camera.draw_grid(screen)
        _draw_snapshot_entities(screen, camera, snap)
        _draw_snap_hud(screen, snap)
        _draw_snap_minimap(screen, snap)
        _draw_train_curve(screen, snap.red_curve, snap.blue_curve, snap.ep,
                          snap.step, speed, paused, snap.total_episodes)
        hint = SMALL_FONT.render(
            "后台全速演算中（不受渲染帧率限制）  P 暂停  S 保存  1/2/4 演示倍速  F 跟随  ESC 退出",
            True, BLACK)
        screen.blit(hint, (10, SCREEN_H - 24))
        pygame.display.flip()
        clock.tick(FPS)
        if snap.finished:
            running = False
    worker.join(timeout=5)
    # 结束画面
    snap = worker.latest_snapshot()
    screen.fill(WHITE)
    t = MENU_FONT.render("训练完成", True, BLACK)
    screen.blit(t, (SCREEN_W // 2 - t.get_width() // 2, 280))
    if snap:
        s = BIG_FONT.render(f"共 {snap.total_episodes} 回合  最终 红 {snap.red_curve[-1] if snap.red_curve else 0:.1f}"
                            f" / 蓝 {snap.blue_curve[-1] if snap.blue_curve else 0:.1f}   按任意键返回",
                            True, BLACK)
        screen.blit(s, (SCREEN_W // 2 - s.get_width() // 2, 360))
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); sys.exit()
            if event.type in (KEYDOWN, MOUSEBUTTONDOWN):
                waiting = False
    print("[train] 后台训练结束")


# ======================================================================
# 8.6 课程训练主循环（分阶段由易到难）
# ======================================================================
def _run_curriculum_mixed(plane_type, model_prefix, stages, episodes_per_stage,
                          render, screen, view, n_workers, envs_per_worker, block_steps,
                          fire_aux_coef=0.0):
    """阶段混合训练：一轮同时训练全部阶段，避免顺序课程的灾难性遗忘。

    实现方式：并行池的每个环境轮流绑定不同阶段（stage_mix_m=阶段数），
    因此每个 rollout 块里同时包含"定点打靶"与"对抗空战"两类经验，
    基础行为不会被后续阶段覆盖。"""
    total_ep = max(60, episodes_per_stage)
    ckpt_red = model_path(f"{model_prefix}_mixed_red")
    print(f"\n===== 阶段混合训练：{len(stages)} 个阶段并行采样 =====", flush=True)
    for i, s in enumerate(stages):
        print(f"      环境组 {i}: {s.name}", flush=True)

    worker = TrainWorker(plane_type, f"{model_prefix}_mixed",
                         episodes=total_ep, n_envs=4,
                         env_factory=lambda st=stages[0]: CurriculumEnv(st, render=False),
                         parallel=True, n_workers=n_workers,
                         envs_per_worker=envs_per_worker, block_steps=block_steps,
                         stage_mix=True, fire_aux_coef=fire_aux_coef)
    worker.start()
    if render:
        clock = pygame.time.Clock()
        paused = False
        speed = 1
        frame = 0
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    worker.stop(); pygame.quit(); sys.exit()
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        worker.stop(); running = False
                    elif event.key == K_p:
                        paused = not paused; worker.set_pause(paused)
                    elif event.key == K_s:
                        worker.request_save()
                view.camera.handle(event)
            frame += 1
            if frame % speed:
                clock.tick(FPS)
                continue
            snap = worker.latest_snapshot()
            if snap is None:
                clock.tick(FPS)
                continue
            screen.fill(WHITE)
            view.camera.draw_grid(screen)
            _draw_snap_hud(screen, snap)
            _draw_train_curve(screen, snap.red_curve, snap.blue_curve, snap.ep,
                              snap.step, speed, paused, snap.total_episodes)
            hint = SMALL_FONT.render("阶段混合训练（全部阶段并行采样）  P 暂停  S 保存  ESC 停止",
                                     True, BLACK)
            screen.blit(hint, (10, SCREEN_H - 24))
            pygame.display.flip()
            clock.tick(FPS)
            if snap.finished:
                running = False
        worker.join(timeout=5)
    else:
        # 无界面：阻塞等待训练线程结束。
        # 注意不能用 worker.join(timeout=10**7)——Windows 上 Thread.join 的 timeout
        # 走 threading 的 Condition.wait，过大的值会抛 OverflowError
        # （实测 `timeout value is too large`），因此改为有界轮询。
        while worker.is_alive():
            worker.join(timeout=5.0)

    summ = worker.stage_summary()
    print(f"----- 混合训练结束：回合{summ['episodes']} 清场{summ['clears']} "
          f"均奖励{summ['avg_reward']:.0f} 均击杀{summ['avg_kills']:.2f} "
          f"均伤害{summ['avg_damage']:.0f}", flush=True)
    import shutil
    for team in ('red', 'blue'):
        src = model_path(f"{model_prefix}_mixed_{team}")
        if os.path.exists(src):
            shutil.copy(src, model_path(f"{model_prefix}_final_{team}"))
    _write_modelset([f"{model_prefix}_final_red", f"{model_prefix}_final_blue"])
    print(f"最终模型已保存：models/{model_prefix}_final_red.pt / _blue.pt", flush=True)
    evaluate_curriculum(plane_type, model_prefix)


def run_curriculum(plane_type, model_prefix, episodes_per_stage=100, n_envs=2,
                   start_stage=0, render=True, episodes_override=None,
                   parallel=True, n_workers=8, envs_per_worker=8, block_steps=256,
                   stage_mix=False, fire_aux_coef=0.0):
    """课程训练。

    两种模式：
      - 顺序课程（stage_mix=False，默认）：按阶段由易到难依次训练，每阶段保存
        models/{prefix}_S{n}.pt。缺点是有**灾难性遗忘**——后段会覆盖前段学到的行为
        （实测：S0~S3 训练时能打出伤害，训练完再评估却全部归零，而 S4~S7 正常）。
      - 阶段混合（stage_mix=True）：并行环境各自采样不同阶段，一轮同时强化全部阶段，
        从根本上避免遗忘，且训练轮转更短。检查点统一保存为 _mixed。
    """
    stages = curriculum_stages(plane_type)
    os.makedirs(MODEL_DIR, exist_ok=True)
    if render:
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        view = TrainView(screen)
        view.camera.follow = 'red'
    else:
        screen, view = None, None

    if stage_mix:
        return _run_curriculum_mixed(plane_type, model_prefix, stages,
                                     episodes_per_stage, render, screen, view,
                                     n_workers, envs_per_worker, block_steps,
                                     fire_aux_coef)

    # 从 start_stage 起，每阶段一个 worker（连续课程：前一阶段检查点作为下一阶段起点）
    stage_summaries = []
    for si in range(start_stage, len(stages)):
        stage = stages[si]
        ckpt = model_path(f"{model_prefix}_S{si + 1}")
        n_ep = episodes_override if episodes_override else (stage.episodes or episodes_per_stage)
        print(f"\n===== 课程阶段 {si + 1}/{len(stages)}: {stage.name} =====", flush=True)
        print(f"      {stage.desc}", flush=True)
        print(f"      红队 {stage.red_team}  蓝队 {stage.blue_team}  地图 {stage.layout}", flush=True)

        # 阶段检查点即本阶段模型名；worker 自动加载已存在的检查点实现断点续训
        worker = TrainWorker(plane_type, f"{model_prefix}_S{si + 1}",
                             episodes=n_ep, n_envs=n_envs,
                             env_factory=lambda st=stage: CurriculumEnv(st, render=False),
                             parallel=parallel, n_workers=n_workers,
                             envs_per_worker=envs_per_worker, block_steps=block_steps)
        worker.start()

        if render:
            clock = pygame.time.Clock()
            paused = False
            speed = 1
            frame = 0
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        worker.stop(); pygame.quit(); sys.exit()
                    if event.type == KEYDOWN:
                        if event.key == K_ESCAPE:
                            worker.stop(); running = False
                        elif event.key == K_p:
                            paused = not paused; worker.set_pause(paused)
                        elif event.key == K_s:
                            worker.request_save()
                        elif event.key == K_1: speed = 1
                        elif event.key == K_2: speed = 2
                        elif event.key == K_4: speed = 4
                    view.camera.handle(event)
                frame += 1
                if frame % speed != 0:
                    clock.tick(FPS)
                    continue
                snap = worker.latest_snapshot()
                if snap is None:
                    clock.tick(FPS)
                    continue
                if view.camera.follow:
                    for (kind, team, x, y, *_rest) in snap.entities:
                        if kind == 'aircraft' and team == view.camera.follow:
                            view.camera.x, view.camera.y = x, y
                            break
                screen.fill(WHITE)
                view.camera.draw_grid(screen)
                _draw_snapshot_entities(screen, view.camera, snap)
                _draw_snap_hud(screen, snap)
                _draw_snap_minimap(screen, snap)
                _draw_train_curve(screen, snap.red_curve, snap.blue_curve, snap.ep,
                                  snap.step, speed, paused, snap.total_episodes)
                hint = SMALL_FONT.render(
                    f"课程阶段 {si + 1}/{len(stages)}: {stage.name}   "
                    "P 暂停  S 保存  1/2/4 倍速  ESC 跳阶段", True, BLACK)
                screen.blit(hint, (10, SCREEN_H - 24))
                pygame.display.flip()
                clock.tick(FPS)
                if snap.finished:
                    running = False
            worker.join(timeout=5)
        else:
            # 同样用有界轮询代替超大 timeout（Windows 上可能溢出）
            while worker.is_alive():
                worker.join(timeout=5.0)
        if worker.stage_passed() and stage.pass_criteria is not None:
            print(f"----- 阶段 {stage.name} ★达标提前晋级（条件 {stage.pass_criteria}）"
                  f"，检查点 → models/{model_prefix}_S{si + 1}_red.pt", flush=True)
        else:
            print(f"----- 阶段 {stage.name} 跑满回合，检查点 → models/{model_prefix}_S{si + 1}_red.pt", flush=True)
        summ = worker.stage_summary()
        stage_summaries.append((si + 1, stage.name, summ))
        print(f"      汇总：回合{summ['episodes']} 清场{summ['clears']} "
              f"清场率{summ['clear_rate']:.0%} 胜率{summ['win_rate']:.0%} "
              f"均奖励{summ['avg_reward']:.0f} 均击杀{summ['avg_kills']:.2f} "
              f"均伤害{summ['avg_damage']:.0f}", flush=True)

    # ---- 挑选最佳阶段检查点作为最终模型 ----
    # 原实现直接把最后一个阶段（S7 多环境对抗评估）复制成 final，而该阶段并未专门
    # 训练、也未必最强。现在按「清场率 → 胜率 → 平均伤害」排序选最佳阶段。
    def score(item):
        _, _, s = item
        return (s['clear_rate'], s['win_rate'], s['avg_damage'])

    best = max(stage_summaries, key=score) if stage_summaries else None
    import shutil
    if best is not None:
        bi, bname, bs = best
        for team in ('red', 'blue'):
            src = model_path(f"{model_prefix}_S{bi}_{team}")
            if os.path.exists(src):
                shutil.copy(src, model_path(f"{model_prefix}_final_{team}"))
        print(f"\n最佳阶段为 {bname}（清场率 {bs['clear_rate']:.0%}、胜率 {bs['win_rate']:.0%}、"
              f"均伤害 {bs['avg_damage']:.0f}）→ 已作为最终模型", flush=True)
    else:
        last_ckpt = f"{model_prefix}_S{len(stages)}"
        for team in ('red', 'blue'):
            src = model_path(f"{last_ckpt}_{team}")
            if os.path.exists(src):
                shutil.copy(src, model_path(f"{model_prefix}_final_{team}"))
    _write_modelset([f"{model_prefix}_final_red", f"{model_prefix}_final_blue"])
    print(f"最终模型已保存：models/{model_prefix}_final_red.pt / _blue.pt", flush=True)
    evaluate_curriculum(plane_type, model_prefix)


def evaluate_curriculum(plane_type, model_prefix, n_episodes=6):
    """用最终模型在所有阶段环境上确定性评估。

    修正：原实现每阶段只创建【一个】环境实例并反复 reset，而 reset 用同一套
    随机种子，等价于把同一场景评 n 次，既看不出泛化也容易全员 0 胜。
    现在每个回合用不同随机种子 + 独立环境，结果才有统计意义。"""
    agent, msg = try_load_agent(f"{model_prefix}_final_red")
    if agent is None:
        print(f"[eval] 无法评估：{msg}")
        return []
    print(f"\n========== 课程评估 {plane_type} / {model_prefix} ==========")
    print(f"评估回合：每阶段 {n_episodes} 次（不同随机种子）")
    print(f"{'阶段':<18}{'清场':>7}{'均伤害':>9}{'均击杀':>8}{'红存活':>8}{'均步数':>8}", flush=True)
    results = []
    for si, stage in enumerate(curriculum_stages(plane_type)):
        wins = 0
        tot_rew = tot_steps = 0.0
        tot_dmg = tot_kills = tot_red = 0.0
        for seed in range(n_episodes):
            random.seed(1000 + si * 100 + seed)
            env = CurriculumEnv(stage, render=False, is_eval=True)
            obs = env.reset()
            done = False
            ep_r = 0.0
            while not done:
                rows = obs.get('red') or []
                if rows:
                    a, _, _ = agent.act(rows[0], deterministic=True)
                    acts = {'red': [a], 'blue': []}
                else:
                    acts = {'red': [], 'blue': []}
                obs, rew, done, _ = env.step(acts)
                ep_r += rew['red']
                tot_dmg += sum(d for (t, _u), d in env.world.damage_credit.items() if t == 'red')
                tot_kills += sum(1 for u in env.world.kills if u.team == 'blue')
            tot_rew += ep_r
            tot_steps += env.steps
            tot_red += len(env.world.aircraft_list_of('red'))
            if env._cleared():
                wins += 1
        row = (stage.name, wins, tot_rew / n_episodes, tot_steps / n_episodes,
               wins / n_episodes, tot_dmg / n_episodes, tot_kills / n_episodes,
               tot_red / n_episodes)
        results.append(row)
        print(f"{row[0]:<18}{row[1]:>4}/{n_episodes}{row[5]:>9.0f}{row[6]:>8.2f}"
              f"{row[7]:>8.2f}{row[3]:>8.0f}", flush=True)
    # 选出最佳阶段（清场数优先，其次平均伤害）
    best = max(results, key=lambda r: (r[1], r[5])) if results else None
    if best is not None:
        print(f"\n最佳阶段：{best[0]}（清场 {best[1]}/{n_episodes}，均伤害 {best[5]:.0f}，"
              f"均击杀 {best[6]:.2f}）")
    return results


def try_load_agent(model_name):
    """安全加载模型：维度/版本不兼容时返回 (None, 中文原因) 而不是崩溃。"""
    p = model_path(model_name)
    if not os.path.exists(p):
        return None, f"模型文件不存在：{p}"
    agent = PPOAgent(OBS_DIM, ACT_DIMS)
    try:
        data = agent.load(p)
    except ValueError as ex:
        return None, str(ex)
    except Exception as ex:
        return None, f"加载失败：{type(ex).__name__}: {ex}"
    ver = data.get('obs_version')
    return agent, f"OK（obs_dim={data.get('obs_dim')} obs_version={ver}）"


def run_normal_mode(screen, layout='duel', map_variants=4):
    """常规模式：红方手动键盘控制 vs 蓝方规则AI（可选现实向地图）。"""
    world = World('standard', red_ai_factory=lambda: ManualController(),
                  blue_ai_factory=lambda: RuleAI('blue'), layout=layout,
                  map_variants=map_variants)
    camera = Camera(SCREEN_W, SCREEN_H)
    camera.follow = 'red'
    hud = HUD()
    minimap = Minimap()
    clock = pygame.time.Clock()
    running = True
    run_no = 1
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                if event.key == K_f:
                    camera.follow = None if camera.follow else 'red'
                if event.key == K_r:          # R 换一局（同地图变体随机）
                    world.reset()
                    run_no += 1
            camera.handle(event)
        world.step()
        camera.update_follow(world, 'red')
        screen.fill(WHITE)
        camera.draw_grid(screen)
        camera.draw_terrain(screen, world)
        for pool in (world.buildings, world.ground, world.aircraft,
                     world.bullets, world.bombs, world.missiles):
            for e in pool.alive_list():
                e.draw(screen, camera)
        hud.draw(screen, world)
        minimap.draw(screen, world)
        cn = REALISTIC_MAPS.get(layout, (layout, '', 1))[0]
        hint = SMALL_FONT.render(
            f"[{cn}] W/S 油门  A/D 转向  Q/E 转头  SPACE 机炮  M 导弹  B 炸弹  F 诱饵  "
            f"R 重开(第{run_no}局)  滚轮缩放  中键拖动  F 跟随  ESC 返回",
            True, BLACK)
        screen.blit(hint, (10, SCREEN_H - 24))
        pygame.display.flip()
        clock.tick(FPS)
        lost = world.base_destroyed()
        if lost:
            print(f"基地被摧毁：{lost} 方落败")
            running = False


def run_watch_mode(screen, red_model, blue_model, layout='duel', map_variants=4):
    """观战模式：加载两个模型对打（可选现实向地图）。"""
    red_agent, rmsg = try_load_agent(red_model)
    blue_agent, bmsg = try_load_agent(blue_model)
    if red_agent is None or blue_agent is None:
        print(f"[watch] 无法开始观战：\n  红方 {red_model}: {rmsg}\n  蓝方 {blue_model}: {bmsg}")
        _modal_message(screen, "模型加载失败", [red_model + ': ' + rmsg.splitlines()[0],
                                                blue_model + ': ' + bmsg.splitlines()[0]])
        return
    print(f"[watch] 红 {red_model} {rmsg}")
    print(f"[watch] 蓝 {blue_model} {bmsg}")
    env = AirCombatEnv('standard', render=False, red_ai_factory=lambda: CommandedController(),
                       blue_ai_factory=lambda: CommandedController(), layout=layout,
                       map_variants=map_variants)
    view = TrainView(screen)
    view.camera.follow = 'red'
    clock = pygame.time.Clock()
    obs = env.reset()
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return
                if event.key == K_f:
                    view.camera.follow = None if view.camera.follow else 'red'
                if event.key == K_r:
                    obs = env.reset()
                    done = False
            view.camera.handle(event)
        act_r, _, _ = red_agent.act(_obs_for(obs, 'red'), deterministic=True)
        act_b, _, _ = blue_agent.act(_obs_for(obs, 'blue'), deterministic=True)
        obs, rewards, done, _ = env.step({'red': act_r, 'blue': act_b})
        view.ep = env.steps // 500
        view.steps = env.steps
        view.draw(env.world)
        clock.tick(FPS)
    # 结束画面
    screen.fill(WHITE)
    t = MENU_FONT.render("对战结束", True, BLACK)
    screen.blit(t, (SCREEN_W // 2 - t.get_width() // 2, 300))
    lost = env.world.base_destroyed()
    red_alive = len(env.world.aircraft_list_of('red'))
    blue_alive = len(env.world.aircraft_list_of('blue'))
    if lost:
        s = BIG_FONT.render(f"{'蓝方' if lost == 'red' else '红方'} 获胜！（基地被毁）按任意键返回", True, BLACK)
    elif red_alive == 0 or blue_alive == 0:
        s = BIG_FONT.render(f"{'蓝方' if red_alive == 0 else '红方'} 获胜！按任意键返回", True, BLACK)
    else:
        s = BIG_FONT.render(f"平局（红剩{red_alive} 蓝剩{blue_alive}）按任意键返回", True, BLACK)
    screen.blit(s, (SCREEN_W // 2 - s.get_width() // 2, 380))
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); sys.exit()
            if event.type in (KEYDOWN, MOUSEBUTTONDOWN):
                waiting = False


def _modal_message(screen, title, lines):
    """简单信息弹窗（按任意键关闭）。"""
    while True:
        screen.fill(WHITE)
        t = MENU_FONT.render(title, True, DARK_RED)
        screen.blit(t, (SCREEN_W // 2 - t.get_width() // 2, 200))
        for i, ln in enumerate(lines):
            s = FONT.render(ln[:90], True, BLACK)
            screen.blit(s, (SCREEN_W // 2 - s.get_width() // 2, 280 + i * 28))
        s = SMALL_FONT.render("按任意键返回", True, GRAY)
        screen.blit(s, (SCREEN_W // 2 - s.get_width() // 2, 400))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); sys.exit()
            if event.type in (KEYDOWN, MOUSEBUTTONDOWN):
                return
        pygame.time.Clock().tick(30)


# ======================================================================
# 9. 入口
# ======================================================================
def bench_training(episodes=10, stage_idx=1):
    """训练吞吐基准：对比串行多环境与跨进程并行，用于验证并行化收益。

    这是「优化程序运行逻辑」的量化依据，可用 `py AWAI\\AWAI.py --bench` 复现。"""
    import glob as _glob
    ep = episodes
    stages = curriculum_stages('standard')
    if not 0 <= stage_idx < len(stages):
        awai_cli.fail(f'参数错误：--stage 必须在 0 到 {len(stages) - 1} 之间，'
                      f'收到 {stage_idx}。')
    if ep <= 0:
        awai_cli.fail(f'参数错误：--episodes 必须是正整数，收到 {ep}。')
    print(f'训练吞吐基准：阶段 {stages[stage_idx].name}，每个环境 {ep} 回合 '
          f'（每回合约 {stages[stage_idx].max_steps} 步）', flush=True)

    def once(label, parallel, n_workers, envs_per_worker, block):
        st = stages[stage_idx]
        t0 = time.time()
        w = TrainWorker('standard', '_bench_tmp', episodes=ep, n_envs=4,
                        env_factory=lambda: CurriculumEnv(st, render=False),
                        parallel=parallel, n_workers=n_workers,
                        envs_per_worker=envs_per_worker, block_steps=block)
        w.start()
        while w.is_alive() and time.time() - t0 < 3600:
            w.join(timeout=5)
        dt = max(1e-6, time.time() - t0)
        for f in _glob.glob(model_path('_bench_tmp*')):
            os.remove(f)
        n_total = (n_workers * envs_per_worker) if parallel else 4
        env_steps = ep * n_total * stages[stage_idx].max_steps / dt
        print(f'  {label:<24} 环境数 {n_total:>3}  耗时 {dt:>6.1f}s  '
              f'{env_steps:>9.0f} 环境步/秒', flush=True)
        return env_steps

    base = once('串行（基线）', False, 1, 1, 1)
    results = [('并行 8×8 blk256', 8, 8, 256), ('并行 8×4 blk128', 8, 4, 128),
               ('并行 16×4 blk256', 16, 4, 256)]
    for label, nw, epw, blk in results:
        s = once(label, True, nw, epw, blk)
        print(f'    → 加速 {s / base:.2f}×', flush=True)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("AWAI 空战 AI 平台 v3")
    map_layout, map_variants = 'duel', 4
    while True:
        choice = main_menu(screen)
        if choice == '常规模式':
            run_normal_mode(screen, map_layout, map_variants)
            screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        elif choice == 'AI 训练':
            setup = training_setup(screen)
            if setup:
                plane_type, model_name, episodes = setup
                print(f"开始后台训练 {plane_type}, 模型前缀: {model_name or 'default'}, "
                      f"设备: {DEVICE}")
                run_train_bg(screen, plane_type, model_name or 'default', episodes, n_envs=2)
                screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        elif choice == '课程训练':
            setup = training_setup(screen)
            if setup:
                plane_type, model_name, episodes = setup
                print(f"开始课程训练 {plane_type}, 前缀: {model_name or 'curriculum'}, 设备: {DEVICE}")
                run_curriculum(plane_type, model_name or 'curriculum', episodes_per_stage=episodes,
                               n_envs=2, render=True)
                screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        elif choice == '选择地图':
            sel = map_setup(screen, '选择对战地图（常规/观战模式生效）')
            if sel:
                map_layout, map_variants = sel
                print(f"当前地图: {map_layout}（{REALISTIC_MAPS[map_layout][0]}），变体数 {map_variants}")
        elif choice == '观战模式':
            sel = watch_setup(screen)
            if sel:
                run_watch_mode(screen, sel[0], sel[1], map_layout, map_variants)
                screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        elif choice == '模型管理':
            model_manager(screen)
        elif choice == '退出':
            pygame.quit()
            sys.exit()


def _print_models():
    """打印 models/ 里的全部模型及其与当前代码的兼容性。"""
    names = list_models()
    if not names:
        print(f'{MODEL_DIR} 里没有任何 .pt 模型。')
        print('先训练一个：py AWAI\\AWAI.py --curriculum standard '
              '--episodes 100 --norender --prefix standard_v3')
        return
    usable, unusable = [], []
    for n in names:
        ok, why = model_status(n)
        (usable if ok else unusable).append((n, why))
    print(f'models/ 共 {len(names)} 个模型；当前代码要求 '
          f'obs_dim={OBS_DIM}、obs_version={OBS_VERSION}。')
    print(f'\n可直接使用（{len(usable)} 个）：')
    for n, why in usable:
        size = os.path.getsize(model_path(n)) / 1024 / 1024
        print(f'  {n:<36}{size:>6.2f} MB  {why}')
    if unusable:
        print(f'\n不可用（{len(unusable)} 个，加载会被中文诊断拒绝而不是崩溃）：')
        for n, why in unusable[:40]:
            print(f'  {n:<36}{"":>6}     {why}')
        if len(unusable) > 40:
            print(f'  ... 其余 {len(unusable) - 40} 个略')


def _print_maps():
    """打印可用地图布局。"""
    print('可用地图布局（菜单「选择地图」或辅助脚本的 --map 参数里使用）：')
    print(f'  {"布局名":<20}{"名称":<14}说明')
    for key, (cn, desc, variants) in REALISTIC_MAPS.items():
        print(f'  {key:<20}{cn:<14}{desc}（变体 {variants} 种）')
    print('\n  山体/丘陵/城区会阻断视线与雷达，低空可借地形遮蔽突防。')


def _print_stages(plane_type='standard'):
    """打印指定机型的课程阶段。"""
    stages = curriculum_stages(plane_type)
    print(f'{plane_type} 的课程阶段（共 {len(stages)} 个）：')
    for i, s in enumerate(stages):
        print(f'  S{i}  {s.name:<18}地图 {s.layout:<20}上限 {s.max_steps} 步')
        print(f'      {s.desc}')


def cli_main(argv):
    """命令行入口：解析参数并分派到对应模式。

    完整的参数说明见模块 docstring（也就是 `py AWAI\\AWAI.py --help` 打印的内容）。
    返回进程退出码；不带任何参数时由 `__main__` 直接进入图形主菜单，不走这里。"""
    os.makedirs(MODEL_DIR, exist_ok=True)

    prog = 'py AWAI\\AWAI.py' if awai_cli.IS_WINDOWS else 'python3 AWAI/AWAI.py'
    parser = awai_cli.Parser(
        prog=prog, add_help=False, allow_abbrev=False,
        description='AWAI v3 空战 AI 训练平台。不带参数运行会进入图形主菜单。')
    parser.add_argument('-h', '--help', action='help', help='显示完整帮助并退出')
    parser.add_argument('-V', '--version', action='version',
                        version=f'AWAI v{awai_cli.AWAI_VERSION}', help='显示版本并退出')

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--curriculum', metavar='机型', choices=sorted(AIRCRAFT_SPECS),
                      help='按 8 阶段课程训练（机型见帮助里的「可用机型」）')
    mode.add_argument('--evaluate', nargs='+', metavar='参数',
                      help='评估模型：--evaluate <机型> [模型前缀]')
    mode.add_argument('--bench', action='store_true',
                      help='训练吞吐基准：串行 vs 跨进程并行')
    mode.add_argument('--list-models', action='store_true',
                      help='列出 models/ 里的模型及其兼容性')
    mode.add_argument('--list-maps', action='store_true', help='列出可用地图布局')
    mode.add_argument('--list-stages', action='store_true', help='列出课程阶段')

    parser.add_argument('--episodes', type=int, metavar='N',
                        help='回合数（课程训练默认 100；--bench 默认 10）')
    parser.add_argument('--n_envs', type=int, metavar='N', default=2,
                        help='单进程环境数，默认 2')
    parser.add_argument('--start', type=int, metavar='S', default=0,
                        help='课程从第 S 阶段开始（断点续训），默认 0')
    parser.add_argument('--norender', action='store_true',
                        help='不渲染窗口（无头训练，更快）')
    parser.add_argument('--prefix', metavar='NAME',
                        help='模型前缀（课程训练默认 <机型>_cur）')
    parser.add_argument('--noparallel', action='store_true',
                        help='关闭跨进程并行，退回单进程多环境')
    parser.add_argument('--stage-mix', action='store_true',
                        help='阶段混合采样（缓解课程顺序训练的灾难性遗忘）')
    parser.add_argument('--workers', type=int, metavar='N', default=8,
                        help='并行 worker 进程数，默认 8')
    parser.add_argument('--envs_per_worker', type=int, metavar='N', default=8,
                        help='每个 worker 的环境数，默认 8')
    parser.add_argument('--block', type=int, metavar='N', default=256,
                        help='并行块长（比 worker 数更影响吞吐），默认 256')
    parser.add_argument('--fire_aux', type=float, metavar='C', default=0.0,
                        help='火控行为克隆辅助系数，默认 0.0')
    parser.add_argument('--stage', type=int, metavar='I', default=1,
                        help='--bench 使用的课程阶段下标，默认 1')

    args = parser.parse_args(argv)

    if args.list_maps:
        _print_maps()
        return 0
    if args.list_models:
        _print_models()
        return 0
    if args.list_stages:
        _print_stages(args.curriculum or 'standard')
        return 0

    if args.bench:
        bench_training(episodes=args.episodes if args.episodes is not None else 10,
                       stage_idx=args.stage)
        return 0

    if args.curriculum:
        episodes = args.episodes if args.episodes is not None else 100
        if episodes <= 0:
            awai_cli.fail(f'参数错误：--episodes 必须是正整数，收到 {episodes}。')
        if not 0 <= args.start < 99:
            awai_cli.fail(f'参数错误：--start 不能为负数，收到 {args.start}。')
        plane_type = args.curriculum
        run_curriculum(plane_type, args.prefix or f'{plane_type}_cur',
                       episodes_per_stage=episodes, n_envs=args.n_envs,
                       start_stage=args.start, render=not args.norender,
                       parallel=not args.noparallel, n_workers=args.workers,
                       envs_per_worker=args.envs_per_worker, block_steps=args.block,
                       stage_mix=args.stage_mix, fire_aux_coef=args.fire_aux)
        return 0

    if args.evaluate:
        if len(args.evaluate) > 2:
            awai_cli.fail('参数错误：--evaluate 最多接受两个参数：<机型> [模型前缀]。')
        plane_type = args.evaluate[0]
        if plane_type not in AIRCRAFT_SPECS:
            awai_cli.fail(f'参数错误：未知机型 {plane_type!r}。\n'
                          f'  可用机型：{" ".join(sorted(AIRCRAFT_SPECS))}')
        prefix = args.evaluate[1] if len(args.evaluate) > 1 else f'{plane_type}_cur'
        evaluate_curriculum(plane_type, prefix)
        return 0

    parser.print_usage(sys.stderr)
    awai_cli.fail('错误：需要指定一个模式（--curriculum / --evaluate / --bench / --list-*）。\n'
                  '      若想打开图形界面，请不要带任何参数运行。')
    return 2


if __name__ == '__main__':
    if len(sys.argv) > 1:
        sys.exit(cli_main(sys.argv[1:]) or 0)
    else:
        main()
