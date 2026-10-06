#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单阶段训练/学习诊断工具：在指定课程阶段上训练并打印伤害、击杀随回合的变化，
用于快速判断"奖励设计是否让模型真的学会了作战"。

用法：
    py AWAI\\train_probe.py                            默认 S1_定点目标，60 回合
    py AWAI\\train_probe.py S5_规则AI对抗 80
    py AWAI\\train_probe.py S3_多目标 100 --load standard_v9_final_red
    py AWAI\\train_probe.py --list-stages               列出全部阶段名

关注点：伤害(dmg) 与 击杀(kills) 是否随回合上升。若长期为 0，说明奖励/关卡
仍不可学（例如索敌奖励畸高导致"只搜索不开火"）。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import awai_cli

if __name__ == '__main__':
    awai_cli.handle_early_exit(__doc__)
awai_cli.check_requirements()

import random
import time

import numpy as np

import AWAI as A

DEFAULT_STAGE = 'S1_定点目标'
DEFAULT_EPISODES = 60


def build_parser():
    p = awai_cli.Parser(prog='py AWAI\\train_probe.py', add_help=False,
                        allow_abbrev=False,
                        description='单阶段学习诊断：奖励设计是否让模型学会作战。')
    p.add_argument('-h', '--help', action='help', help='显示本帮助并退出')
    p.add_argument('stage', nargs='?', default=DEFAULT_STAGE, metavar='阶段名',
                   help=f'课程阶段名，默认 {DEFAULT_STAGE}')
    p.add_argument('episodes', nargs='?', type=int, default=DEFAULT_EPISODES,
                   metavar='回合数', help=f'训练回合数，默认 {DEFAULT_EPISODES}')
    p.add_argument('--load', metavar='模型名',
                   help='从这个检查点继续训练；省略则从随机初始化开始')
    p.add_argument('--plane', metavar='机型', default='standard',
                   choices=sorted(A.AIRCRAFT_SPECS),
                   help='机型，默认 standard')
    p.add_argument('--list-stages', action='store_true', help='列出全部阶段后退出')
    return p


def main(argv):
    args = build_parser().parse_args(argv)

    if args.list_stages:
        return A.cli_main(['--list-stages'])

    if args.episodes <= 0:
        awai_cli.fail(f'参数错误：回合数必须是正整数，收到 {args.episodes}。')

    stages = A.curriculum_stages(args.plane)
    names = [s.name for s in stages]
    if args.stage not in names:
        awai_cli.fail(f'参数错误：未知阶段 {args.stage!r}。\n'
                      f'  可用阶段：{names}\n'
                      f'  用 --list-stages 可以看到每个阶段在练什么。')
    stage = next(s for s in stages if s.name == args.stage)

    random.seed(1234)
    env = A.CurriculumEnv(stage, render=False)
    agent = A.PPOAgent(A.OBS_DIM, A.ACT_DIMS)
    if args.load:
        got, msg = A.try_load_agent(args.load)
        if got is None:
            awai_cli.fail(f'加载失败：{args.load}\n  {msg}\n\n'
                          f'  可选的模型：\n{A.suggested_models()}')
        agent = got
        print(f'已加载 {args.load}')

    print(f'=== 训练探针 {stage.name}：{args.episodes} 回合 ===', flush=True)
    print(f'奖励系数: 索敌={stage.hunt_bonus}(一次性) 锁敌={stage.lock_bonus}/{stage.lock_acquire_bonus}(一次性) '
          f'伤害={stage.damage_coef} 接近={stage.approach_coef} 击杀倍率={stage.kill_mult}', flush=True)

    t0 = time.time()
    hist = []
    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        ep_r = dmg = kills = steps = 0
        pending = None
        rew = {'red': 0.0}
        while not done:
            rows = obs.get('red') or []
            acts, lps, vals = agent.act_batch(rows) if rows else ([], [], [])
            if pending is not None:
                po, pa, plp, pv = pending
                agent.remember(po, pa, plp, pv, rew['red'], done)
            obs, rew, done, _ = env.step({'red': list(acts), 'blue': []})
            pending = (rows[0], acts[0], lps[0], vals[0]) if rows else None
            dmg += sum(d for (t, u), d in env.world.damage_credit.items() if t == 'red')
            kills += sum(1 for u in env.world.kills if u.team == 'blue')
            ep_r += rew['red']
            steps += 1
            # 与 TrainWorker 一致：长回合做中途更新，避免每回合只更新一次导致发散
            if len(agent.b_obs) >= 500:
                rows_now = obs.get('red') or []
                lv = agent.act(rows_now[0])[2] if rows_now else 0.0
                agent.update(lv)
        terminal = env.world.base_destroyed() is not None or not env.world.aircraft_list_of('red')
        rows_end = obs.get('red') or []
        lv = 0.0 if terminal or not rows_end else agent.act(rows_end[0])[2]
        agent.update(lv)
        hist.append((ep_r, dmg, kills, steps))
        if ep % 10 == 0 or ep == 1:
            w = hist[-10:]
            print(f'  Ep{ep:>4} 奖励{np.mean([x[0] for x in w]):>8.0f} '
                  f'伤害{np.mean([x[1] for x in w]):>7.1f} 击杀{np.mean([x[2] for x in w]):>5.2f} '
                  f'步数{np.mean([x[3] for x in w]):>6.0f}  ({time.time()-t0:.0f}s)', flush=True)

    n = min(10, len(hist))
    first_d, last_d = np.mean([x[1] for x in hist[:n]]), np.mean([x[1] for x in hist[-n:]])
    first_k, last_k = np.mean([x[2] for x in hist[:n]]), np.mean([x[2] for x in hist[-n:]])
    print(f'\n首 {n} 回合:  伤害 {first_d:.1f}  击杀 {first_k:.2f}')
    print(f'末 {n} 回合:  伤害 {last_d:.1f}  击杀 {last_k:.2f}')
    verdict = '学会作战（输出上升）' if (last_d > first_d * 1.2 + 1 or last_k > first_k + 0.2) \
        else '输出未见上升，需检查奖励/关卡设计'
    print(f'判定：{verdict}')
    print(f'总耗时 {time.time()-t0:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]) or 0)
