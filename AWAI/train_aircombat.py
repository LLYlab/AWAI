#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""纯空战专项训练任务 —— 本项目的核心训练任务。

设计动机（实测依据）：
  标准课程 8 个阶段里有 4 个是"打靶"（layout=range，单机对静止/直线地面靶标），
  真正含对手的空战阶段只有 S4~S7。阶段混合训练给每个阶段等权，等于把一半算力
  花在不需要空战能力的打靶上。实测后果：对称 1v1 竞技场评估中，
  混合训练的 v9 胜率 0%（464 步被打光、均伤害 0），而同期 v7 有 60% 胜率。

本任务只训练"与考核同分布"的空战环境（custom_stages），并用同一评估器汇报胜率。

用法：
    py AWAI\\train_aircombat.py                      默认 120 回合，前缀 aircombat
    py AWAI\\train_aircombat.py 120 ac_v1            120 回合，输出前缀 ac_v1
    py AWAI\\train_aircombat.py -n 120 --prefix ac_v1
    py AWAI\\train_aircombat.py --eval-only ac_v1_red

环境变量 AC_MODE=duel_only（默认）时只训练 1v1 对称空战；设为其它值会额外加入
1v2 与复杂地形阶段（实测会显著拉低胜率，仅作对照）。
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

import AWAI as A

DEFAULT_EPISODES = 120
DEFAULT_PREFIX = 'aircombat'
RED_SPAWN = (0.30, 0.5)
BLUE_SPAWN = (0.70, 0.5)


def build_parser():
    p = awai_cli.Parser(prog='py AWAI\\train_aircombat.py', add_help=False,
                        allow_abbrev=False,
                        description='纯空战专项训练（只训练与考核同分布的空战环境）。')
    p.add_argument('-h', '--help', action='help', help='显示本帮助并退出')
    p.add_argument('episodes', nargs='?', type=int, default=DEFAULT_EPISODES,
                   metavar='回合数', help=f'训练回合数，默认 {DEFAULT_EPISODES}')
    p.add_argument('prefix', nargs='?', default=DEFAULT_PREFIX, metavar='输出前缀',
                   help=f'模型输出前缀，默认 {DEFAULT_PREFIX}')
    p.add_argument('-n', '--episodes', dest='episodes_opt', type=int,
                   metavar='N', help='回合数（与位置参数等价）')
    p.add_argument('--prefix', dest='prefix_opt', metavar='NAME',
                   help='模型输出前缀（与位置参数等价）')
    p.add_argument('--eval-only', metavar='模型名',
                   help='不训练，只对给定模型跑一遍胜率评估')
    return p


def _base(name, desc, red, blue, layout='range', steps=1800, variants=1, close=0.0):
    return A.StageSpec(name=name, desc=desc, red_team=red, blue_team=blue,
                       layout=layout, max_steps=steps, map_variants=variants,
                       red_spawn=RED_SPAWN, blue_spawn=BLUE_SPAWN,
                       cleared_bonus=200, damage_coef=2.0, hunt_bonus=10,
                       lock_bonus=0.05, lock_acquire_bonus=10,
                       combat_approach_coef=2.0, aim_fire_bonus=0.3, close_bonus=close)


def aircombat_stages():
    """空战课程。默认只保留"与考核完全同分布"的 1v1 对称空战，
    因为实测：加入 1v2 与复杂地形阶段会把胜率从 60% 拉低到 0%
    （混合难度过高 + 奖励尺度差异把已学到的空战策略冲掉）。"""
    mode = os.environ.get('AC_MODE', 'duel_only')
    if mode == 'duel_only':
        return [_base('AC_1v1对称', '1v1 对称空战（与考核同分布）',
                      [('standard', A.CTRL_NEURAL)], [('standard', A.CTRL_RULE)])]
    return [
        _base('AC1_基础对抗', '1v1 对称空战',
              [('standard', A.CTRL_NEURAL)], [('standard', A.CTRL_RULE)]),
        _base('AC3_同型竞技', '1v1 同机型（含基地的经典战场）',
              [('standard', A.CTRL_NEURAL)], [('standard', A.CTRL_RULE)],
              layout='duel', variants=4),
        _base('AC4_复杂地形', '1v1 山脊地形对抗',
              [('standard', A.CTRL_NEURAL)], [('standard', A.CTRL_RULE)],
              layout='mountain_corridor', variants=3),
    ]


def report_winrate(model, n=30):
    """用与 bench_duel 相同口径汇报胜率。"""
    agent, msg = A.try_load_agent(model)
    if agent is None:
        print(f'  {model}: 加载失败 ({msg.splitlines()[0]})')
        return False
    wins = losses = draws = 0
    dmg = 0.0
    for seed in range(n):
        random.seed(seed)
        st = _base('bench', '', [('standard', A.CTRL_NEURAL)],
                   [('standard', A.CTRL_RULE)])
        env = A.CurriculumEnv(st, render=False, is_eval=True)
        obs = env.reset()
        done = False
        while not done:
            rows = obs.get('red') or []
            if rows:
                a, _, _ = agent.act(rows[0], deterministic=True)
                acts = {'red': [a], 'blue': []}
            else:
                acts = {'red': [], 'blue': []}
            obs, rew, done, _ = env.step(acts)
            dmg += sum(x for (t, u), x in env.world.damage_credit.items() if t == 'red')
        red = env.world.aircraft_list_of('red')
        blue = env.world.aircraft_list_of('blue')
        if not blue and red:
            wins += 1
        elif not red and blue:
            losses += 1
        elif not red and not blue:
            draws += 1
        else:
            rp = sum(x.hp for x in red) / max(1.0, sum(x.max_hp for x in red))
            bp = sum(x.hp for x in blue) / max(1.0, sum(x.max_hp for x in blue))
            if abs(rp - bp) < 0.05:
                draws += 1
            elif rp > bp:
                wins += 1
            else:
                losses += 1
    print(f'  胜率评估（{n} 局）：胜 {wins} 负 {losses} 平 {draws} → '
          f'**胜率 {wins/n:.0%}**  均伤害 {dmg/n:.1f}', flush=True)
    return True


def main(argv):
    args = build_parser().parse_args(argv)

    if args.eval_only:
        return 0 if report_winrate(args.eval_only) else awai_cli.fail(
            f'模型无法评估：{args.eval_only}\n可选的模型：\n{A.suggested_models()}')

    episodes = args.episodes_opt if args.episodes_opt is not None else args.episodes
    prefix = args.prefix_opt or args.prefix
    if episodes <= 0:
        awai_cli.fail(f'参数错误：回合数必须是正整数，收到 {episodes}。')

    stages = aircombat_stages()
    print(f'=== 纯空战专项训练：{len(stages)} 个空战阶段并行采样，{episodes} 回合 ===', flush=True)
    for i, s in enumerate(stages):
        print(f'    环境组 {i}: {s.name}  ({s.layout})', flush=True)
    print(f'    输出前缀 {prefix}（含 fire_aux 行为克隆 0.6）', flush=True)
    t0 = time.time()
    worker = A.TrainWorker('standard', prefix, episodes=episodes, n_envs=4,
                           env_factory=lambda: A.CurriculumEnv(stages[0], render=False),
                           parallel=True, n_workers=8, envs_per_worker=8,
                           block_steps=256, stage_mix=True,
                           custom_stages=stages, fire_aux_coef=0.6)
    worker.start()
    while worker.is_alive() and time.time() - t0 < 7200:
        worker.join(timeout=5)
    s = worker.stage_summary()
    print(f'  训练完成 {s["episodes"]} 回合，耗时 {time.time()-t0:.0f}s', flush=True)
    print(f'  清场 {s["clears"]}  均击杀 {s["avg_kills"]:.2f}  均伤害 {s["avg_damage"]:.1f}',
          flush=True)
    report_winrate(f'{prefix}_red')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]) or 0)
