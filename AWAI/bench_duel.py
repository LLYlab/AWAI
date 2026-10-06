#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""1v1 空战碾压评估器：神经网络（红）vs 硬编码规则 AI（蓝）。

设计说明（修正了一个长期误导性的基准）：
  原先用 `duel` 布局 + `_cleared()` 判胜，但 `duel` 有基地与防空，而 `_cleared()`
  在对抗阶段要求"蓝方所有飞机被消灭"。实测蓝方残血恒为 100、清场恒为 0/30，
  说明真正的胜负手是**基地**而不是空战，导致评估基准与"空战谁更强"脱节。
  现在改为**对称空战竞技场**：`range` 布局（无基地/无防空/无地面干扰），
  仅红蓝各 1 架标准机对打，用"击落对手"决定胜负。这才是碾压的判据。

判据：
  胜 = 蓝方被击落且红方存活；负 = 红方被击落；平 = 到时间限制双方都存活（按血量判优）。

用法：
    py AWAI\\bench_duel.py                                       默认模型
    py AWAI\\bench_duel.py standard_v9_final_red                  指定模型
    py AWAI\\bench_duel.py standard_v9_final_red -n 40            指定局数
    py AWAI\\bench_duel.py a.pt b.pt -n 20                        多模型对比
    py AWAI\\bench_duel.py standard_v9_final_red 40               局数也可直接跟在后面

不带模型名时会自动挑一个真实存在的模型（优先 modelset.txt 里存在的条目）。
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

import AWAI as A

N_EP = 30
MAX_STEPS = 1800        # 30 秒；足够分出胜负但不会无限拖时间
RED_SPAWN = (0.30, 0.5)
BLUE_SPAWN = (0.70, 0.5)


def build_parser():
    p = awai_cli.Parser(prog='py AWAI\\bench_duel.py', add_help=False,
                        allow_abbrev=False,
                        description='1v1 对称空战评估：神经网络 vs 硬编码规则 AI。')
    p.add_argument('-h', '--help', action='help', help='显示本帮助并退出')
    p.add_argument('models', nargs='*', metavar='模型名',
                   help='要评估的模型，可以给多个；省略则自动挑一个默认模型')
    p.add_argument('-n', '--episodes', type=int, metavar='N',
                   help=f'每个模型评估多少局，默认 {N_EP}')
    p.add_argument('--seed0', type=int, default=0, metavar='S',
                   help='起始随机种子，默认 0')
    p.add_argument('--list-models', action='store_true', help='列出可用模型后退出')
    return p


def duel_once(agent, seed, max_steps=MAX_STEPS):
    """对称空战一局：红=神经网络，蓝=规则AI，无基地/防空/地面干扰。"""
    random.seed(seed)
    st = A.StageSpec(name='duel_air', desc='1v1 空战竞技场',
                     red_team=[('standard', A.CTRL_NEURAL)],
                     blue_team=[('standard', A.CTRL_RULE)],
                     layout='range', max_steps=max_steps, map_variants=1,
                     red_spawn=RED_SPAWN, blue_spawn=BLUE_SPAWN)
    env = A.CurriculumEnv(st, render=False, is_eval=True)
    obs = env.reset()
    done = False
    dmg = kills = steps = 0
    while not done:
        rows = obs.get('red') or []
        if rows:
            a, _, _ = agent.act(rows[0], deterministic=True)
            acts = {'red': [a], 'blue': []}
        else:
            acts = {'red': [], 'blue': []}
        obs, rew, done, _ = env.step(acts)
        dmg += sum(x for (t, u), x in env.world.damage_credit.items() if t == 'red')
        kills += sum(1 for u in env.world.kills if u.team == 'blue')
        steps += 1
    red = env.world.aircraft_list_of('red')
    blue = env.world.aircraft_list_of('blue')
    red_hp = sum(x.hp for x in red)
    blue_hp = sum(x.hp for x in blue)
    red_max = sum(x.max_hp for x in red) or 100.0
    blue_max = sum(x.max_hp for x in blue) or 100.0
    if not blue and red:
        result = 'win'
    elif not red and blue:
        result = 'loss'
    elif not red and not blue:
        result = 'draw'
    else:
        rp, bp = red_hp / red_max, blue_hp / blue_max
        if abs(rp - bp) < 0.05:
            result = 'draw'
        else:
            result = 'win' if rp > bp else 'loss'
    return {'result': result, 'dmg': dmg, 'kills': kills, 'steps': steps,
            'red_hp': red_hp, 'blue_hp': blue_hp,
            'red_alive': len(red), 'blue_alive': len(blue)}


def evaluate(model, n_ep=N_EP, seed0=0, verbose=True):
    agent, msg = A.try_load_agent(model)
    if agent is None:
        print(f'  {model}: 加载失败 — {msg.splitlines()[0]}')
        return None
    res = [duel_once(agent, seed0 + i) for i in range(n_ep)]
    wins = sum(1 for r in res if r['result'] == 'win')
    losses = sum(1 for r in res if r['result'] == 'loss')
    draws = n_ep - wins - losses
    killed_blue = sum(1 for r in res if r['blue_alive'] == 0)
    out = {
        'model': model, 'n': n_ep, 'wins': wins, 'losses': losses, 'draws': draws,
        'win_rate': wins / n_ep, 'kill_rate': killed_blue / n_ep,
        'avg_dmg': sum(r['dmg'] for r in res) / n_ep,
        'avg_steps': sum(r['steps'] for r in res) / n_ep,
        'survive_rate': sum(1 for r in res if r['red_alive'] > 0) / n_ep,
        'avg_red_hp': sum(r['red_hp'] for r in res) / n_ep,
        'avg_blue_hp': sum(r['blue_hp'] for r in res) / n_ep,
        'res': res,
    }
    if verbose:
        print(f'  {model}')
        print(f'    胜 {wins} 负 {losses} 平 {draws}   '
              f'胜率 {out["win_rate"]:>6.1%}   击落对手率 {out["kill_rate"]:>5.0%}')
        print(f'    均伤害 {out["avg_dmg"]:>7.1f}  均步数 {out["avg_steps"]:>6.0f}   '
              f'红存活率 {out["survive_rate"]:>5.0%}   残血 红 {out["avg_red_hp"]:>5.1f} / 蓝 {out["avg_blue_hp"]:>5.1f}')
    return out


def main(argv):
    args = build_parser().parse_args(argv)

    if args.list_models:
        return A.cli_main(['--list-models'])

    n_ep = args.episodes
    models = []
    for a in args.models:
        # 向后兼容旧写法：直接跟在后面的纯数字当作局数
        if a.isdigit():
            if n_ep is not None:
                awai_cli.fail('参数错误：局数给了两次，请只保留一个。')
            n_ep = int(a)
        else:
            models.append(a)
    if n_ep is None:
        n_ep = N_EP
    if n_ep <= 0:
        awai_cli.fail(f'参数错误：局数必须是正整数，收到 {n_ep}。')

    if not models:
        default = A.default_model_name()
        if default is None:
            awai_cli.fail('models/ 里没有任何模型。请先训练一个：\n'
                          '  py AWAI\\AWAI.py --curriculum standard --episodes 100 '
                          '--norender --prefix standard_v3')
        models = [default]

    print(f'=== 1v1 空战碾压评估：神经网络 vs 硬编码规则AI（对称竞技场，{n_ep} 局）===')
    print('    规则：击落对手=胜；无基地/防空/地面干扰，胜负只由空战决定')

    results = []
    for m in models:
        r = evaluate(m, n_ep, args.seed0)
        if r:
            results.append(r)
        print()

    if not results:
        awai_cli.fail(f'所有模型都加载失败。当前代码要求 obs_dim={A.OBS_DIM}、'
                      f'obs_version={A.OBS_VERSION}。\n可选的模型：\n{A.suggested_models()}')

    if len(results) > 1:
        print('=== 汇总（按胜率排序）===')
        for r in sorted(results, key=lambda x: -x['win_rate']):
            print(f'  {r["model"]:<28} 胜率 {r["win_rate"]:>6.1%}  '
                  f'击落率 {r["kill_rate"]:>5.0%}  均伤害 {r["avg_dmg"]:>7.1f}  '
                  f'存活率 {r["survive_rate"]:>5.0%}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]) or 0)
