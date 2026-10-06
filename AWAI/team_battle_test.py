#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""10v10 团战互殴测试：标准机 × 10 神经网络 AI（红） vs 10 规则 AI（蓝）。
10 种不同出生阵型，每种跑 N 回合，统计胜负/剩余/击杀/伤害。

用法：
    py AWAI\\team_battle_test.py                                 1 回合，默认模型，range 地图
    py AWAI\\team_battle_test.py 2 standard_v9_final_red urban
    py AWAI\\team_battle_test.py -n 2 --model standard_v9_final_red --map urban
    py AWAI\\team_battle_test.py 2 urban standard_v9_final_red    顺序无所谓，自动识别

位置参数可以不按顺序给：纯数字算回合数、已知地图名算地图、其余算模型名。
不带模型名时会自动挑一个真实存在的模型（优先 modelset.txt 里存在的条目）。

结果写入 report/team_battle_10v10.json（可用 --out 改路径）。

注意：模型在 1v1 课程上训练，10v10 属跨规模泛化测试，结果仅作参考。
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

import datetime
import json
import math
import random

import AWAI as A

N = 10                      # 每队飞机数
REPORT_DIR = os.path.join(_HERE, 'report')
FORMATIONS = ['line', 'wedge', 'column', 'arrow', 'double_col',
              'circle', 'grid', 'wing', 'echelon', 'random']


def build_parser():
    p = awai_cli.Parser(prog='py AWAI\\team_battle_test.py', add_help=False,
                        allow_abbrev=False,
                        description='10v10 团战互殴测试：10 种阵型 × N 回合。')
    p.add_argument('-h', '--help', action='help', help='显示本帮助并退出')
    p.add_argument('items', nargs='*', metavar='参数',
                   help='位置参数：[回合数] [模型名] [地图布局]，顺序任意')
    p.add_argument('-n', '--episodes', type=int, metavar='N',
                   help='每种阵型的回合数，默认 1')
    p.add_argument('-m', '--model', metavar='模型名', help='要评估的模型')
    p.add_argument('--map', dest='layout', metavar='布局', help='地图布局，默认 range')
    p.add_argument('--seed', type=int, default=20260814, metavar='S',
                   help='随机种子，默认 20260814')
    p.add_argument('--out', metavar='路径',
                   help='JSON 输出路径，默认 report/team_battle_10v10.json')
    p.add_argument('--list-models', action='store_true', help='列出可用模型后退出')
    p.add_argument('--list-maps', action='store_true', help='列出可用地图后退出')
    return p


def formation_offsets(name, n=N):
    """10 种出生阵型的偏移列表（相对各自阵营中心）。"""
    def f_line():
        return [((i - (n - 1) / 2) * 130, (i - (n - 1) / 2) * 30) for i in range(n)]

    def f_wedge():
        out, idx = [], 0
        for r in range(4):               # 行: 1,2,3,4 架 → 10 架
            for j in range(1 + r):
                dx = (r - 1.5) * 170
                dy = (j - r / 2) * 150
                out.append((dx, dy))
                idx += 1
        return out[:n]

    def f_column():
        return [((i - (n - 1) / 2) * 45, (i - (n - 1) / 2) * 120) for i in range(n)]

    def f_arrow():
        out = []
        for i in range(4):               # 中列 4 架
            out.append((0, (i - 1.5) * 140))
        for k in range(3):               # 两翼各 3 架斜线
            out.append((180 + k * 150, (-60 - k * 130)))
            out.append((180 + k * 150, (60 + k * 130)))
        return out[:n]

    def f_double_col():
        out = []
        for k in range(5):
            out.append((-95, (k - 2) * 120))
        for k in range(5):
            out.append((95, (k - 2) * 120))
        return out[:n]

    def f_circle():
        return [(420 * math.cos(i / n * 2 * math.pi), 420 * math.sin(i / n * 2 * math.pi))
                for i in range(n)]

    def f_grid():
        out = []
        for row in range(4):
            for col in range(3):
                if len(out) >= n:
                    break
                out.append(((col - 1) * 180, (row - 1.5) * 140))
        return out[:n]

    def f_wing():
        out = []
        for i in range(2):               # 中心双机
            out.append(((-60 + i * 120), 0))
        for k in range(4):               # 左右翼外扩
            out.append((-260 - k * 160, (-80 - k * 40)))
            out.append((260 + k * 160, (80 + k * 40)))
        return out[:n]

    def f_echelon():
        return [((i - (n - 1) / 2) * 150, -(i - (n - 1) / 2) * 85) for i in range(n)]

    def f_random():
        out, tries = [], 0
        while len(out) < n and tries < 2000:
            tries += 1
            dx = random.uniform(-420, 420)
            dy = random.uniform(-420, 420)
            if all(math.hypot(dx - ox, dy - oy) > 90 for ox, oy in out):
                out.append((dx, dy))
        return out[:n]

    table = {'line': f_line, 'wedge': f_wedge, 'column': f_column, 'arrow': f_arrow,
             'double_col': f_double_col, 'circle': f_circle, 'grid': f_grid,
             'wing': f_wing, 'echelon': f_echelon, 'random': f_random}
    return table[name]()


def run_battle(agent, offsets, episode, layout):
    """跑一回合 10v10，返回统计。"""
    stage = A.StageSpec(name='10v10', desc='团战互殴',
                        red_team=[('standard', A.CTRL_NEURAL)] * N,
                        blue_team=[('standard', A.CTRL_RULE)] * N,
                        layout=layout, max_steps=3000, formation=offsets,
                        red_spawn=(0.35, 0.5), blue_spawn=(0.65, 0.5))
    env = A.CurriculumEnv(stage, render=False)
    obs = env.reset()
    done = False
    steps = 0
    red_dmg = blue_dmg = 0.0
    red_kills = blue_kills = 0
    while not done and steps < 3000:
        reds = obs.get('red') or []
        if reds:
            # 合批推理：10 架飞机一次前向，避免每步 10 次单条推理（实测慢约 10 倍）
            acts, _, _ = agent.act_batch(reds, deterministic=True)
            actions = {'red': list(acts), 'blue': []}
        else:
            actions = {'red': [], 'blue': []}
        obs, rew, done, _ = env.step(actions)
        steps += 1
        for (team, uid), dmg in env.world.damage_credit.items():
            if team == 'red':
                red_dmg += dmg
            else:
                blue_dmg += dmg
        for u in env.world.kills:
            if u.team == 'blue':
                red_kills += 1
            else:
                blue_kills += 1
        if steps % 500 == 0:
            print(f'    [ep{episode}] 步数 {steps}  红剩 {len(env.world.aircraft_list_of("red"))}'
                  f' 蓝剩 {len(env.world.aircraft_list_of("blue"))}', flush=True)
    red_alive = len(env.world.aircraft_list_of('red'))
    blue_alive = len(env.world.aircraft_list_of('blue'))
    if red_alive == 0 and blue_alive > 0:
        winner = 'blue'
    elif blue_alive == 0 and red_alive > 0:
        winner = 'red'
    elif red_alive == blue_alive:
        winner = 'draw'
    else:
        winner = 'red' if red_alive > blue_alive else 'blue'
    return {'steps': steps, 'winner': winner,
            'red_alive': red_alive, 'blue_alive': blue_alive,
            'red_kills': red_kills, 'blue_kills': blue_kills,
            'red_dmg': round(red_dmg), 'blue_dmg': round(blue_dmg)}


def main(argv):
    args = build_parser().parse_args(argv)

    if args.list_models:
        return A.cli_main(['--list-models'])
    if args.list_maps:
        return A.cli_main(['--list-maps'])

    # 位置参数顺序任意：纯数字=回合数，已知地图名=地图，其余=模型名
    episodes, model, layout = args.episodes, args.model, args.layout
    for item in args.items:
        if item.isdigit() and episodes is None:
            episodes = int(item)
        elif item in A.REALISTIC_MAPS:
            layout = layout or item
        elif model is None:
            model = item
        else:
            awai_cli.fail(f'参数错误：无法识别的位置参数 {item!r}（多半是地图名写错了）。\n'
                          f'  可用地图：{"  ".join(A.REALISTIC_MAPS)}\n'
                          f'  用法：py AWAI\\team_battle_test.py [回合数] [模型名] [地图布局]')

    episodes = 1 if episodes is None else episodes
    layout = layout or 'range'
    if episodes <= 0:
        awai_cli.fail(f'参数错误：回合数必须是正整数，收到 {episodes}。')
    if layout not in A.REALISTIC_MAPS:
        awai_cli.fail(f'参数错误：未知地图 {layout!r}。\n'
                      f'  可用地图：{"  ".join(A.REALISTIC_MAPS)}')

    if not model:
        model = A.default_model_name()
        if model is None:
            awai_cli.fail('models/ 里没有任何模型。请先训练一个：\n'
                          '  py AWAI\\AWAI.py --curriculum standard --episodes 100 '
                          '--norender --prefix standard_v3')

    agent, msg = A.try_load_agent(model)
    if agent is None:
        awai_cli.fail(f'模型无法加载：{model}\n  {msg}\n\n'
                      f'  当前代码要求 obs_dim={A.OBS_DIM}、obs_version={A.OBS_VERSION}。\n'
                      f'  可选的模型：\n{A.suggested_models()}')

    random.seed(args.seed)
    print(f'=== 10v10 团战互殴测试（标准机，地图 {layout}）===', flush=True)
    print(f'红方: {N}×神经网络AI（{model}，{msg}）  蓝方: {N}×规则AI  '
          f'每阵型 {episodes} 回合', flush=True)

    rows = []
    for fname in FORMATIONS:
        offs = formation_offsets(fname)
        reds_w = blues_w = draws = 0
        r_alive = b_alive = r_kill = b_kill = r_dmg = b_dmg = 0
        steps_sum = 0
        print(f'\n[阵型] {fname}  {offs[:3]}...', flush=True)
        for ep in range(episodes):
            s = run_battle(agent, offs, ep, layout)
            if s['winner'] == 'red':
                reds_w += 1
            elif s['winner'] == 'blue':
                blues_w += 1
            else:
                draws += 1
            r_alive += s['red_alive']; b_alive += s['blue_alive']
            r_kill += s['red_kills']; b_kill += s['blue_kills']
            r_dmg += s['red_dmg']; b_dmg += s['blue_dmg']
            steps_sum += s['steps']
            print(f'    ep{ep}: 胜方={s["winner"]} 红剩{s["red_alive"]}/蓝剩{s["blue_alive"]} '
                  f'击杀 {s["red_kills"]}:{s["blue_kills"]} 伤害 {s["red_dmg"]}:{s["blue_dmg"]} '
                  f'步数 {s["steps"]}', flush=True)
        rows.append({
            'formation': fname, 'episodes': episodes,
            'red_wins': reds_w, 'blue_wins': blues_w, 'draws': draws,
            'avg_red_alive': round(r_alive / episodes, 1),
            'avg_blue_alive': round(b_alive / episodes, 1),
            'red_kills': r_kill, 'blue_kills': b_kill,
            'red_dmg': r_dmg, 'blue_dmg': b_dmg,
            'avg_steps': round(steps_sum / episodes),
        })
        print(f'    → 红胜 {reds_w} 蓝胜 {blues_w} 平 {draws}  '
              f'平均存活 {rows[-1]["avg_red_alive"]}:{rows[-1]["avg_blue_alive"]} '
              f'总击杀 {r_kill}:{b_kill} 伤害 {r_dmg}:{b_dmg}', flush=True)

    print('\n========== 汇总表 ==========', flush=True)
    print(f'{"阵型":<11}{"红胜":>4}{"蓝胜":>4}{"平":>4}{"平均存活(红:蓝)":>16}{"击杀(红:蓝)":>14}{"伤害(红:蓝)":>14}{"均步数":>8}', flush=True)
    for r in rows:
        print(f'{r["formation"]:<11}{r["red_wins"]:>4}{r["blue_wins"]:>4}{r["draws"]:>4}'
              f'{str(r["avg_red_alive"])+":"+str(r["avg_blue_alive"]):>16}'
              f'{str(r["red_kills"])+":"+str(r["blue_kills"]):>14}'
              f'{str(r["red_dmg"])+":"+str(r["blue_dmg"]):>14}'
              f'{r["avg_steps"]:>8}', flush=True)
    total_red = sum(r['red_wins'] for r in rows)
    total_blue = sum(r['blue_wins'] for r in rows)
    total_draw = sum(r['draws'] for r in rows)
    total_rk = sum(r['red_kills'] for r in rows)
    total_bk = sum(r['blue_kills'] for r in rows)
    total_rd = sum(r['red_dmg'] for r in rows)
    total_bd = sum(r['blue_dmg'] for r in rows)
    print(f'{"合计":<11}{total_red:>4}{total_blue:>4}{total_draw:>4}'
          f'{"":>16}{str(total_rk)+":"+str(total_bk):>14}{str(total_rd)+":"+str(total_bd):>14}', flush=True)

    out = args.out or os.path.join(REPORT_DIR, 'team_battle_10v10.json')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                   'model': model, 'n': N, 'layout': layout,
                   'episodes_per_formation': episodes,
                   'rows': rows}, f, ensure_ascii=False, indent=1)
    print(f'结果已保存: {out}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]) or 0)
