#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 AWAI 课程训练检查页（report/AWAI_training_report.html）。

用最终模型确定性评估各阶段环境，输出分数表 + 逐回合奖励图 + 模型清单 + 地图清单。

用法：
    py AWAI\\report_gen.py                      默认每阶段评估 4 回合
    py AWAI\\report_gen.py --eval 8             每阶段评估 8 回合（更准，更慢）
    py AWAI\\report_gen.py --out D:\\out.html    指定输出路径

会自动发现可评估的模型（models/ 下的 *_final_red）。
"""
import os
import sys

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import awai_cli

if __name__ == '__main__':
    awai_cli.handle_early_exit(__doc__)
awai_cli.check_requirements()

import datetime
import html

import AWAI as A

N_EVAL = 4          # 每阶段评估回合数（可用 --eval 覆盖）
REPORT_DIR = os.path.join(_HERE, 'report')


def discover_models():
    """自动发现可评估的模型前缀：优先 modelset.txt，其次扫描 models/ 里的 *_final_red。

    原实现硬编码 ('standard', 'standard_cur') 与 ('missile_plane', 'missile_plane_cur')，
    这两个前缀的模型文件都不存在，导致报告页只会显示报错。"""
    prefixes = []
    for name in A._read_modelset():
        base = name[:-4] if name.endswith('_red') else name
        if base.endswith('_final'):
            base = base[:-6]
        if base and base not in prefixes and os.path.exists(A.model_path(f'{base}_final_red')):
            prefixes.append(base)
    if os.path.isdir(A.MODEL_DIR):
        for fn in sorted(os.listdir(A.MODEL_DIR)):
            if fn.endswith('_final_red.pt'):
                base = fn[:-len('_final_red.pt')]
                if base not in prefixes:
                    prefixes.append(base)
    out = []
    for base in prefixes:
        pt = 'missile_plane' if base.startswith('missile_plane') else 'standard'
        out.append((pt, base))
    return out


def evaluate_type(plane_type, prefix):
    """对最终模型逐阶段评估，返回结构化数据。"""
    agent, msg = A.try_load_agent(f"{prefix}_final_red")
    if agent is None:
        return {'plane_type': plane_type, 'prefix': prefix, 'error': msg}
    stages = A.curriculum_stages(plane_type)
    rows = []
    for si, stage in enumerate(stages):
        env = A.CurriculumEnv(stage, render=False, is_eval=True)
        eps = []
        for _ in range(N_EVAL):
            obs = env.reset()
            done = False
            ep_r = 0.0
            while not done:
                rows_obs = obs.get('red') or []
                if rows_obs:
                    a, _, _ = agent.act(rows_obs[0], deterministic=True)
                    acts = {'red': [a], 'blue': []}
                else:
                    acts = {'red': [], 'blue': []}
                obs, rew, done, _ = env.step(acts)
                ep_r += rew['red']
            eps.append({'reward': round(ep_r, 1), 'steps': env.steps,
                        'cleared': env._cleared()})
        rows.append({
            'stage': stage.name,
            'desc': stage.desc,
            'wins': sum(1 for e in eps if e['cleared']),
            'avg_reward': round(sum(e['reward'] for e in eps) / len(eps), 1),
            'avg_steps': round(sum(e['steps'] for e in eps) / len(eps)),
            'cleared_rate': sum(1 for e in eps if e['cleared']) / len(eps),
            'episodes': eps,
        })
    return {'plane_type': plane_type, 'prefix': prefix, 'stages': rows}


def models_inventory():
    out = []
    d = A.MODEL_DIR
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if fn.endswith('.pt'):
                p = os.path.join(d, fn)
                out.append({'name': fn, 'kb': round(os.path.getsize(p) / 1024),
                            'time': datetime.datetime.fromtimestamp(
                                os.path.getmtime(p)).strftime('%m-%d %H:%M')})
    return out


# ---------- 图表（纯 SVG，无需 JS） ----------
def bar_chart(eps, width=620, height=150):
    """逐回合奖励柱状图。"""
    if not eps:
        return '<p style="color:#999">无数据</p>'
    vals = [e['reward'] for e in eps]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    pad_l, pad_b = 40, 22
    cw = (width - pad_l - 10) / len(vals)
    parts = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    parts.append(f'<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-5}" y2="{height-pad_b}" stroke="#999"/>')
    for i, v in enumerate(vals):
        x = pad_l + i * cw
        h = max(3, (v - lo) / span * (height - pad_b - 20))
        y = height - pad_b - h
        color = '#2a7' if eps[i]['cleared'] else ('#4af' if v >= 0 else '#f55')
        parts.append(f'<rect x="{x+2:.1f}" y="{y:.1f}" width="{max(2,cw-4):.1f}" height="{h:.1f}" fill="{color}">'
                     f'<title>回合{i+1}: {v:.1f} 步数{eps[i]["steps"]} {"清场" if eps[i]["cleared"] else ""}</title></rect>')
    # 零线
    if lo < 0 < hi:
        y0 = height - pad_b - (0 - lo) / span * (height - pad_b - 20)
        parts.append(f'<line x1="{pad_l}" y1="{y0:.1f}" x2="{width-5}" y2="{y0:.1f}" stroke="#bbb" stroke-dasharray="4,3"/>')
    parts.append(f'<text x="4" y="{height-pad_b-6}" font-size="11" fill="#666">最低{lo:.0f}</text>')
    parts.append(f'<text x="{width-90}" y="{height-pad_b-6}" font-size="11" fill="#666">最高{hi:.0f}</text>')
    parts.append(f'<text x="{width-90}" y="{height-4}" font-size="10" fill="#999">共{len(vals)}回合</text>')
    parts.append('</svg>')
    return ''.join(parts)


def build_html(data, inventory):
    es = []
    es.append('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">'
              '<title>AWAI 课程训练检查页</title>'
              '<style>body{font-family:"Microsoft YaHei",sans-serif;margin:24px;background:#f7f8fa;color:#222}'
              'h1{font-size:26px}h2{font-size:20px;border-left:4px solid #2a7;padding-left:10px;margin-top:34px}'
              'table{border-collapse:collapse;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.08)}'
              'th,td{border:1px solid #e2e4e8;padding:7px 14px;text-align:center;font-size:14px}'
              'th{background:#eef2f7}.stage-block{margin:16px 0}.warn{background:#fff6e6;border:1px solid #f0c36d;'
              'padding:10px 14px;border-radius:6px}.green{color:#2a7;font-weight:bold}.red{color:#d33}'
              'code{background:#eef;padding:2px 6px;border-radius:4px}</style></head><body>')
    es.append('<h1>✈ AWAI 课程训练检查页</h1>')
    es.append(f'<p>生成时间：{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}　'
              f'评估方式：最终模型确定性推理，每阶段 {N_EVAL} 回合　'
              f'观测布局版本 v{A.OBS_VERSION}（{A.OBS_DIM} 维）</p>')
    # 结论速览：按实际评估数据自动生成，不再写死文案
    tot_stages = sum(len(m.get('stages', [])) for m in data)
    cleared = sum(1 for m in data for r in m.get('stages', []) if r['wins'] > 0)
    avg_rew = [r['avg_reward'] for m in data for r in m.get('stages', [])]
    if tot_stages:
        es.append(f'<div class="warn"><b>结论速览：</b>共评估 {len(data)} 个模型 / {tot_stages} 个阶段，'
                  f'其中 <b>{cleared}</b> 个阶段至少清场一次，'
                  f'平均奖励区间 {min(avg_rew):.1f} ~ {max(avg_rew):.1f}。'
                  f'奖励已改为“战斗输出主导”（默认对每点伤害给分、索敌仅首次发现奖励一次、'
                  f'每目标首次锁定奖励一次），若清场率仍偏低，应优先加长训练回合数。'
                  f'具体见下方逐阶段数据。</div>')
    else:
        errs = [m.get('error', '') for m in data]
        if any('不兼容' in e for e in errs):
            es.append('<div class="warn"><b>现有模型与当前观测布局不兼容。</b>'
                      f'当前为 v{A.OBS_VERSION}（{A.OBS_DIM} 维），'
                      '观测维度变更后旧检查点无法直接加载，请重新训练生成兼容模型：'
                      '<code>py AWAI\\AWAI.py --curriculum standard --episodes 80 --norender --prefix standard_v3</code>'
                      '<br>（历史检查点仍保留在 <code>models/</code>，仅作记录。）</div>')
        else:
            es.append('<div class="warn"><b>暂无可用模型。</b>先运行课程训练生成检查点：'
                      '<code>py AWAI\\AWAI.py --curriculum standard --episodes 80 --norender --prefix standard_v3</code>'
                      '</div>')

    for m in data:
        es.append(f'<h2>{m["plane_type"]}（{m.get("prefix", "")}）</h2>')
        if 'error' in m:
            es.append(f'<p class="red">{html.escape(m["error"])}</p>')
            continue
        es.append('<table><tr><th>阶段</th><th>胜率</th><th>平均奖励</th><th>平均步数</th><th>清场率</th></tr>')
        for r in m['stages']:
            wr = f'{r["wins"]}/{N_EVAL}'
            cls = 'green' if r['wins'] > 0 else ''
            es.append(f'<tr><td style="text-align:left">{html.escape(r["stage"])}<br>'
                      f'<span style="color:#999;font-size:12px">{html.escape(r["desc"])}</span></td>'
                      f'<td class="{cls}">{wr}</td><td>{r["avg_reward"]}</td>'
                      f'<td>{r["avg_steps"]}</td><td>{r["cleared_rate"]*100:.0f}%</td></tr>')
        es.append('</table>')
        for r in m['stages']:
            es.append(f'<div class="stage-block"><b>{html.escape(r["stage"])}</b>'
                      f'（均奖励 {r["avg_reward"]}，清场 {r["cleared_rate"]*100:.0f}%）')
            es.append(bar_chart(r['episodes']))
            es.append('</div>')

    es.append('<h2>模型清单（models/）</h2>')
    es.append(f'<table><tr><th>模型文件</th><th>大小</th><th>更新时间</th></tr>')
    for it in inventory:
        es.append(f'<tr><td style="text-align:left"><code>{html.escape(it["name"])}</code></td>'
                  f'<td>{it["kb"]} KB</td><td>{it["time"]}</td></tr>')
    es.append(f'</table><p>共 {len(inventory)} 个模型文件。</p>')

    es.append('<h2>地图清单（现实向地形）</h2>')
    es.append('<table><tr><th>布局名</th><th>名称</th><th>地形特征</th><th>变体数</th><th>地形块数</th></tr>')
    for layout, (cn, desc, nv) in A.REALISTIC_MAPS.items():
        tm = A.build_terrain_map(layout, 0)
        kinds = {}
        for f in tm.features:
            k = A.TERRAIN_NAMES.get(f.kind, '?')
            kinds[k] = kinds.get(k, 0) + 1
        kstr = ' '.join(f'{k}×{v}' for k, v in kinds.items()) or '无'
        es.append(f'<tr><td style="text-align:left"><code>{layout}</code></td><td>{cn}</td>'
                  f'<td style="text-align:left">{html.escape(desc)}</td><td>{nv}</td><td>{kstr}</td></tr>')
    es.append('</table><p>山体/丘陵/城区会<b>阻断视线与雷达</b>（低空可借地形遮蔽突防），'
              '地面单位藏身地形内还会降低被发现概率。</p>')

    es.append('<h2>下一步操作指引</h2>'
              '<ul>'
              '<li><b>继续加训</b>：<code>py .\\AWAI\\AWAI.py --curriculum standard --episodes 200 --norender --prefix standard_v3b</code></li>'
              '<li><b>课程训练（带界面）</b>：主菜单 → 课程训练 → 输入模型前缀</li>'
              '<li><b>观战</b>：<code>py .\\AWAI\\watch_show.py &lt;模型名&gt; &lt;地图布局&gt;</code>，'
              '或在主菜单先“选择地图”再进观战模式</li>'
              '<li><b>团战测试</b>：<code>py .\\AWAI\\team_battle_test.py 1 &lt;模型名&gt; mountain_corridor</code></li>'
              '<li><b>单独评估</b>：<code>py .\\AWAI\\AWAI.py --evaluate standard &lt;模型前缀&gt;</code></li>'
              '<li><b>观察训练</b>：<code>py .\\AWAI\\train_probe.py &lt;阶段名&gt; &lt;回合数&gt;</code></li>'
              '</ul>')
    es.append('</body></html>')
    return '\n'.join(es)


def build_parser():
    p = awai_cli.Parser(prog='py AWAI\\report_gen.py', add_help=False,
                        allow_abbrev=False,
                        description='生成 HTML 训练检查页（评估 models/ 下的最终模型）。')
    p.add_argument('-h', '--help', action='help', help='显示本帮助并退出')
    p.add_argument('--eval', type=int, default=N_EVAL, metavar='N',
                   help=f'每阶段评估回合数，默认 {N_EVAL}')
    p.add_argument('--out', metavar='路径',
                   help='HTML 输出路径，默认 report/AWAI_training_report.html')
    p.add_argument('--list-models', action='store_true', help='列出可用模型后退出')
    return p


def main(argv):
    global N_EVAL
    args = build_parser().parse_args(argv)

    if args.list_models:
        return A.cli_main(['--list-models'])
    if args.eval <= 0:
        awai_cli.fail(f'参数错误：--eval 必须是正整数，收到 {args.eval}。')
    N_EVAL = args.eval

    data = []
    models = discover_models()
    if not models:
        print('[report] 未发现可评估的模型（models/ 下没有 *_final_red.pt）。'
              '请先训练：py AWAI\\AWAI.py --curriculum standard --episodes 100 '
              '--norender --prefix standard_v3')
    for pt, prefix in models:
        print(f'[report] 评估 {pt} / {prefix} （每阶段 {N_EVAL} 回合）...', flush=True)
        data.append(evaluate_type(pt, prefix))
    inv = models_inventory()
    page = build_html(data, inv)
    out = args.out or os.path.join(REPORT_DIR, 'AWAI_training_report.html')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(page)
    print(f'[report] 已生成 {out} （{len(inv)} 个模型，'
          f'{sum(len(m.get("stages", [])) for m in data)} 个阶段）', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]) or 0)
