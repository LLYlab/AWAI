#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实时观战：神经网络 AI（红）vs 规则 AI（蓝），1v1 战场。

用法：
    py AWAI\\watch_show.py                              默认模型 + 经典战场
    py AWAI\\watch_show.py standard_v9_final_red
    py AWAI\\watch_show.py standard_v9_final_red urban
    py AWAI\\watch_show.py --list-models                列出可用模型（含兼容性）
    py AWAI\\watch_show.py --list-maps                  列出可用地图

不带模型名时会自动挑一个**真实存在**的模型：优先 modelset.txt 里存在的条目，
否则取版本号最大的 `*_final_red`。

操作：ESC 退出；F 切换跟随/自由相机；R 重开一局；滚轮缩放；中键拖动。
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

import pygame
from pygame.locals import *

import AWAI as A

FPS = 60


def build_parser():
    p = awai_cli.Parser(prog='py AWAI\\watch_show.py', add_help=False,
                        allow_abbrev=False,
                        description='实时观战：神经网络 AI（红）vs 规则 AI（蓝）。')
    p.add_argument('-h', '--help', action='help', help='显示本帮助并退出')
    p.add_argument('model', nargs='?', metavar='模型名',
                   help='模型名（可省略，示例：standard_v9_final_red）')
    p.add_argument('layout', nargs='?', default='duel', metavar='地图布局',
                   help='地图布局，默认 duel')
    p.add_argument('--list-models', action='store_true', help='列出可用模型后退出')
    p.add_argument('--list-maps', action='store_true', help='列出可用地图后退出')
    return p


def main(argv):
    args = build_parser().parse_args(argv)

    # 这两项直接复用主程序的打印逻辑，避免两处列表各自过期
    if args.list_models:
        return A.cli_main(['--list-models'])
    if args.list_maps:
        return A.cli_main(['--list-maps'])

    if args.layout not in A.REALISTIC_MAPS:
        awai_cli.fail(f'参数错误：未知地图 {args.layout!r}。\n'
                      f'  可用地图：{"  ".join(A.REALISTIC_MAPS)}\n'
                      f'  用 --list-maps 可以看到每个地图的中文名与地形特征。')

    model = args.model or A.default_model_name()
    if model is None:
        awai_cli.fail('models/ 里没有任何模型。请先训练一个：\n'
                      '  py AWAI\\AWAI.py --curriculum standard --episodes 100 '
                      '--norender --prefix standard_v3')
    agent, msg = A.try_load_agent(model)
    if agent is None:
        awai_cli.fail(f'模型无法加载：{model}\n  {msg}\n\n'
                      f'  当前代码要求 obs_dim={A.OBS_DIM}、obs_version={A.OBS_VERSION}。\n'
                      f'  可选的模型：\n{A.suggested_models()}')
    print(f'已加载模型 {model} — {msg}', flush=True)

    # S5 是 1v1 规则 AI 对抗阶段，最适合观战；用名字查找而不是写死下标
    stages = A.curriculum_stages('standard')
    stage = next((s for s in stages if '对抗' in s.name), stages[-1])
    env = A.CurriculumEnv(stage, render=False)
    obs = env.reset()

    screen = pygame.display.set_mode((A.SCREEN_W, A.SCREEN_H))
    pygame.display.set_caption(
        f'神经网络 AI ({model}) vs 规则 AI — {A.REALISTIC_MAPS[args.layout][0]}')
    camera = A.Camera(A.SCREEN_W, A.SCREEN_H)
    camera.follow = 'red'
    hud = A.HUD()
    minimap = A.Minimap()
    clock = pygame.time.Clock()
    ep = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                if event.key == K_f:
                    camera.follow = None if camera.follow else 'red'
                if event.key == K_r:
                    obs = env.reset()
            camera.handle(event)
        # 决策（确定性策略，稳定展示）
        rows = obs.get('red') or []
        if rows:
            a, _, _ = agent.act(rows[0], deterministic=True)
            acts = {'red': [a], 'blue': []}
        else:
            acts = {'red': [], 'blue': []}
        obs, rew, done, _ = env.step(acts)
        # 相机跟随红机
        if camera.follow:
            r = env.world.aircraft_of('red')
            if r:
                camera.x, camera.y = r.x, r.y
        # 渲染
        screen.fill(A.WHITE)
        camera.draw_grid(screen)
        camera.draw_terrain(screen, env.world)
        for pool in (env.world.buildings, env.world.ground, env.world.aircraft,
                     env.world.bullets, env.world.bombs, env.world.missiles):
            for e in pool.alive_list():
                e.draw(screen, camera)
        hud.draw(screen, env.world)
        minimap.draw(screen, env.world)
        red = env.world.aircraft_of('red')
        blue = env.world.aircraft_of('blue')
        txt = A.FONT.render(
            f'{model}  第{ep + 1}场  步数{env.steps}  '
            f'红血{int(red.hp) if red else 0}  蓝血{int(blue.hp) if blue else 0}', True, A.BLACK)
        screen.blit(txt, (10, 70))
        hint = A.SMALL_FONT.render('ESC 退出   F 跟随   R 重开   滚轮缩放   中键拖动', True, A.GRAY)
        screen.blit(hint, (10, 92))
        pygame.display.flip()
        clock.tick(FPS)
        if done:
            ep += 1
            if env._cleared():
                print(f'第{ep}场：红方（神经网络）胜', flush=True)
            elif not env.world.aircraft_list_of('red'):
                print(f'第{ep}场：蓝方（规则AI）胜', flush=True)
            else:
                print(f'第{ep}场：超时（红{len(env.world.aircraft_list_of("red"))} '
                      f'vs 蓝{len(env.world.aircraft_list_of("blue"))}）', flush=True)
            obs = env.reset()
    pygame.quit()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]) or 0)
