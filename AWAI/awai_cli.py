#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AWAI 命令行公共支持 —— 只依赖标准库。

本模块必须在导入 pygame / numpy / torch **之前**使用，因此可以保证：

* `--help` 在依赖没装全的机器上也能正常打印，而不是先崩一段 traceback；
* 缺少依赖时给出中文安装指引（含本机 `py -3` 写法），而不是 ImportError 堆栈；
* Windows 下把「重定向输出」的编码改成 UTF-8，避免把中文日志写进文件时报编码错。

AWAI.py 与 AWAI/ 下的各辅助脚本共用本模块。本模块不导入任何第三方库。
"""

import argparse
import os
import re
import sys

AWAI_VERSION = '3.0'

# 实测可用的最低 Python 版本（见 README「环境要求」）。
MIN_PYTHON = (3, 8)

# (import 名, pip 名, 版本建议)
REQUIRED_PACKAGES = (
    ('pygame', 'pygame', '>=2.1'),
    ('numpy', 'numpy', '>=1.20'),
    ('torch', 'torch', '>=1.10'),
)

IS_WINDOWS = os.name == 'nt'
# 本机的 `python` 可能是 Windows 商店占位符，所以统一推荐 py 启动器。
PY = 'py -3' if IS_WINDOWS else 'python3'
SCRIPT_DIR = 'AWAI\\' if IS_WINDOWS else 'AWAI/'

# 提示信息里的「空行」。不用空串是因为 PowerShell 会把 stderr 的空行包装成
# RemoteException 噪声（详见 check_requirements 的说明）。
_BLANK = ' '


def setup_console():
    """Windows：输出被重定向到管道/文件时改用 UTF-8，避免中文日志触发 UnicodeEncodeError。

    直接连控制台时不改动 —— PEP 528 已让控制台正确渲染 Unicode。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            enc = (getattr(stream, 'encoding', '') or '').lower().replace('-', '')
            if not stream.isatty() and enc != 'utf8':
                stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass


def eprint(*args):
    """打印到 stderr（错误信息与正常输出分离，便于被其它脚本调用）。"""
    print(*args, file=sys.stderr)


def handle_early_exit(usage, argv=None, version=AWAI_VERSION):
    """在导入第三方库之前处理 `-h` / `--help` / `--version`：打印后以 0 退出。

    未命中时静默返回，调用方继续正常启动。usage 通常直接传模块的 `__doc__`。

    注意：只有在「本文件是被直接运行的主脚本」时才应调用它，
    否则被 `import` 时会把主脚本的 --help 抢走。"""
    setup_console()
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(a in ('-h', '--help') for a in argv):
        print((usage or '').strip())
        sys.exit(0)
    if any(a in ('-V', '--version') for a in argv):
        print(f'AWAI v{version}')
        sys.exit(0)


def check_requirements(packages=REQUIRED_PACKAGES, python_min=MIN_PYTHON):
    """自检 Python 版本与第三方依赖；不满足时打印中文指引并以 1 退出。

    必须放在 `import pygame` 等语句之前调用。

    注意：整个提示用**一次** eprint 输出，且分隔行用纯空格而不是空串 ——
    PowerShell 会把 native 命令 stderr 的**空行**包装成一条
    `System.Management.Automation.RemoteException`，在控制台里哗啦一片噪声；
    纯空格行既看着是空行，又不会被包装。"""
    setup_console()
    if sys.version_info < python_min:
        eprint('\n'.join([
            f'错误：AWAI 需要 Python {python_min[0]}.{python_min[1]} 或更高版本，'
            f'当前是 {sys.version.split()[0]}。',
            f'      当前解释器：{sys.executable}',
            '      请从 https://www.python.org/downloads/ 安装较新版本后重试。',
            f'      本机推荐的启动方式：{PY}',
        ]))
        sys.exit(1)

    missing = []
    for mod, pip_name, spec in packages:
        try:
            __import__(mod)
        except ImportError:
            missing.append(f'{pip_name}{spec}')

    if missing:
        lines = ['错误：缺少运行 AWAI 所需的依赖：']
        lines += [f'        - {name}' for name in missing]
        lines += [
            _BLANK,
            '      一条命令装齐（建议先建虚拟环境）：',
            f'        {PY} -m pip install -r requirements.txt',
            _BLANK,
            '      或者只装缺的那几个：',
            f'        {PY} -m pip install {" ".join(missing)}',
            _BLANK,
            '      装完用这条命令确认：',
            f'        {PY} -c "import pygame, numpy, torch; print(\'ok\')"',
        ]
        eprint('\n'.join(lines))
        sys.exit(1)


class Parser(argparse.ArgumentParser):
    """argparse 的中文化包装：参数出错时用中文说明，并提示去哪儿看帮助。"""

    def error(self, message):
        setup_console()
        eprint(f'参数错误：{translate_arg_error(message)}')
        eprint(f'用 "{self.prog} --help" 查看全部可用参数。')
        sys.exit(2)


# argparse 的报错是英文的，这里按最常见的几种句式翻译成中文。
# 每条是 (正则, 替换模板)，按顺序匹配，命中即返回。
_ERROR_PATTERNS = (
    (r'^unrecognized arguments: (.+)$', r'无法识别的参数：\1'),
    (r'^argument (\S+): expected one argument$', r'参数 \1 缺少取值'),
    (r'^argument (\S+): expected at least one argument$', r'参数 \1 至少需要一个取值'),
    (r'^argument (\S+): expected (\d+) arguments?$', r'参数 \1 需要 \2 个取值'),
    (r"^argument (\S+): invalid choice: '(.*)' \(choose from (.*)\)$",
     r'参数 \1 的取值 "\2" 无效，可选：\3'),
    (r"^argument (\S+): invalid int value: '(.*)'$", r'参数 \1 需要整数，但收到 "\2"'),
    (r"^argument (\S+): invalid float value: '(.*)'$", r'参数 \1 需要数字，但收到 "\2"'),
    (r'^argument (\S+): not allowed with argument (\S+)$', r'参数 \1 不能与 \2 同时使用'),
    (r'^argument (\S+): ignored explicit argument (.*)$', r'参数 \1 不接受取值'),
    (r'^the following arguments are required: (.+)$', r'缺少必需参数：\1'),
    (r'^ambiguous option: (\S+) could match (.+)$', r'参数缩写 \1 有歧义，可能是：\2'),
)


def translate_arg_error(message):
    """把 argparse 的英文报错翻成中文；没命中的原样返回。"""
    for pattern, repl in _ERROR_PATTERNS:
        if re.match(pattern, message):
            return re.sub(pattern, repl, message)
    return message


def fail(message, code=2):
    """打印一条中文错误信息并以指定退出码结束。"""
    setup_console()
    eprint(message)
    sys.exit(code)
