#!/usr/bin/env python3
"""采样包含 metainfo/data_list 的 nuScenes 标注；只读取可信的 pickle 文件。"""

import argparse
import math
import pickle
import random
from pathlib import Path


def select_indices(total, count, method, seed):
    """无放回采样，返回按原始顺序排列的下标。"""
    if not 1 <= count <= total:
        raise ValueError(f'采样数量必须在 1 到 {total} 之间，收到 {count}')
    if method == 'first':
        return list(range(count))
    if method == 'uniform':
        if count == 1:
            return [0]
        return [i * (total - 1) // (count - 1) for i in range(count)]
    if method == 'random':
        return sorted(random.Random(seed).sample(range(total), count))
    raise ValueError(f'未知采样方式: {method}')


def describe(value):
    if isinstance(value, dict):
        return f'dict，字段: {", ".join(map(str, value))}'
    if isinstance(value, (list, tuple)):
        return f'{type(value).__name__}，长度: {len(value)}'
    if hasattr(value, 'shape') and hasattr(value, 'dtype'):
        return f'{type(value).__name__}，shape={value.shape}，dtype={value.dtype}'
    return f'{type(value).__name__}，示例: {str(value)[:100]}'


def print_structure(data):
    print('顶层结构:')
    for key, value in data.items():
        print(f'  {key}: {describe(value)}')
    print(f'元信息: {data.get("metainfo", {})}')
    records = data['data_list']
    if records:
        print('首帧结构:')
        for key, value in records[0].items():
            print(f'  {key}: {describe(value)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='?', type=Path,
                        default=Path('nuscenes_infos_val.pkl'), help='原始 pkl 路径')
    parser.add_argument('-o', '--output', type=Path, help='输出路径，禁止覆盖已有文件')
    size = parser.add_mutually_exclusive_group()
    size.add_argument('-n', '--count', type=int, help='采样帧数，未指定数量或比例时默认 100')
    size.add_argument('--ratio', type=float, help='采样比例 (0, 1]，帧数向上取整')
    parser.add_argument('--method', choices=['random', 'uniform', 'first'],
                        default='random', help='随机 / 均匀 / 取前 N 帧，默认 random')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，默认 42')
    parser.add_argument('--inspect', action='store_true', help='只查看结构，不写文件')
    args = parser.parse_args()

    if not args.inspect:
        if args.output is None:
            parser.error('采样时必须用 -o 指定输出路径')
        if args.input.resolve() == args.output.resolve():
            parser.error('输出路径不能与输入路径相同')
        if args.output.exists():
            parser.error(f'输出文件已存在，拒绝覆盖: {args.output}')
    if args.ratio is not None and not (0 < args.ratio <= 1):
        parser.error('--ratio 必须在 (0, 1] 范围内且为有限数值')

    try:
        print(f'正在读取: {args.input} ({args.input.stat().st_size / 1024**2:.2f} MiB)',
              flush=True)
        with args.input.open('rb') as handle:
            data = pickle.load(handle)
        if not isinstance(data, dict) or not isinstance(data.get('data_list'), list):
            raise ValueError('需要顶层为 dict 且 data_list 为 list 的标注文件')
        print_structure(data)
        if args.inspect:
            return

        total = len(data['data_list'])
        count = math.ceil(total * args.ratio) if args.ratio is not None else (
            args.count if args.count is not None else 100)
        indices = select_indices(total, count, args.method, args.seed)
        sampled = dict(data)
        sampled['data_list'] = [data['data_list'][i] for i in indices]
        # 保留原 sample_idx/token、全部标注字段和元信息，不重新编号或修改路径。
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as handle:
            try:
                # protocol=4 兼容 Python 3.4+，不依赖 mmengine/mmcv。
                pickle.dump(sampled, handle, protocol=4)
            except BaseException:
                args.output.unlink()
                raise
        print(f'完成: {total} -> {count} 帧；方式={args.method}；seed={args.seed}')
        print(f'已保存: {args.output} ({args.output.stat().st_size / 1024**2:.2f} MiB)')
    except (OSError, ValueError, pickle.UnpicklingError, EOFError, ImportError) as exc:
        parser.exit(1, f'错误: {exc}\n')


if __name__ == '__main__':
    main()
