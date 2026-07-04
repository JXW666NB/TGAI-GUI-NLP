"""
.TG 模型打包工具（ONNX Runtime 版）
==============================
将 TGAI ONNX 模型 (tgai.onnx + tokenizer.json) 打包为 .TG 文件。
.TG 本质是 ZIP 格式，手机端 TG CHAT 可一键导入并自动部署。

用法:
    python scripts/pack_tg.py --model exported/tgai.onnx --tokenizer exported/tokenizer.json --out TGAI-4000.tg
    python scripts/pack_tg.py --dir exported/ --out TGAI-4000.tg
"""
import argparse
import zipfile
import os
import json
from pathlib import Path


def find_files(directory: str) -> tuple:
    """从目录自动查找模型文件和 tokenizer.json"""
    dir_path = Path(directory)
    model_path = None
    tokenizer_path = None

    for ext in ['.onnx', '.pte']:
        candidates = list(dir_path.glob(f'*{ext}'))
        if candidates:
            model_path = str(candidates[0])
            break

    tok_candidate = dir_path / 'tokenizer.json'
    if tok_candidate.exists():
        tokenizer_path = str(tok_candidate)

    return model_path, tokenizer_path


def validate_file(path: str) -> bool:
    if not os.path.exists(path):
        print(f"  [错误] 文件不存在: {path}")
        return False
    if Path(path).stat().st_size < 1024:
        print(f"  [警告] {path} 文件过小，可能无效")
    return True


def pack(model: str, tokenizer: str, output: str, meta: dict = None):
    """打包为 .TG 文件"""
    print(f"\n打包 TGAI 模型 (ONNX):")
    print(f"  模型:       {model}")
    print(f"  tokenizer:  {tokenizer}")
    print(f"  输出:       {output}")

    if not all(validate_file(p) for p in [model, tokenizer]):
        return False

    if not output.lower().endswith('.tg'):
        output += '.tg'

    model_path = Path(model)
    model_name = model_path.name
    is_onnx = model_name.endswith('.onnx')

    # 检测外部数据文件 (.onnx.data)
    data_path = Path(str(model_path) + '.data')
    has_external_data = is_onnx and data_path.exists()

    sizes = {
        'model': os.path.getsize(model),
        'tokenizer': os.path.getsize(tokenizer),
    }
    extra_files = {}
    if has_external_data:
        sizes['model_data'] = data_path.stat().st_size
        extra_files[str(data_path)] = data_path.name

    total = sum(sizes.values())
    print(f"  总大小: {total / 1024 / 1024:.1f} MB")
    if has_external_data:
        print(f"  (含外部数据: {data_path.name}, {sizes['model_data'] / 1024 / 1024:.1f} MB)")

    manifest = {
        'format': 'tgai-onnx-1' if is_onnx else 'tgai-executorch-1',
        'engine': 'onnx' if is_onnx else 'executorch',
        'files': {
            'model': model_name,
            'tokenizer.json': Path(tokenizer).name,
        },
        'sizes': sizes,
        'has_external_data': has_external_data,
        **(meta or {}),
    }
    if has_external_data:
        manifest['files']['model_data'] = data_path.name

    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        zf.write(model, model_name)
        if has_external_data:
            zf.write(str(data_path), data_path.name)
        zf.write(tokenizer, 'tokenizer.json')

    output_size = os.path.getsize(output)
    print(f"  打包完成: {output} ({output_size / 1024 / 1024:.1f} MB)")
    print(f"\n将 {output} 传到手机后，在 TG CHAT「模型」页点击导入即可。")
    return True


def main():
    parser = argparse.ArgumentParser(description='打包 TGAI 模型为 .TG 文件')
    parser.add_argument('--model', type=str, help='模型文件路径 (.onnx 或 .pte)')
    parser.add_argument('--tokenizer', type=str, help='tokenizer.json 路径')
    parser.add_argument('--dir', type=str, help='模型目录（自动查找模型和 tokenizer）')
    parser.add_argument('--out', type=str, required=True, help='输出 .TG 文件路径')
    parser.add_argument('--name', type=str, help='模型名称')
    parser.add_argument('--author', type=str, default='', help='作者信息')
    parser.add_argument('--description', type=str, default='', help='模型描述')
    args = parser.parse_args()

    if args.dir:
        model, tokenizer = find_files(args.dir)
        if model is None:
            print("[错误] 目录中未找到 .onnx 或 .pte 模型文件")
            return
        if tokenizer is None:
            print("[错误] 目录中未找到 tokenizer.json")
            return
    elif args.model and args.tokenizer:
        model = args.model
        tokenizer = args.tokenizer
    else:
        print("[错误] 请指定 --dir 或同时指定 --model --tokenizer")
        return

    meta = {}
    if args.name:
        meta['name'] = args.name
    if args.author:
        meta['author'] = args.author
    if args.description:
        meta['description'] = args.description

    pack(model, tokenizer, args.out, meta)


if __name__ == '__main__':
    main()
