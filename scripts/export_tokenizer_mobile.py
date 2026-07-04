"""
导出 TGAI 分词器为手机端兼容格式
========================================
用法:
    python export_tokenizer_mobile.py --tokenizer tgai_nlp/checkpoints/tokenizer.json --out tgai_nlp/checkpoints/tokenizer_mobile.json
"""
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', required=True, help='Python tokenizer JSON 路径')
    parser.add_argument('--out', required=True, help='输出路径')
    args = parser.parse_args()

    with open(args.tokenizer, 'r', encoding='utf-8') as f:
        data = json.load(f)

    token_to_id = data['token_to_id']

    # 从 HF tokenizer 导出 merges（如果存在）
    merges = []
    hf_path = data.get('hf_path')
    if hf_path and Path(hf_path).exists():
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(hf_path)
            # HF BPE merges 格式: "word1 word2"
            merges = [f"{m[0]} {m[1]}" for m in tok.model.get_merges()]
            print(f"  [导出] 从 HF tokenizer 导出 {len(merges)} 条 merges")
        except Exception as e:
            print(f"  [警告] 无法读取 HF merges: {e}")

    # 如果没有 merges，生成一个简单的 char-level fallback
    if not merges:
        print("  [警告] 没有 merges，使用 char-level fallback（编码效果可能略有差异）")
        # 收集所有单字符 token 作为基础词汇
        chars = sorted([t for t in token_to_id if len(t) == 1])
        # 构建简单的二元组 merges（按词频启发式，这里直接按字母序）
        for i in range(len(chars) - 1):
            merges.append(f"{chars[i]} {chars[i+1]}")

    mobile_data = {
        "special_ids": {
            "pad": token_to_id.get("<PAD>", 0),
            "unk": token_to_id.get("<UNK>", 1),
            "bos": token_to_id.get("<BOS>", 2),
            "eos": token_to_id.get("<EOS>", 3),
        },
        "token_to_id": token_to_id,
        "merges": merges,
    }

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(mobile_data, f, ensure_ascii=False, indent=2)

    print(f"  [完成] 已导出: {args.out}")
    print(f"  词汇量: {len(token_to_id)}, merges: {len(merges)}")

if __name__ == '__main__':
    main()
