# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llama_tokenizer_dir', default=None, type=str, required=True)
    parser.add_argument('--chinese_sp_model_file', default='./chinese_sp.model', type=str)
    parser.add_argument('--jieba_word_freq_file', default='data/vocab/word_freq.txt', type=str)
    args = parser.parse_args()

    llama_tokenizer_dir = args.llama_tokenizer_dir
    chinese_sp_model_file = args.chinese_sp_model_file
    jieba_vocab_file = args.jieba_word_freq_file
    # load
    llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
    chinese_sp_model = spm.SentencePieceProcessor()
    chinese_sp_model.Load(chinese_sp_model_file)

    llama_spm = sp_pb2_model.ModelProto()
    llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
    chinese_spm = sp_pb2_model.ModelProto()
    chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())

    # print number of tokens
    print(len(llama_tokenizer), len(chinese_sp_model))
    print(llama_tokenizer.all_special_tokens)
    print(llama_tokenizer.all_special_ids)
    print(llama_tokenizer.special_tokens_map)

    ## Add Chinese tokens to LLaMA tokenizer
    llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)

    # Read jieba vocab and sort by freq
    with open(jieba_vocab_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        word_freqs = [line.strip().split() for line in lines]
        word_freqs.sort(key=lambda x: int(x[1]), reverse=True)

    top_words = word_freqs[:20000]
    print('jieba top10 freq words:', top_words[:10])

    jieba_vocab_set = set([i[0] for i in top_words if i])
    print('jieba_vocab_set', len(jieba_vocab_set))
    print('jieba_vocab head:', list(jieba_vocab_set)[:3])

    print(len(llama_spm_tokens_set))
    print(f"Before:{len(llama_spm_tokens_set)}")
    added_set = set()
    for p in chinese_spm.pieces:
        piece = p.piece
        if piece not in llama_spm_tokens_set:
            # print('picec', piece)
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            llama_spm.pieces.append(new_p)
            added_set.add(piece)
    print(f"[add chinese tokens]New model pieces: {len(llama_spm.pieces)}")
    for p in jieba_vocab_set:
        piece = p
        if piece not in llama_spm_tokens_set and piece not in added_set:
            # print('jieba new picec', piece)
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            llama_spm.pieces.append(new_p)
    print(f"[add jieba tokens]New model pieces: {len(llama_spm.pieces)}")

    ## Save
    output_sp_dir = 'merged_tokenizer_sp'
    output_hf_dir = 'merged_tokenizer_hf'  # the path to save Chinese-LLaMA tokenizer
    os.makedirs(output_sp_dir, exist_ok=True)
    with open(output_sp_dir + '/chinese_llama.model', 'wb') as f:
        f.write(llama_spm.SerializeToString())
    tokenizer = LlamaTokenizer(vocab_file=output_sp_dir + '/chinese_llama.model')

    tokenizer.save_pretrained(output_hf_dir)
    print(f"Chinese-LLaMA tokenizer has been saved to {output_hf_dir}")

    # Test
    llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
    chinese_llama_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)
    print(tokenizer.all_special_tokens)
    print(tokenizer.all_special_ids)
    print(tokenizer.special_tokens_map)
    print('old len:', len(tokenizer), ' new len:', len(chinese_llama_tokenizer))
    text = '''this is a test, hello world. thisisatesthelloworld, 
慕容复来到河边，姑苏慕容氏在外面丢了人。
1号店一周岁了，我们一古脑儿买了10斤零食。
巴塞罗那足球俱乐部简称巴萨（Barça），是一家位于西班牙加泰罗尼亚巴塞罗那的足球俱乐部，于1899年由瑞士企业家胡安·甘伯所创立，世界球坛顶级足球俱乐部之一。俱乐部主场可容纳接近十万名观众，是全欧洲最大及世界第二大的足球场。
白日依山尽，黄河入海流。欲穷千里目，更上一层楼。'''
    print("Test text:\n", text)
    print(f"Tokenized by LLaMA tokenizer:{llama_tokenizer.tokenize(text)}")
    print(f"Tokenized by Chinese-LLaMA tokenizer:{chinese_llama_tokenizer.tokenize(text)}")


if __name__ == '__main__':
    main()
