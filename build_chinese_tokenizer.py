# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Build chinese tokenizer from corpus txt

# train sentencepiece model from `corpus.txt` and makes `m.model` and `m.vocab`
# `m.vocab` is just a reference. not used in the segmentation.
# spm.SentencePieceTrainer.train('--input=data/pretrain/tianlongbabu.txt --model_prefix=m --vocab_size=20000')
"""
import sentencepiece as spm


def main():
    spm.SentencePieceTrainer.train(
        input='data/pretrain/tianlongbabu.txt',
        model_prefix='chinese_sp',
        shuffle_input_sentence=False,
        train_extremely_large_corpus=True,
        max_sentence_length=16384,
        pad_id=3,
        model_type="BPE",
        vocab_size=50000,
        split_digits=True,
        split_by_unicode_script=True,
        byte_fallback=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        normalization_rule_name="nfkc",
    )

    # makes segmenter instance and loads the model file (m.model)
    sp = spm.SentencePieceProcessor()
    sp.load('chinese_sp.model')

    # encode: text => id
    print(sp.encode_as_pieces('慕容复来到河边,this is a test'))
    print(sp.encode_as_ids('this is a test'))

    # decode: id => text
    print(sp.decode_pieces(['▁This', '▁is', '▁a', '▁t', 'est']))
    # print(sp.decode_ids([209, 31, 9, 375, 586]))


if __name__ == '__main__':
    main()
