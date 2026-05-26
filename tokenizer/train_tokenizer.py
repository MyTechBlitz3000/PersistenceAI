import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="datasets/data.txt",
    model_prefix="tokenizer/tokenizer",
    vocab_size=32000,
    model_type="bpe",
    character_coverage=1.0
)

print("Tokenizer trained.")
