import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("tokenizer/tokenizer.model")

text = "Hello PersistenceAI"

tokens = sp.encode(text)

print(tokens)

decoded = sp.decode(tokens)

print(decoded)
