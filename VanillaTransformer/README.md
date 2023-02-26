# *Vanilla* Transformer from scrach with JAX/Flax

 This is a first notebook of a tutorial to understand implementation of Transformer Models with JAX/Flax. As a first step, this notebook shows how to implement *Vanilla* Transformer. Transformer Encoder Decoder Model, aka Vanilla Transformer was originally introduced in "Attention is all you need"[1] and is the origin of all derived transformers. Let's understand Transformer and how to use JAX/Flax!

The main contents are the following.
1. Implement *Vanilla* Transformer from scratch.
1. Using the Multi30k dataset, train the transformer to translate German to English.
1. Translate some sentences with a greedy search translator.
1. Plot attention matrixes.

The key features of this implementation are the following.
1. Implment *Vanilla* Transformer with original class of MultiHeadAttention and PositionalEncoder.
1. Made the translator that can tranlate batch by batch, not a sentence by a sentence.
1. By padding, all source/target sentences to max sequence length, fix an input shape of the jit compiled function, and minimize the number of compile. That enabled it to accelerate training very much!
1. Usually, both query and key sequeces are masked at the same time and masked position are changed to `jnp.finfo(dtype).min` in attention calculation with JAX/Flax implementation. In this implementation, only key sequences are masked and masked positions are changed to `-jnp.inf` and query sequences are masked in a loss function as common implementations.

The points that can be added in the future.
1. Use sentencepiece tokenizer in stead of SpaCy tokenizer.
1. Implement a beam search tokenizer from scratch.
1. Implement a functin that calculate BLEU score from scratch.

# Reference
- Paper

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017). Attention is All you Need. Advances in Neural Information Processing Systems, 2017-Decem, 5999–6009. 

- GitHub
1. https://github.com/bentrevett/pytorch-seq2seq
1. https://gist.github.com/enakai00/2371a25acb0bd7cd80ccd72c89364db9