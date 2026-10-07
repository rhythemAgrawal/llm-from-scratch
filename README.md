# LLM from scratch

My follow-along of Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) lecture series, worked through in 2025.

This is course work, not original research. I typed the code out while following the lectures. Comments that are questions or reminders to myself are left as they were.

## What is here

| File | Lecture | What it covers |
|---|---|---|
| `micrograd.ipynb` | Building micrograd | A scalar autograd engine (`Value`), backpropagation, a small MLP and its training loop |
| `makemore_1.ipynb` | Building makemore | Character-level bigram model, first by counting and then as a one-layer neural network |
| `makemore_2.ipynb` | Building makemore Part 2: MLP | Character embeddings, an MLP language model, learning-rate search |
| `makemore_3.ipynb` | Building makemore Part 3: Activations & Gradients, BatchNorm | Activation statistics, initialisation, batch normalisation |
| `makemore_4_backprop.ipynb` | Building makemore Part 4: Becoming a Backprop Ninja | The lecture's exercise notebook with my solutions: gradients of the Part 3 network written by hand and checked against PyTorch (see status below) |
| `gpt.ipynb`, `bigram.py`, `gpt.py` | Let's build GPT: from scratch, in code, spelled out | Self-attention, multi-head attention and a small character-level transformer trained on Tiny Shakespeare |
| `tokenization.ipynb` | Let's build the GPT Tokenizer | Byte-pair encoding: training merges, `encode` and `decode` |
| `gpt2.py` | Let's reproduce GPT-2 (124M) | GPT-2 architecture, loading the pretrained weights, and a basic training loop (see status below) |

## Status

`gpt2.py` stops partway through the GPT-2 lecture. It has:

- the GPT-2 (124M) architecture, with weight tying and the scaled initialisation
- loading of the pretrained weights from Hugging Face
- a simple data loader and a 50-step training loop on Tiny Shakespeare, with per-step timing
- `F.scaled_dot_product_attention` in place of the hand-written attention
- `torch.autocast` and `torch.compile`, tried on Apple's MPS backend and left commented out

It does not have the learning-rate schedule, weight decay, gradient accumulation, multi-GPU training, the FineWeb-Edu training run or the HellaSwag evaluation.

`makemore_4_backprop.ipynb` is partly done:

- Exercise 1 (backpropagating through every step of the network by hand): done, all 26 gradients match PyTorch.
- Exercise 2 (the cross-entropy gradient in one step): done, matches PyTorch.
- Exercise 3 (the batch-normalisation gradient in one step): attempted, does not match yet.
- Exercise 4 (training the network with the hand-written backward pass): not done.

There is no code here for makemore Part 5 (WaveNet).

## Running it

Written and run on an Apple Silicon Mac. `gpt.py` and `gpt2.py` assume the MPS device and will need small changes to run elsewhere.

```bash
pip install -r requirements.txt
python bigram.py
```

`micrograd.ipynb` draws computation graphs and needs the Graphviz system package as well as the Python one.

`names.txt` and `input.txt` are the datasets the lectures use, included so the code runs as is.

## Credits

All of the model code follows Andrej Karpathy's lectures and repositories:
[nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero),
[micrograd](https://github.com/karpathy/micrograd),
[makemore](https://github.com/karpathy/makemore),
[ng-video-lecture](https://github.com/karpathy/ng-video-lecture),
[minbpe](https://github.com/karpathy/minbpe) and
[build-nanogpt](https://github.com/karpathy/build-nanogpt).

- `names.txt` comes from the makemore repository.
- `input.txt` is Tiny Shakespeare, from the [char-rnn](https://github.com/karpathy/char-rnn) repository.
