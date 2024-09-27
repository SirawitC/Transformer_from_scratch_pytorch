# Transformer from scratch using Pytorch

This repository provides a step-by-step implementation of the Transformer architecture from scratch using PyTorch. The Transformer model, introduced in the seminal paper "Attention is All You Need," has become the foundation for state-of-the-art natural language processing (NLP) models such as BERT and GPT. In this repository, we break down the core components of the Transformer, including multi-head self-attention, positional encoding, and layer normalization, offering a clear and intuitive understanding of how the model functions. Whether you're a student or researcher looking to deepen your understanding of Transformers or an engineer exploring custom implementations, this repository will guide you through the essential building blocks of this powerful architecture.

> This repository is heavily inspired by the YouTube video by [Umar Jamil](https://www.youtube.com/@umarjamilai), and we would really like to acknowledge his valuable contribution to making Transformers accessible to a broader audience.

## Table of Contents

- [Introduction](#Introduction)
- [Core Components](#Core-Components)
  - [Tokenizer](#Tokenizer)
  - [Input Embedding](#Input-Embedding)
  - [Positional Encoding](#Positional-Encoding)
  - [Multi-Head Attention](#Multi-Head-Attention)
  - [FeedForward Block](#FeedForward-Block)
  - [Residual Connection](#Residual-Connection)
  - [Layer Normalization](#Layer-Normalization)
- [Transformer Model](#Transformer-Model)
  - [Encoder](#Encoder)
  - [Decoder](#Decoder)
- [Training Loop](#Training-Loop)
- [Inference](#Inference)
- [License](#license)

## Introduction

When talking about something called a "Transformer", surely, each individual would have a different image in their mind. Many might associate this word with nostalgic movies/toys/cartoons regarding alien robots like Optimus Prime, while those with an electrical engineering background may think of a passive component for altering the voltage. Recently, computer and ML scientists, have also come up with a new definition of this word. Transformer architecture originally proposed in the groundbreaking paper "[Attention is all you need](https://arxiv.org/abs/1706.03762)", was initially introduced to address several limitations and flaws of recurrent neural networks (RNNs), especially in the context of tasks like neural machine translation (NMT) and other sequence-to-sequence tasks. Since then it has become a revolutionary foundation for the field of natural language processing and beyond, serving as the backbone of many modern AI models. A marvelous technology like chatGPT or even a more accurate language translator that we take for granted, would not be possible without this vital architecture as its constituent. It would not be far-fetched to say that this transformer model is a spark for this new era of artificial intelligence technology.

Now without further ado, let's get to know this marvel of technology thoroughly starting with the overall architecture of a transformer.

<p align="center">
  <img src="./img/transformer.png" alt="transformer" width="350" height="450"/>
</p>
<b><i><p align="center">The Transformer Architecture - from Attention is all you need paper</p></i></b>

## Core Components

### Tokenizer

### Input Embedding

<p align="center">
  <img src="./img/input_embed.png" alt="input_embed"/>
</p>

### Positional Encoding

### Multi-Head Attention

### FeedForward Block

### Residual Connection

### Layer Normalization

## Transformer Model

### Encoder

### Decoder

## Training Loop

## Inference

## License

This code is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
