---
title: "Finding EQ Knobs Inside a Neural Audio Codec with Sparse Autoencoders"
date: 2026-03-17
tags: [deep-learning, mechanistic-interpretability, sparse-autoencoders, moshi, mimi, audio, quantization, speech-to-speech]
---

# Finding EQ Knobs Inside a Neural Audio Codec with Sparse Autoencoders

Last month I [squeezed PersonaPlex onto a 12GB GPU](/blog/2026-02-15-personaplex-4bit-quantization) with uniform 4-bit quantization. It works -- real-time full-duplex speech on an RTX 4070. But the quantization was blind. Every layer got the same 4-bit treatment regardless of what it does. That bothered me.

What if we could *see inside the model*, figure out which weights encode high-frequency audio detail versus speech intelligibility, and quantize accordingly? Crush the parts that don't matter for your use case. Keep precision where it counts.

Turns out Anthropic published the playbook for this. Their [monosemantic features](https://transformer-circuits.pub/2023/monosemantic-features) and [scaling monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) papers show how to decompose a neural network into interpretable features using Sparse Autoencoders. They did it on Claude. I did it on a neural audio codec.

## The Idea

PersonaPlex is built on [Moshi](https://github.com/kyutai-labs/moshi), which has three main components: a Mimi audio codec (encoder/decoder), a 7B temporal transformer, and a smaller depth transformer. Mimi is the part that touches raw audio -- it compresses 24kHz waveforms into 512-dimensional latent vectors at 25Hz, then a residual vector quantizer discretizes them into tokens.

Those 512-dimensional latent vectors are where audio properties like frequency content, amplitude, and timbre get encoded. If you can decompose that space into interpretable features, you know exactly what each dimension is doing. And if you know what each dimension is doing, you know which ones you can throw away.

The target application: make audio sound like it's coming through a radio. A radio signal is band-limited (~300Hz-3kHz), compressed, and low-fidelity. If we can find the "EQ knob" inside Mimi's latent space and turn it, we've proven the features are real. Then we can use that same map to guide quantization.

## Training the SAE

A Sparse Autoencoder takes a dense representation, expands it into a much wider hidden layer, and forces most of those hidden units to be zero (via an L1 penalty). The surviving active units tend to be monosemantic -- each one represents a single interpretable concept.

I followed the architecture from [Learning Interpretable Features in Audio Latent Spaces via Sparse Autoencoders](https://arxiv.org/abs/2510.23802), with one key modification from the audio domain: RMS normalization after the ReLU activation. This prevents out-of-distribution artifacts when you later manipulate features.

```python
class AudioSAE(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=4096):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.encoder(x))
        rms = torch.sqrt(torch.mean(h**2, dim=-1, keepdim=True) + 1e-8)
        h = h / rms
        return h

    def decode(self, h):
        return self.decoder(h)
```

Loss is MSE reconstruction plus L1 sparsity: `L = ||x - x̂||² + λ||h||₁`

The pipeline:

1. Download ~500 LibriSpeech utterances (~68 minutes of speech)
2. Run each through Mimi's encoder, hook the encoder transformer output (pre-quantizer)
3. Save the 512-dim latent vectors to disk (102,099 frames total)
4. Normalize to zero mean / unit variance
5. Train the SAE on those vectors (no Mimi in VRAM during training)

Training took under a minute on the RTX 4070:

```
Epoch   5/80 | loss=0.03312 recon=0.02697 | sparsity=0.304 alive=4096/4096 (100.0%)
Epoch  40/80 | loss=0.00834 recon=0.00414 | sparsity=0.326 alive=4096/4096 (100.0%)
Epoch  80/80 | loss=0.00502 recon=0.00123 | sparsity=0.328 alive=4096/4096 (100.0%)
```

100% alive neurons, reconstruction loss converged to 0.001. The SAE successfully decomposes Mimi's 512-dim latent space into 4096 sparse features.

## Finding the EQ Features

Having sparse features is useless if you can't interpret them. I used linear probes -- simple classifiers that map SAE features to known acoustic properties extracted from the raw audio with librosa:

- **Spectral centroid** (the "center of mass" of the frequency spectrum -- this is the EQ proxy)
- **RMS amplitude** (loudness)

Each property gets discretized into 20 bins, then a linear classifier `p(class) = softmax(W @ features + b)` tries to predict the bin from SAE activations alone.

Results:

| Property | Accuracy | Random Baseline |
|---|---|---|
| Spectral centroid | **48.1%** | 5.0% |
| RMS amplitude | **46.1%** | 5.0% |

The features are **linearly decodable** for audio properties. The SAE learned real structure, not noise.

The weight matrix W is the interesting part. For each SAE feature, W tells you exactly which acoustic property classes it pushes toward. Feature 3744, for example, has high positive weight for centroid bin 0 (lowest frequencies) and negative weight everywhere else. It's a "low frequency" detector. Feature 3214 peaks at bins 12-14 (high frequencies). These are monosemantic EQ features.

## Steering: The Radio Effect

This is the "Golden Gate Claude" moment but for audio. Anthropic showed you can clamp a feature to make Claude obsessively talk about the Golden Gate Bridge. Same principle: clamp the low-centroid features to make audio sound like a radio.

The control vector comes directly from the probe weights:

```python
radio_vector = probe.get_control_vector(low_class) - probe.get_control_vector(high_class)
radio_vector = radio_vector / radio_vector.norm()
```

Then during inference, hook into Mimi's encoder transformer, intercept the latent vectors, run them through the SAE, add the scaled control vector, decode back, and let Mimi's decoder reconstruct the audio:

```python
def steering_hook(module, input, output):
    latents = output.last_hidden_state
    flat = latents.reshape(-1, 512)

    # Normalize (SAE was trained on normalized data)
    flat_norm = (flat - norm_mean) / norm_std

    # SAE encode → steer → SAE decode
    h = sae.encode(flat_norm)
    h_steered = h + alpha * radio_vector
    h_steered = F.relu(h_steered)  # features must be non-negative
    reconstructed = sae.decode(h_steered)

    # Denormalize back
    reconstructed = reconstructed * norm_std + norm_mean
    output.last_hidden_state = reconstructed.reshape(latents.shape)
    return output
```

The `alpha` parameter controls how hard you turn the knob.

| Alpha | Spectral Centroid | Effect |
|---|---|---|
| 0 (SAE round-trip only) | 2611 Hz | Nearly identical to original |
| 1 | 2463 Hz | Subtle warmth |
| 2 | 2303 Hz | Noticeable low-pass |
| 5 | 1872 Hz | Clear band-limiting |
| **10** | **1241 Hz** | **Sounds like a radio** |

Original audio centroid: 2832 Hz. At alpha=10, it drops to 1241 Hz -- a 56% shift toward the low-frequency, band-limited profile of AM radio. And it's intelligible. You can understand every word.

## The Normalization Bug

One gotcha that cost me time: the SAE was trained on normalized latents (zero mean, unit variance), but the steering hook initially passed raw Mimi latents straight through. The SAE saw out-of-distribution inputs and produced garbage. Every alpha value sounded like unintelligible mush -- you could hear the cadence of speech but no words.

Fix was straightforward: normalize before SAE encode, denormalize after SAE decode. Always match your inference-time preprocessing to your training-time preprocessing. A boring bug, but it's the kind of thing that makes you question whether the entire approach works until you find it.

## Where This Gets Interesting: Non-Uniform Quantization

The radio effect is fun, but the real payoff is the importance map. The SAE + probe gives us a way to score every one of Mimi's 512 latent dimensions by how much it contributes to speech intelligibility versus high-frequency detail.

The composition is simple: SAE encoder weights tell you which latent dimensions activate which features. Probe weights tell you which features correspond to which spectral centroid ranges. Multiply them together and you get a direct map from latent dimensions to "does this carry speech content or just sparkle?"

The result:

| Category | Dimensions | % of latent space |
|---|---|---|
| **Expendable** (importance < 0.3) | 285 | 56% |
| **Moderate** (0.3 - 0.7) | 220 | 43% |
| **Critical** (>= 0.7) | 7 | 1% |

**56% of Mimi's latent dimensions are expendable for speech intelligibility.** Only 7 out of 512 are critical. That's a lot of weight budget going toward frequencies you don't need if your quality target is "radio" rather than "studio."

Tracing this back to model weights gives you a non-uniform quantization plan:

| Strategy | Mimi Size | Savings |
|---|---|---|
| Uniform 8-bit | 76 MB | baseline |
| Uniform 4-bit | 38 MB | 50% |
| SAE-guided ~3-bit avg | 29 MB | 62% |
| SAE-guided ~2.5-bit avg | 23 MB | 70% |

## The Bigger Picture

Mimi is only 1.1% of PersonaPlex's 7B parameters. The real VRAM hog is the temporal transformer. But the same technique applies: train SAEs on the transformer's residual stream, find which layers and attention heads encode acoustic fidelity versus semantic content, and quantize accordingly.

Projected numbers for the full model:

| Strategy | VRAM |
|---|---|
| Uniform 4-bit (what I have today) | ~3.5 GB |
| SAE-guided ~3-bit avg | ~2.6 GB |
| SAE-guided ~2.5-bit avg | ~2.2 GB |

2.2 GB puts full-duplex speech-to-speech on a Jetson Orin Nano. Or a phone. The quality floor is "radio" -- band-limited but fully intelligible. For a voice assistant, that's all you need.

The insight that makes this work: **if you know what each part of the model does, you know what you can afford to lose.** Uniform quantization treats every weight as equally important. Mechanistic interpretability says they're not.

## What's Next

Train SAEs on the 7B temporal transformer's residual stream, per-layer. Identify which attention heads specialize in acoustic detail versus semantic content. Build the non-uniform quantization and measure the actual quality/size tradeoff end-to-end.

The code is at [eq-personaplex](https://github.com/brianmatzelle/eq-personaplex). The full pipeline (data download → activation extraction → SAE training → probe training → feature analysis → steering) runs in about 15 minutes on an RTX 4070.

## Links

- **Anthropic's monosemantic features**: [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features)
- **Scaling monosemanticity**: [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
- **Audio SAE paper**: [Learning Interpretable Features in Audio Latent Spaces via Sparse Autoencoders](https://arxiv.org/abs/2510.23802)
- **PersonaPlex**: [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1)
- **4-bit quantized PersonaPlex**: [brianmatzelle/personaplex-7b-v1-bnb-4bit](https://huggingface.co/brianmatzelle/personaplex-7b-v1-bnb-4bit)
- **Moshi / Mimi**: [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi)
- **Previous post**: [Squeezing a 14GB Speech Model onto a 12GB GPU](/blog/2026-02-15-personaplex-4bit-quantization)
