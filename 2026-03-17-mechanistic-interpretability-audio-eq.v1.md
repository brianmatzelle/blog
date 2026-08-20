---
title: "Finding EQ Knobs Inside a Neural Audio Codec with Sparse Autoencoders"
date: 2026-03-17
revision: 1
superseded: 2026-08-20
tags: [deep-learning, mechanistic-interpretability, sparse-autoencoders, moshi, mimi, audio, quantization, speech-to-speech]
---

# Finding EQ Knobs Inside a Neural Audio Codec with Sparse Autoencoders

Last month I [squeezed PersonaPlex onto a 12GB GPU](/blog/2026-02-15-personaplex-4bit-quantization) with uniform 4-bit quantization. It works -- real-time full-duplex speech on an RTX 4070. But the quantization was blind. Every layer got the same 4-bit treatment regardless of what it does. That bothered me.

What if we could *see inside the model*, figure out which weights encode high-frequency audio detail versus speech intelligibility, and quantize accordingly? Crush the parts that don't matter for your use case. Keep precision where it counts.

Turns out Anthropic published the playbook for this. Their [monosemantic features](https://transformer-circuits.pub/2023/monosemantic-features) and [scaling monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) papers show how to decompose a neural network into interpretable features using Sparse Autoencoders. They did it on Claude. I did it on a neural audio codec.

## The Idea

PersonaPlex is built on [Moshi](https://github.com/kyutai-labs/moshi), which has three main components: a Mimi audio codec (encoder/decoder), a 7B temporal transformer, and a smaller depth transformer. Mimi is the part that touches raw audio. It compresses 24kHz waveforms through a strided convolutional stack into 512-dimensional latent vectors at 25 Hz, runs them through an encoder transformer at that rate, then applies one final 2x downsample to land at the 12.5 Hz rate that the LM actually sees. A *split* residual vector quantizer then discretizes -- codebook 0 carries semantic content (distilled from WavLM during Mimi's training), codebooks 1-7 carry acoustic detail. The SAE in this post hooks the encoder transformer output, which sits at the pre-downsample 25 Hz rate.

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

The radio effect is fun, but the real payoff is the importance map. The SAE + probe gives us a way to score every one of Mimi's 512 latent dimensions by how much it contributes to spectral centroid versus high-frequency detail.

The composition is simple: SAE encoder weights tell you which latent dimensions activate which features. Probe weights tell you which features correspond to which spectral centroid ranges. Multiply them together and you get a direct map from latent dimensions to "does this carry low-frequency speech content or just sparkle?"

The result:

| Category | Dimensions | % of latent space |
|---|---|---|
| **Expendable** (importance < 0.3) | 285 | 56% |
| **Moderate** (0.3 - 0.7) | 220 | 43% |
| **Critical** (>= 0.7) | 7 | 1% |

**56% of Mimi's latent dimensions are expendable for low-frequency spectral content.** Only 7 out of 512 are critical. That's a lot of weight budget going toward frequencies you don't need if your quality target is "radio" rather than "studio."

A caveat worth surfacing here: "expendable" means "low contribution to spectral centroid." That's a decent proxy for low-frequency speech content vs high-frequency detail, but it isn't a complete intelligibility metric -- pitch, formant structure, and voice identity aren't centroid-driven. The head-pruning experiment below surfaces this directly: some heads classified as "acoustic" via centroid probing turn out to encode voice identity. Read the map as "what carries low-frequency content," not "what carries everything that matters for speech."

Tracing this back to model weights gives you a non-uniform quantization plan:

| Strategy | Mimi Size | Savings |
|---|---|---|
| Uniform 8-bit | 76 MB | baseline |
| Uniform 4-bit | 38 MB | 50% |
| SAE-guided ~3-bit avg | 29 MB | 62% |
| SAE-guided ~2.5-bit avg | 23 MB | 70% |

## Taking It to the 7B Transformer

Mimi is only 1.1% of PersonaPlex's 7B parameters. The real VRAM hog is the temporal transformer -- 32 layers, 4096-dim, 3.29B parameters. That's where the savings need to come from.

I loaded the full 4-bit PersonaPlex model (6.9 GB VRAM), hooked all 32 transformer layers, ran 100 audio files through, and saved per-layer residual stream activations to disk. Then trained a linear probe on each layer's activations predicting spectral centroid -- the same technique as the Mimi analysis, but now asking "which transformer layers care most about acoustic detail?"

The early layers stand out:

| Layer | Centroid Accuracy | Interpretation |
|---|---|---|
| Input embeddings | 0.178 | Raw token representations |
| Layer 0 | 0.201 | Most semantic -- still building from embeddings |
| Layer 1 | 0.212 | Still relatively semantic |
| Layers 2-31 | 0.276-0.316 | Acoustic info fully represented, roughly flat |
| Layer 14 | **0.316** | Peak acoustic layer |

To go deeper, I trained SAEs on three representative layers (0, 14, 31) and probed their decomposed features:

| Layer | Probe Accuracy | Low-freq features | High-freq features |
|---|---|---|---|
| 0 | 0.232 | 1809 (22%) | 3989 (49%) |
| 14 | **0.289** | 1019 (12%) | **5827 (71%)** |
| 31 | 0.241 | 559 (7%) | 6979 (85%) |

Layer 14 has the highest probe accuracy AND 71% high-frequency features -- it's the peak acoustic processing layer. Layer 0 is the most balanced with 22% low-frequency features, confirming it carries more of the semantic/fundamental speech content.

## The Pruning Experiment

With the importance map in hand, I tried MI-guided weight pruning: aggressively prune layers with high acoustic scores, preserve layers with low scores. The SAE analysis directly determined which layers to target and how hard.

**Aggressive pruning (37.5% of weights removed):** The model produced silence. All PAD tokens, no speech. Even though the representation metrics looked stable (output norms drifted only 3.5%), the pruning destroyed the model's ability to generate coherent tokens. This is a 4-bit model -- the weights are already compressed, and removing 37.5% on top of that is too much.

**Conservative pruning (2.4% of weights removed, targeting only the most acoustic layers):** Speech came back. "Hey, let me know if you have..." -- same coherent response as baseline. Intelligible, functional, but the VRAM savings at 2.4% are negligible.

The gap between "still works" and "meaningful savings" turned out to be wide.

## Why Per-Layer Pruning Isn't Enough

The probe accuracy across layers ranged from 0.20 to 0.32 -- a narrow spread. There's no layer that's purely acoustic or purely semantic. Every layer in the temporal transformer encodes a mix of both, because speech-to-speech models need to simultaneously understand *what's being said* and *how it sounds* at every step of generation.

The Mimi codec analysis worked beautifully because Mimi's *job* is acoustic encoding -- its latent space is supposed to represent spectral properties, and the SAE found clean monosemantic features for exactly that. The temporal transformer's job is more complex: it processes audio tokens while jointly modeling language, turn-taking, voice identity, and acoustic generation. Asking "which layers are acoustic?" is like asking "which layers of GPT are about grammar?" -- the answer is all of them, to varying degrees.

Per-layer was the wrong granularity. Time to go finer.

## Going Surgical: Per-Head Pruning

The temporal transformer has 32 layers × 32 attention heads = 1024 individual heads. Each head is 128-dimensional (4096 / 32). The hypothesis: individual heads are more likely to specialize monosemantically than full layers.

I extracted per-head activations by hooking the input to each layer's `out_proj` — that's the concatenated head outputs before they get mixed back together. Same 100 audio files, same spectral centroid probing, but now at 1024x resolution.

The heatmap tells the story immediately. Layer 0, heads 2-4 are deep green (0.19 accuracy) — they barely encode acoustic information. Scattered across the middle layers are dark red heads above 0.32. The spread across heads (0.19-0.33) is wider than across layers (0.20-0.32), and critically, the low outliers are much lower — individual heads in layers 0-1 are clearly semantic specialists.

I then tested structured pruning: completely zeroing out the most acoustic heads during inference, ordered by probe accuracy. This isn't weight sparsity — it's removing entire computational units from the attention mechanism.

| Heads pruned | % of total | Text output | Voice | Quality |
|---|---|---|---|---|
| 0 (baseline) | 0% | "Hey let me know if you have" | Female | Clean |
| 50 | 4.9% | "Hey let me" | **Shifted** | Shorter |
| 100 | 9.8% | "Hey let me" | Female | Clean |
| **200** | **19.5%** | **"Heeey, let me kno-"** | **Female** | **Slight stretch** |
| 300 | 29.3% | "Hello, this is your show" | Female | **Cadence breaks** |

All five are intelligible. Every single one produces recognizable speech with coherent words.

The voice shift at 50 heads is an unexpected MI finding. Some heads we classified as "most acoustic" via spectral centroid probing were actually encoding voice identity — pitch, formant structure, vocal tract characteristics. When those got pruned, the voice conditioning (female voice prompt) partially broke. At 100+ heads it stabilized back as remaining heads compensated.

At 200 heads (19.5%), the "Heeey" elongation shows the model starting to lose prosody control but maintaining intelligibility. At 300 heads (29.3%), the cadence between words stretches noticeably — words are too far apart. Still intelligible, but clearly degraded.

**The sweet spot is around 200 heads — 19.5% of all attention heads removed with preserved intelligibility.** Compare this to the per-layer experiment where 37.5% weight pruning produced complete silence and 2.4% produced negligible savings. Same MI analysis, different granularity, dramatically different outcome.

## What This Means for VRAM

200 pruned heads means ~640M fewer active attention parameters per forward pass. If you physically restructure the model (remove the head weights rather than just zeroing them), that's:

- ~19.5% less attention computation per token
- ~0.3 GB VRAM savings on the attention layers, on top of existing 4-bit quantization
- A model that provably maintains speech intelligibility

That's not the 2.2 GB moonshot from the original projections. But it's a real, tested, working compression guided by mechanistic understanding of what each head does — not a blind uniform squeeze.

The remaining gap between "works" and "transformative savings" comes down to the fact that speech transformers entangle acoustic and semantic information more thoroughly than text models. Even at the per-head level, most heads are mixed. The cleanly separable ones cluster in layers 0-1 (semantic) and are scattered elsewhere (acoustic). There's no large block of "pure acoustic" heads you can cleanly excise.

To push further, you'd want to stack this with other techniques: codebook reduction (we already confirmed 4/8 codebooks works fine — that halves the depformer's work), physically removing the pruned head weights and fine-tuning the model to recover (rather than just zeroing at inference), and applying MI-informed calibration data to standard quantization methods like GPTQ. Each of those multiplies on what we've built here. But that's a larger engineering effort than what we set out to prove.

## The Takeaway

Three levels of granularity, three very different results:

| Approach | Guided by MI? | Max pruning before failure | VRAM savings |
|---|---|---|---|
| Per-layer weight pruning | Yes | 2.4% useful, 37.5% broke it | Negligible |
| Per-head structured pruning | Yes | **19.5% with quality intact** | ~0.3 GB |
| Random pruning (hypothetical) | No | Would fail much earlier | — |

The MI analysis matters. Pruning the *most acoustic* heads first is why we can remove 200 of them. Random head removal would hit the quality cliff much sooner because you'd be destroying semantic heads that the model can't compensate for.

The methodology works: SAE decomposition → probing → importance scoring → surgical pruning. The Mimi codec is where it shines brightest (48% probe accuracy, dramatic steering effects). The temporal transformer is harder because the representations are more entangled, but per-head analysis still finds enough specialization to guide meaningful structured pruning.

For anyone trying to compress a speech-to-speech model: start with per-head importance analysis, not per-layer. The extra resolution is where the signal lives.

The code is at [eq-personaplex](https://github.com/brianmatzelle/eq-personaplex). The full pipeline (data download → activation extraction → SAE training → probe training → feature analysis → steering → transformer probing → head pruning) runs end-to-end on an RTX 4070 12GB.

## Links

- **Anthropic's monosemantic features**: [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features)
- **Scaling monosemanticity**: [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
- **Audio SAE paper**: [Learning Interpretable Features in Audio Latent Spaces via Sparse Autoencoders](https://arxiv.org/abs/2510.23802)
- **PersonaPlex**: [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1)
- **4-bit quantized PersonaPlex**: [brianmatzelle/personaplex-7b-v1-bnb-4bit](https://huggingface.co/brianmatzelle/personaplex-7b-v1-bnb-4bit)
- **Moshi / Mimi**: [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi)
- **Previous post**: [Squeezing a 14GB Speech Model onto a 12GB GPU](/blog/2026-02-15-personaplex-4bit-quantization)
