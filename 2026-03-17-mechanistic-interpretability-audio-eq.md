---
title: "Finding EQ Knobs Inside a Neural Audio Codec with Sparse Autoencoders"
date: 2026-08-20
revision: 2
revised: 2026-08-20
revisionNote: "Controlled rerun with baselines, objective metrics, and direct ablation tests. Some claims strengthened, some corrected, one refuted."
tags: [deep-learning, mechanistic-interpretability, sparse-autoencoders, moshi, mimi, audio, quantization, speech-to-speech]
---

# Finding EQ Knobs Inside a Neural Audio Codec with Sparse Autoencoders

> **Revision note (v2, August 2026).** Five months after publishing this, I reran the whole experiment on a DGX Spark with the controls and objective metrics the original lacked: raw-activation probe baselines, a properly sparse TopK SAE, difference-in-means and DSP steering baselines, ground-truth Whisper WER, ECAPA speaker similarity, and — for the first time — a direct ablation test of the importance map. The verdict cut both ways:
>
> - **Corrected:** the original SAE was not actually sparse (~2,750 of 4,096 features active per frame), and the probing and steering results don't need an SAE at all — raw Mimi latents do just as well.
> - **Confirmed, surprisingly:** the "56% of latent dimensions are expendable" claim survived its first direct ablation test decisively, and the SAE-composed importance map beats both random selection and a raw-probe map. This is where the SAE genuinely earns its place.
> - **Refuted:** "only 7 dimensions are critical" — keeping only the critical set destroys speech entirely.
>
> Sections below are corrected in place, with the new evidence inline. The original version of this post is preserved unchanged and viewable via the version switcher. Rerun code: scripts 30–36 in [eq-personaplex](https://github.com/brianmatzelle/eq-personaplex). For the quantization-transfer story, see the [May follow-up](/blog/2026-05-16-mi-guided-mixed-precision).

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

**v2 correction: this log was a red flag, not a success.** In v1 I read "100% alive neurons, reconstruction loss converged to 0.001" as the SAE working. It's the opposite. `sparsity=0.328` means only 32.8% of activations are *zero* -- about 2,750 of 4,096 features fire on every frame. That is not a sparse autoencoder; it's a slightly-regularized overcomplete linear basis that reconstructs almost perfectly *because* nothing forces it to be sparse. 100% alive neurons with a weak L1 penalty is a symptom, not health. The rerun reproduced this exactly on 6.3× more data (L0 = 2,903), then replaced the recipe with a TopK SAE ([Gao et al. 2024](https://arxiv.org/abs/2406.04093)) that enforces sparsity by construction:

| SAE | Dictionary | L0 per frame | Val FVU | Dead features |
|---|---|---|---|---|
| ReLU + L1 (v1 recipe) | 4,096 | 2,903 | 0.005 | 0 |
| TopK (v2) | 8,192 | **32** | 0.142 | 0 |

The TopK SAE explains 86% of latent variance through exactly 32 active features per frame. When I talk about "sparse features" below, honest numbers only exist for the v2 TopK model; the v1 model's "features" were dense mixtures.

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

**v2: this number replicates -- but it's not the SAE's number.** The rerun added the baseline v1 skipped: probing the raw 512-dim latents directly, with utterance-level splits, trained on train-clean-100 and scored on held-out test-clean (disjoint speakers):

| Feature space | Centroid acc | ±1 bin | Pitch acc | Centroid ridge R² |
|---|---|---|---|---|
| Raw latents (512) | 0.428 | 0.805 | 0.479 | 0.869 |
| ReLU SAE (4,096) | 0.424 | 0.800 | 0.508 | 0.886 |
| TopK SAE (8,192) | 0.417 | 0.814 | **0.566** | 0.843 |

Raw latents decode spectral centroid exactly as well as either SAE. The 48.1% was real (and, to my mild surprise, *not* inflated by the original random-frame split -- utterance-level splits give the same in-dataset number; the drop to ~0.43 is cross-dataset transfer). But it demonstrates linear structure in Mimi's latents, not anything the SAE added. The one probing task where the sparse dictionary genuinely wins is **pitch** (0.566 vs 0.479 raw).

In v1 I pointed at individual features -- "feature 3744 is a low-frequency detector" -- as monosemantic EQ features. With 2,750 features active per frame, that reading doesn't survive: probe weights on a dense code identify directions, not units. The TopK model is where feature-level claims could be made honestly, and I haven't yet done that analysis per-feature.

## Steering: The Radio Effect

This is the "Golden Gate Claude" moment but for audio. Anthropic showed you can clamp a feature to make Claude obsessively talk about the Golden Gate Bridge. Same principle: clamp the low-centroid features to make audio sound like a radio.

The control vector comes directly from the probe weights:

```python
radio_vector = probe.get_control_vector(low_class) - probe.get_control_vector(high_class)
radio_vector = radio_vector / radio_vector.norm()
```

Then during inference, hook into Mimi's encoder transformer, intercept the latent vectors, run them through the SAE, add the scaled control vector, decode back, and let Mimi's decoder reconstruct the audio. The `alpha` parameter controls how hard you turn the knob.

| Alpha | Spectral Centroid | Effect |
|---|---|---|
| 0 (SAE round-trip only) | 2611 Hz | Nearly identical to original |
| 2 | 2303 Hz | Noticeable low-pass |
| 5 | 1872 Hz | Clear band-limiting |
| **10** | **1241 Hz** | **Sounds like a radio** |

Original audio centroid: 2832 Hz. At alpha=10, it drops to 1241 Hz -- a 56% shift toward the low-frequency, band-limited profile of AM radio. And it's intelligible. You can understand every word.

**v2: the effect replicates; the mechanism attribution doesn't.** The rerun calibrated four latent-space methods to the *same* centroid shift and scored them on 50 held-out clips with Whisper-large-v3-turbo WER against LibriSpeech ground truth and ECAPA speaker similarity -- plus a plain Butterworth band-pass (300–3000 Hz) as the known-answer reference:

| Method | Centroid (Hz) | WER | Speaker sim |
|---|---|---|---|
| Original audio | 1,594 | 0.019 | 1.000 |
| Mimi round-trip (codec floor) | 1,442 | 0.018 | 0.923 |
| Raw-latent probe direction | 736 | 0.021 | **0.752** |
| ReLU-SAE probe direction (v1 method) | 774 | 0.027 | **0.752** |
| Difference-in-means | 707 | 0.020 | 0.707 |
| DSP band-pass | 1,184 | 0.024 | 0.643 |
| TopK-SAE probe direction | 794 | 0.027 | 0.622 |

Three things. First, "it's intelligible" holds everywhere -- WER barely moves in any arm. Second, the same probe trained on raw latents, steered without any SAE round-trip, causes *identical* collateral damage to the v1 method. The SAE contributes nothing to the radio effect; the linear direction was in the latents all along. (This matches what [AxBench](https://arxiv.org/abs/2501.17148) found for LLM steering.) Third -- the genuinely new result -- every latent-space method preserves speaker identity *better* than an actual DSP band-pass at matched centroid shift. Latent EQ beats waveform EQ. That's a better headline than the one I originally wrote.

## The Normalization Bug

One gotcha that cost me time: the SAE was trained on normalized latents (zero mean, unit variance), but the steering hook initially passed raw Mimi latents straight through. The SAE saw out-of-distribution inputs and produced garbage. Every alpha value sounded like unintelligible mush -- you could hear the cadence of speech but no words.

Fix was straightforward: normalize before SAE encode, denormalize after SAE decode. Always match your inference-time preprocessing to your training-time preprocessing. A boring bug, but it's the kind of thing that makes you question whether the entire approach works until you find it.

## Where This Gets Interesting: The Importance Map

The radio effect is fun, but the real payoff is the importance map. The SAE + probe gives us a way to score every one of Mimi's 512 latent dimensions by how much it contributes to spectral centroid versus high-frequency detail.

The composition is simple: SAE encoder weights tell you which latent dimensions activate which features. Probe weights tell you which features correspond to which spectral centroid ranges. Multiply them together and you get a direct map from latent dimensions to "does this carry low-frequency speech content or just sparkle?"

The result:

| Category | Dimensions | % of latent space |
|---|---|---|
| **Expendable** (importance < 0.3) | 285 | 56% |
| **Moderate** (0.3 - 0.7) | 220 | 43% |
| **Critical** (>= 0.7) | 7 | 1% |

**v2: this claim was never tested in v1 -- and it passed.** The rerun recomputed the map from retrained artifacts (298/512 expendable -- same picture) and then actually ablated, with the control v1 never ran:

| Ablation (50 clips) | WER | Speaker sim |
|---|---|---|
| Mimi round-trip (floor) | 0.018 | 0.923 |
| **SAE-map expendable 298, mean-ablated** | **0.024** | **0.723** |
| Raw-probe map, bottom-298 (no SAE) | 0.036 | 0.492 |
| Random 298 dims (3 seeds) | 0.085–0.152 | 0.261–0.422 |
| Keep only "critical" dims | 0.986 | −0.020 |

Ablating the map's 298 "expendable" dimensions costs almost nothing. Random same-size sets are catastrophic. And -- this is the part that redeems the SAE -- an importance map built the same way from *raw-latent* probe weights selects a much worse set (speaker similarity 0.492 vs 0.723). Composing probe weights through the SAE's encoder is the one step in this whole pipeline where the SAE demonstrably earns its place. After the probing and steering results above, I did not expect that.

Two corrections in the same table, though. "Only 7 dimensions are critical" is refuted: ablate everything except the critical set and you get WER 0.99 -- noise. The critical dims are necessary, nowhere near sufficient. And keep reading the map as what it is: "expendable for low-frequency spectral content" measured on clean read speech, not a universal intelligibility guarantee. Pitch, formant structure, and voice identity aren't centroid-driven; the speaker-similarity drop from 0.92 to 0.72 under ablation is real, audible coloration.

The non-uniform quantization plan this implied (SAE-guided ~3-bit Mimi at 29 MB vs 38 MB uniform 4-bit) is unchanged as arithmetic, but see the [May follow-up](/blog/2026-05-16-mi-guided-mixed-precision) for what happened when the thesis met actual quantization experiments on the transformer: right sign, wrong magnitude.

## Taking It to the 7B Transformer

Mimi is only 1.1% of PersonaPlex's 7B parameters. The real VRAM hog is the temporal transformer -- 32 layers, 4096-dim, 3.29B parameters. That's where the savings need to come from.

I loaded the full 4-bit PersonaPlex model (6.9 GB VRAM), hooked all 32 transformer layers, ran 100 audio files through, and saved per-layer residual stream activations to disk. Then trained a linear probe on each layer's activations predicting spectral centroid. The early layers stand out as most semantic (0.20–0.21 accuracy); layers 2–31 sit in a roughly flat 0.28–0.32 band with layer 14 at the peak. There's no layer that's purely acoustic or purely semantic -- every layer encodes a mix, because speech-to-speech models need to simultaneously understand *what's being said* and *how it sounds* at every step of generation.

The Mimi codec analysis worked because Mimi's *job* is acoustic encoding. The temporal transformer's job is more complex: it processes audio tokens while jointly modeling language, turn-taking, voice identity, and acoustic generation. Asking "which layers are acoustic?" is like asking "which layers of GPT are about grammar?" -- the answer is all of them, to varying degrees.

Per-layer was the wrong granularity. Time to go finer.

## Going Surgical: Per-Head Analysis

The temporal transformer has 32 layers × 32 attention heads = 1024 individual heads. The hypothesis: individual heads are more likely to specialize than full layers. I probed all 1024 heads, then tested structured pruning -- zeroing the most-acoustic heads at inference, ordered by probe accuracy:

| Heads pruned | % of total | Text output | Voice | Quality |
|---|---|---|---|---|
| 0 (baseline) | 0% | "Hey let me know if you have" | Female | Clean |
| 50 | 4.9% | "Hey let me" | **Shifted** | Shorter |
| **200** | **19.5%** | **"Heeey, let me kno-"** | **Female** | **Slight stretch** |
| 300 | 29.3% | "Hello, this is your show" | Female | **Cadence breaks** |

The voice shift at 50 heads is an unexpected MI finding: heads classified as "most acoustic" via centroid probing were actually encoding voice identity -- pitch, formant structure, vocal tract characteristics. Probes find correlates, not functions.

**v2: the ranking is real; the audio-level claim is weaker than I wrote it.** v1's takeaway table included a row that read "Random pruning (hypothetical): would fail much earlier." That was an assumption, not a result. The rerun tested it in the closest controlled setting available (the May post's quantize-dequantize harness: 512 heads at 3-bit under different rankings, same budget):

| Ranking (512 heads @ 3-bit) | Final-hidden cos vs bf16 |
|---|---|
| Sensitivity ranking, least-sensitive first | **0.728** |
| Weight-norm, smallest first | 0.605 |
| Random (3 seeds) | 0.538–0.621 |
| Wrong direction (most-sensitive first) | 0.530 |

The informed ranking clearly beats random and a cheap weight-norm heuristic on representation fidelity -- the v1 intuition survives its controls, and the ranking is uncorrelated with weight norms (r = −0.10), so it's not magnitude in disguise. But the same rerun exposed why none of my audio-level quality claims here were ever properly grounded: the only resynthesis path available for scoring (teacher-forced re-prediction through the LM) has a **ground-truth WER floor of 0.48 with no intervention at all**. Every recipe's WER is indistinguishable from that floor. The v1 quality column ("Clean", "Slight stretch") was me listening to a handful of generations -- directionally honest, but a metric it is not. Measuring whether head-level interventions audibly matter needs real autoregressive generation as the eval vehicle, which is the top of the follow-up list.

## The Takeaway

v1 ended with "the methodology works." v2's honest version: **the phenomena were real, but I attributed them to the wrong component, in both directions.**

| Claim (v1) | Verdict (v2 rerun) |
|---|---|
| Probes decode centroid at 48% from SAE features | Replicated -- but raw latents match it; the SAE isn't the reason |
| Radio-effect steering stays intelligible | Replicated -- raw probe direction is identical; latent EQ beats DSP EQ on speaker identity |
| The SAE was sparse and monosemantic | Refuted -- L0 ≈ 2,750/4,096; TopK replacement fixes it |
| 56% of latent dims expendable | **Confirmed by direct ablation** -- and the SAE-composed map beats raw-probe and random maps |
| Only 7 dims critical | Refuted -- keeping only those destroys speech |
| Informed head ranking beats random | Confirmed at the representation level; audio-level effect unmeasurable in the teacher-forced pipeline |

If you're doing interpretability work on audio models, the transferable lessons: report L0, not "alive neurons"; probe the raw activations first and make the SAE beat that number before crediting it; calibrate interventions to matched effect size and score them with task metrics (WER, speaker similarity), not listening; and when you build an importance map, ablate it against a random map of the same size before you publish the percentage. The one result here that cleared every one of those bars -- the importance map -- is also the one I almost didn't test.

The code, including the full v2 rerun pipeline (scripts 30–36: TopK SAE, probe matrix, steering shootout, ablation tests, head-ranking controls), is at [eq-personaplex](https://github.com/brianmatzelle/eq-personaplex). The v1 experiments ran on an RTX 4070 12GB; the rerun ran on a DGX Spark (GB10, 128 GB unified memory) with 6.3× the training data and full-bf16 models.

## Links

- **Anthropic's monosemantic features**: [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features)
- **Scaling monosemanticity**: [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
- **TopK SAEs**: [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093)
- **Steering baselines beat SAEs in LLMs**: [AxBench](https://arxiv.org/abs/2501.17148)
- **Audio SAE paper**: [Learning Interpretable Features in Audio Latent Spaces via Sparse Autoencoders](https://arxiv.org/abs/2510.23802)
- **PersonaPlex**: [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1)
- **Moshi / Mimi**: [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi)
- **Previous post**: [Squeezing a 14GB Speech Model onto a 12GB GPU](/blog/2026-02-15-personaplex-4bit-quantization)
- **Follow-up**: [MI for Quantization: Right Sign, Wrong Magnitude](/blog/2026-05-16-mi-guided-mixed-precision)
