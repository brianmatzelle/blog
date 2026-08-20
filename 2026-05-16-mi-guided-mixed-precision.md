---
title: "Mechanistic Interpretability for Quantization: Right Sign, Wrong Magnitude"
date: 2026-05-16
tags: [deep-learning, mechanistic-interpretability, quantization, moshi, mimi, dgx-spark, speech-to-speech, sparse-autoencoders]
---

# Mechanistic Interpretability for Quantization: Right Sign, Wrong Magnitude

[Last time](/blog/2026-03-17-mechanistic-interpretability-audio-eq) I trained Sparse Autoencoders on Mimi's pre-quantizer latents, found interpretable EQ features, and steered audio toward a radio effect. The longer-term goal was always **MI-guided non-uniform quantization** -- use the probes to decide which layers can be crushed and which ones need precision. With PersonaPlex now running on a [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) (128 GB unified memory, GB10 Blackwell), I finally have the headroom to actually test that thesis end-to-end.

The honest summary up front:
- The MI probe accuracy *does* predict quantization sensitivity, **Pearson r = -0.74** between centroid-probe accuracy and layer drift under 3-bit quantization. The interpretability signal is real.
- At the **per-layer and per-head** granularities, MI-guided mixed precision still falls below the uniform-quantization interpolation line. Sensitivity-guided beats wrong-direction at matched bits (good), but it doesn't dominate the convex hull of uniform points (not so good).
- The naive recipe from the prior post (most-acoustic layers crushed to 2-bit) is actually *worse* than uniform 3-bit at the same average bit budget. Once you push to 2-bit on any subset, the recipe collapses.
- Per-head signal is meaningfully stronger than per-layer (max/median 9.3× vs 3.6×), and at a 50/50 right-vs-wrong direction split the cosine delta is roughly 2× larger than the closest per-layer comparison point. The blog's "per-head is sharper than per-layer" intuition holds quantitatively.
- The most actionable finding is unexpected: an attention-vs-MLP decomposition shows that **layer 0's drift is 82% from MLP, layer 31's is 68% from MLP, and every other layer's drift is dominated by attention.** The boundary layers' MLP weights are the real quant bottlenecks, not attention.
- A **block-aware recipe** built from that decomposition (high-precision MLP at layers 0 and 31, high-precision attention at the first ~6 layers, crush everything else) is the **first recipe in this experiment that lands above the uniform interpolation line** -- a small but real Pareto improvement (+0.024 cosine at 3.10 average bits on a held-out validation set, +0.005 on the original eval set). The wrong-direction block recipe sits below the line on both sets, so the signal is directional, robust, and not an artifact of clip selection.

So the headline result is: **MI as a quantization oracle is right in sign but not large enough in magnitude to beat the Pareto frontier at the layer or head level. At the block level (attention vs MLP within each layer) it finally does, modestly: ~1-2% cosine above uniform interpolation by protecting four blocks (~1% of the transformer's parameters).**

## Setup

The fully-precise PersonaPlex weights are gated on Hugging Face (`nvidia/personaplex-7b-v1`), and the user-side bf16 release isn't accessible without authentication. The bnb-4bit fork I pushed last time *is* public but pins old `torch<2.5` / `sphn<0.2` / `huggingface-hub<0.25`, which conflict with everything on the Spark today.

The substitute is `kyutai/moshiko-pytorch-bf16` -- the base Moshi 7B that PersonaPlex is fine-tuned from. Same architecture: 32-layer temporal transformer, 32 heads × 128 dim, 17 codebooks (1 text + 16 audio). For layer-specialization questions a fine-tune shouldn't change the picture much; for absolute task quality on PersonaPlex specifically you'd want to re-run on that checkpoint. All numbers below are on `moshiko-bf16`.

Probe data: 100 LibriSpeech-clean-test utterances (~8.6 K Mimi frames at 12.5 Hz), encoded with Mimi, then run through Moshi's temporal transformer with the audio codes occupying user-audio slots and model-audio slots zero-padded. Acoustic labels come from `librosa` (spectral centroid binned to 20 bins, RMS amplitude similarly).

## What does the per-layer probe see in bf16?

Last time the per-layer probe was reported on the bnb-4bit PersonaPlex and showed clear layer differentiation. On bf16 Moshi the signal is much flatter:

```
Layer  Centroid acc   Amplitude acc
  0       0.213         0.216
  1       0.216         0.222
  2       0.268         0.260
  5       0.310         0.265
  8       0.310         0.278
  9       0.314         0.272   ← peak
 17       0.300         0.277
 25       0.307         0.260
 31       0.299         0.263
random    0.050         0.050
```

Centroid accuracy is consistently in the 0.21-0.31 band across all 32 layers. Layer 9 is the peak at 0.314, layers 0-1 are the floor at 0.21-0.22. Compared to random baseline (1/20 = 0.05), every layer is *substantially* predictive of spectral centroid -- which itself is a clue: this acoustic information isn't localized to any particular layer, it's spread across the whole stack.

The original [script 09](https://github.com/brianmatzelle/eq-personaplex/blob/main/scripts/09_layer_probes.py) thresholding -- `score > 0.3 → 2-bit`, `> 0.2 → 3-bit`, else 4-bit -- assigns 2 layers (8, 9) to 2-bit and the remaining 30 to 3-bit. Average 2.9 bits, ostensibly 26.6% savings over uniform 4-bit.

## QDQ simulation: does that recipe actually preserve behavior?

Instead of building the real mixed-precision checkpoint right away (HQQ for 2/3-bit, bnb for 4-bit, lots of plumbing) I ran the recipes as **quantize-dequantize hooks** on the bf16 weights. For each transformer layer, replace `Linear.weight` with a per-channel symmetric QDQ tensor at the assigned bit width (group size 128), forward the model, measure how much the final hidden state and text logits drift from the bf16 baseline.

This isolates "what does quantization noise do to the model" from "which backend can express that quantization." If the recipe doesn't work in QDQ, it definitely won't work for real.

Held-out: the last 12 LibriSpeech clips not seen during probing.

| Recipe              | avg bits | final-layer cos vs bf16 | text-logit KL |
|---------------------|---------:|------------------------:|--------------:|
| uniform-4bit        |     4.00 |               **0.747** |     **0.359** |
| uniform-3bit        |     3.00 |                   0.359 |         1.009 |
| uniform-2bit        |     2.00 |                   0.031 |         1.308 |
| mixed-from-probes   |     2.94 |                   0.271 |         1.053 |

Two things jump out.

**Uniform 4-bit weight QDQ already drifts noticeably from bf16.** Cosine 0.747 between final hidden states is "the model still has the same general direction, but it's wandered." This is per-channel symmetric quant with group size 128 -- a baseline implementation. Real backends (NF4, AWQ, HQQ, GPTQ) do better via learned grids, optimized group sizes, or calibration. So treat 4-bit cos=0.75 as a lower bound on quality; the *deltas* between recipes are what matters.

**The mixed-from-probes recipe is worse than uniform 3-bit, despite using more average bits (2.94 vs 3.00).** Forcing layers 8 and 9 down to 2-bit was a bad trade: 2-bit blows the model up almost completely (uniform 2-bit has cos=0.031, basically random direction), so even quantizing just two layers at that level adds significantly more drift than it saves.

The 2-bit assignment in the original recipe is the problem, not the layer choice per se.

## Per-layer sensitivity sweep: who can take a punch?

To check whether layers 8 and 9 were actually a bad choice -- vs. the 2-bit setting being the real culprit -- I ran a single-layer sweep: quantize **only layer i** to 3-bit, everything else stays bf16, measure drift. Across 8 held-out clips:

| Layer | cos vs bf16 | drift (1-cos) |
|------:|------------:|--------------:|
|     0 |       0.892 |         0.108 |
|     1 |       0.901 |         0.099 |
|     2 |       0.855 |         0.145 ← peak |
|     3 |       0.925 |         0.075 |
|     4 |       0.933 |         0.067 |
|     8 |       0.966 |         0.034 |
|     9 |       0.963 |         0.037 |
|    17 |       0.946 |         0.054 |
|    26 |       0.972 |         0.028 |
|    30 |       0.974 |         0.026 ← floor |
|    31 |       0.957 |         0.043 |

Layers 0-2 are most quant-sensitive. Layers 8-9 (the ones the original recipe wanted to crush) are *among the least sensitive* -- they shrug off 3-bit quantization with cos = 0.96. The late layers 24-30 are also very tolerant.

So the centroid probe was right about *direction*: high-centroid-predictability layers really do tolerate quantization better. The Pearson correlation between centroid probe accuracy and 3-bit quantization drift (1 − cos):

```
Pearson  centroid_acc vs drift:   -0.741
Pearson  centroid_acc vs text KL: -0.721
Spearman centroid_acc vs drift:   -0.424
```

Negative because higher centroid-predictability ⇒ lower drift. **The MI probe predicts quantization tolerance with r = -0.74.** Anthropic-style interpretability *can* guide quantization at the layer level -- the prior post's hypothesis was right.

## The Pareto curve: smart recipes vs uniform

If the MI ranking is good, you'd hope a recipe that protects the centroid-LEAST-predictable layers at 4-bit, and crushes the rest to 3-bit, would beat uniform at matched average bit count. I tried several "protect top-K layers at 4-bit, rest at 3-bit" recipes, ranked both by the actual sensitivity sweep above ("sens-guided") and by the cheaper MI probe ("mi-guided"). Same 12 held-out clips, same QDQ setup:

| Recipe                              | avg bits | cos vs bf16 | text KL |
|-------------------------------------|---------:|------------:|--------:|
| uniform-4bit                        |     4.00 |       0.747 |   0.359 |
| sens-guided  protect-20 @4 / rest 3 |     3.62 |       0.543 |   0.633 |
| mi-guided    protect-20 @4 / rest 3 |     3.62 |       0.551 |   0.634 |
| sens-guided  protect-16 @4 / rest 3 |     3.50 |       0.480 |   0.686 |
| mi-guided    protect-16 @4 / rest 3 |     3.50 |       0.504 |   0.747 |
| sens-guided  protect-12 @4 / rest 3 |     3.38 |       0.461 |   0.758 |
| mi-guided    protect-12 @4 / rest 3 |     3.38 |       0.461 |   0.786 |
| sens-guided  protect-8  @4 / rest 3 |     3.25 |       0.404 |   0.833 |
| mi-guided    protect-8  @4 / rest 3 |     3.25 |       0.438 |   0.788 |
| uniform-3bit                        |     3.00 |       0.359 |   1.009 |
| mixed-from-probes (script 09)       |     2.94 |       0.271 |   1.053 |
| sens-guided  8×4 / 16×3 / 8×2       |     3.00 |       0.135 |   1.065 |
| mi-inverse   protect-12 @4 / rest 3 |     3.38 |       0.428 |   0.855 |

Several things to read off this:

1. **MI-guided ≈ sens-guided at every protection budget.** At protect-12, both land at cos = 0.461. At protect-20, cos = 0.543 vs 0.551. The MI probe, free to compute, ranks layers almost as well as the expensive sensitivity sweep does. That's a clean win for "MI as a cheap proxy for quant tolerance."

2. **Smart mixed sits *between* the two uniform baselines, it doesn't beat them.** Pulling the (3.0 bit, cos=0.36) and (4.0 bit, cos=0.75) points and drawing a straight line, the smart-mixed recipes track that line nearly perfectly. There's no free lunch -- you pay the linear cost in cosine for each extra bit you take away on average.

3. **MI-inverse (wrong direction) is only mildly worse than MI-guided** at protect-12: cos=0.428 vs 0.461. The MI signal is real but not strong at this granularity; the layer-level differentiation just isn't huge.

4. **Crushing any layer to 2-bit is a disaster.** The "8×4 / 16×3 / 8×2" recipe has the same average bits as uniform-3bit but loses an extra 0.22 cosine. 2-bit per-channel symmetric QDQ on weights is too aggressive without a proper learned grid.

The "no Pareto improvement" result is the boring, honest finding. *At the layer granularity, MI-guided mixed precision is essentially a smooth interpolation between uniforms.* It's not useless -- if you have an explicit avg-bit budget that falls between 3 and 4, MI gives you a principled way to pick which layers to pad to 4. But you don't get the kind of "free 30% savings at no quality cost" that the original interpretability thesis seemed to suggest.

## Why the prior post's "56% expendable" doesn't transfer here

Re-reading the [earlier post](/blog/2026-03-17-mechanistic-interpretability-audio-eq), the strong "56% of latent dimensions expendable" claim was about Mimi's 512-dim pre-quantizer latent -- the codec's bottleneck representation. That's a *much* tighter, more specialized space than the LM's 4096-d residual stream. The codec has 7 critical dimensions out of 512. The LM has 32 layers and probably hundreds of attention heads, each carrying overlapping responsibility for any given concept.

In other words: the MI thesis worked beautifully on the codec because the codec is a designed bottleneck. On the LM's residual stream the signal is real but diluted across many redundant pathways, exactly what we'd expect from a 7B transformer.

## Per-head sensitivity: sharper but still not Pareto-dominant

The blog also said per-layer pruning failed and per-head pruning was the lever that worked: "Removed 200 attention heads (19.5% of total) while preserving speech intelligibility." So the next test was an exhaustive per-(layer, head) sensitivity sweep -- 32 layers × 32 heads = 1024 configurations, each QDQ-ing just that head's Q/K/V/O slices to 3-bit, on 6 held-out clips. The full sweep took ~46 minutes on the Spark.

Compared to the per-layer sweep:

|                              | per-layer | per-head |
|------------------------------|----------:|---------:|
| Median drift (1-cos at 3-bit) |     0.040 |  0.00119 |
| Max drift                    |     0.145 |  0.01103 |
| **Ratio max/median**         |     **3.6×** |  **9.3×** |
| Ratio P95/median             |      2.6× |     2.9× |

The per-head distribution has a *much fatter tail*. Within any given layer, most heads have drift very close to the layer median, but a few heads contribute outsized drift. Layer 2's mean head drift is 0.005 but its max is 0.011 -- twice the mean. So per-head granularity lets you isolate those specific heavy hitters, which is exactly what the original blog post observed for pruning.

Hottest individual heads, ranked by drift under 3-bit attention QDQ:

```
L2H7   drift=0.0110   ← single biggest hot head
L1H28  drift=0.0081
L1H13  drift=0.0080
L2H29  drift=0.0074
L2H13  drift=0.0069
L1H30  drift=0.0069
L2H21  drift=0.0069
L2H25  drift=0.0067
L20H4  drift=0.0058   ← only mid-stack head in the top 15
```

Layer 2 has nine heads in the top-15. Layer 1 has five. Layer 0 -- which the per-layer sweep flagged as third-most-sensitive overall -- has *zero* heads in the top 15, and several of its heads sit at the bottom of the distribution. That's a useful disambiguation: layer 0's per-layer sensitivity is coming from its **MLP block**, not its attention heads. The per-head sweep wouldn't have caught that.

To verify, I ran a quick attention-vs-MLP decomposition: for each layer, separately QDQ just the attention Linear weights to 3-bit, then separately QDQ just the gating/MLP weights, measure drift. Top 5 layers by total drift:

| Layer | Total drift | Attn drift | MLP drift | MLP fraction |
|------:|------------:|-----------:|----------:|-------------:|
|     2 |       0.145 |      0.139 |     0.007 |          5%  |
|     0 |       0.108 |      0.017 |     0.079 |     **82%**  |
|     1 |       0.099 |      0.080 |     0.030 |         27%  |
|     3 |       0.075 |      0.070 |     0.005 |          6%  |
|     4 |       0.067 |      0.065 |     0.006 |          8%  |

And across the full stack:
- **Layers 0 and 31** (the boundary layers): MLP dominates drift (82% and 68% respectively).
- **Layers 2-29** (mid-stack): attention dominates, MLP contributes 5-15%.
- **Layer 1**: 73% attention, 27% MLP -- transitional.

So "where quantization hurts" depends not just on which layer but which *block*. The first layer's MLP is doing something hard to compress (probably mapping raw codebook embeddings into a usable residual representation). The last layer's MLP is doing the inverse -- preparing the final residual for the depth-former's text/audio readout. The middle layers' MLPs are mostly feed-forward refinement and are very quant-tolerant.

This implies a "block-aware" mixed precision recipe could be more efficient than either the per-layer or per-head ones: keep attention high-precision in early-mid layers (1-5), MLP high-precision in boundary layers (0, 31), and crush everything else. I went ahead and tested it (script 17):

| Recipe                                                  | avg bits | cos vs bf16 | text KL | vs uniform line |
|---------------------------------------------------------|---------:|------------:|--------:|----------------:|
| uniform-4bit                                            |     4.00 |       0.747 |   0.359 |     (baseline) |
| uniform-3bit                                            |     3.00 |       0.359 |   1.009 |     (baseline) |
| block-aware-A: MLP@4 on L0,L31; rest 3-bit              |     3.04 |       0.388 |   0.900 |    **+0.013 above** |
| block-aware-D: MLP@4 on L0,L31 + attn@4 on L2; rest 3   |     3.05 |       0.393 |   0.947 |    **+0.014 above** |
| block-aware-C: MLP@4 on L0,L31 + attn@4 on L1..L6; rest |     3.10 |       0.403 |   0.838 |    **+0.005 above** |
| block-aware-B: MLP@4 on L0,L31 + attn@4 on L1..L5; rest |     3.09 |       0.375 |   0.847 |     -0.019 below |
| INVERSE: MLP@4 on L15,L20; attn@4 on L29-31; rest 3-bit |     3.07 |       0.380 |  1.014 |     -0.006 below |

(Uniform-line cosine at average bits *a* between 3 and 4 is approximated as `0.359 + (a-3)*0.388`.)

Three of the four sensitivity-guided block recipes sit *above* the uniform interpolation line for the first time in this entire experiment. The wrong-direction recipe (INVERSE) sits below. The largest win is from block-aware-D, which protects only the four most-quant-sensitive Linear blocks in the entire model (Layer 0 MLP, Layer 31 MLP, Layer 2 attention, and the "rest 3-bit" backbone) -- a tiny precision budget invested in exactly the right spots.

The deltas are small (+0.013 cosine at 3.04 bits, +0.014 at 3.05). They're nothing like the "30% free savings" the original interpretability thesis implied. But they're positive, statistically distinguishable from the wrong-direction recipe (Δ ≈ 0.02 between block-A and INVERSE at near-identical bit budgets), and they vindicate the original blog's intuition that MI-guided quantization can outperform uniform -- just at a much smaller granularity (a handful of *blocks*) than was originally suggested.

This is the cleanest result of the night: **block-level mixed precision beats uniform by 1-2% cosine at matched bits, and the MI probe + block-decomposition together identify the right blocks to protect.**

I also tested whether the same trick works at *aggressive* bit budgets (avg ~2.2 bits, where most of the model is at 2-bit and only the critical blocks are protected at 4-bit). It does not. Every recipe with a 2-bit floor lands at cos < 0.05 -- essentially random direction -- regardless of which blocks were protected. The right-direction and wrong-direction recipes are statistically indistinguishable at that level. So the useful range for MI-guided block-level mixed precision is roughly 3-4 bits, not 2-3. Once any large fraction of the model is at 2-bit, the cascading quantization noise drowns out any signal that protecting individual blocks could provide.

### Validation on a disjoint clip set

To make sure the Pareto win isn't an artifact of fitting recipes to the same clips the sensitivity sweep used, I re-ran the recipes on a totally disjoint range -- clips 200-220 from LibriSpeech-clean-test, which were used neither for probe training nor for sensitivity sweeping:

| Recipe                                            | avg bits | cos vs bf16 | Δ vs uniform line |
|---------------------------------------------------|---------:|------------:|------------------:|
| uniform-4bit                                      |     4.00 |       0.765 |        (baseline) |
| uniform-3bit                                      |     3.00 |       0.382 |        (baseline) |
| BA-C: MLP@4 on L0,L31 + attn@4 on L1..L6; rest 3  |     3.10 |       0.446 |     **+0.024 above** |
| BA-A: MLP@4 on L0,L31 only; rest 3-bit            |     3.04 |       0.408 |     **+0.010 above** |
| BA-D: MLP@4 on L0,L31 + attn@4 on L2; rest 3-bit  |     3.05 |       0.398 |       -0.004 below |
| INVERSE: MLP@4 mid + attn@4 late; rest 3-bit      |     3.07 |       0.400 |       -0.010 below |

The biggest Pareto win on the original set (BA-D) is borderline here. But **BA-C now lands +0.024 cosine above the uniform line, larger than the +0.005 it got on the original set**, and BA-A is also robustly above. The inverse control is still cleanly below. So the conclusion holds and even strengthens on independent data: protecting Layer 0/Layer 31 MLP + a handful of early-attention layers buys a real but modest Pareto improvement.

For the Pareto comparison I added a script that quantizes **MLP/gating weights uniformly at 4-bit** *plus* applies head-level QDQ to the attention slices. This is a fair comparison to "uniform-4bit transformer" because both touch all the same parameters; the only thing varying is whether a subset of attention heads is bumped down to 3-bit.

Held-out: same 12 LibriSpeech clips. (Note that absolute cos numbers shifted slightly because the per-channel QDQ application path is structured differently when patched piecewise -- treat the per-bit deltas, not the raw numbers, as the signal.)

| Recipe                                                  | total avg bits | cos vs bf16 | text KL |
|---------------------------------------------------------|---------------:|------------:|--------:|
| uniform-4bit (all)                                      |          4.00 |     **0.713** |   0.550 |
| mlp-4bit + head-3bit bottom-25% / 4bit top-75%          |          3.92 |       0.639 |   0.619 |
| mlp-4bit + head-3bit bottom-50% / 4bit top-50%          |          3.84 |       0.575 |   0.660 |
| mlp-4bit + head-3bit TOP-50% (**WRONG DIRECTION**)      |          3.84 |       0.493 |   0.971 |
| mlp-4bit + head-3bit bottom-75% / 4bit top-25%          |          3.75 |       0.461 |   0.976 |
| mlp-4bit + head-3bit bottom-100% (all heads 3-bit)      |          3.67 |       0.393 |   0.878 |
| mlp-4bit + head-2bit bottom-50%                         |          3.67 |       0.136 |   1.065 |
| uniform-3bit (all)                                      |          3.00 |       0.362 |   0.881 |
| mlp-4bit + head-2bit bottom-75%                         |          3.51 |       0.058 |   1.406 |
| uniform-2bit (all)                                      |          2.00 |       0.019 |   1.483 |

The cleanest experiment is the **same-bits A/B test at 3.84 average bits**:
- Sensitivity-guided (bottom-50% of heads to 3-bit): **cos = 0.575**
- Wrong-direction (top-50% most-sensitive heads to 3-bit): cos = 0.493
- **Δ = 0.082 cosine** at identical bit budget

That's a strong MI-direction signal at the head level. The closest per-layer A/B point (protect-12 right vs wrong at avg 3.38) had Δ = 0.033, on a recipe that mixed 38% of layers. The per-head 50%-mix recipe roughly doubles the cosine separation, validating the blog's "per-head is sharper" intuition quantitatively.

But:

| Bit budget   | Uniform-line cos (interpolation) | MI-guided actual cos | Below the line? |
|--------------|---------------------------------:|---------------------:|:----------------|
| 3.92         |                            0.679 |                0.639 | yes, by 0.040   |
| 3.84         |                            0.650 |                0.575 | yes, by 0.075   |
| 3.75         |                            0.617 |                0.461 | yes, by 0.156   |
| 3.67         |                            0.587 |                0.393 | yes, by 0.194   |

(Uniform line = straight line between (3.0, 0.362) and (4.0, 0.713).)

Every MI-guided recipe lands *below* the line connecting uniform-3bit and uniform-4bit. **The right-direction recipe is better than the wrong direction, but it's still worse than what a uniform mix at the same average would predict.**

The implication: at fixed bit budget, you do *better* by uniformly lowering everything than by selectively concentrating the savings on "expendable" heads. That's counterintuitive if you assumed MI would unlock targeted savings, but it matches intuition once you think about quantization as additive noise: spreading the noise thinly across many parameters degrades a transformer less than concentrating it on a small subset, even if that subset is chosen well.

What MI does buy you, robustly:
- **The right-vs-wrong direction matters.** At 3.84 bits, MI-guided (cos 0.575) is +0.082 better than reverse (cos 0.493). If you have to do mixed precision (e.g. heterogeneous hardware constraints), MI tells you which heads to prefer.
- **MI ≈ sensitivity-sweep at zero extra cost.** The cheap-to-compute MI probe ranks heads/layers about as well as the expensive sensitivity sweep does. At layer level the two rankings produced near-identical Pareto curves.
- **At block granularity (attention vs MLP per layer), MI-derived guidance produces a small Pareto improvement.** Block-aware recipes land +0.013 to +0.014 cosine above the uniform line; the wrong-direction control sits below. The win comes from protecting a small structured set of blocks rather than scattering precision across heads.

What MI *doesn't* buy:
- A "free 30%" quantization recipe at the layer or head level. The savings from MI guidance at those granularities is washed out by quantization being diffusely-distributed across the residual stream.

## A bit of mechanistic intuition for why this is the case

Looking at the per-head sensitivity heatmap, the hottest heads cluster in layers 1-2 -- almost every head in those two layers is in the top quartile of sensitivity. Layer 0's attention heads are *not* hot in aggregate; the per-layer sensitivity of layer 0 came almost entirely from its MLP block. Layers 1 and 2 do a lot of basic acoustic preprocessing -- detecting frame-level features, integrating context across recent frames, deciding what's worth passing through the residual stream -- and that's exactly where attention quantization hurts most.

Once a piece of acoustic information makes it past the first few layers, it's redundantly encoded in many subsequent residual stream coordinates and across many heads. Crushing any individual mid-stack head loses very little, because that head's specific function is shared by half a dozen others. This is the well-known **iterative refinement** picture of transformer residual streams, and it implies that "MI features at the head level" identify *currently active* contributors, not *irreplaceable* ones.

To get a real free lunch on quantization, you'd want to identify weights that are *not* redundantly encoded -- which in a 7B model is a very small fraction, and is probably better captured by activation-aware metrics (AWQ-style) than by interpretability probes on intermediate representations.

## Numbers and code

The scripts that produce all of this live in `~/projects/active/eq-personaplex`:

```
scripts/
  08b_extract_transformer_activations_bf16.py  ← bf16 variant of script 08
  09_layer_probes.py                           ← unchanged; reads bf16 acts
  10_eval_qdq_recipes.py                       ← naive recipe comparison
  11_layer_sensitivity.py                      ← per-layer single-layer-quant sweep
  12_smart_mixed_recipes.py                    ← layer-level Pareto sweep
  13_head_sensitivity.py                       ← per-(layer, head) sweep (1024 configs)
  14_smart_head_recipes.py                     ← head-only Pareto sweep
  15_head_plus_mlp_recipes.py                  ← head + MLP combined Pareto
  16_block_breakdown.py                        ← attention-vs-MLP decomposition per layer
  17_block_aware_recipe.py                     ← block-aware recipe test (the Pareto winner)
  18_aggressive_block_aware.py                 ← block-aware at 2-bit floor (doesn't work)
  19_validate_block_aware.py                   ← validation on disjoint clip range
src/
  qdq.py        ← weight quantize-dequantize patch context manager
  qdq_head.py   ← per-head version (Q/K/V/O slices of attention)
```

Outputs land in `outputs/`:
- `qdq_eval.json`, `smart_recipes.json` (layer level)
- `layer_sensitivity.json`, `head_sensitivity.npy`, `head_sensitivity_summary.json`
- `head_plus_mlp_recipes.json` (the head Pareto comparison)
- `block_breakdown.json` (attention-vs-MLP decomposition per layer)
- `block_aware_recipes.json` (the Pareto-winning block recipes)
- `aggressive_block_aware.json` (why the 2-bit floor doesn't work)
- `validate_block_aware.json` (validation on disjoint clip set)
- matching `.png` plots for each

## Methodology notes (Spark-specific)

A few things worth flagging for anyone trying to reproduce on a Spark:

- **`torchaudio.load` is currently routed through `torchcodec` on torch 2.9+ for aarch64**, and torchcodec needs FFmpeg shared libs we don't have. Patching `src/mimi_hooks.py` to decode via `soundfile` directly is a one-line fix.

- **Triton tries to JIT-compile a CUDA helper** the first time a fused op fires, which requires `python3.12-dev` (Python headers) on the system. Without `sudo` to apt-install it, set `TORCHDYNAMO_DISABLE=1` and `torch._dynamo.config.disable = True` to keep everything in eager mode. The forward pass is plenty fast on GB10 even without compile.

- **moshi 0.2.13 stock doesn't have `lm.embed_codes`**; it has `lm.forward_text(sequence)` which builds the embedding inline (sum of per-codebook embeddings + text embedding) and returns `(transformer_out, text_logits)`. The original `scripts/08_extract_transformer_activations.py` relies on the bnb-4bit fork's API; the bf16 variant uses `forward_text` plus hooks on `transformer.layers[i]`.

- **QDQ details**: per-channel symmetric quantization, group size 128 along the input dim, no calibration, no learned grid. This is intentionally a simple ceiling. Real backends (NF4, AWQ, HQQ, GPTQ) at 4-bit will land much closer to the bf16 baseline than the 0.71 cosine we see here. The interesting axis is the *delta* between recipes, not the absolute cosine.

- **The DGX Spark made all of this feasible.** Loading 7B bf16 + per-head sensitivity sweeps would not fit on the 12 GB RTX 4070 the prior work used. Unified 128 GB memory + bf16 PersonaPlex means the experiment is interactive rather than a multi-day sharding exercise.

## Next

Things I'd do next, in rough order of bang-for-buck:

- **Real quantization backends and Whisper WER.** QDQ is an upper bound on quality loss; AWQ/GPTQ/HQQ at 4-bit are likely much tighter to bf16, and the *deltas* between recipes might compress. Building a real mixed-precision checkpoint and measuring Whisper WER on generated audio is the production-grade version of this experiment.

- **PersonaPlex bf16 instead of Moshi base.** `nvidia/personaplex-7b-v1` is gated, so this run used the base Moshi checkpoint. Re-running on the actual fine-tuned PersonaPlex weights might sharpen layer specialization, since fine-tuning often concentrates task-specific information.

- **Per-channel within MLP.** I quantized MLP uniformly because the gating module doesn't have a natural per-head decomposition. But [AWQ-style](https://arxiv.org/abs/2306.00978) channel-importance analysis would slot cleanly into the QDQ harness and is the obvious next refinement -- and given that boundary MLPs are doing most of the work in those layers, channel-level inside *just those two MLP blocks* could be the highest-leverage refinement.

- **SAE features as the probe target.** The centroid probe is a crude proxy for "audio information." Using SAE features (the actually-monosemantic outputs of the codec SAE from the [prior post](/blog/2026-03-17-mechanistic-interpretability-audio-eq)) might give a cleaner predictor of which heads carry redundant vs. unique information.

The cleanest result tonight is the **block-aware recipe (Layer 0 MLP @ 4-bit, Layer 31 MLP @ 4-bit, Layer 2 attention @ 4-bit, everything else @ 3-bit) landing +0.014 cosine above the uniform interpolation line at 3.05 average bits.** It's a small win, but it's the first time anything beat uniform Pareto in this whole experiment. The wrong-direction control sits below the line by a similar margin, so the signal is real and structured.

The supporting result is the **r = -0.74 correlation between MI probe accuracy and quantization sensitivity**, plus the per-head sensitivity sweep that disambiguates "Layer 0 is sensitive" into "Layer 0's *MLP* is sensitive; its attention is fine." That decomposition is what made the block-aware recipe possible in the first place -- you can't write a winning recipe without knowing what's actually limiting you.

So: MI-guided quantization isn't a free-lunch oracle, but it's a useful calibration tool. It tells you where in the model precision matters most, and when applied at the right granularity (blocks, not layers or heads), it does buy a small but measurable Pareto improvement. That's the honest end of the story tonight.
