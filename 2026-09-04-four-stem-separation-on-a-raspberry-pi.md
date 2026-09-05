---
title: "Four-Stem Music Separation, Live, on a Raspberry Pi 4"
date: 2026-09-04
tags: [deep-learning, audio, source-separation, raspberry-pi, onnx, distillation, demucs, dgx-spark, termtv]
---

# Four-Stem Music Separation, Live, on a Raspberry Pi 4

I have a terminal IPTV/Spotify app called termtv that runs on a Raspberry Pi 4 plugged into a CRT. It has a DJ deck in it -- a chop-and-screw sampler, a tempo-synced echo, a bunch of knobs on an `--af` chain. The knob I wanted most was the one that takes the drums out of whatever is playing, so a screwed song can play drumless under the deck's haze. Then the bass. Then the vocals, so the sampler can chop an acapella.

**Try it first.** The exact model the Pi runs, in your browser. Press the sample clip or open a file, then pull a stem's slider down to take it out of the song, live; three at 0 isolates the fourth. In Chrome or Edge you can paste a YouTube link, open it in a new tab and capture that tab's audio. Nothing leaves your machine.

<div data-demo="stems"><a href="https://matzelle.co/blog/2026-09-04-four-stem-separation-on-a-raspberry-pi">▶ there is a live, in-browser demo of the model in this post on matzelle.co</a></div>

That is the whole model: 3.4 MB, 232 ms of lookahead, nothing else. On the Pi it is the knob: 0.86M parameters, distilled from Demucs, running in a child process at about two thirds of one Cortex-A72 core, with four mask heads on one body so any combination of drums / bass / other / vocals comes out of the same forward pass. It sits in the Spotify receiver's PCM path in front of the sampler, so pads, chops and the screw all work on the stemmed song.

Before writing this up I went looking for prior art, because I was fairly sure someone had done it. As far as I can find, nobody has -- not this small, on this class of CPU, in a live audio path. It is not a surprising result. Every ingredient is in the literature. It is more like a branch on the tree that hadn't been grown yet, and I find it a particularly interesting one, so here is how it was done and where it sits honestly against the published work.

If you would rather watch than read, there is a three-minute animated explainer [at the bottom of this post](#explainer): the mix as a sum, the spectrogram, the mask, the 54 bands, the two GRU passes, the lookahead as a delay, the four heads, the training, and the Pi's time budget, in that order.

## What existed before

Music source separation (MSS) is a mature field. Offline, the state of the art is very good: BS-RoFormer, SCNet-large, HT-Demucs all sit around 9-10 dB SDR on MUSDB18-HQ, and they are all tens of millions of parameters with seconds of context. They are what every "AI stem splitter" website runs.

Real-time is a different and much smaller literature, because a causal model with milliseconds of context loses several dB against the same architecture run offline. The published real-time models, with the numbers as they report them:

| model | params | stems | algorithmic latency | SDR (MUSDB18-HQ) | timed on |
|---|---|---|---|---|---|
| HS-TasNet (2024) | 42M | 4 | 23 ms | 4.65 | 4 CPU cores under 50% |
| HS-TasNet-Small | 16M | 4 | 23 ms | 4.48 | -- |
| RT-STT (Nov 2025) | 383K | 4 | 23 ms | 5.17 | RTX 3080Ti |
| Online SCNet | 4.36M | 4 | 92 ms | 7.09 | i7-13700H, one thread, RTF 0.43 |
| Band-SCNet (Interspeech 2025) | 2.59M | 4 | 92 ms | 7.79 | i7-13700H, one thread, RTF 0.48 |
| MMDenseNet karaoke (2024) | ~574K | 2 (accompaniment only) | 0.65-1.2 s | ~13.7 (accompaniment) | Xeon, 4 cores |

Band-SCNet is the strongest of these, and it needs about half of one modern laptop-class thread. RT-STT is the smallest, but it was only timed on a GPU and it lands 3 dB lower. Every one of them was benchmarked on an x86 desktop or an NVIDIA card.

On the commercial side, Algoriddim's Neural Mix (AudioShake's models on Apple's Neural Engine), VirtualDJ's Stems (local GPU) and Serato Stems all do real-time separation, all on hardware with a matrix unit. The one ADC 2025 talk about ONNX in DJ software was about exporting Demucs v4 for C++ hosts -- a 42M-parameter model that, per several reports, doesn't fit on a phone, let alone a Pi.

So the niche was: **a four-stem separator small enough to run on an ARM CPU with no accelerator, in real time, alongside a whole TUI, a sampler and an LLM.** Nothing I found occupies it.

## Why the niche exists

Two reasons, and they point in opposite directions.

**The Pi has no memory bandwidth to spare.** A batch-1 model under onnxruntime on a Cortex-A72 is memory-bound: every 11.6 ms STFT frame re-reads all the weights. A model that is small in FLOPs but has a big weight matrix per frame (anything with a wide fully-connected layer over 1025 bins) is slow here regardless of its parameter count. That rules out most of the real-time designs above, which are built to hit 23 ms on a desktop and don't care about bandwidth.

**The published models are fighting for a latency I don't need.** Hearing aids, live PA, and DJ decks scratching by hand all need the 23-92 ms these papers optimize for. My receiver path already carries about 1.5 s of pipe latency (mpv's read-ahead plus the demuxer cache), and the app already compensates the deck's visualizers for it. So the model can be handed a couple of hundred milliseconds of lookahead essentially for free, and lookahead is exactly what a causal model needs to get back most of what it loses on drums.

That second point is the whole trade. I want to be clear about it up front: **186 ms is not low latency by this literature's standard.** It is low enough for a player that is already buffering, and it is the reason the model does better than the 23 ms ones. If you need 23 ms, this design is not for you.

## How it was built

### The model

The student is a band-split GRU mask estimator. The spectrogram (2048-point Hann, hop 512, 44.1 kHz, 1025 bins) is split into 54 bands, narrow at the bottom where a kick and a bass note share a few bins and wide at the top, in four groups of equal width so each group is one batched linear layer. Each band gets a small per-band projection to a D-dimensional feature, then two blocks of:

- a **causal time-GRU**, weights shared across all 54 bands, bands folded into the batch. This is the part that streams: its hidden state carries across chunks.
- a **bidirectional band-GRU** across the 54 bands *within* a frame, carrying nothing between frames.

Then a per-band linear back to `stems × 2 channels × bins`, and a sigmoid. That is the whole thing. It exports to ONNX as plain `Gemm` / `GRU` / `LayerNormalization` nodes with no data-dependent control flow.

Sharing the GRU weights across bands is the BSRNN idea, and on this hardware it is the point: the weights are read once per frame and reused 54 times, so the cost becomes compute, which the Pi has about 10 GFLOPS per core of. At d=48 with one head it is 0.31M parameters. At d=64 with four heads it is 0.86M.

### Lookahead as a delay, not a backward RNN

The model is fully causal. The lookahead is a shift in the *target*: the mask emitted at frame t is applied to frame t − 16, so it has seen 16 frames (186 ms) of the future by the time it is used. Training shifts the target the same way and that is the only place the lookahead exists. No bidirectional time layer, no chunked attention -- just a delay line in the runtime.

### One body, four complement heads

This is the part I haven't seen elsewhere. Every published four-stem real-time model emits four stems. This one emits four *complements*: head `s` is trained against `mix − stem_s`, the song with that stem taken out. In the runtime each head has a level in [0, 1] and the removals add:

```
out = X · max(0, 1 − Σ_s level_s · (1 − mask_s))
```

which is exact for ideal ratio masks. So one forward pass gives you the drums out, or the drums and the vocals out, or three stems out to isolate the fourth (the acapella, or the drum break, for the sampler). The four heads share the entire body; only the last per-band linear is per stem, so four stems cost one stem plus three output layers. Measured on the Pi, four stems up costs the same as one.

Why complements rather than stems? Because the knob is *remove this*, and a "keep everything but drums" target puts the loss where the ear is: on what is left playing. It also means the teacher's stems not summing exactly to the mix costs nothing.

### Beat-grid conditioning

Every band's input gets five conditioning values appended: sin/cos of beat phase, sin/cos of bar phase, and a grid confidence. On the Pi these come from termtv's own beat-grid estimator (the one the sampler quantizes to), and training computes them with the *same* estimator on each crop so the model sees the same failure modes in training as in production. A fraction of crops get the grid dropped to zero so the model never depends on it. I did not have an ablation for how much this buys when I built it; it was cheap to add because the grid was already there. The eval at the end of this post now has one: 0.07 dB. Almost nothing.

### Distillation from Demucs

Training data is two kinds:

- **True stems** from MUSDB18-HQ, remixed on the fly: each stem, with some probability, is swapped for another song's (its own crop, gain, maybe channels swapped). The standard MSS augmentation, an effectively unbounded set of mixes with exact targets.
- **Pseudo-labelled clips**: 6000 clips from FMA plus my own library, run through `htdemucs` on the DGX Spark (86 minutes for the four-stem relabel) to produce teacher stems.

The loss is L1 on the masked complex spectrogram, at two scales: a linear term and a power-law-compressed term (c = 0.5) so quiet bins -- hats, room, tails -- count at all. One thing learned the hard way: under a (4, 1) linear/compressed weighting the linear term is owned by the kick and the bass, and the **vocals head sat at identity for its first 4000 steps.** Resuming from that checkpoint with the vocals head's compressed weight raised to 4 moved it from 1.8 to 6.9 dB in the next three thousand steps.

The shipped model (`v3`) is d=64, four heads, 40k steps at about 0.95 s/step on the Spark -- 11 hours.

### The runtime, and the GIL

`infer.py` is numpy plus onnxruntime, single-threaded, `process(block)` in and `block` out, with the levels riding on every call. The streaming export nulls the offline model at 48-60 dB per stem.

The first deployment ran it on the audio tap thread and produced a one-second cut-out every 6-10 seconds. It wasn't CPU: the tap used about 45% of a core and the UI 55%. It was the **GIL**. Every onnxruntime call and every large numpy op had to win the interpreter back from the render thread, up to a switch interval each, and that was enough for mpv's 1 s read-ahead to drain, pause for cache, re-buffer, and resume. mpv reports this as `paused-for-cache` over its IPC socket, which is the thing to watch.

The fix was to move the model into a child process, pipelined one block deep -- the tap collects the *previous* reply rather than waiting on this one. The tap's per-block cost dropped to about 0.2 ms (a pipe write and a read of what is already there) at the price of one more block of latency, 232 → 279 ms end to end. The app also drops the interpreter switch interval to 1 ms for what stays on the thread.

Measured on the Pi with the whole app up: the d=64 four-head graph runs one 23.2 ms block in about 15.5 ms of child CPU, versus about 11.5 for the d=48 one-head. Roughly 67% of one A72 core, in its own process, for four stems.

### The same model, in a browser tab

The demo at the top of this post is the same `student.onnx`, byte for byte, running in your browser under onnxruntime-web. The runtime around it -- the STFT framing, the sixteen-frame delay line, the mask combination, the overlap-add -- is a port of the Pi's numpy runtime to plain JavaScript, and it nulls against the Python one to better than 128 dB, which is the float32 noise floor. The model lives in a web worker on the single-threaded WASM build; an AudioWorklet ferries 1024-sample blocks to it over a MessagePort and plays the returned blocks out of a ring. Single-threaded on purpose: the threaded build needs cross-origin isolation, which would break every embed on the page, and the Pi runs it single-threaded anyway.

The numbers, per 23.2 ms block of audio, same model and WASM build:

| core | model per block | whole pipeline |
|---|---|---|
| Cortex-X925 (the Spark's CPU, under Node) | 1.4 ms | 8 % of real time |
| Cortex-A72 (the Pi, in its own Firefox) | 21 ms | 1.05x real time |

So the Pi in a browser is just over the edge: the model alone fits, the JavaScript FFTs around it tip it into dropouts. Native onnxruntime does the same block in 15.5 ms on that core and leaves room, which is the whole reason the Pi's runtime is numpy + onnxruntime and not this. A laptop has far more headroom than it needs; the demo prints what it measures on yours.

Two honest caveats about the demo. It runs without the beat grid -- the conditioning is zeros, the mode a fraction of the training crops saw -- which the eval below puts at 0.07 dB worse than the Pi's path. And a web page cannot read the audio out of a YouTube embed; the iframe is another origin, and that is the end of it. What works is tab capture: open the video in its own tab, then capture that tab with its audio, in Chrome or Edge. The source tab is muted locally and the stemmed version plays here, about 0.3 s late (the model's 232 ms plus the worklet's buffer). Firefox and Safari cannot capture tab audio; the bundled clip and the file picker work everywhere. The bundled clip is a Creative Commons track from the Free Music Archive that was **not** in the distillation set -- credited in the demo -- so what you hear is the model on a song it never saw.

## Where it lands

museval -- BSSEval v4, one-second frames, the scorer the MUSDB literature reports -- on the isolated stems, `mix − complement`, over all 50 songs of the MUSDB18-HQ test set at full length, with the beat grid on. The headline column is the SiSEC convention: the median over a song's frames, then the median over songs. The mean over songs is beside it.

| head | SDR (median) | SDR (mean) | SIR | SAR |
|---|---|---|---|---|
| drums | 4.61 | 5.13 | 9.9 | 3.5 |
| bass | 3.61 | 3.61 | 8.0 | 3.3 |
| other | 2.72 | 2.70 | 2.3 | 3.1 |
| vocals | 4.22 | 3.69 | 11.0 | 3.1 |
| **mean** | **3.79** | 3.78 | | |

So the number that belongs next to the table at the top is **3.79 dB**, and it is the lowest number on it: 0.9 dB under HS-TasNet (42M parameters), 1.4 under RT-STT (383K, but timed on a GPU), 4 dB under Band-SCNet. I did not beat anyone. What the 0.86M parameters and the 186 ms of lookahead buy is that the model runs on a Pi, not that it separates better than the models that don't. As a two-source vocals/accompaniment separator, the karaoke framing, the accompaniment scores 10.6 dB (MMDenseNet karaoke: ~13.7) and the vocals 3.9.

The SIR and SAR columns say where the 3.8 dB goes. Interference is handled -- 10 dB on drums and 11 on vocals means what is in the isolated stem is mostly that stem -- but artifacts sit at about 3 dB on every head. That is the price of the complement formulation: each head learns to make the song *without* its stem sound right, so the isolated stem is the leftover, and every error in the complement lands in it at full size. "Other" is the exception in the SIR column: at 2.3 dB it is barely separated at all.

Without the beat grid -- zero conditioning, the demo's mode -- the same eval reads 3.72 dB (drums 4.50, bass 3.64, other 2.65, vocals 4.08). The grid is worth 0.07 dB on this set. That is the ablation I did not have when I wrote the conditioning section, and it says two things: the browser demo is not meaningfully worse than the Pi, and the grid was not worth the plumbing.

The metrics I actually tune by, re-run over the same 50 full songs (the first version of this post had them over 12 songs at 60 s; nothing moved by more than a dB):

| head | SDR (complement) | attenuation | damage |
|---|---|---|---|
| drums | 9.12 | −9.8 dB | −11.9 dB |
| bass | 10.45 | −9.0 dB | −13.1 dB |
| other | 8.24 | −7.2 dB | −11.6 dB |
| vocals | 11.24 | −8.6 dB | −14.5 dB |

"Attenuation" is the least-squares amount of the stem still present in the output and "damage" is what was done to everything that is *not* the stem. Those two are what a knob that *removes* a stem is judged by: the drums go 10 dB down and the rest of the song stays 12 dB clean of collateral, and that is what the demo sounds like. They read far higher than the museval numbers because the complement is most of the mix, which is why the first version of this post refused to put them next to the published table, and why this version has a number that can go there.

## The honest summary

What is genuinely uncommon here:

- **Four complement heads on one body with additive removal.** Any subset of stems from one forward pass; three at full isolates the fourth. I did not find this formulation in the real-time literature.
- **A sub-1M-parameter four-stem separator running in real time on an ARM CPU with no accelerator**, in a live audio path, next to a TUI, a sampler and a local LLM. No published or open-source equivalent found.
- **Distilling a streaming student from Demucs labels** at this size for this target. Distillation itself is standard; the destination isn't.

What is not novel: the band-split GRU (BSRNN), lookahead-as-delay, mask-based separation, the remix augmentation, pseudo-labelling from a big teacher. All borrowed, all cited below.

What it trades away: **latency, and separation quality.** 186 ms of model lookahead and 279 ms end to end, against 23-92 ms for the published real-time models; 3.8 dB of museval SDR against their 4.7 to 7.8. The latency trade is what the model is built on, and it only works because the player already had the buffer. The quality gap is the size of the model, and a bigger one does not fit in the Pi's time budget.

If you have a device with an ARM CPU, a second or so of buffer you were already paying for, and want stems, this is the point on the curve I couldn't find anyone else standing on. The code is in `drumsep/` of the termtv repo; the model is `student.onnx`, 0.86M parameters, and the training pipeline is one shell script on the Spark.

## The three-minute version

The same story as the sections above, animated. Silent, with captions in place of narration, made with manim.

<div class="explainer" id="explainer"><video controls preload="metadata" playsinline poster="https://matzelle.co/demos/stems/explainer.jpg" src="https://matzelle.co/demos/stems/explainer.mp4"><a href="https://matzelle.co/demos/stems/explainer.mp4">the explainer video (mp4, 8 MB, 3:11)</a></video></div>

Nine chapters: a mix is a sum; time × frequency; the mask; 54 bands; the body's two passes; lookahead as a delay; one body, four heads; where the answers come from; fitting it on the Pi.

## References

- Stöter, Liutkus & Ito, [The 2018 Signal Separation Evaluation Campaign](https://arxiv.org/abs/1804.06267) -- museval, the BSSEval v4 scorer and its median-over-frames-then-songs convention.
- Luo & Yu, [Music Source Separation with Band-Split RNN](https://arxiv.org/pdf/2209.15174) -- the band-split GRU with shared weights.
- Yang et al., [Band-SCNet: A Causal, Lightweight Model for High-Performance Real-Time Music Source Separation](https://www.isca-archive.org/interspeech_2025/yang25d_interspeech.pdf), Interspeech 2025.
- [Towards Practical Real-Time Low-Latency Music Source Separation (RT-STT)](https://arxiv.org/abs/2511.13146), arXiv 2511.13146.
- Venkatesh et al., [Real-time Low-latency Music Source Separation using Hybrid Spectrogram-TasNet](https://arxiv.org/pdf/2402.17701).
- [Improving Real-Time Music Accompaniment Separation with MMDenseNet](https://arxiv.org/pdf/2407.00657), arXiv 2407.00657.
- [Moises-Light: Resource-efficient Band-split U-Net](https://arxiv.org/html/2510.06785), arXiv 2510.06785 (offline, 5M / 2.2M params, for scale).
- Rouard, Massa & Défossez, [Hybrid Transformers for Music Source Separation](https://github.com/facebookresearch/demucs) -- the `htdemucs` teacher.
- [Converting Source Separation Models to ONNX for Real Time Usage in DJ Software](https://conference.audio.dev/session/2025/converting-source-separation-models-to-onnx-for-real-time-usage-in-dj-software/), ADC 2025.
- [Algoriddim djay and VirtualDJ introduce realtime stem separation](https://musictech.com/news/algoriddim-djay-virtual-dj-stem-separation/), MusicTech.
