---
title: "Squeezing a 14GB Speech Model onto a 12GB GPU with 4-bit Quantization"
date: 2026-02-15
tags: [deep-learning, quantization, bitsandbytes, moshi, nvidia, personaplex, rtx-4070, speech-to-speech]
---

# Squeezing a 14GB Speech Model onto a 12GB GPU with 4-bit Quantization

NVIDIA released [PersonaPlex](https://huggingface.co/nvidia/personaplex-7b-v1), a real-time full-duplex speech-to-speech model. You talk, it talks back -- simultaneously, with persona control through voice conditioning and text prompts. It's built on [Kyutai's Moshi](https://github.com/kyutai-labs/moshi) architecture: a temporal transformer (32 layers, 4096 dim), a depth transformer ("depformer"), and a Mimi audio codec. 7B parameters total.

The problem: it needs ~14GB VRAM in bf16. My RTX 4070 has 12GB. This post is about the two approaches I tried to make it fit, why the first one was technically correct but practically useless, and how 4-bit quantization got it running in real-time at 9.6GB.

The quantized model is published at [brianmatzelle/personaplex-7b-v1-bnb-4bit](https://huggingface.co/brianmatzelle/personaplex-7b-v1-bnb-4bit).

## Attempt 1: CPU Offloading (Don't Bother)

The obvious first move: use `accelerate`'s `infer_auto_device_map` to split the model across GPU and CPU. Keep as many layers on GPU as possible, offload the rest to system RAM.

This required three patches to the Moshi codebase before it would even start:

**1. OOM on KV cache allocation.** `infer_auto_device_map(max_memory=None)` packs the GPU completely full of model weights, leaving zero room for the KV cache (~1.5GiB for 32 layers at 48MiB each). Fix: reserve 2GiB of headroom before computing the device map.

```python
free_mem = torch.cuda.mem_get_info(0)[0]
usable_gpu = max(free_mem - 2 * 1024**3, 0)
device_map = infer_auto_device_map(model, max_memory={0: usable_gpu, "cpu": "32GiB"})
```

**2. Meta tensors in streaming state.** accelerate's `dispatch_model` replaces CPU-offloaded parameters with meta tensors (zero-memory placeholders). Moshi's `_init_streaming_state` reads `self.in_proj_weight.device` to decide where to allocate the KV cache. It got `meta`, and allocated the cache on the meta device -- a tensor that doesn't exist. Fix: a `_resolve_device()` helper that checks accelerate's execution hooks before falling back to parameter inspection.

**3. torch.compile vs accelerate hooks.** `torch._dynamo` traces through the model graph and sees meta tensors everywhere. It can't compile a model whose weights don't exist on a real device. Fix: disable `torch.compile` and CUDA graphs at runtime when CPU offloading is active.

After all that, the server started. It loaded. It connected.

And every response took **40 seconds**.

CPU offloading works by shuttling layer weights between system RAM and GPU memory on every forward pass. For a language model doing autoregressive generation (one token at a time, hundreds of times per response), that's hundreds of CPU-to-GPU transfers per second. The PCIe bus becomes the bottleneck. For a real-time speech model that needs to keep up with live audio, it's hopeless.

| Metric | CPU Offload | Target for Real-Time |
|--------|-------------|---------------------|
| Response latency | ~40s | <500ms |
| Missed audio frames | 39s worth | 0 |
| Usable for conversation | No | Yes |

CPU offloading is the right answer for batch inference or offline evaluation. For real-time streaming, the entire model needs to live on the GPU.

## Attempt 2: 4-bit NF4 Quantization (This Is the One)

The math: 7B parameters in bf16 = 14GB. In 4-bit = 3.5GB. Even with overhead from quantization metadata and the non-quantized components, it should fit in 12GB.

I used [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) with NF4 (Normal Float 4-bit) quantization. The approach: recursively walk the model, replace every `nn.Linear` whose dimensions exceed 4096 with a `bnb.nn.Linear4bit`, then load the original weights through the quantized modules.

```python
def _replace_linears_with_4bit(model, compute_dtype=torch.bfloat16, min_size=4096):
    import bitsandbytes as bnb
    for name, child in list(model.named_children()):
        if isinstance(child, torch.nn.Linear):
            if max(child.in_features, child.out_features) < min_size:
                continue
            bnb_linear = bnb.nn.Linear4bit(
                child.in_features, child.out_features,
                bias=child.bias is not None,
                compute_dtype=compute_dtype, quant_type="nf4",
                compress_statistics=True,
            )
            bnb_linear.weight = bnb.nn.Params4bit(
                child.weight.data, requires_grad=False,
                quant_type="nf4", compress_statistics=True,
            )
            if child.bias is not None:
                bnb_linear.bias = child.bias
            setattr(model, name, bnb_linear)
        else:
            _replace_linears_with_4bit(child, compute_dtype, min_size)
```

Simple in theory. Four bugs in practice.

### Bug 1: Gating layers pass raw weights to F.linear

Moshi's `ActivationGating` module has a custom forward path that calls `F.linear(x, self.linear_in.weight)` directly. This bypasses the module's `forward()` method, which means bitsandbytes never gets a chance to dequantize the weights. You get garbage or a crash.

Fix: detect quantized weights and route through the module instead of the raw kernel:

```python
def forward(self, x):
    if hasattr(self.linear_in.weight, 'quant_type'):
        # Route through module forward() so bnb can dequantize
        x = self.linear_in(x)
        B, T, _ = x.shape
        x = x.view(B, T, 2, -1)
        x = self.activation(x[..., 0, :]) * x[..., 1, :]
        return self.linear_out(x)
    return gating_forward_kernel(self.linear_in.weight, self.linear_out.weight, self.activation, x)
```

### Bug 2: Depformer's multi_linear reshapes raw weights

The depformer (depth transformer) has a `multi_linear` function that reshapes weight tensors directly. Same problem -- bitsandbytes stores weights in a compressed format that can't be reshaped.

Rather than patching every raw-weight path in the depformer, I limited quantization to only `model.transformer` (the main temporal transformer). The depformer is much smaller (6 layers, 1024 dim vs 32 layers, 4096 dim), so keeping it in bf16 costs maybe 200MB. The Mimi audio codec, embeddings, and output heads also stay in bf16. Audio quality matters more than a few hundred megabytes.

### Bug 3: Duplicate in_proj parameters double memory usage

Moshi's attention module does this:

```python
in_proj = nn.Linear(embed_dim, 3 * embed_dim)
self.in_proj_weight = in_proj.weight  # registered as a parameter
self.in_proj_bias = in_proj.bias
```

This stores the weight twice: once as `in_proj.weight` (child module) and once as `in_proj_weight` (registered parameter). In bf16, that's an extra 3.2GB of wasted memory. With quantization, it's worse -- `in_proj_weight` is a reference to the original bf16 tensor, while the quantized module has its own compressed copy.

Fix: refactor to use `self.in_proj` as a proper `nn.Module`, remove the aliases, and add state dict key remapping for backwards compatibility with existing checkpoints:

```python
# Old checkpoint keys:  self_attn.in_proj_weight
# New module structure: self_attn.in_proj.weight
def _remap_state_dict_keys(state_dict):
    return {k.replace("self_attn.in_proj_weight", "self_attn.in_proj.weight"): v
            for k, v in state_dict.items()}
```

### Bug 4: KV cache allocated as uint8

After quantization, `self.in_proj.weight.dtype` returns `torch.uint8` -- that's bitsandbytes' internal storage format, not the actual compute type. The KV cache allocation reads this dtype and creates uint8 tensors. Attention over uint8 values does not produce coherent speech.

Fix: detect quantized weights and use the module's `compute_dtype` instead:

```python
weight = self.in_proj.weight
if hasattr(weight, 'quant_type'):
    dtype = getattr(self.in_proj, 'compute_dtype', torch.bfloat16)
else:
    dtype = weight.dtype
```

## The Result

After all four fixes, the server starts cleanly:

```
INFO - Loading model with 4-bit NF4 quantization
INFO - Quantized 128 linear layers to 4-bit NF4
INFO - GPU memory after quantized load: 9.6 GiB
INFO - warming up the model
INFO - Access the Web UI directly at https://10.165.161.216:8998
```

| | Original (bf16) | CPU Offload | **4-bit NF4** |
|---|---|---|---|
| VRAM | ~14 GiB | ~10 GiB | **9.6 GiB** |
| Min GPU | A100 / H100 | RTX 4070 (12GB) | **RTX 4070 (12GB)** |
| torch.compile | Yes | No | **Yes** |
| CUDA graphs | Yes | No | **Yes** |
| Real-time capable | Yes | No (~40s latency) | **Yes** |

9.6GB out of 12GB. torch.compile and CUDA graphs both work. Real-time full-duplex conversation with voice cloning on a consumer GPU.

## How Ours Compares to the GGUF Quantization

There's one other quantized version of PersonaPlex out there: [Codes4Fun/personaplex-7b-v1-q4_k-GGUF](https://huggingface.co/Codes4Fun/personaplex-7b-v1-q4_k-GGUF). It uses Q4_K quantization in GGUF format, built for [moshi.cpp](https://github.com/Codes4Fun/moshi.cpp).

They solve different problems.

| | **Ours (bnb NF4)** | **GGUF (Q4_K)** |
|---|---|---|
| Format | PyTorch `.pt` | GGUF `.gguf` |
| Quantization | NF4 (bitsandbytes) | Q4_K (k-quant) |
| Runtime | Original Moshi Python server | moshi.cpp (C++ port) |
| Web UI | Built-in (ships with Moshi) | Separate / BYO |
| What's quantized | Main transformer only | Entire model |
| Audio codec (Mimi) | bf16 (full precision) | Quantized |
| Depformer | bf16 (full precision) | Quantized |
| Embeddings & heads | bf16 (full precision) | Quantized |
| torch.compile | Yes | N/A |
| CUDA graphs | Yes | N/A |

The key difference is **selective quantization**. We only quantize the 32-layer temporal transformer -- the big dense thing that accounts for most of the parameters. Everything that directly touches audio stays in bf16:

- **Mimi** (the audio encoder/decoder) compresses and reconstructs waveforms. Quantization artifacts here become audible distortion.
- **Depformer** (depth transformer) generates the audio codebook tokens that Mimi decodes. It's small (6 layers, 1024 dim) and errors cascade into the audio output.
- **Embeddings and output heads** map between token IDs and continuous representations. They're small and sensitive to precision.

The GGUF version quantizes all of it. That's the standard approach for text LLMs (where GGUF and llama.cpp dominate), and it works well there because text tokens are discrete and robust to small perturbations. But speech models have a tighter quality budget -- the output is a continuous waveform, and quantization noise in the codec or the depth transformer can produce audible artifacts.

The other practical difference: our version runs on the **original Moshi server** with its built-in WebSocket streaming and browser UI. You start the server, open `https://localhost:8998`, and talk. The GGUF version requires moshi.cpp, which is a separate C++ reimplementation of the inference engine. If you're already in the llama.cpp ecosystem and want to integrate PersonaPlex into a C++ pipeline, GGUF is the right choice. If you want the full real-time voice chat experience with minimal setup, ours is what you want.

## Usage

```bash
# Clone and install
git clone https://huggingface.co/brianmatzelle/personaplex-7b-v1-bnb-4bit
cd personaplex-7b-v1-bnb-4bit
pip install moshi/.
pip install bitsandbytes

# Run
SSL_DIR=$(mktemp -d)
python -m moshi.server --ssl "$SSL_DIR" --quantize-4bit
```

Open `https://localhost:8998`. Talk. It talks back.

## What I'd Do Differently

The `in_proj` refactor (Bug 3) was the messiest change. The original code stored `in_proj_weight` as a direct reference to the linear layer's weight tensor, which made sense before quantization -- it avoids a module call overhead in the hot path. But it meant the weight appeared twice in the module's parameter list, doubling memory. If I were designing this from scratch, I'd just use `self.in_proj = nn.Linear(...)` everywhere and trust torch.compile to eliminate the overhead.

The gating fix (Bug 1) is also fragile. It checks for a `quant_type` attribute to decide whether to route through the module or the raw kernel. A cleaner approach would be to always route through the module and let the non-quantized path stay fast via torch.compile. But I didn't want to change the non-quantized behavior for a model I haven't tested extensively in that configuration.

## Links

- **Quantized model**: [brianmatzelle/personaplex-7b-v1-bnb-4bit](https://huggingface.co/brianmatzelle/personaplex-7b-v1-bnb-4bit)
- **Base model**: [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1)
- **Paper**: [PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models](https://arxiv.org/abs/2602.06053) (Roy et al., 2026)
- **GGUF alternative**: [Codes4Fun/personaplex-7b-v1-q4_k-GGUF](https://huggingface.co/Codes4Fun/personaplex-7b-v1-q4_k-GGUF)
