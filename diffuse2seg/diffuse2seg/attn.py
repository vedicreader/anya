"""Step 1: self-attention extraction from SD2 via single-step denoising."""
import torch, torch.nn.functional as F, numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler

# stabilityai/stable-diffusion-2-base is gated; this is an ungated SD2.1 with the identical UNet.
SD2 = "friedrichor/stable-diffusion-2-1-realistic"


def load_pipe(model=SD2, device="cpu", dtype=torch.float32):
    "Load the SD2 pipeline (VAE + UNet + text encoder) for feature extraction."
    pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=dtype, safety_checker=None)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe.to(device)


def _to_latent(pipe, img, size, device, dtype):
    "VAE-encode a PIL image (resized square) into a scaled SD latent."
    x = np.asarray(img.convert("RGB").resize((size, size), Image.BICUBIC), np.float32) / 127.5 - 1.0
    x = torch.from_numpy(x).permute(2, 0, 1)[None].to(device, dtype)
    with torch.no_grad(): z = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor
    return z


class _Grab:
    "Attention processor that caches softmax(QK^T) for the self-attention it wraps."
    def __init__(self, store, key): self.store, self.key = store, key
    def __call__(self, attn, hidden, encoder_hidden_states=None, attention_mask=None, **kw):
        h = hidden
        q = attn.to_q(h); k = attn.to_k(h); v = attn.to_v(h)
        q, k, v = (attn.head_to_batch_dim(t) for t in (q, k, v))
        p = attn.get_attention_scores(q, k, attention_mask)          # (B*heads, N, N)
        nh = attn.heads; N = p.shape[1]
        self.store[self.key] = p.view(-1, nh, N, N).mean(1).mean(0)  # aggregate heads+batch -> (N,N)
        out = torch.bmm(p, v)
        out = attn.batch_to_head_dim(out)
        out = attn.to_out[0](out); out = attn.to_out[1](out)
        return out


def _targets(unet):
    "The first two self-attentions of the highest-resolution UNet decoder block."
    up = unet.up_blocks[-1]                                          # full-latent-res CrossAttnUpBlock2D
    return [up.attentions[i].transformer_blocks[0].attn1 for i in (0, 1)]


def extract_affinity(pipe, img, size=512, t=100, w=(0.85, 0.15), seed=0):
    "Return aggregated NxN affinity A and latent grid (Hf,Wf) from one denoise step."
    device, dtype = pipe.device, pipe.dtype
    z = _to_latent(pipe, img, size, device, dtype)
    Hf, Wf = z.shape[-2:]
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(z.shape, generator=g).to(device, dtype)
    zt = pipe.scheduler.add_noise(z, noise, torch.tensor([t]))
    emb = pipe.encode_prompt("", device, 1, False)[0]               # empty-prompt conditioning
    store = {}
    for i, m in enumerate(_targets(pipe.unet)): m.set_processor(_Grab(store, i))
    with torch.no_grad(): pipe.unet(zt, t, encoder_hidden_states=emb)
    A = sum(wl * store[i] for i, wl in enumerate(w))                # w-weighted layer aggregation
    return A.float().cpu().numpy(), (Hf, Wf)
