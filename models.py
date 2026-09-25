"""Pinned timm DeiT with a single official ToMe merge after attention, before MLP.

Integration adapted from facebookresearch/ToMe tome/patch/timm.py (Meta,
CC-BY-NC-4.0; see vendor/tome/LICENSE). Supports timm 1.0.20's layer-scale API.
All conditions use the same explicit attention kernel, including the full baseline.
"""
import timm
import torch
from torch import nn
from vendor.tome.merge import bipartite_soft_matching, merge_wavg

URLS = {
    "teacher": "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
    "student": "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
}


class ExperimentModel(nn.Module):
    def __init__(self, net, merge_block=2, merge_r=98):
        super().__init__()
        self.net, self.merge_block, self.merge_r = net, merge_block, merge_r
        if not 0 <= merge_block < len(net.blocks):
            raise ValueError("merge_block is zero-based and must exist")
        if not 0 <= merge_r <= net.patch_embed.num_patches // 2:
            raise ValueError("Invalid single-site merge budget")
        for block in net.blocks:
            block.attn.fused_attn = False

    def forward(self, images, merge=False, capture=False):
        net = self.net
        x = net.norm_pre(net.patch_drop(net._pos_embed(net.patch_embed(images))))
        size, trace = None, {}
        for index, block in enumerate(net.blocks):
            a = block.attn
            h = block.norm1(x)
            b, n, c = h.shape
            qkv = a.qkv(h).reshape(b, n, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = a.q_norm(q), a.k_norm(k)
            scores = (q * a.scale) @ k.transpose(-2, -1)
            if size is not None:
                scores = scores + size.log()[:, None, None, :, 0]
            attn = a.attn_drop(scores.softmax(-1))
            y = (attn @ v).transpose(1, 2).reshape(b, n, c)
            x = x + block.drop_path1(block.ls1(a.proj_drop(a.proj(a.norm(y)))))
            if index == self.merge_block:
                if capture:
                    trace["before"] = x
                if merge and self.merge_r:
                    fn, unmerge = bipartite_soft_matching(k.mean(1), self.merge_r, class_token=True)
                    x, size = merge_wavg(fn, x, size)
                    if capture:
                        trace.update(merge=fn, unmerge=unmerge, size=size, after=x)
                elif capture:
                    trace["after"] = x
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        logits = net.forward_head(net.norm(x))
        return (logits, trace) if capture else logits


def build(role, cfg, pretrained=True):
    name = "deit_small_patch16_224" if role == "teacher" else "deit_tiny_patch16_224"
    net = timm.create_model(name, pretrained=False, num_classes=2, drop_path_rate=cfg["drop_path"])
    if pretrained:
        state = torch.hub.load_state_dict_from_url(URLS[role], map_location="cpu", check_hash=True, weights_only=True)["model"]
        state = {k: v for k, v in state.items() if not k.startswith("head.")}
        mismatch = net.load_state_dict(state, strict=False)
        if set(mismatch.missing_keys) != {"head.weight", "head.bias"} or mismatch.unexpected_keys:
            raise RuntimeError(f"Pretrained weight mismatch: {mismatch}")
    return ExperimentModel(net, cfg["merge_block"], cfg["merge_r"])
