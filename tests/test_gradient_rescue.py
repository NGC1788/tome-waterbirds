import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from timm.models.vision_transformer import VisionTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run
from models import ExperimentModel
from experiments import gradient_rescue as probe

torch.set_num_threads(2)


def tiny(drop_path=0.):
    return ExperimentModel(VisionTransformer(img_size=224, patch_size=16, embed_dim=24, depth=4,
                             num_heads=3, num_classes=2, drop_path_rate=drop_path), merge_block=2, merge_r=98)


def config(arm="standard"):
    cfg = json.loads((run.HERE/"config.json").read_text())
    cfg.update(workers=0, batch_size=3, accumulation_steps=2, amp=False)
    cfg["gradient_intervention"] = {"arm": arm, "strength": 1.}
    return cfg


def test_detached_suffix_matches_full_gradient_without_parameter_or_rng_changes():
    torch.manual_seed(14)
    model = tiny().eval()
    logits, trace = model(torch.randn(2,3,224,224), capture=True)
    teacher = torch.randn(2,2)
    labels = torch.tensor([0,1])
    loss, _, _ = run.loss_fn(logits, labels, teacher, config())
    expected, = torch.autograd.grad(loss, trace["before"])
    rng = torch.random.get_rng_state().clone()
    actual, _ = probe.full_reference_gradient(model, trace["before"], labels, teacher, config(), torch.device("cpu"))
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-4)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(p.grad is None for p in model.parameters())


def test_residual_is_nullspace_and_controls_match_norm():
    torch.manual_seed(7)
    model = tiny()
    _, trace = model(torch.randn(2,3,224,224), merge=True, capture=True)
    grad = torch.randn_like(trace["before"])
    rng = torch.random.get_rng_state().clone()
    vectors, diagnostic = probe.intervention_vectors(grad, trace, seed=41)
    assert torch.equal(rng, torch.random.get_rng_state())
    for key in ("residual", "shuffled"):
        sums = trace["merge"](vectors[key], mode="sum")
        torch.testing.assert_close(sums, torch.zeros_like(sums), atol=5e-6, rtol=0)
    norms = [vectors[key].flatten(1).norm(dim=1) for key in probe.ARMS[1:]]
    torch.testing.assert_close(norms[0], norms[1])
    torch.testing.assert_close(norms[0], norms[2])
    assert not torch.equal(vectors["residual"], vectors["shuffled"])
    for value in vectors.values():
        assert torch.count_nonzero(value[:,0]) == 0  # protected CLS
    assert 0 <= diagnostic["residual_energy_fraction"] <= 1


def test_backward_intervention_keeps_forward_and_suffix_gradients_unchanged():
    torch.manual_seed(13)
    model = tiny()
    logits, trace = model(torch.randn(2,3,224,224), merge=True, capture=True)
    loss = logits.square().mean()
    vectors, _ = probe.intervention_vectors(torch.randn_like(trace["before"]), trace, seed=3)
    augmented = probe.inject_gradient(loss, trace["before"], vectors["residual"], strength=.7)
    assert torch.equal(augmented, loss)
    original_boundary, = torch.autograd.grad(loss, trace["before"], retain_graph=True)
    modified_boundary, = torch.autograd.grad(augmented, trace["before"], retain_graph=True)
    torch.testing.assert_close(modified_boundary-original_boundary, .7*vectors["residual"], atol=1e-6, rtol=1e-5)
    parameters = dict(model.named_parameters())
    original = torch.autograd.grad(loss, parameters.values(), retain_graph=True)
    modified = torch.autograd.grad(augmented, parameters.values())
    for name, a, b in zip(parameters, original, modified):
        if name.startswith(("net.head.", "net.norm.", "net.blocks.3.", "net.blocks.2.mlp.", "net.blocks.2.norm2.")):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert any(not torch.equal(a,b) for a,b in zip(original,modified))


def test_cost_matched_standard_preserves_original_training_with_dropout():
    class Images(Dataset):
        def __init__(self):
            self.x = torch.randn(5,3,224,224)
        def __len__(self): return 5
        def __getitem__(self, i): return self.x[i], i%2, i%4, i
    torch.manual_seed(37)
    a = tiny(drop_path=.2)
    b = copy.deepcopy(a)
    teacher = tiny().requires_grad_(False).eval()
    before = {n:p.detach().clone() for n,p in teacher.named_parameters()}
    data = Images()
    cfg = config()
    oa, ob = run.optimizer_for(a,cfg), run.optimizer_for(b,cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    sa, pa = run.train_epoch(a, teacher, data, cfg, torch.device("cpu"), oa, scaler, 1, True)
    sb, pb = probe.train_epoch(b, teacher, data, cfg, torch.device("cpu"), ob, scaler, 1, True)
    for x,y in zip(a.parameters(),b.parameters()): torch.testing.assert_close(x,y,atol=0,rtol=0)
    for x,y in zip(pa,pb): torch.testing.assert_close(x,y,atol=0,rtol=0)
    assert sa["loss"] == sb["loss"]
    assert all(torch.equal(before[n],p) and p.grad is None for n,p in teacher.named_parameters())
    assert np.isfinite(sb["residual_energy_fraction"])
