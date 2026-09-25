import copy
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from timm.models.vision_transformer import VisionTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run
from data import Waterbirds, audit, metrics
from models import ExperimentModel, build

torch.set_num_threads(2)


def tiny():
    return ExperimentModel(VisionTransformer(img_size=224, patch_size=16, embed_dim=24, depth=4,
                                            num_heads=3, num_classes=2), merge_block=2, merge_r=98)


@pytest.fixture
def cfg():
    c = json.loads((run.HERE/"config.json").read_text())
    c.update(workers=0, batch_size=3, eval_batch_size=4, accumulation_steps=2,
             teacher_epochs=2, student_epochs=2, warmup_epochs=0, amp=False, train_eval_every=1)
    return c


@pytest.fixture
def data_root(tmp_path):
    rng = np.random.default_rng(41)
    rows = []
    # Five train examples, four per held-out split: exercises an uneven accumulation window.
    for s in range(3):
        for i in range(5 if s == 0 else 4):
            name = f"{s}_{i}.png"
            Image.fromarray(rng.integers(0, 256, (240, 300, 3), dtype=np.uint8)).save(tmp_path/name)
            g = i % 4
            rows.append(dict(img_filename=name, y=g//2, place=g%2, split=s))
    with (tmp_path/"metadata.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    return tmp_path


def test_full_path_matches_native_timm_outputs_and_parameter_gradients():
    torch.manual_seed(7)
    model = tiny().eval()
    x = torch.randn(2, 3, 224, 224)
    native = model.net(x)
    native.square().sum().backward()
    native_grads = {n:p.grad.clone() for n,p in model.named_parameters()}
    model.zero_grad()
    actual = model(x, merge=False)
    torch.testing.assert_close(actual, native, atol=1e-6, rtol=1e-5)
    actual.square().sum().backward()
    for n, p in model.named_parameters():
        torch.testing.assert_close(p.grad, native_grads[n], atol=2e-5, rtol=2e-4)


def test_merge_preserves_cls_mass_and_has_group_shared_gradient():
    torch.manual_seed(3)
    model = tiny().eval()
    _, trace = model(torch.randn(2, 3, 224, 224), merge=True, capture=True)
    assert trace["before"].shape[1] == 197
    assert trace["after"].shape[1] == 99
    torch.testing.assert_close(trace["before"][:, 0], trace["after"][:, 0])
    torch.testing.assert_close(trace["size"].sum(1), torch.full((2,1), 197.))
    g = torch.autograd.grad(trace["after"].sum(), trace["before"])[0]
    expected = trace["unmerge"](1/trace["size"]).expand_as(g)
    torch.testing.assert_close(g, expected)
    assert torch.isfinite(g).all()


def test_wga_below_chance_and_missing_group_not_silently_dropped():
    logits = torch.tensor([[5.,0.], [0.,5.], [0.,5.], [5.,0.]])
    labels, groups = torch.tensor([0,0,1,1]), torch.arange(4)
    m = metrics(logits, labels, groups, [.25]*4)
    assert m["accuracy"] == .5 and m["wga"] == 0
    assert metrics(logits[:3],labels[:3],groups[:3],[.25]*4)["wga"] is None


def test_augmentations_match_across_model_rng_and_data_audit(data_root):
    ds = Waterbirds(data_root, 0, train=True, seed=8)
    ds.epoch = 2
    a = ds[0][0]
    torch.rand(2000)
    b = ds[0][0]
    assert torch.equal(a,b)
    ds.epoch = 3
    assert not torch.equal(a,ds[0][0])
    result = audit(data_root, require_official=False)
    assert result["counts"][0] == [2,1,1,1]
    with pytest.raises(ValueError, match="official"):
        audit(data_root)


def test_kd_direction_and_frozen_teacher(cfg):
    student = torch.tensor([[1.,-1.]],requires_grad=True)
    teacher = torch.tensor([[3.,-3.]])
    total, _, kd = run.loss_fn(student,torch.tensor([0]),teacher,cfg)
    expected = torch.nn.functional.kl_div(torch.log_softmax(student/2,-1),torch.softmax(teacher/2,-1),reduction="batchmean")*4
    torch.testing.assert_close(kd,expected.detach())
    total.backward()
    assert teacher.grad is None and student.grad is not None


def test_uneven_microbatch_accumulation_matches_full_batch(data_root,cfg):
    from torch.utils.data import TensorDataset
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.fc = torch.nn.Linear(3,2)
        def forward(self,x,merge=False): return self.fc(x)
    torch.manual_seed(5)
    a=Net(); b=copy.deepcopy(a)
    ds=TensorDataset(torch.randn(5,3),torch.tensor([0,1,0,1,1]),torch.tensor([0,1,2,3,3]),torch.arange(5))
    ca={**cfg,"grad_clip":1e6}
    opt_a=torch.optim.SGD(a.parameters(),lr=.01)
    opt_b=torch.optim.SGD(b.parameters(),lr=.01)
    scaler=torch.amp.GradScaler("cuda",enabled=False)
    run.train_epoch(a,None,ds,ca,torch.device("cpu"),opt_a,scaler,1,False)
    loss=torch.nn.functional.cross_entropy(b(ds.tensors[0]),ds.tensors[1]); loss.backward(); opt_b.step()
    for p,q in zip(a.parameters(),b.parameters()): torch.testing.assert_close(p,q)


def test_training_resume_and_teacher_immutable(tmp_path,data_root,cfg,monkeypatch):
    original_audit=run.audit
    monkeypatch.setattr(run,"audit",lambda root:original_audit(root,require_official=False))
    monkeypatch.setattr(run,"build",lambda role,cfg,pretrained:tiny())
    def args(method):
        return SimpleNamespace(out=str(tmp_path/method),method=method,data=str(data_root),teacher=None,resume=False)
    teacher_args=args("teacher")
    run.fit(teacher_args,cfg,torch.device("cpu"))
    tpath=Path(teacher_args.out)/"best.pt"; before=run.sha(tpath)
    kd_args=args("tome_kd"); kd_args.teacher=str(tpath)
    # Interrupt at the start of epoch 2; resume must match an uninterrupted run exactly.
    original_epoch=run.train_epoch
    def interrupt(*a,**kw):
        if a[-2]==2: raise RuntimeError("test interruption")
        return original_epoch(*a,**kw)
    monkeypatch.setattr(run,"train_epoch",interrupt)
    with pytest.raises(RuntimeError,match="test interruption"):
        run.fit(kd_args,cfg,torch.device("cpu"))
    monkeypatch.setattr(run,"train_epoch",original_epoch)
    kd_args.resume=True
    run.fit(kd_args,cfg,torch.device("cpu"))
    other=args("tome_kd"); other.out=str(tmp_path/"uninterrupted"); other.teacher=str(tpath)
    run.fit(other,cfg,torch.device("cpu"))
    a=torch.load(Path(kd_args.out)/"last.pt",weights_only=True)
    b=torch.load(Path(other.out)/"last.pt",weights_only=True)
    for key in a["model"]: torch.testing.assert_close(a["model"][key],b["model"][key],rtol=0,atol=0)
    assert run.sha(tpath)==before
    assert not list(Path(kd_args.out).glob("*test*"))
