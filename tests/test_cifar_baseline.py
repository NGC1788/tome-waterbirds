import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import cifar_baseline as cb


class TinyNet(torch.nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 8),
                                          torch.nn.BatchNorm1d(8), torch.nn.ReLU(),
                                          torch.nn.Linear(8, num_classes))

    def forward(self, x):
        return self.layers(x), {}


class TinyData(torch.utils.data.Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        # Augmentation consumes RNG, so resume/pairing tests exercise data randomness.
        return torch.full((3, 4, 4), index/12) + torch.rand(3, 4, 4)*.01, index % 4


def test_official_schedule_and_kl_gradient():
    assert [cb.learning_rate(e) for e in [1,150,151,180,181,210,211,240]] == pytest.approx(
        [.05,.05,.005,.005,.0005,.0005,.00005,.00005])
    student = torch.tensor([[1.,-.5,2.],[0.,2.,-1.]], requires_grad=True)
    teacher = torch.tensor([[2.,0.,-1.],[1.,.5,-2.]])
    labels = torch.tensor([0,1])
    loss, ce, kd = cb.objective(student, labels, teacher)
    t = 4.
    q, p = (teacher/t).softmax(1), (student/t).softmax(1)
    expected_kd = (q*(q.log() - p.log())).sum(1).mean()*t*t
    torch.testing.assert_close(kd, expected_kd)
    expected_grad = (.1*(student.softmax(1)-F.one_hot(labels,3)) + .9*t*(p-q))/2
    torch.testing.assert_close(torch.autograd.grad(loss, student)[0], expected_grad)
    assert cb.objective(student, labels)[0].item() == ce.item()


@pytest.fixture
def tiny_setup(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    monkeypatch.setitem(cb.RECIPE, "epochs", 2)
    monkeypatch.setattr(cb, "resnet8x4", TinyNet)
    monkeypatch.setattr(cb, "resnet32x4", TinyNet)
    calls = []
    def dataset(root, train, transform=None):
        calls.append(train)
        return TinyData()
    monkeypatch.setattr(cb.datasets, "CIFAR100", dataset)
    data = tmp_path/"data"
    (data/"cifar-100-python").mkdir(parents=True)
    (data/"cifar-100-python/train").write_bytes(b"synthetic train fixture")
    (data/"cifar-100-python/test").write_bytes(b"synthetic test fixture")
    cb.seed_all(10)
    torch.save(TinyNet().state_dict(), data/"resnet32x4.pt")
    def args(method, folder):
        return argparse.Namespace(device="cpu", data=str(data), seed=0, workers=0,
                                  method=method, out=str(folder), resume=True)
    return args, calls, data


def test_resume_is_exact_with_momentum_and_random_augmentation(tiny_setup, monkeypatch, tmp_path):
    args, calls, data = tiny_setup
    cb.fit(args("kd", tmp_path/"straight"))
    original = cb.train_epoch
    def interrupt(*a, **k):
        if a[5] == 2:
            raise RuntimeError("test interruption")
        return original(*a, **k)
    monkeypatch.setattr(cb, "train_epoch", interrupt)
    with pytest.raises(RuntimeError, match="test interruption"):
        cb.fit(args("kd", tmp_path/"resumed"))
    monkeypatch.setattr(cb, "train_epoch", original)
    cb.fit(args("kd", tmp_path/"resumed"))
    a = torch.load(tmp_path/"straight/last.pt", weights_only=True)
    b = torch.load(tmp_path/"resumed/last.pt", weights_only=True)
    for key in a["student"]:
        torch.testing.assert_close(a["student"][key], b["student"][key], rtol=0, atol=0)
    for key in a["optimizer"]["state"]:
        torch.testing.assert_close(a["optimizer"]["state"][key]["momentum_buffer"],
                                   b["optimizer"]["state"][key]["momentum_buffer"], rtol=0, atol=0)
    assert all(calls), "Training must not instantiate the test dataset"
    assert [r["first_batch_sha256"] for r in a["history"]] == [r["first_batch_sha256"] for r in b["history"]]


def test_paired_runs_final_report_and_identity_guard(tiny_setup, tmp_path):
    args, calls, data = tiny_setup
    runs = tmp_path/"runs"
    for method in ("ce", "kd"):
        cb.fit(args(method, runs/f"{method}_seed0"))
    assert all(calls)
    teacher_before = cb.sha(data/"resnet32x4.pt")
    final_args = argparse.Namespace(device="cpu", data=str(data), runs=str(runs), seed=0, workers=0)
    cb.finish(final_args)
    result = json.loads((runs/"report_seed0/results.json").read_text())
    assert [r["method"] for r in result["records"]] == ["teacher", "ce", "kd"]
    assert result["records"][1]["teacher_images"] == 0
    assert result["records"][2]["teacher_images"] == 24
    assert teacher_before == cb.sha(data/"resnet32x4.pt")
    assert calls.count(False) == 1
    cb.finish(final_args)  # Existing report does not evaluate test again.
    assert calls.count(False) == 1
    path = runs/"ce_seed0"
    state = torch.load(path/"last.pt", weights_only=True)
    state["spec"]["method"] = "kd"
    torch.save(state, path/"last.pt")
    done = json.loads((path/"training_complete.json").read_text())
    done["checkpoint_sha256"] = cb.sha(path/"last.pt")
    cb.atomic_json(path/"training_complete.json", done)
    with pytest.raises(ValueError, match="method/seed"):
        cb.finish(final_args)
    assert calls.count(False) == 1


def test_official_models_shape_and_frozen_teacher():
    from vendor.mdistiller.resnet import resnet32x4, resnet8x4
    torch.set_num_threads(1)
    teacher = resnet32x4(num_classes=100).eval().requires_grad_(False)
    student = resnet8x4(num_classes=100).train()
    before = cb.model_hash(teacher)
    x, y = torch.randn(2,3,32,32), torch.tensor([0,1])
    with torch.no_grad():
        target, _ = teacher(x)
    z, _ = student(x)
    cb.objective(z, y, target)[0].backward()
    assert z.shape == target.shape == (2,100)
    assert all(p.grad is None for p in teacher.parameters())
    assert before == cb.model_hash(teacher)
    assert all(torch.isfinite(p.grad).all() for p in student.parameters() if p.grad is not None)
