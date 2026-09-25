from pathlib import Path
import json
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.probe_logit_shift import auc, align_margins, diagnose, make_folds


def test_auc_ties_and_invariance_to_constant_shift():
    y = np.array([0,0,1,1])
    assert auc(np.ones(4), y) == .5
    assert auc(np.array([0.,1.,2.,3.]), y) == 1
    assert auc(np.array([3.,2.,1.,0.]), y) == 0
    s = np.array([0.,1.,1.,2.])
    assert auc(s, y) == .875
    assert auc(s-4, y) == auc(s, y)


def test_held_out_shift_does_not_fit_held_out_examples():
    off = np.zeros(8)
    folds = np.array([0,0,0,0,1,1,1,1])
    on = np.array([100.,100.,100.,100.,-2.,-2.,-2.,-2.])
    aligned, shifts = align_margins(off, on, folds)
    assert shifts == {"0": -2., "1": 100.}
    np.testing.assert_array_equal(aligned, [102.,102.,102.,102.,-102.,-102.,-102.,-102.])


def test_constant_shift_recovers_reference_predictions_out_of_fold():
    groups = np.repeat(np.arange(4), 2)
    labels = groups//2
    off = np.array([-2.,1.,-1.,2.,-1.,2.,-3.,1.])
    on = off - 4
    folds = make_folds(groups, count=2)
    result = diagnose(off, on, labels, groups, folds)
    assert result["on"]["wga"] == 0
    assert result["off"] == result["aligned_on_oof"]
    assert result["auc_off"] == result["auc_on"]
    assert result["delta_by_group"]["all"]["median"] == -4
    assert result["delta_by_group"]["all"]["fraction_negative"] == 1
    for g in range(4):
        assert sorted(folds[groups == g]) == [0,1]
    with pytest.raises(ValueError):
        make_folds(groups, count=3)


def test_saved_prediction_cli_reads_all_checkpoint_modes(tmp_path):
    from scripts.probe_logit_shift import main
    directory = tmp_path/"runs"/"merge_probe_seed0"
    directory.mkdir(parents=True)
    groups = torch.arange(4).repeat_interleave(2)
    margin = torch.tensor([-2.,1.,-1.,2.,-1.,2.,-3.,1.])
    common = {"labels": groups//2, "groups": groups, "sample_ids": torch.arange(8)}
    pred = {"merge_off": {**common, "logits": torch.stack([torch.zeros(8), margin], -1)},
            "merge_on": {**common, "logits": torch.stack([torch.zeros(8), margin-4], -1)}}
    records = []
    for kind in ("fixed_epoch", "native_best"):
        for method in ("kd", "tome_kd"):
            torch.save(pred, directory/f"{kind}_{method}_validation_predictions.pt")
            records.append({"checkpoint_kind": kind, "method": method, "epoch": 100,
                            "checkpoint_sha256": "synthetic-fixture"})
    (directory/"results.json").write_text(json.dumps({"split": "validation", "seed": 0, "records": records}))
    output = tmp_path/"out"
    main(["--runs", str(tmp_path/"runs"), "--out", str(output), "--seeds", "0", "--folds", "2"])
    result = json.loads((output/"results.json").read_text())
    assert result["split"] == "validation" and len(result["records"]) == 4
    assert len((output/"summary.csv").read_text().splitlines()) == 5
    assert all(r["wga_off_percent"] == r["aligned_on_oof_wga_percent"] for r in result["records"])
