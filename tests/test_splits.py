"""Smoke tests for gwr.splits.get_split."""

import numpy as np
import pytest

from gwr.splits import KINDS, composition_ids, get_split

SEEDS = range(42, 52)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_nonempty_disjoint(kind, seed):
    sp = get_split(kind, seed, verbose=False)
    tr, va, te = sp.train_idx, sp.val_idx, sp.test_idx
    assert len(tr) > 0 and len(va) > 0 and len(te) > 0
    assert len(sp.X_train) == len(tr) and len(sp.X_val) == len(va) and len(sp.X_test) == len(te)
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    assert len(tr) + len(va) + len(te) == len(sp.df)


@pytest.mark.parametrize("seed", SEEDS)
def test_composition_group_no_leakage(seed):
    sp = get_split("composition_group", seed, verbose=False)
    g = composition_ids(sp.df, sp.feature_cols)
    gtr, gva, gte = set(g[sp.train_idx]), set(g[sp.val_idx]), set(g[sp.test_idx])
    assert not (gtr & gte) and not (gtr & gva) and not (gva & gte)
    # whole compositions held out: all 81 temperatures each
    for idx in (sp.train_idx, sp.val_idx, sp.test_idx):
        counts = np.bincount(g[idx])
        assert set(counts[counts > 0]) == {81}
