"""Tests for synthetic data generation."""

from fincheck.data.synthetic import SyntheticConfig, generate_transactions


def test_determinism():
    a = generate_transactions(SyntheticConfig(n_background=300, seed=7))
    b = generate_transactions(SyntheticConfig(n_background=300, seed=7))
    assert [t.id for t in a] == [t.id for t in b]
    assert [t.amount for t in a] == [t.amount for t in b]


def test_different_seed_differs():
    a = generate_transactions(SyntheticConfig(n_background=300, seed=7))
    b = generate_transactions(SyntheticConfig(n_background=300, seed=8))
    assert [t.amount for t in a] != [t.amount for t in b]


def test_contains_laundering_labels():
    txns = generate_transactions(SyntheticConfig(n_background=300, seed=7))
    laundering = [t for t in txns if t.is_laundering]
    assert len(laundering) > 0
    assert len(laundering) < len(txns)


def test_no_self_loops():
    txns = generate_transactions(SyntheticConfig(n_background=300, seed=7))
    assert all(t.src != t.dst for t in txns)


def test_unique_ids():
    txns = generate_transactions(SyntheticConfig(n_background=300, seed=7))
    ids = [t.id for t in txns]
    assert len(ids) == len(set(ids))


def test_size_scales_with_config():
    small = generate_transactions(SyntheticConfig(n_background=100, seed=1))
    big = generate_transactions(SyntheticConfig(n_background=1000, seed=1))
    assert len(big) > len(small)
