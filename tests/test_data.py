"""Locating data/ and results/ in a repository checkout."""

from __future__ import annotations

import pytest

from tessa import data_path, repo_root, results_dir


def _fake_repo(tmp_path):
    (tmp_path / "src" / "tessa").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "data" / "images" / "qcat").mkdir(parents=True)
    (tmp_path / "data" / "images" / "qcat" / "qcat.png").write_bytes(b"")
    (tmp_path / "data" / "a").mkdir()
    (tmp_path / "data" / "b").mkdir()
    (tmp_path / "data" / "a" / "twice.txt").write_text("")
    (tmp_path / "data" / "b" / "twice.txt").write_text("")
    (tmp_path / "results").mkdir()
    return tmp_path


def test_the_checkout_holds_this_package():
    root = repo_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "tessa" / "__init__.py").is_file()


def test_tessa_repo_selects_a_checkout(tmp_path, monkeypatch):
    root = _fake_repo(tmp_path)
    monkeypatch.setenv("TESSA_REPO", str(root))
    assert repo_root() == root.resolve()
    assert results_dir() == root.resolve() / "results"


def test_a_bare_name_is_found_anywhere_under_data(tmp_path, monkeypatch):
    root = _fake_repo(tmp_path)
    monkeypatch.setenv("TESSA_REPO", str(root))
    want = root.resolve() / "data" / "images" / "qcat" / "qcat.png"
    assert data_path("qcat.png") == want
    assert data_path("images/qcat/qcat.png") == want


def test_a_missing_or_ambiguous_name_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("TESSA_REPO", str(_fake_repo(tmp_path)))
    with pytest.raises(FileNotFoundError, match="not found"):
        data_path("nothing.png")
    with pytest.raises(FileNotFoundError, match="ambiguous"):
        data_path("twice.txt")
    assert data_path("a/twice.txt").name == "twice.txt"
