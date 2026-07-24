from __future__ import annotations

import pytest

from sft import hami_cuda_bootstrap


def test_bootstrap_cuda_context_initializes_local_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setattr(
        hami_cuda_bootstrap.torch.cuda,
        "set_device",
        lambda rank: calls.append(("set_device", rank)),
    )
    monkeypatch.setattr(
        hami_cuda_bootstrap.torch.cuda,
        "init",
        lambda: calls.append(("init",)),
    )
    monkeypatch.setattr(
        hami_cuda_bootstrap.torch,
        "device",
        lambda kind, rank: (kind, rank),
    )
    monkeypatch.setattr(
        hami_cuda_bootstrap.torch,
        "empty",
        lambda size, *, device: calls.append(("empty", size, device)),
    )

    assert hami_cuda_bootstrap.bootstrap_cuda_context() == 2
    assert calls == [
        ("set_device", 2),
        ("init",),
        ("empty", 1, ("cuda", 2)),
    ]


def test_bootstrap_cuda_context_requires_local_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    with pytest.raises(RuntimeError, match="LOCAL_RANK is required"):
        hami_cuda_bootstrap.bootstrap_cuda_context()


def test_main_launches_env_selected_target(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        hami_cuda_bootstrap,
        "bootstrap_cuda_context",
        lambda: calls.append(("bootstrap",)),
    )
    monkeypatch.setenv(
        hami_cuda_bootstrap.TARGET_MODULE_ENV,
        "sft.label_token_matrix_infer",
    )
    monkeypatch.setattr(
        hami_cuda_bootstrap.runpy,
        "run_module",
        lambda module, *, run_name: calls.append(("run_module", module, run_name)),
    )

    hami_cuda_bootstrap.main()

    assert calls == [
        ("bootstrap",),
        ("run_module", "sft.label_token_matrix_infer", "__main__"),
    ]


@pytest.mark.parametrize("target", ["", "   ", "sft.hami_cuda_bootstrap"])
def test_main_rejects_invalid_target(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    monkeypatch.setattr(hami_cuda_bootstrap, "bootstrap_cuda_context", lambda: 0)
    monkeypatch.setenv(hami_cuda_bootstrap.TARGET_MODULE_ENV, target)

    with pytest.raises(RuntimeError, match="SFT_HAMI_BOOTSTRAP_TARGET_MODULE"):
        hami_cuda_bootstrap.main()
