from __future__ import annotations

from vllm_omni.diffusion.envs import PackagesEnvChecker


class _RaisingPlatform:

    def get_device_count(self) -> int:
        raise RuntimeError("cuda init failed")

    def has_flash_attn_package(self) -> bool:
        raise AssertionError("should not be called when device probe fails")


class _EmptyPlatform:

    def get_device_count(self) -> int:
        return 0

    def has_flash_attn_package(self) -> bool:
        raise AssertionError("should not be called when no devices exist")


def test_check_flash_attn_returns_false_when_device_probe_fails(monkeypatch) -> None:
    checker = object.__new__(PackagesEnvChecker)

    monkeypatch.setattr("vllm_omni.diffusion.envs.current_omni_platform", _RaisingPlatform())

    assert checker._check_flash_attn() is False


def test_check_flash_attn_returns_false_when_no_devices_exist(monkeypatch) -> None:
    checker = object.__new__(PackagesEnvChecker)

    monkeypatch.setattr("vllm_omni.diffusion.envs.current_omni_platform", _EmptyPlatform())

    assert checker._check_flash_attn() is False
