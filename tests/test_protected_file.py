from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from microtensor_miner_controller.protected_file import (
    ProtectedFileError,
    read_root_service_file,
)


def _metadata(metadata: os.stat_result, **changes: int) -> SimpleNamespace:
    fields = {
        name: getattr(metadata, name)
        for name in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_gid",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    }
    fields.update(changes)
    return SimpleNamespace(**fields)


class ProtectedFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "protected"
        self.path.write_bytes(b"safe payload")
        self.path.chmod(0o640)
        self.metadata_patchers: list[Any] = []
        if os.geteuid() != 0:
            real_lstat = Path.lstat
            real_fstat = os.fstat

            def root_service(metadata: os.stat_result) -> SimpleNamespace:
                return _metadata(metadata, st_uid=0, st_gid=os.getegid())

            self.metadata_patchers = [
                mock.patch.object(
                    Path,
                    "lstat",
                    lambda path: root_service(real_lstat(path)),
                ),
                mock.patch(
                    "microtensor_miner_controller.protected_file.os.fstat",
                    side_effect=lambda descriptor: root_service(real_fstat(descriptor)),
                ),
            ]
            for patcher in self.metadata_patchers:
                patcher.start()

    def tearDown(self) -> None:
        for patcher in reversed(self.metadata_patchers):
            patcher.stop()
        self.temporary.cleanup()

    def test_exact_root_service_group_mode_0640_file_is_read(self) -> None:
        self.assertEqual(
            read_root_service_file(self.path, label="fixture", maximum_bytes=64),
            b"safe payload",
        )

    def test_relative_missing_symlink_hardlink_and_directory_are_refused(self) -> None:
        with self.assertRaisesRegex(ProtectedFileError, "absolute"):
            read_root_service_file(Path("relative"), label="fixture", maximum_bytes=64)
        with self.assertRaisesRegex(ProtectedFileError, "unavailable or unsafe"):
            read_root_service_file(
                self.path.with_name("missing"), label="fixture", maximum_bytes=64
            )

        symlink = self.path.with_name("symlink")
        symlink.symlink_to(self.path)
        with self.assertRaisesRegex(ProtectedFileError, "regular non-symlink"):
            read_root_service_file(symlink, label="fixture", maximum_bytes=64)

        hardlink = self.path.with_name("hardlink")
        hardlink.hardlink_to(self.path)
        with self.assertRaisesRegex(ProtectedFileError, "exactly one hard link"):
            read_root_service_file(self.path, label="fixture", maximum_bytes=64)
        hardlink.unlink()

        directory = self.path.with_name("directory")
        directory.mkdir()
        directory.chmod(0o640)
        with self.assertRaisesRegex(ProtectedFileError, "regular non-symlink"):
            read_root_service_file(directory, label="fixture", maximum_bytes=64)

    def test_wrong_owner_group_and_mode_are_refused_without_content_leak(self) -> None:
        secret = self.path.read_text()
        real_lstat = Path.lstat
        real_fstat = os.fstat
        cases = (
            ("owner", {"st_uid": 1}, "owned by root"),
            ("group", {"st_gid": os.getegid() + 1}, "effective service group"),
            ("mode", {"st_mode": stat_mode(0o600)}, "exactly 0640"),
        )
        for label, changes, message in cases:
            with self.subTest(label=label):
                self._stop_portability_patchers()
                with (
                    mock.patch.object(
                        Path,
                        "lstat",
                        lambda path, changes=changes: _metadata(
                            real_lstat(path),
                            **({"st_uid": 0, "st_gid": os.getegid()} | changes),
                        ),
                    ),
                    mock.patch(
                        "microtensor_miner_controller.protected_file.os.fstat",
                        side_effect=lambda descriptor, changes=changes: _metadata(
                            real_fstat(descriptor),
                            **({"st_uid": 0, "st_gid": os.getegid()} | changes),
                        ),
                    ),
                    self.assertRaisesRegex(ProtectedFileError, message) as raised,
                ):
                    read_root_service_file(self.path, label="fixture", maximum_bytes=64)
                self.assertNotIn(secret, str(raised.exception))

    def test_missing_nofollow_and_oversize_are_refused(self) -> None:
        with (
            mock.patch(
                "microtensor_miner_controller.protected_file.os.O_NOFOLLOW",
                new=None,
            ),
            self.assertRaisesRegex(ProtectedFileError, "cannot securely open"),
        ):
            read_root_service_file(self.path, label="fixture", maximum_bytes=64)
        with self.assertRaisesRegex(ProtectedFileError, "8-byte limit"):
            read_root_service_file(self.path, label="fixture", maximum_bytes=8)

    def test_short_read_descriptor_mutation_and_path_replacement_are_refused(self) -> None:
        self._stop_portability_patchers()
        real_lstat = Path.lstat
        real_fstat = os.fstat

        with (
            mock.patch(
                "microtensor_miner_controller.protected_file.os.read",
                return_value=b"",
            ),
            self.assertRaisesRegex(ProtectedFileError, "changed while it was read"),
        ):
            read_root_service_file(self.path, label="fixture", maximum_bytes=64)

        fstat_calls = 0

        def changed_fstat(descriptor: int) -> SimpleNamespace:
            nonlocal fstat_calls
            fstat_calls += 1
            observed = real_fstat(descriptor)
            return _metadata(
                observed,
                st_mtime_ns=observed.st_mtime_ns + (1 if fstat_calls == 2 else 0),
            )

        with (
            mock.patch(
                "microtensor_miner_controller.protected_file.os.fstat",
                side_effect=changed_fstat,
            ),
            mock.patch(
                "microtensor_miner_controller.protected_file.os.close",
                wraps=os.close,
            ) as close,
            self.assertRaisesRegex(ProtectedFileError, "changed while it was read"),
        ):
            read_root_service_file(self.path, label="fixture", maximum_bytes=64)
        close.assert_called_once()

        lstat_calls = 0

        def changed_lstat(path: Path) -> SimpleNamespace:
            nonlocal lstat_calls
            lstat_calls += 1
            observed = real_lstat(path)
            return _metadata(
                observed,
                st_ctime_ns=observed.st_ctime_ns + (1 if lstat_calls == 2 else 0),
            )

        with (
            mock.patch.object(Path, "lstat", changed_lstat),
            self.assertRaisesRegex(ProtectedFileError, "changed while it was read"),
        ):
            read_root_service_file(self.path, label="fixture", maximum_bytes=64)

    def _stop_portability_patchers(self) -> None:
        for patcher in reversed(self.metadata_patchers):
            patcher.stop()
        self.metadata_patchers = []


def stat_mode(mode: int) -> int:
    return (os.stat(__file__).st_mode & ~0o7777) | mode


if __name__ == "__main__":
    unittest.main()
