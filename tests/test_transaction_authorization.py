from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from helpers import base_env

from microtensor_miner_controller.backend import MicrotensorBackend
from microtensor_miner_controller.config import (
    AUTHORIZED_HOTKEY_SS58,
    BITTENSOR_VERSION,
    BITTENSOR_WALLET_VERSION,
    FINNEY_GENESIS_HASH,
    FINNEY_RUNTIME_CODE_HASH,
    FINNEY_RUNTIME_SPEC_VERSION,
    FINNEY_TRANSACTION_VERSION,
    SUBSTRATE_INTERFACE_VERSION,
    ControllerConfig,
)
from microtensor_miner_controller.errors import AuthorizationRefused, PreflightError


class Amount:
    def __init__(self, rao: int) -> None:
        self.rao = rao


class FakeSnapshot:
    def __init__(self, uid: int | None = 32) -> None:
        self.uid = uid

    def uid_of(self, hotkey: str) -> int | None:
        del hotkey
        return self.uid


class FakeSubstrate:
    def __init__(
        self,
        *,
        fee: int | tuple[Any, ...] = 0,
        initial_deposit: Any = 0,
        field_deposit: Any = 0,
        specs: tuple[Any, ...] = (FINNEY_RUNTIME_SPEC_VERSION,),
        transaction_versions: tuple[Any, ...] = (FINNEY_TRANSACTION_VERSION,),
        code_hashes: tuple[str, ...] = (FINNEY_RUNTIME_CODE_HASH,),
        genesis_hash: str = FINNEY_GENESIS_HASH,
        nonces: tuple[Any, ...] = (0,),
        mutate_call: bool = False,
        signed_mutation: tuple[str, Any] | None = None,
        charged_fee: Any = 0,
        receipt_finalized: bool = True,
        receipt_success: bool = True,
        receipt_hash: str | None = None,
        submit_error: bool = False,
    ) -> None:
        self.fees = fee if isinstance(fee, tuple) else (fee,)
        self.constants = {
            "InitialDeposit": initial_deposit,
            "FieldDeposit": field_deposit,
        }
        self.specs = specs
        self.transaction_versions = transaction_versions
        self.code_hashes = code_hashes
        self.genesis_hash = genesis_hash
        self.nonces = nonces
        self.mutate_call = mutate_call
        self.signed_mutation = signed_mutation
        self.submit_error = submit_error
        self.heads = 0
        self.identity_reads = 0
        self.nonce_reads = 0
        self.payment_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []
        self.submit_calls: list[dict[str, Any]] = []
        canonical_extrinsic_hash = "0x" + "ab" * 32
        self.receipt = SimpleNamespace(
            extrinsic_hash=receipt_hash or canonical_extrinsic_hash,
            block_hash="0x" + "cd" * 32,
            finalized=receipt_finalized,
            is_success=receipt_success,
            total_fee_amount=charged_fee,
        )

    @staticmethod
    def _at(values: tuple[Any, ...], index: int) -> Any:
        return values[min(index, len(values) - 1)]

    def get_block_hash(self, block_id: int) -> str:
        if block_id != 0:
            raise AssertionError("only genesis may be queried by the authorization guard")
        return self.genesis_hash

    def get_chain_head(self) -> str:
        self.heads += 1
        return f"0x{self.heads:064x}"

    def get_block_runtime_info(self, block_hash: str) -> dict[str, Any]:
        del block_hash
        index = self.identity_reads
        self.identity_reads += 1
        return {
            "specVersion": self._at(self.specs, index),
            "transactionVersion": self._at(self.transaction_versions, index),
        }

    def rpc_request(self, method: str, params: list[str]) -> dict[str, str]:
        if method != "state_getStorageHash" or params[0] != "0x3a636f6465":
            raise AssertionError("unexpected runtime identity RPC")
        return {"result": self._at(self.code_hashes, self.identity_reads - 1)}

    def compose_call(self, **kwargs: Any) -> Any:
        value = {
            "call_module": kwargs["call_module"],
            "call_function": kwargs["call_function"],
            "call_args": kwargs["call_params"],
        }
        if self.mutate_call:
            value["call_function"] = "transfer_allow_death"
        return SimpleNamespace(value=value)

    def get_constant(self, module: str, name: str, *, block_hash: str) -> Any:
        self.constant_request = (module, name, block_hash)
        return self.constants[name]

    def get_account_next_index(self, address: str) -> Any:
        if address != AUTHORIZED_HOTKEY_SS58:
            raise AssertionError("nonce requested for a different signer")
        value = self._at(self.nonces, self.nonce_reads)
        self.nonce_reads += 1
        return value

    def get_payment_info(self, **kwargs: Any) -> dict[str, Any]:
        kwargs["era"].setdefault("current", 1234)
        self.payment_calls.append({**kwargs, "era": dict(kwargs["era"])})
        return {"partial_fee": self._at(self.fees, len(self.payment_calls) - 1)}

    def create_signed_extrinsic(self, **kwargs: Any) -> Any:
        self.create_calls.append({**kwargs, "era": dict(kwargs["era"])})
        call = kwargs["call"].value
        account = "0x" + kwargs["keypair"].public_key.hex()
        value = {
            "account_id": account,
            "address": account,
            "asset_id": {"tip": kwargs["tip"], "asset_id": kwargs["tip_asset_id"]},
            "call": call,
            "call_args": call["call_args"],
            "call_function": call["call_function"],
            "call_module": call["call_module"],
            "era": dict(kwargs["era"]),
            "mode": "Disabled",
            "nonce": kwargs["nonce"],
            "signature": "0x" + "11" * 64,
            "signature_version": 1,
            "tip": kwargs["tip"],
        }
        if self.signed_mutation is not None:
            field, mutated = self.signed_mutation
            value[field] = mutated
        return SimpleNamespace(value=value, data="0x1234", extrinsic_hash=b"\xab" * 32)

    def submit_extrinsic(self, **kwargs: Any) -> Any:
        self.submit_calls.append(kwargs)
        if self.submit_error:
            raise RuntimeError("simulated ambiguous transport failure")
        return self.receipt


class FakeSubtensor:
    def __init__(self, substrate: FakeSubstrate) -> None:
        self.substrate = substrate


class FakeClient:
    def __init__(
        self, subtensor: FakeSubtensor, *, uid: int | tuple[int | None, ...] | None = 32
    ) -> None:
        self.netuid = 92
        self.subtensor = subtensor
        self.uids = uid if isinstance(uid, tuple) else (uid,)
        self.registration_reads = 0

    def snapshot(self, *, refresh: bool = False) -> FakeSnapshot:
        if not refresh:
            raise AssertionError("authorization registration reads must be refreshed")
        index = min(self.registration_reads, len(self.uids) - 1)
        self.registration_reads += 1
        return FakeSnapshot(self.uids[index])


class TransactionAuthorizationTests(unittest.TestCase):
    def _backend(
        self,
        root: Path,
        *,
        dry_run: bool = False,
        fee: int | tuple[Any, ...] = 0,
        initial_deposit: Any = 0,
        field_deposit: Any = 0,
        specs: tuple[Any, ...] = (FINNEY_RUNTIME_SPEC_VERSION,),
        transaction_versions: tuple[Any, ...] = (FINNEY_TRANSACTION_VERSION,),
        code_hashes: tuple[str, ...] = (FINNEY_RUNTIME_CODE_HASH,),
        genesis_hash: str = FINNEY_GENESIS_HASH,
        nonces: tuple[Any, ...] = (0,),
        uid: int | tuple[int | None, ...] | None = 32,
        mutate_call: bool = False,
        signed_mutation: tuple[str, Any] | None = None,
        charged_fee: Any = 0,
        receipt_finalized: bool = True,
        receipt_success: bool = True,
        receipt_hash: str | None = None,
        submit_error: bool = False,
    ) -> tuple[MicrotensorBackend, FakeClient, FakeSubtensor]:
        env = base_env(root, dry_run=dry_run)
        env["MMC_SOURCE_TEMPLATE"] = "https:github.com/vandungtech/mt92/releases/download/r{round}"
        if not dry_run:
            env["MMC_GITHUB_TOKEN_FILE"] = str(root / "github.token")
        config = ControllerConfig.from_env(env)
        substrate = FakeSubstrate(
            fee=fee,
            initial_deposit=initial_deposit,
            field_deposit=field_deposit,
            specs=specs,
            transaction_versions=transaction_versions,
            code_hashes=code_hashes,
            genesis_hash=genesis_hash,
            nonces=nonces,
            mutate_call=mutate_call,
            signed_mutation=signed_mutation,
            charged_fee=charged_fee,
            receipt_finalized=receipt_finalized,
            receipt_success=receipt_success,
            receipt_hash=receipt_hash,
            submit_error=submit_error,
        )
        subtensor = FakeSubtensor(substrate)
        client = FakeClient(subtensor, uid=uid)
        backend = MicrotensorBackend(config)
        backend._client = client
        backend._wallet = SimpleNamespace(
            hotkey=SimpleNamespace(ss58_address=AUTHORIZED_HOTKEY_SS58, public_key=b"\x01" * 32)
        )
        backend._hotkey = AUTHORIZED_HOTKEY_SS58
        return backend, client, subtensor

    def test_exact_zero_cost_direct_hotkey_call_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend, client, subtensor = self._backend(Path(temporary))
            backend._submit_authorized_commitment("mt1|probe")

        substrate = subtensor.substrate
        self.assertEqual(client.registration_reads, 3)
        self.assertEqual(len(substrate.payment_calls), 3)
        self.assertEqual(len(substrate.create_calls), 1)
        self.assertEqual(len(substrate.submit_calls), 1)
        estimated = substrate.payment_calls[-1]
        signed = substrate.create_calls[0]
        self.assertEqual(estimated["nonce"], signed["nonce"])
        self.assertEqual(estimated["era"], signed["era"])
        self.assertEqual(estimated["tip"], 0)
        self.assertEqual(signed["tip"], 0)
        self.assertIsNone(estimated["tip_asset_id"])
        self.assertIsNone(signed["tip_asset_id"])
        self.assertTrue(substrate.submit_calls[0]["wait_for_inclusion"])
        self.assertTrue(substrate.submit_calls[0]["wait_for_finalization"])
        self.assertEqual(
            signed["call"].value,
            {
                "call_module": "Commitments",
                "call_function": "set_commitment",
                "call_args": {
                    "netuid": 92,
                    "info": {"fields": [[{"Raw9": b"mt1|probe"}]]},
                },
            },
        )

    def test_nonzero_estimated_fee_stops_before_signing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend, _, subtensor = self._backend(Path(temporary), fee=1)
            with self.assertRaisesRegex(AuthorizationRefused, "estimated transaction fee is 1"):
                backend._submit_authorized_commitment("mt1|probe")
        self.assertEqual(subtensor.substrate.create_calls, [])

    def test_nonzero_required_deposit_stops_before_signing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend, _, subtensor = self._backend(Path(temporary), field_deposit=1)
            with self.assertRaisesRegex(AuthorizationRefused, "required commitment deposit is 1"):
                backend._submit_authorized_commitment("mt1|probe")
        self.assertEqual(subtensor.substrate.create_calls, [])

    def test_registration_change_stops_before_signing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend, _, subtensor = self._backend(Path(temporary), uid=(32, None))
            with self.assertRaisesRegex(AuthorizationRefused, "registration at UID 32"):
                backend._submit_authorized_commitment("mt1|probe")
        self.assertEqual(subtensor.substrate.create_calls, [])

    def test_runtime_change_stops_before_signing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend, _, subtensor = self._backend(Path(temporary), specs=(452, 453))
            with self.assertRaisesRegex(AuthorizationRefused, "runtime spec is 453"):
                backend._submit_authorized_commitment("mt1|probe")
        self.assertEqual(subtensor.substrate.create_calls, [])

    def test_different_composed_call_stops_before_signing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend, _, subtensor = self._backend(Path(temporary), mutate_call=True)
            with self.assertRaisesRegex(AuthorizationRefused, "composed transaction differs"):
                backend._submit_authorized_commitment("mt1|probe")
        self.assertEqual(subtensor.substrate.create_calls, [])

    def test_nonzero_receipt_fee_is_a_post_submission_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend, _, subtensor = self._backend(Path(temporary), charged_fee=1)
            with self.assertRaisesRegex(AuthorizationRefused, "nonzero fee"):
                backend._submit_authorized_commitment("mt1|probe")
        self.assertEqual(len(subtensor.substrate.submit_calls), 1)

    def test_unexpected_signed_extrinsic_field_is_refused_before_submit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend, _, subtensor = self._backend(
                Path(temporary), signed_mutation=("mev_extrinsic", object())
            )
            with self.assertRaisesRegex(AuthorizationRefused, "fields differ"):
                backend._submit_authorized_commitment("mt1|probe")
        self.assertEqual(len(subtensor.substrate.create_calls), 1)
        self.assertEqual(subtensor.substrate.submit_calls, [])

    def test_pinned_chain_identity_mismatches_stop_before_signing(self) -> None:
        cases = (
            ({"genesis_hash": "0x" + "00" * 32}, "chain genesis"),
            ({"transaction_versions": (2,)}, "runtime transaction version"),
            ({"code_hashes": ("0x" + "00" * 32,)}, "runtime code hash"),
        )
        for kwargs, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                backend, _, subtensor = self._backend(Path(temporary), **kwargs)
                with self.assertRaisesRegex(AuthorizationRefused, message):
                    backend._submit_authorized_commitment("mt1|probe")
                self.assertEqual(subtensor.substrate.create_calls, [])

    def test_authorization_amounts_require_integral_runtime_types(self) -> None:
        for value in (0.0, Decimal("0"), "0", True):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(AuthorizationRefused, "exact integer"),
            ):
                MicrotensorBackend._authorization_integer(value, "amount")
        self.assertEqual(MicrotensorBackend._authorization_integer(0, "amount"), 0)

    def test_nonce_change_after_estimate_stops_before_signing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend, _, subtensor = self._backend(Path(temporary), nonces=(0, 1))
            with self.assertRaisesRegex(AuthorizationRefused, "nonce changed"):
                backend._submit_authorized_commitment("mt1|probe")
        self.assertEqual(subtensor.substrate.create_calls, [])

    def test_refreshed_nonzero_fee_stops_before_signing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend, _, subtensor = self._backend(Path(temporary), fee=(0, 1))
            with self.assertRaisesRegex(AuthorizationRefused, "refreshed transaction fee is 1"):
                backend._submit_authorized_commitment("mt1|probe")
        self.assertEqual(subtensor.substrate.create_calls, [])

    def test_final_nonzero_fee_stops_after_signing_before_submit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend, _, subtensor = self._backend(Path(temporary), fee=(0, 0, 1))
            with self.assertRaisesRegex(AuthorizationRefused, "final transaction fee is 1"):
                backend._submit_authorized_commitment("mt1|probe")
        self.assertEqual(len(subtensor.substrate.create_calls), 1)
        self.assertEqual(subtensor.substrate.submit_calls, [])

    def test_signed_extrinsic_mutations_are_refused_before_submit(self) -> None:
        cases = (
            ("nonce", 1),
            ("tip", 1),
            ("mode", "Enabled"),
            ("asset_id", {"tip": 0, "asset_id": 7}),
            ("call", {"call_module": "Balances", "call_function": "transfer"}),
        )
        for field, value in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                backend, _, subtensor = self._backend(
                    Path(temporary), signed_mutation=(field, value)
                )
                with self.assertRaises(AuthorizationRefused):
                    backend._submit_authorized_commitment("mt1|probe")
                self.assertEqual(subtensor.substrate.submit_calls, [])

    def test_submission_and_receipt_ambiguity_are_authorization_refusals(self) -> None:
        cases = (
            ({"submit_error": True}, "outcome is ambiguous"),
            ({"receipt_finalized": False}, "did not prove finalization"),
            ({"receipt_success": False}, "did not prove success"),
            ({"receipt_hash": "0x" + "ee" * 32}, "does not identify"),
        )
        for kwargs, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                backend, _, subtensor = self._backend(Path(temporary), **kwargs)
                with self.assertRaisesRegex(AuthorizationRefused, message):
                    backend._submit_authorized_commitment("mt1|probe")
                self.assertEqual(len(subtensor.substrate.submit_calls), 1)

    def test_all_transaction_dependencies_are_exactly_pinned(self) -> None:
        versions = {
            "bittensor": BITTENSOR_VERSION,
            "bittensor-wallet": BITTENSOR_WALLET_VERSION,
            "async-substrate-interface": SUBSTRATE_INTERFACE_VERSION,
        }
        with patch(
            "microtensor_miner_controller.backend.importlib.metadata.version",
            side_effect=versions.__getitem__,
        ) as version:
            MicrotensorBackend._verify_transaction_dependencies()
        self.assertEqual({call.args[0] for call in version.call_args_list}, set(versions))
        versions["bittensor-wallet"] = "4.1.2"
        with (
            patch(
                "microtensor_miner_controller.backend.importlib.metadata.version",
                side_effect=versions.__getitem__,
            ),
            self.assertRaisesRegex(PreflightError, "bittensor-wallet"),
        ):
            MicrotensorBackend._verify_transaction_dependencies()

    def test_last_mile_refuses_hotkey_alias_and_custom_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend, _, subtensor = self._backend(Path(temporary))
            backend._hotkey = "5Different"
            with self.assertRaisesRegex(AuthorizationRefused, "authorized UID-32"):
                backend._submit_authorized_commitment("mt1|probe")
            self.assertEqual(subtensor.substrate.create_calls, [])
        with tempfile.TemporaryDirectory() as temporary:
            backend, _, subtensor = self._backend(Path(temporary))
            backend.config = replace(backend.config, endpoint="wss://example.invalid")
            with self.assertRaisesRegex(AuthorizationRefused, "no custom endpoint"):
                backend._submit_authorized_commitment("mt1|probe")
            self.assertEqual(subtensor.substrate.create_calls, [])

    def test_publish_refuses_wrong_round_source_before_payload_or_signing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend, _, subtensor = self._backend(Path(temporary))
            packaged = SimpleNamespace(
                hotkey=AUTHORIZED_HOTKEY_SS58,
                round_index=7,
                source="https:github.com/vandungtech/mt92/releases/download/r8",
            )
            with (
                patch.object(backend, "validate_commitment") as validate,
                self.assertRaisesRegex(AuthorizationRefused, "authorized round release"),
            ):
                backend.publish(packaged)
            validate.assert_not_called()
            self.assertEqual(subtensor.substrate.create_calls, [])

    def test_dry_run_never_reaches_signing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend, _, subtensor = self._backend(Path(temporary), dry_run=True)
            with self.assertRaisesRegex(AuthorizationRefused, "MMC_DRY_RUN=true"):
                backend._submit_authorized_commitment("mt1|probe")
        self.assertEqual(subtensor.substrate.create_calls, [])


if __name__ == "__main__":
    unittest.main()
