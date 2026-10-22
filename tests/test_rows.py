import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "refusal_rows_under_test", ROOT / "payload" / "refusal_rows.py"
)
rows = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rows)


class RowsTests(unittest.TestCase):
    def test_streaming_update_checks_strength_before_slot_removal(self):
        state = rows.RefusalRows(2, 8, "cpu")
        state.set_request(0, 1, "live")
        state.check_update("live", 1)
        with self.assertRaisesRegex(ValueError, "cannot change"):
            state.check_update("live", 2.5)
        self.assertEqual(state._values["live"], 1)
        self.assertEqual(float(state.request_values[0]), 1)
        state.remove_request("live")
        state.set_request(1, 1, "live")
        state.check_update("live", 1)
        with self.assertRaisesRegex(ValueError, "no existing"):
            state.check_update("unknown", 1)

    def test_final_gpu_boundaries_override_cpu_bounds_and_reorder(self):
        state = rows.RefusalRows(4, 16, "cpu")
        state.set_request(3, 2.5, "a")
        state.set_request(1, 0, "b")
        batch = SimpleNamespace(
            num_reqs=2,
            num_tokens_after_padding=8,
            query_start_loc=torch.tensor([0, 2, 7]),
            idx_mapping=torch.tensor([3, 1]),
            query_start_loc_np=[0, 4, 7],
            num_scheduled_tokens=[4, 3],
            is_padding=torch.tensor([False, False, False, False, True, False, False, True]),
        )
        pointer = state.rows.data_ptr()
        result = state.fill(batch)
        torch.testing.assert_close(result, torch.tensor([2.5, 2.5, 0, 0, 0, 0, 0, 0]))
        batch.idx_mapping = torch.tensor([1, 3])
        result = state.fill(batch)
        torch.testing.assert_close(result, torch.tensor([0, 0, 2.5, 2.5, 0, 2.5, 2.5, 0]))
        self.assertEqual(pointer, result.data_ptr())

    def test_remove_reuse_and_dummy_rows_are_zero(self):
        state = rows.RefusalRows(2, 8, "cpu")
        state.set_request(0, 1, "a")
        state.remove_request("a")
        self.assertEqual(float(state.request_values[0]), 0)
        state.set_request(0, 4, "b")
        state.rows.fill_(4)
        pointer = state.rows.data_ptr()
        self.assertTrue(torch.equal(state.dummy(6), torch.zeros(6)))
        self.assertEqual(pointer, state.rows.data_ptr())

    def test_invalid_strength_and_capacity_reject(self):
        state = rows.RefusalRows(2, 8, "cpu")
        for value in (float("nan"), float("inf"), -1, 4.01):
            with self.subTest(value=value), self.assertRaises(ValueError):
                state.set_request(0, value, "a")
        with self.assertRaises(ValueError):
            state.set_request(2, 1, "a")
        with self.assertRaises(ValueError):
            state.dummy(9)

    def test_live_request_strength_cannot_change(self):
        state = rows.RefusalRows(2, 8, "cpu")
        state.set_request(0, 1, "streaming")
        with self.assertRaises(ValueError):
            state.set_request(0, 2.5, "streaming")
        state.set_request(1, 1, "streaming")
        self.assertEqual(float(state.request_values[0]), 0)
        self.assertEqual(float(state.request_values[1]), 1)


if __name__ == "__main__":
    unittest.main()
