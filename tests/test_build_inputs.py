# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Staticduo contributors.
"""Fail-closed checks for the public build inputs, without Docker or CUDA."""
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InputTests(unittest.TestCase):
    def test_pinned_source_labels_are_required(self):
        module = load('prepare_build')
        info = {'Id': 'sha256:example', 'Architecture': 'arm64', 'Config': {
            'Labels': {'local.spark3.' + key + '.tree': value
                       for key, value in module.EXPECTED_TREES.items()}}}
        self.assertEqual(module.verify_image(info), info['Id'])
        info['Architecture'] = 'amd64'
        with self.assertRaisesRegex(ValueError, 'ARM64'):
            module.verify_image(info)
        info['Architecture'] = 'arm64'
        info['Config']['Labels']['local.spark3.vllm.tree'] = 'different'
        with self.assertRaisesRegex(ValueError, 'vllm'):
            module.verify_image(info)

    def test_untrusted_download_never_reaches_zip_parser(self):
        module = load('fetch_directions')
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'untrusted.pt'
            source.write_bytes(b'not the pinned artifact')
            output = Path(directory) / 'directions.safetensors'
            with patch.object(module.zipfile, 'ZipFile') as parser:
                with self.assertRaisesRegex(ValueError, 'reviewed revision'):
                    module.extract(source, output)
                parser.assert_not_called()
            self.assertFalse(output.exists())

    def test_fetched_artifact_if_available(self):
        module = load('fetch_directions')
        source = ROOT / 'build/downloads/ablation_directions.pt'
        if not source.exists():
            self.skipTest('optional pinned author download not present')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'directions.safetensors'
            receipt = module.extract(source, output)
            self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), module.OUTPUT_SHA)
            self.assertEqual(receipt['bytes'], 781776)


if __name__ == '__main__':
    unittest.main()
