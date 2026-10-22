# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Staticduo contributors.
import json
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import render_recipe

class RecipeTests(unittest.TestCase):
    def test_native_recipe_pin_and_defaults(self):
        runtime=json.loads((ROOT/'runtime.json').read_text())
        site=json.loads((ROOT/'examples/site.example.json').read_text())
        recipe=render_recipe.render(runtime,site)
        for text in ['runtime: vllm-distributed','builder: docker-pull','user: "0:0"',
                     'VLLM_REFUSAL_LAMBDA_INIT: "2.5"','--revision '+render_recipe.EXPECTED_REVISION]:
            self.assertIn(text,recipe)
        self.assertNotIn('{model_revision}',recipe)
        self.assertEqual(recipe.count('memory_limit:'),1)
        self.assertNotIn('192.0.2.10',recipe)
        runtime['serve_args'][runtime['serve_args'].index('1048576')]='262144'
        with self.assertRaises(ValueError): render_recipe.render(runtime,site)

if __name__ == '__main__': unittest.main()
