# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Staticduo contributors.
import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import launch
import refusal

RUNTIME = json.loads((ROOT / 'runtime.json').read_text())
SITE = json.loads((ROOT / 'examples/site.example.json').read_text())
PROBES = {'page_size': '65536', 'uvm_pack': 'Y', 'drm_modeset': 'Y',
          'mem_available': 'MemAvailable: 12582912 kB', 'device': 'present',
          'model_readonly': 'yes', 'cache_writable': 'yes'}

class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.model = self.root / 'model'; self.model.mkdir()
        (self.model / 'config.json').write_text(json.dumps({
            'model_type': 'deepseek_v41', 'architectures': ['DeepseekV41ForCausalLM'],
            'text_config': {'hidden_size': 5120, 'num_hidden_layers': 40},
            'quantization_config': {'quant_method': 'fp8', 'expert_dtype': 'fp4'}}))
        self.site = copy.deepcopy(SITE); self.site['model_path'] = str(self.model)
        self.sitepath = self.root / 'site.json'; self.sitepath.write_text(json.dumps(self.site))

    def tearDown(self):
        self.tmp.cleanup()

    def native(self, rank):
        return ['--nnodes','3','--node-rank',str(rank),'--master-addr','192.0.2.10',
                '--master-port','25000'] + (['--headless'] if rank else [])

    def test_all_ranks_preserve_rendezvous_once_and_isolate_cache(self):
        for rank in range(3):
            with self.subTest(rank=rank):
                plan = launch.build_plan(self.site, RUNTIME, model=self.model,
                    node_rank=rank, probes=PROBES, distributed_args=self.native(rank))
                args=plan['argv']; env=plan['environment']
                for flag in ['--nnodes','--node-rank','--master-addr','--master-port',
                             '--distributed-executor-backend']:
                    self.assertEqual(args.count(flag),1)
                self.assertEqual(args[args.index('--master-addr')+1], '192.0.2.10')
                self.assertEqual('--headless' in args,rank>0)
                if rank: self.assertNotIn('--port',args)
                self.assertIn(f'/rank{rank}/',env['B12X_COMPILE_CACHE_DIR'])
                self.assertEqual(json.loads(env['SPARKNET_ROCE_PEER_HCAS']),self.site['nodes'][rank]['roce_peer_hcas'])
                self.assertEqual(env['VLLM_REFUSAL_DIRS'],'/opt/sparkrun-refusal/refusal_dirs_1017.safetensors')
                self.assertEqual(env['VLLM_REFUSAL_LAMBDA_INIT'],'2.5')

    def test_actual_cli_accepts_native_values_for_each_rank(self):
        for rank in range(3):
            out=io.StringIO()
            with contextlib.redirect_stdout(out):
                result=launch.main(['--model',str(self.model),'--revision',launch.EXPECTED_MODEL_REVISION,
                    '--site',str(self.sitepath),'--runtime-config',str(ROOT/'runtime.json'),
                    '--profile','maximum1M','--dry-run',*self.native(rank)], probes_provider=lambda: PROBES)
            self.assertEqual(result,0)
            self.assertEqual(json.loads(out.getvalue())['rank'],rank)

    def test_gates_reject_bad_host_and_model(self):
        for key,value in [('page_size','4096'),('uvm_pack','N'),('drm_modeset','N'),
                          ('mem_available','MemAvailable: 1 kB'),('device','missing'),
                          ('model_readonly','no'),('cache_writable','no')]:
            with self.subTest(key=key), self.assertRaises(launch.LaunchError):
                launch.preflight({**PROBES,key:value})
        (self.model/'config.json').write_text('{}')
        with self.assertRaises(launch.LaunchError):
            launch.build_plan(self.site,RUNTIME,model=self.model,node_rank=0,probes=PROBES)

    def test_binary_helper_sends_zero_or_two_point_five(self):
        for state,value in [('0',0.0),('1',2.5)]:
            with patch.object(refusal,'request_state',return_value={'lambda':value}) as request:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(refusal.main([state]),0)
                self.assertEqual(request.call_args.args[2],value)

if __name__ == '__main__': unittest.main()
