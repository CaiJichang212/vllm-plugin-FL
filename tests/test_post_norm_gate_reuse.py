# SPDX-License-Identifier: Apache-2.0
"""Reuse complete real42 numeric evidence after a proof-local-only correction."""
import ast,hashlib,json,os
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

class ReusedReal42(unittest.TestCase):
 def inputs(self):
    p=read(os.environ['C500_FROZEN_PROTOCOL'])
    receipts=[Path(f) for f in p['declaration']['evidence'] if f.endswith('numerical-reuse.json')]
    self.assertEqual(len(receipts),1)
    receipt=read(receipts[0]);old=Path(receipt['source'])
    for name,digest in receipt['files'].items():self.assertEqual(sha(name),digest)
    self.assertEqual(receipt['current_refs'],p['source_refs'])
    self.assertEqual(receipt['prior_refs']['gems_B'],p['source_refs']['gems_B'])
    return old,receipt
 def test_no_numeric_or_dispatch_change(self):
    old,receipt=self.inputs();old_runtime=Path(receipt['runtime'])
    for file in ('vllm_fl/models/minicpm5_post_norm_gate.py','vllm_fl/models/minicpm5_ffn_fusion.py'):
      self.assertEqual(sha(ROOT/file),sha(old_runtime/'plugin'/file))
    def forward(path,strip=False):
      tree=ast.parse(path.read_text())
      layer=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RopeKVDecoderLayer')
      fn=next(n for n in layer.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
      if strip:
        for n in ast.walk(fn):
          if isinstance(n,ast.If):n.body=[v for v in n.body if not (isinstance(v,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('layer','phase') for t in v.targets))]
      return ast.dump(fn,include_attributes=False)
    file='vllm_fl/models/minicpm5_rope_kv.py'
    self.assertEqual(forward(ROOT/file,True),forward(old_runtime/'plugin'/file))
    current_gems=ROOT.parent/'gems'
    if not current_gems.exists():current_gems=ROOT.parent/'FlagGems'
    file='src/flag_gems/runtime/backend/_metax/fused/post_norm_gate.py'
    self.assertEqual(sha(current_gems/file),sha(old_runtime/'gems'/file))
 def test_complete_original_numeric_and_cleanup_evidence(self):
    old,receipt=self.inputs();d=old/'validation'
    self.assertEqual(read(d/'result.json')['status'],'passed')
    tests=read(d/'payload/validation.json');self.assertTrue(tests['passed']);self.assertEqual(tests['tests_run'],1);self.assertEqual(tests['skipped'],0)
    life=read(d/'payload/real-model-lifecycle.json')
    self.assertEqual(life['source_refs'],receipt['prior_refs']);self.assertEqual(life['completed_requests'],4)
    self.assertTrue(life['shutdown']);self.assertTrue(life['observer_restored'])
    self.assertEqual({r['layer'] for r in life['captured']},set(range(42)))
    self.assertTrue(all(r['exact'] and r['norm_reference'] for r in life['captured']))
    samples=[json.loads(v) for v in (d/'resources.jsonl').read_text().splitlines()]
    cleanup=[v for v in samples if v.get('phase')=='after_exit'][-3:]
    self.assertEqual(len(cleanup),3);self.assertTrue(all(v['cleanup_clear'] and not v.get('owned_processes') for v in cleanup))
    self.assertEqual(samples[0]['cgroup']['oom'],samples[-1]['cgroup']['oom'])
