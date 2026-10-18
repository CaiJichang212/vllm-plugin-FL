"""CPU-only whole-table address and fallback contract; no GPU import."""
import ast
from pathlib import Path
import unittest
import torch

ROOT=Path(__file__).resolve().parents[3]
OPS=ROOT/'vllm_fl/dispatch/backends/vendor/metax/impl/attention/ops/rope_kv.py'
MODEL=ROOT/'vllm_fl/models/minicpm5_rope_kv.py'

def functions(path,names,ns):
    tree=ast.parse(path.read_text())
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns

class Table(unittest.TestCase):
    def setUp(self):
        self.ns=functions(OPS,{'_table_supported','_check_layout','_fused_rope_cache_table','fused_rope_cache'},{'torch':torch})
    def test_addresses_and_storage_offset(self):
        for dtype in (torch.bfloat16,torch.float16,torch.float32):
            for rows,cols,off in ((128,1,0),(280,2,13),(300,2,15)):
                backing=torch.arange(5000,dtype=torch.float32).to(dtype)
                t=torch.as_strided(backing,(9,128),(rows,cols),off)
                self.assertTrue(self.ns['_table_supported'](t))
                c,s=t.chunk(2,-1)
                for i in range(9):
                    for d in range(64):
                        self.assertEqual(c.storage_offset()+i*c.stride(0)+d*c.stride(1),off+i*rows+d*cols)
                        self.assertEqual(s.storage_offset()+i*s.stride(0)+d*s.stride(1),off+i*rows+(64+d)*cols)
                        self.assertEqual(float(s[i,d]),float(backing[off+i*rows+(64+d)*cols]))
    def test_fallback_domain(self):
        class Sub(torch.Tensor):pass
        for t in (torch.empty(0,128),torch.empty(9,64),torch.empty(9,128,dtype=torch.int64),torch.empty(9,128).as_subclass(Sub),torch.empty(128).expand(9,128),torch.as_strided(torch.empty(1024),(9,128),(2,1))):
            self.assertFalse(self.ns['_table_supported'](t))
        self.assertFalse(self.ns['_table_supported'](torch.empty(9,128),torch.empty(1).as_subclass(Sub)))
    def test_whole_and_original_dispatch(self):
        calls=[]
        self.ns['_fused_rope_cache']=lambda *a,**k:calls.append((a,k)) or a[0]
        q=torch.empty(0,16,128);k=torch.empty(0,2,128);v=k.clone();pos=torch.empty(0,dtype=torch.long);t=torch.empty(9,128);kc=torch.empty(1,16,2,128);vc=kc.clone();slots=pos
        self.assertIs(self.ns['_fused_rope_cache_table'](q,k,v,pos,t,kc,vc,slots),q)
        self.assertIs(calls[-1][0][4],t);self.assertIs(calls[-1][0][5],t)
        self.assertEqual(calls[-1][1],dict(num_warps=8,whole_table=True))
        c,s=t.chunk(2,-1);self.ns['fused_rope_cache'](q,k,v,pos,c,s,kc,vc,slots,num_warps=8)
        self.assertEqual(calls[-1][1],dict(num_warps=8))
        with self.assertRaises(ValueError):self.ns['_fused_rope_cache_table'](q,k,v,pos,t[:,:64],kc,vc,slots)
        self.assertEqual(len(calls),2)
    def test_kernel_offset_and_plugin_fallback(self):
        source=OPS.read_text();self.assertIn('SIN_OFFSET: tl.constexpr = 0',source)
        self.assertIn('SIN + SIN_OFFSET + pos * SS + (d % 64) * SD',source)
        self.assertIn('SIN_OFFSET=64 * cos.stride(1) if whole_table else 0',source)
        tree=ast.parse(MODEL.read_text());f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='fused_attention')
        f.decorator_list=[]
        events=[]
        class Cache:
            def unbind(self,axis):return ('kc','vc')
        class Impl:
            def forward(self,*a,**k):events.append('attention')
        from types import SimpleNamespace as N
        layer=N(impl=Impl());meta=N(slot_mapping='slots');positions=N(flatten=lambda:'pos')
        ns=dict(get_attention_context=lambda name:(meta,layer,Cache(),None),NUM_WARPS=8,trace=lambda *a:None,
                _table_supported=lambda *a:True,_fused_rope_cache_table=lambda *a,**k:events.append('whole'),fused_rope_cache=lambda *a,**k:events.append('split'))
        exec(compile(ast.Module(body=[f],type_ignores=[]),'<model>','exec'),ns)
        class Whole:
            def chunk(self,*a,**k):events.append('chunk');return ('cos','sin')
        ns['fused_attention'](positions,'q','k','v',Whole(),'out','layer');self.assertEqual(events,['whole','attention'])
        events.clear();ns['_table_supported']=lambda *a:False
        ns['fused_attention'](positions,'q','k','v',Whole(),'out','layer');self.assertEqual(events,['chunk','split','attention'])

if __name__=='__main__':unittest.main()
