"""CPU address/alias and forward fallback contract; no GPU initialization."""
import ast
import os
from pathlib import Path
import types
import unittest
import torch

SOURCE = Path(__file__).resolve().parents[3] / 'vllm_fl/models/minicpm5_rope_kv.py'


def load():
    tree=ast.parse(SOURCE.read_text())
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_qkv_head_views']
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='RopeKVAttention')
    nodes.append(cls)
    ns=dict(torch=torch,LlamaAttention=object,os=os)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),ns)
    return ns


class Views(unittest.TestCase):
    def test_independent_address_map_and_alias(self):
        helper=load()['_qkv_head_views']
        for m in (1,4,8,15,16,17,1023,1024,1025):
            for padding,offset in ((0,0),(37,11)):
                with self.subTest(m=m,padding=padding,offset=offset):
                    stride=2560+padding
                    storage=torch.arange(offset+m*stride,dtype=torch.int32).to(torch.bfloat16)
                    x=storage.as_strided((m,2560),(stride,1),offset)
                    views=helper(x)
                    for y,heads,start in zip(views,(16,2,2),(0,2048,2304)):
                        self.assertEqual(tuple(y.shape),(m,heads,128))
                        self.assertEqual(y.stride(),(stride,128,1))
                        self.assertEqual(y.storage_offset(),offset+start)
                        self.assertEqual(y.untyped_storage().data_ptr(),x.untyped_storage().data_ptr())
                        expected=(offset+start+torch.arange(m)[:,None,None]*stride
                                  +torch.arange(heads)[None,:,None]*128+torch.arange(128)[None,None,:])
                        actual=(y.storage_offset()+torch.arange(m)[:,None,None]*y.stride(0)
                                +torch.arange(heads)[None,:,None]*y.stride(1)+torch.arange(128)[None,None,:]*y.stride(2))
                        self.assertTrue(torch.equal(actual,expected))
                        self.assertTrue(torch.equal(y.view(torch.int16),storage[expected].view(torch.int16)))
                        y[-1,-1,-1]=7
                        self.assertEqual(float(x[-1,start+heads*128-1]),7)
                        x[0,start]=-9
                        self.assertEqual(float(y[0,0,0]),-9)
                    old=x.split((2048,256,256),dim=-1)
                    for a,b in zip(views,[z.view(m,h,128) for z,h in zip(old,(16,2,2))]):
                        self.assertTrue(torch.equal(a.view(torch.int16),b.view(torch.int16)))
                        self.assertEqual((a.stride(),a.storage_offset()),(b.stride(),b.storage_offset()))

    def test_unsupported_retains_dispatch(self):
        helper=load()['_qkv_head_views']
        class Sub(torch.Tensor):pass
        base=torch.empty(4,2560,dtype=torch.bfloat16)
        for x in (base.as_subclass(Sub),base.float(),base[:,::2],base.t(),base[:0],base[:,:2559],
                  base.as_strided((4,2560),(1,1)),base.expand(2,4,2560)):
            self.assertIsNone(helper(x))

    def test_forward_supported_and_separated_shapes(self):
        ns=load();x=torch.randn(4,2560,dtype=torch.bfloat16)
        ns.update(get_attention_context=lambda _: (None,None,None,None),
                  get_forward_context=lambda:types.SimpleNamespace(cudagraph_runtime_mode=None),
                  trace=lambda *a:None)
        module=ns['RopeKVAttention']();module.q_size=2048;module.kv_size=256
        module.qkv_proj=lambda _: (x,None);module.o_proj=lambda y:(y,None)
        module.rotary_emb=lambda positions,q,k:(q,k)
        module.attn=types.SimpleNamespace(layer_name='test')
        module.rotary_emb.cos_sin_cache=torch.empty(1)
        captured=[]
        def fused(pos,q,k,v,table,out,name):
            captured.append((q,k,v));out.copy_(q)
        ns.update(fused_attention=fused,fallback_reason=lambda *a:None)
        y=module.forward(torch.arange(4),None)
        self.assertTrue(torch.equal(y,x[:,:2048]))
        self.assertEqual([tuple(v.shape) for v in captured[0]],[(4,16,128),(4,2,128),(4,2,128)])
        class Attention:
            layer_name='test'
            def __call__(self,q,k,v):
                self.shapes=[tuple(t.shape) for t in (q,k,v)];return q
        module.attn=Attention();ns['fallback_reason']=lambda *a:'original-fallback'
        y=module.forward(torch.arange(4),None)
        self.assertEqual(module.attn.shapes,[(4,2048),(4,256),(4,256)])
        self.assertTrue(torch.equal(y,x[:,:2048]))


if __name__=='__main__':unittest.main()
