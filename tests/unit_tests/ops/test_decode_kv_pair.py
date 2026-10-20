"""Address proof on host; actual BF16 KV mutation and FA2 test in owned worker."""
import ast
import itertools
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

SOURCE = Path(__file__).resolve().parents[3] / 'vllm_fl/models/minicpm5_rope_kv.py'


def functions(torch):
    tree = ast.parse(SOURCE.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('decode_kv_pair', 'original_kv_pair')]
    ns = {'torch': torch}
    exec(compile(tree, str(SOURCE), 'exec'), ns)
    return ns


class Tensor:
    def __init__(self, shape=(2,3,16,2,128), stride=(12288,4096,256,128,1), offset=37):
        self.shape, self.strides, self.offset = shape, stride, offset
        self.ndim, self.dtype = len(shape), 'bf16'
        self.device = 'cuda'
        self.requires_grad = False
        self.views = []
    @property
    def device(self): return self._device
    @device.setter
    def device(self, value):
        class Device:
            def __init__(self, kind): self.type = kind
            def __eq__(self, other): return self.type == other.type
        self._device = Device(value)
    def stride(self): return self.strides
    def storage_offset(self): return self.offset
    def as_strided(self, shape, stride, offset):
        self.views.append((shape,stride,offset))
        return Tensor(shape,stride,offset)
    def unbind(self, dim): return ('fallback',dim)


class HostContract(unittest.TestCase):
    def test_each_address_matches_independent_axis_selection(self):
        ns=functions(NS(Tensor=Tensor,bfloat16='bf16'))
        for offset,blocks,gap,layer in itertools.product((0,37),(1,3),(0,1024),(0,41)):
            stride=(blocks*4096+gap,4096,256,128,1)
            cache=Tensor((2,blocks,16,2,128),stride,offset)
            x=Tensor((4,2560))
            views=ns['decode_kv_pair'](cache,x,'decode',f'model.layers.{layer}.self_attn.attn')
            self.assertEqual(len(cache.views),2)
            for which,view in enumerate(views):
                for idx in itertools.product(range(blocks),range(16),range(2),range(128)):
                    expected=offset+sum(i*s for i,s in zip((which,*idx),stride))
                    actual=view.offset+sum(i*s for i,s in zip(idx,view.strides))
                    self.assertEqual(actual,expected)

    def test_all_domain_boundaries_keep_public_fallback(self):
        ns=functions(NS(Tensor=Tensor,bfloat16='bf16'))
        for kind in ('phase','layer','name','m1','width','dtype','device','grad','xgrad','subclass','xsubclass','rank','first','tail','empty','stride'):
            cache=Tensor();x=Tensor((4,2560));phase='decode';name='model.layers.0.self_attn.attn'
            if kind=='phase':phase='mixed'
            if kind=='layer':name='model.layers.42.self_attn.attn'
            if kind=='name':name=None
            if kind=='m1':x.shape=(1,2560)
            if kind=='width':x.shape=(4,2048)
            if kind=='dtype':cache.dtype='fp32'
            if kind=='device':cache.device='cpu'
            if kind=='grad':cache.requires_grad=True
            if kind=='xgrad':x.requires_grad=True
            if kind=='subclass':cache=type('Sub',(Tensor,),{})()
            if kind=='xsubclass':x=type('Sub',(Tensor,),{})((4,2560))
            if kind=='rank':cache.ndim=4
            if kind=='first':cache.shape=(3,3,16,2,128)
            if kind=='tail':cache.shape=(2,3,16,2,64)
            if kind=='empty':cache.shape=(2,0,16,2,128)
            if kind=='stride':cache.strides=(12288,4096,256,128,0)
            self.assertEqual(ns['decode_kv_pair'](cache,x,phase,name),('fallback',0),kind)
            self.assertEqual(cache.views,[],kind)

    def test_common_outer_and_candidate_branch_are_traceable_for_all_layers(self):
        import sys
        tree=ast.parse(SOURCE.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RopeKVAttention')
        cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward']
        cls.bases=[];cls.decorator_list=[]
        unit=ast.Module(body=[cls],type_ignores=[])
        ns=functions(NS(Tensor=Tensor,bfloat16='bf16'))
        ns.update(qkv_fallback_reason=lambda *a:None,
                  torch=NS(empty=lambda *a,**k:NS(view=lambda *a:'output')))
        # Helper's globals keep the tensor facade; forward uses allocation stub.
        helper=functions(NS(Tensor=Tensor,bfloat16='bf16'))['decode_kv_pair']
        ns['get_attention_context']=lambda name:(None,None,Tensor(),None)
        def fused(self,positions,hidden,output,name):
            helper(Tensor(),Tensor((4,2560)),'decode',name)
            return True
        ns['fused_qkv_attention']=fused
        exec(compile(unit,str(SOURCE),'exec'),ns)
        outer=[];seen=set()
        branch_line=next(n.lineno for n in ast.walk(tree) if isinstance(n,ast.Assign)
                         and any(isinstance(t,ast.Name) and t.id=='key' for t in n.targets))
        def profile(frame,event,arg):
            if event=='call' and frame.f_code.co_qualname=='RopeKVAttention.forward':
                outer.append(frame.f_code.co_filename)
        def trace(frame,event,arg):
            if event=='line' and frame.f_code.co_qualname=='decode_kv_pair' and frame.f_lineno==branch_line:
                local=frame.f_locals
                seen.add((tuple(local['x'].shape),local['phase'],local['layer']))
            return trace
        previous_profile=sys.getprofile();previous_trace=sys.gettrace()
        try:
            sys.setprofile(profile);sys.settrace(trace)
            for layer in range(42):
                obj=NS(attn=NS(layer_name=f'model.layers.{layer}.self_attn.attn'),
                       o_proj=lambda value:('projected',None),q_size=2048)
                hidden=NS(shape=(4,2048),device='cuda',dtype='bf16')
                self.assertEqual(ns['RopeKVAttention'].forward(obj,None,hidden),'projected')
        finally:
            sys.settrace(previous_trace);sys.setprofile(previous_profile)
        self.assertEqual(outer,[str(SOURCE)]*42)
        self.assertEqual(seen,{((4,2560),'decode',layer) for layer in range(42)})

    def test_real_call_uses_qkv_and_transfer_context(self):
        tree=ast.parse(SOURCE.read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='decode_kv_pair']
        self.assertEqual(len(calls),1)
        for prefills in (0,1):
            scope=dict(cache='kv',qkv='qkv',metadata=NS(num_prefills=prefills),layer_name='model.layers.41.self_attn.attn')
            args=[eval(compile(ast.Expression(n),str(SOURCE),'eval'),{},scope) for n in calls[0].args]
            self.assertEqual(args,['kv','qkv','decode' if prefills==0 else 'mixed',scope['layer_name']])
        outer=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RopeKVAttention')
        forward=next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='fused_qkv_attention' for n in ast.walk(forward)))
        self.assertEqual(functions(NS())['decode_kv_pair'].__code__.co_filename,str(SOURCE))


class DeviceContract(unittest.TestCase):
    def test_actual_bf16_views_mutation_and_attention_exact(self):
        import torch
        from flash_attn import flash_attn_interface as fa
        ns=functions(torch)
        torch.manual_seed(20261021)
        with torch.inference_mode():
            x=torch.empty((4,2560),device='cuda',dtype=torch.bfloat16)
            for offset,gap in ((0,0),(37,1024),(40,1024)):
                strides=(16*4096+gap,4096,256,128,1)
                raw=torch.randn((offset+2*strides[0]+128,),device='cuda',dtype=torch.bfloat16)
                cache=raw.as_strided((2,16,16,2,128),strides,offset)
                saved=raw.clone()
                a=cache.unbind(0)
                b=ns['decode_kv_pair'](cache,x,'decode','model.layers.41.self_attn.attn')
                for left,right in zip(a,b):
                    self.assertEqual(left.data_ptr(),right.data_ptr())
                    self.assertEqual(left.stride(),right.stride())
                    self.assertTrue(torch.equal(left,right))
                b[1][0,0,0,0]=1.25
                self.assertEqual(a[1][0,0,0,0].item(),1.25)
                # Restore then check FA2 reads identical cache values/layouts.
                raw.copy_(saved)
                q=torch.randn((4,1,16,128),device='cuda',dtype=torch.bfloat16)
                table=torch.tensor([[0,1,2,3]]*4,device='cuda',dtype=torch.int32)
                lens=torch.tensor([1,21,41,61],device='cuda',dtype=torch.int32)
                def attend(pair):
                    out=fa.flash_attn_with_kvcache(q,*pair,block_table=table,cache_seqlens=lens,softmax_scale=.125,causal=True)
                    torch.cuda.synchronize();return out
                # Keep arbitrary-offset alias coverage, but never submit a
                # non-vector-aligned cache to native FA2. The prior alias
                # candidate preserved an illegal-address failure at BF16
                # offset 37; offset 40 exercises nonzero aligned FA2 input.
                if all(t.data_ptr() % 16 == 0 for t in (*a, *b)):
                    self.assertTrue(torch.equal(attend(a),attend(b)))
                self.assertTrue(torch.equal(raw,saved))
                fallback=ns['decode_kv_pair'](cache,x,'mixed',None)
                self.assertTrue(all(torch.equal(t,u) for t,u in zip(a,fallback)))


if __name__=='__main__': unittest.main()
