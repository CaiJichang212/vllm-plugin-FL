"""CPU address proof and owned-worker actual BF16/FA2 equivalence."""
import ast
import itertools
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

SOURCE = Path(__file__).resolve().parents[3] / 'vllm_fl/dispatch/backends/vendor/metax/impl/attention/flash_attn.py'


def functions(torch):
    tree = ast.parse(SOURCE.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("decode_query_view", "original_query_view")]
    ns = {'torch': torch}
    exec(compile(tree, str(SOURCE), 'exec'), ns)
    return ns


class Tensor:
    def __init__(self, shape=(4,16,128), stride=(2560,128,1), offset=40):
        self.shape, self.strides, self.offset = shape, stride, offset
        self.ndim, self.device, self.dtype = len(shape), NS(type='cuda'), 'bf16'
        self.requires_grad = False
        self.views = []
    def stride(self): return self.strides
    def as_strided(self, shape, stride):
        self.views.append((shape,stride)); return Tensor(shape,stride,self.offset)


class HostContract(unittest.TestCase):
    def test_addresses_and_single_view_for_padded_prefix(self):
        ns = functions(NS(Tensor=Tensor,bfloat16='bf16'))
        for rows,step,head,offset,layer in itertools.product((4,7),(2048,2560,3072),(128,136),(0,40,56),(0,41)):
            q=Tensor((rows,16,128),(step,head,1),offset)
            out=ns['decode_query_view'](q,4,4,lambda *_:self.fail('fallback'),'decode',f'model.layers.{layer}.self_attn.attn')
            self.assertEqual(len(q.views),1)
            self.assertEqual(out.strides,(step,16*head,head,1))
            # Independent address enumeration of slice then leading-axis split.
            expected=[offset+b*step+h*head+d for b in range(4) for h in range(16) for d in range(128)]
            actual=[out.offset+sum(i*s for i,s in zip(idx,out.strides)) for idx in itertools.product(*(range(n) for n in out.shape))]
            self.assertEqual(actual,expected)
    def test_all_domain_guards_retain_original_path(self):
        ns=functions(NS(Tensor=Tensor,bfloat16='bf16')); calls=[]
        def fallback(*args): calls.append(args);return 'original'
        ns['original_query_view']=fallback
        for kind in ('mixed','spec','short','dtype','device','grad','subclass','negative','tail','layer','unknown'):
            q=Tensor();tokens=decodes=4;phase='decode';name='model.layers.0.self_attn.attn'
            if kind=='mixed':phase='mixed'
            if kind=='spec':tokens=8
            if kind=='short':q.shape=(3,16,128)
            if kind=='dtype':q.dtype='fp32'
            if kind=='device':q.device.type='cpu'
            if kind=='grad':q.requires_grad=True
            if kind=='subclass':q=type('Sub',(Tensor,),{})()
            if kind=='negative':q.strides=(-2560,128,1)
            if kind=='tail':q.strides=(2560,128,2)
            if kind=='layer':name='model.layers.42.self_attn.attn'
            if kind=='unknown':name=None
            self.assertEqual(ns['decode_query_view'](q,tokens,decodes,None,phase,name),'original')
            self.assertEqual(q.views,[])
            self.assertEqual(calls[-1][-1],'outside_ordinary_m4_decode')
    def test_fallback_keeps_prefix_and_reshape(self):
        ns=functions(NS()); seen=[]
        class Prefix:
            def __getitem__(self,key):seen.append(key);return 'prefix'
        self.assertEqual(ns['original_query_view'](Prefix(),3,1,lambda q,n:(q,n),'test'),('prefix',1))
        self.assertEqual(seen,[slice(None,3)])
    def test_common_outer_and_branch_have_same_traceable_file(self):
        # The A/B observer requires both targets in one file. A helper-only
        # outer target cannot possibly be called by the unmodified A source.
        import sys
        tree=ast.parse(SOURCE.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FlashAttentionImpl')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        # Compile the real method unchanged, deferring dependency annotations.
        cls.body=[method];cls.bases=[];cls.decorator_list=[]
        unit=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),cls],type_ignores=[])
        ast.fix_missing_locations(unit);ns={}
        exec(compile(unit,str(SOURCE),'exec'),ns)
        forward=ns['FlashAttentionImpl'].forward
        self.assertEqual(forward.__qualname__,'FlashAttentionImpl.forward')
        calls=[]
        def profile(frame,event,arg):
            if event=='call' and frame.f_code.co_qualname=='FlashAttentionImpl.forward':
                calls.append(frame.f_code.co_filename)
        previous=sys.getprofile()
        try:
            sys.setprofile(profile)
            # Real method early profiling path, no tensor/library import.
            out=NS(fill_=lambda value: ('profile',value))
            self.assertEqual(forward(None,None,None,None,None,None,None,output=out),('profile',0))
        finally:sys.setprofile(previous)
        self.assertEqual(calls,[str(SOURCE)])
        helper=functions(NS(Tensor=Tensor,bfloat16='bf16'))['decode_query_view']
        self.assertEqual(helper.__code__.co_filename,forward.__code__.co_filename)

    def test_forward_calls_helper_with_actual_metadata(self):
        p=SOURCE;tree=ast.parse(p.read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='decode_query_view']
        self.assertEqual(len(calls),1)
        c=calls[0]
        for prefills in (0,1):
            values=dict(query='q',num_decode_tokens=4,attn_metadata=NS(num_decodes=4,num_prefills=prefills),reshape_query_for_spec_decode='original',layer=NS(layer_name='model.layers.41.self_attn.attn'))
            args=[eval(compile(ast.Expression(a),str(p),'eval'),{},values) for a in c.args]
            self.assertEqual(args,['q',4,4,'original','decode' if prefills==0 else 'mixed','model.layers.41.self_attn.attn'])


class DeviceContract(unittest.TestCase):
    def test_actual_bf16_view_and_public_fa2_outputs_exact(self):
        import torch
        from vllm.v1.attention.backends.utils import reshape_query_for_spec_decode as original
        from flash_attn import flash_attn_interface as fa
        ns=functions(torch);torch.manual_seed(20261021)
        with torch.inference_mode():
            for rows,row_stride in ((4,2560),(7,2560),(7,3072)):
                raw=torch.randn((40+rows*row_stride+32,),device='cuda',dtype=torch.bfloat16)
                q=raw.as_strided((rows,16,128),(row_stride,128,1),40)
                saved=raw.clone()
                a=original(q[:4],4)
                b=ns['decode_query_view'](q,4,4,original,'decode','model.layers.41.self_attn.attn')
                self.assertEqual(a.stride(),b.stride());self.assertEqual(a.data_ptr(),b.data_ptr())
                self.assertTrue(torch.equal(a,b))
                kv=torch.randn((2,16,16,2,128),device='cuda',dtype=torch.bfloat16)
                saved_kv=kv.clone()
                table=torch.tensor([[0,1,2,3]]*4,device='cuda',dtype=torch.int32)
                lens=torch.tensor([1,21,41,61],device='cuda',dtype=torch.int32)
                def attend(x):
                    out=fa.flash_attn_with_kvcache(x,kv[0],kv[1],block_table=table,cache_seqlens=lens,softmax_scale=.125,causal=True)
                    torch.cuda.synchronize();return out
                self.assertTrue(torch.equal(attend(a),attend(b)))
                self.assertTrue(torch.equal(raw,saved));self.assertTrue(torch.equal(kv,saved_kv))
                # Out-of-domain route uses the real original implementation.
                self.assertTrue(torch.equal(ns['decode_query_view'](q,4,4,original,'mixed',None),a))


if __name__=='__main__':unittest.main()
