"""Host proof of conservative byte bounds; actual FA2 BF16 equality in owned worker."""
import ast
from pathlib import Path
from types import SimpleNamespace as N
import unittest

SOURCE = Path(__file__).resolve().parents[3] / 'vllm_fl/dispatch/backends/vendor/metax/impl/attention/ops/decode_output.py'


def functions(torch, fa=None):
    tree = ast.parse(SOURCE.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
             ('_interval', '_disjoint', '_generic_disjoint', '_m4_disjoint', 'try_decode_into')]
    fa = fa or N(flash_attn_with_kvcache=object(), maybe_contiguous=object())
    ns = dict(torch=torch, fa=fa, flash_attn=N(__version__='2.6.3'),
              _PUBLIC=fa.flash_attn_with_kvcache, _CONTIGUOUS=fa.maybe_contiguous,
              os=N(environ={}), _count=lambda _: None)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), ns)
    return ns


class Tensor:
    def __init__(self, shape, stride, ptr, size=2, dtype='bf16'):
        self.shape, self.strides, self.ptr, self.size = shape, stride, ptr, size
        self.ndim, self.dtype = len(shape), dtype
        self.device = DEVICE
        self.requires_grad = False
    def data_ptr(self): return self.ptr
    def stride(self): return self.strides
    def element_size(self): return self.size
    def is_contiguous(self):
        step = 1
        for n, s in zip(reversed(self.shape), reversed(self.strides)):
            if n != 1 and s != step: return False
            step *= n
        return True
    def as_strided(self, shape, stride):
        return Tensor(shape, stride, self.ptr, self.size, self.dtype)


DEVICE = N(type='cuda', index=0)
TORCH = N(Tensor=Tensor, bfloat16='bf16', int32='i32')


def tensors():
    return [Tensor((4,1,16,128),(2560,2048,128,1),100000),
            Tensor((3,16,2,128),(4096,256,128,1),200000),
            Tensor((3,16,2,128),(4096,256,128,1),300000),
            Tensor((4,16,128),(2048,128,1),400000),
            Tensor((4,2),(4,1),500000,4,'i32'),
            Tensor((4,),(1,),600000,4,'i32')]


def call(ns, ts, **context):
    return ns['try_decode_into'](ns['_PUBLIC'], *ts, .125, (-1,-1), None, 0., None, **context)


class HostContract(unittest.TestCase):
    def test_bounds_against_independent_address_enumeration(self):
        import itertools
        ns = functions(TORCH)
        q,k,v,o,t,l = tensors()
        for qs in ((2560,2048,128,1), (6000,1,256,1), (1,999,1,1)):
            q.strides = qs
            for width, ts, ls in ((1,1,1), (2,4,2), (9,32,3)):
                t.shape, t.strides, l.strides = (4,width),(ts,1),(ls,)
                # Independent enumeration of every element byte, including offset views.
                spans=[]
                for x in (q,k,v,t,l):
                    addresses=[x.ptr+sum(i*s for i,s in zip(idx,x.strides))*x.size
                               for idx in itertools.product(*(range(n) for n in x.shape))]
                    spans.append((min(addresses),max(addresses)+x.size))
                for a,b in spans:
                    for begin in (a-16384,a-16383,a-1,a,a+1,b-1,b,b+1):
                        end=begin+16384
                        expected=all(end<=lo or begin>=hi for lo,hi in spans)
                        self.assertEqual(ns['_m4_disjoint'](begin,end,q,qs,k,v,3,t,t.shape,t.strides,l,l.strides),expected)

    def test_real_dispatch_aliases_fallbacks_and_no_retry(self):
        ns=functions(TORCH); launches=[]
        def launch(*args):
            launches.append(args)
            return args[12], None
        ns['fa'].flash_attn_cuda=N(fwd_kvcache=launch)
        original=ns['_generic_disjoint']; generic=[]
        def fallback(*args): generic.append(True); return original(*args)
        ns['_generic_disjoint']=fallback
        ts=tensors()
        for phase,name,expected_generic in [('decode','model.layers.0.self_attn.attn',False),
                ('decode','model.layers.41.self_attn.attn',False),('mixed','model.layers.0.self_attn.attn',True),
                ('decode','model.layers.42.self_attn.attn',True),('decode','custom',True),(None,None,True)]:
            generic.clear();self.assertTrue(call(ns,ts,phase=phase,layer_name=name))
            self.assertEqual(bool(generic),expected_generic)
        # Every input overlap is rejected before submission; adjacent boundaries pass.
        for source in (0,1,2,4,5):
            ts=tensors();ts[3].ptr=ts[source].ptr
            before=len(launches)
            self.assertFalse(call(ns,ts,phase='decode',layer_name='model.layers.3.self_attn.attn'))
            self.assertEqual(len(launches),before)
        ts=tensors();ts[3].ptr=ts[1].ptr+3*8192
        self.assertTrue(call(ns,ts,phase='decode',layer_name='model.layers.3.self_attn.attn'))
        # Existing version/type/layout guards remain authoritative.
        for change in ('version','dtype','stride','public'):
            ns=functions(TORCH);ts=tensors()
            ns['fa'].flash_attn_cuda=N(fwd_kvcache=lambda *a:self.fail('guard launched'))
            if change=='version': ns['flash_attn'].__version__='unknown'
            if change=='dtype':ts[0].dtype='f32'
            if change=='stride':ts[0].strides=(2560,2048,128,-1)
            if change=='public':ns['fa'].flash_attn_with_kvcache=object()
            self.assertFalse(call(ns,ts,phase='decode',layer_name='model.layers.3.self_attn.attn'))
        ns=functions(TORCH);count=[]
        def failed(*args): count.append(1);raise RuntimeError('submitted')
        ns['fa'].flash_attn_cuda=N(fwd_kvcache=failed)
        with self.assertRaisesRegex(RuntimeError,'submitted'):
            call(ns,tensors(),phase='decode',layer_name='model.layers.3.self_attn.attn')
        self.assertEqual(count,[1])

    def test_forward_propagates_actual_context(self):
        source = SOURCE.parents[1] / 'flash_attn.py'
        tree = ast.parse(source.read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'try_decode_into']
        self.assertEqual(len(calls), 1)
        kwargs = {k.arg: k.value for k in calls[0].keywords}
        for prefills, expected in ((0, 'decode'), (1, 'mixed')):
            local = dict(attn_metadata=N(num_prefills=prefills),
                         layer=N(layer_name='model.layers.41.self_attn.attn'))
            evaluate = lambda value: eval(compile(ast.Expression(value), str(source), 'eval'), {}, local)
            self.assertEqual(evaluate(kwargs['phase']), expected)
            self.assertEqual(evaluate(kwargs['layer_name']), local['layer'].layer_name)
            local['layer'] = N()
            self.assertIsNone(evaluate(kwargs['layer_name']))

    def test_existing_generic_proof_matches_reference(self):
        ns=functions(TORCH)
        for m in (1,2,3,4,8):
            q,k,v,o,t,l=tensors()
            q.shape=(m,1,16,128);t.shape=(m,2);l.shape=(m,)
            for begin in (0,q.ptr,q.ptr+1,k.ptr-64,v.ptr+20000,t.ptr,l.ptr,999999):
                end=begin+4096
                expected=all(ns['_disjoint'](begin,end,x,x.shape,x.stride()) for x in (q,k,v,t,l))
                self.assertEqual(ns['_generic_disjoint'](begin,end,q,q.shape,q.stride(),k,k.shape,v,v.shape,t,t.shape,t.stride(),l,l.shape,l.stride()),expected)


class DeviceContract(unittest.TestCase):
    def test_fa2_exact_output_inputs_and_sentinels(self):
        import torch
        from flash_attn import flash_attn_interface as fa
        import flash_attn
        ns=functions(torch,fa);ns['flash_attn']=flash_attn
        torch.manual_seed(20261021)
        with torch.inference_mode():
            for m,row in ((1,2560),(2,2560),(3,2560),(4,2560),(4,3000)):
                raw=torch.randn((37+m*row+31,),device='cuda',dtype=torch.bfloat16)
                q=raw.as_strided((m,1,16,128),(row,2048,128,1),37)
                kv=torch.randn((2,3,16,2,128),device='cuda',dtype=torch.bfloat16)
                table=torch.tensor([[0,1]]*m,device='cuda',dtype=torch.int32)
                lengths=torch.tensor([min(29,1+9*i) for i in range(m)],device='cuda',dtype=torch.int32)
                a=torch.full((m*2048+19,),-13.,device='cuda',dtype=torch.bfloat16)
                b=a.clone();out_a=a[:m*2048].view(m,16,128);out_b=b[:m*2048].view(m,16,128)
                before=[x.clone() for x in (raw,kv,table,lengths)]
                args=[q,kv[0],kv[1],out_a,table,lengths]
                self.assertTrue(call(ns,args))  # Original generic alias proof, same native ABI.
                args[3]=out_b
                self.assertTrue(call(ns,args,phase='decode',layer_name='model.layers.41.self_attn.attn'))
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(a,b));self.assertTrue(torch.all(b[-19:]==-13.))
                for x,saved in zip((raw,kv,table,lengths),before):self.assertTrue(torch.equal(x,saved))


if __name__ == '__main__': unittest.main()
