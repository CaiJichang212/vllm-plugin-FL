"""CPU tests execute the real forward with fake launch/attention objects."""
import ast
from pathlib import Path
from types import SimpleNamespace as N
import unittest

SOURCE=Path(__file__).resolve().parents[3]/'vllm_fl/models/minicpm5_rope_kv.py'


class Contract(unittest.TestCase):
    def forward(self,reason=None,hit=True,error=None):
        events=[]
        class T:
            shape=(4,2048);device='cuda';dtype='bf16'
            def view(self,*shape):return self
            def as_strided(self,*args):return self
            def split(self,*args,**kw):return (self,self,self)
        def proj(x):events.append('projection');return T(),None
        def out(x):events.append('out');return T(),None
        def candidate(*args):
            events.append('fusion')
            if error:raise error
            return hit
        ns=dict(torch=N(empty=lambda *a,**kw:T(),empty_like=lambda *a:T()),
                get_attention_context=lambda _: (None,None,None,None),
                get_forward_context=lambda:N(cudagraph_runtime_mode=None),
                qkv_fallback_reason=lambda *a:reason,fused_qkv_attention=candidate,
                fallback_reason=lambda *a:None,os=N(environ={}),trace=lambda *a:None,
                fused_attention=lambda *a:events.append('old_rope'),)
        tree=ast.parse(SOURCE.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RopeKVAttention');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_paired_qkv_attention_outputs')
        exec(compile(ast.Module(body=[helper,fn],type_ignores=[]),str(SOURCE),'exec'),ns)
        layer=N(qkv_proj=proj,o_proj=out,q_size=2048,kv_size=256,attn=N(layer_name='layer'),rotary_emb=N(cos_sin_cache=N(to=lambda _:None)))
        return lambda:ns['forward'](layer,positions=N(device='cuda'),hidden_states=T()),events

    def test_supported_and_fallback_and_submitted_error(self):
        call,events=self.forward();call();self.assertEqual(events,['fusion','out'])
        for reason,hit in [('hooks',True),(None,False),('prefill',True)]:
            call,events=self.forward(reason,hit);call()
            self.assertEqual(events[-3:],['projection','old_rope','out'])
        call,events=self.forward(error=RuntimeError('submitted'))
        with self.assertRaisesRegex(RuntimeError,'submitted'):call()
        self.assertEqual(events,['fusion'])

    def test_real_guard_hooks_dispatch_grad_and_decode_domain(self):
        import sys
        import types
        from unittest.mock import patch
        calls=[]
        # The real guard imports these interfaces; no framework/GPU initialization.
        fusion=types.ModuleType('contract.minicpm5_ffn_fusion')
        fusion._hooked=lambda m:bool(getattr(m,'hook',False))
        hooks=types.ModuleType('torch.nn.modules.module')
        mods=types.ModuleType('torch.nn.modules');mods.module=hooks
        linear=types.ModuleType('vllm.model_executor.layers.linear')
        class Method:pass
        linear.UnquantizedLinearMethod=Method
        q=N(quant_method=Method(),bias=None,tp_size=1,hook=False,changed=False)
        rotary=N(hook=False)
        identity=lambda m:(id(m),getattr(m,'changed',False))
        q.weight=None;rotary.cos_sin_cache=None
        self_attention=N(qkv_proj=q,rotary_emb=rotary,attn=None,_qkv_fusion_saved=(identity(q),identity(rotary)))
        state=N(grad=False,inference=True)
        fake_torch=N(is_grad_enabled=lambda:state.grad,is_inference_mode_enabled=lambda:state.inference)
        tree=ast.parse(SOURCE.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='qkv_fallback_reason')
        ns=dict(__package__='contract',_dispatch_identity=identity,torch=fake_torch,
                fallback_reason=lambda *a:None,get_forward_context=lambda:N(cudagraph_runtime_mode=None),os=N(environ={}))
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),ns)
        metadata=N(num_prefills=0,num_prefill_tokens=0,num_decodes=4,num_decode_tokens=4,num_actual_tokens=4,slot_mapping=None)
        op=types.ModuleType('flag_gems.runtime.backend._metax.fused.qkv_rope_cache');op.supported=lambda *a:True
        with patch.dict(sys.modules,{'flag_gems.runtime.backend._metax.fused.qkv_rope_cache':op,'contract.minicpm5_ffn_fusion':fusion,'torch.nn.modules':mods,'torch.nn.modules.module':hooks,'vllm.model_executor.layers.linear':linear}):
            invoke=lambda:ns['qkv_fallback_reason'](self_attention,N(ndim=1),N(shape=(4,2048)),metadata,None)
            self.assertIsNone(invoke())
            for obj,attr,value,reason in [(q,'hook',True,'hooks'),(q,'changed',True,'dispatch_changed'),(metadata,'num_prefills',1,'prefill_or_profile'),(metadata,'num_decode_tokens',5,'not_single_token_decode'),(state,'grad',True,'autograd'),(q,'bias',object(),'projection')]:
                old=getattr(obj,attr,None);setattr(obj,attr,value)
                self.assertEqual(invoke(),reason)
                setattr(obj,attr,old)
            hooks._global_forward_hooks={1:object()};self.assertEqual(invoke(),'hooks')

    def test_transfer_wraps_projection_write_attention(self):
        tree=ast.parse(SOURCE.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fused_qkv_attention')
        self.assertEqual([n.id for n in fn.decorator_list],['maybe_transfer_kv_layer'])
        self.assertFalse(any(isinstance(n,ast.Try) for n in ast.walk(fn)))


if __name__=='__main__':unittest.main()
