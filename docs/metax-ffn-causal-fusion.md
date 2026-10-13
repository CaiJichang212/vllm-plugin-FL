# Experimental MiniCPM gate/up SiLU fusion

Base plugin: 910375835be4ca0ad30b1ac5135402f84b3d7b55. Requires the explicit
FlagGems gate_up_silu candidate descended from U
(a1304d30b78c80411e8534185e7d0526d26fdd2d).

Bind the existing LlamaMLP instance after RopeKVDecoderLayer construction.
The original bound forward and Parameter/state_dict identities remain intact.
Bypass only gate_up_proj and act_fn; invoke down_proj normally. Unsupported
module replacement, changed dispatch, hooks, adapters, quantization, gradients,
bias, TP/PP, batch invariance, compile or graph mode uses complete original FFN.
The operator guards layout/dtype/alignment and supports the predeclared row domain
1 through 1024. Weight contents and pointers are never cached.

CPU fallback tests are outer-repository tests; they are not vLLM/GPU validation.
No GPU build, numerical or performance result is claimed by this commit.
