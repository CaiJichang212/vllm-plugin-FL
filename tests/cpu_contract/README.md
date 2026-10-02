# Iluvatar paged prefill CPU contract checks

Run from this repository without importing PyTorch, vLLM or FlagGems:

```sh
python3 tests/cpu_contract/test_iluvatar_paged_prefill.py -v
```

The tests load the actual adapter and FlagGems selector modules with standard-library
runtime stubs. The platform routing test compiles the actual method AST because
loading the entire platform module initializes unrelated device integrations.
Do not run through pytest on a busy GPU: the repository root conftest imports the
device runtime.

These checks cover launch guards, argument identity, inherited KV update/layout,
pure-decode fallback, propagation of candidate failures, legacy/vendor routing,
and explicit backend selection. They do not establish numerical correctness,
negative-slot KV behavior, Graph compatibility or performance. Those require the
separate real-runtime verification and integration drivers after installation.
