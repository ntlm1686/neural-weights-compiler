"""
NeuralCompiler: Three-Tier Neural Weight Compiler
==================================================
Program / Task Specification -> Neural IR -> Architecture Mapping -> Executable Weights
Supports:
  - Stage 1: Deterministic MLP Weight Backend (tanh / ReLU) with automatic hidden-unit allocation
  - Stage 2: Deterministic Transformer Circuit Backend (W_Q, W_K, W_V, W_O, W_fc, W_proj)
             with automatic residual-stream subspace (register/bus) allocation and architecture transfer.
"""

from .ir import MLPProgram, TransformerProgram
from .mlp_backend import MLPCompilerBackend, ExecutableMLP
from .transformer_backend import TransformerCircuitBackend, ExecutableTransformer

__all__ = [
    "MLPProgram",
    "TransformerProgram",
    "MLPCompilerBackend",
    "ExecutableMLP",
    "TransformerCircuitBackend",
    "ExecutableTransformer",
]
