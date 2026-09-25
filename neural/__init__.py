"""
P7 — Neural architecture package.

The paper-faithful encoder backbone shared by both policy pathways:

    P5 NeuralState
        ↓  neural.tensors            (numpy → torch, alignment-checked)
    GAT × 3                          [PAPER] Eqs. 7-8
        ↓  neural.gat
    Transformer × 3, 8 heads         [PAPER] Eq. 11
        ↓  neural.transformer
    latent port / vessel embeddings  →  P8 (sigmoid head), P9 (LSTM decoder)

P7 owns NO policy decision, NO PPO, NO training, and NO service generation.

Public API
----------
    from neural import NeuralBackbone, ArchitectureConfig, paper_config
"""

from .backbone import (
    BackboneOutput,
    NeuralBackbone,
    load_backbone,
    save_backbone,
)
from .config import (
    PAPER_ARCHITECTURE,
    PAPER_NODE_FEATURE_DIM,
    PAPER_VESSEL_FEATURE_DIM,
    ArchitectureConfig,
    paper_config,
)
from .gat import EdgeFeatureAttention, GATLayer, GATStack
from .masks import (
    DecoderPhaseMask,
    MASK_FILL_VALUE,
    all_available,
    already_selected_port_mask,
    apply_edge_mask,
    apply_mask_to_logits,
    decoder_phase_mask,
    describe_mask,
    draft_feasible_port_mask,
    edge_padding_mask,
    encoder_only_port_mask,
    global_node_mask,
    masked_log_softmax,
    none_available,
    to_pytorch_padding_mask,
)
from .tensors import (
    GLOBAL_NODE_OFFSET,
    RESERVED_EDGE_FEATURE_DIM,
    GraphTensors,
    edge_to_ports,
    neural_state_to_tensors,
    verify_edge_alignment,
)
from .transformer import (
    EncoderOutput,
    TransformerEncoderStack,
    joint_encoder_forward,
)

__all__ = [
    # config
    "ArchitectureConfig",
    "PAPER_ARCHITECTURE",
    "PAPER_NODE_FEATURE_DIM",
    "PAPER_VESSEL_FEATURE_DIM",
    "paper_config",
    # tensors
    "GraphTensors",
    "GLOBAL_NODE_OFFSET",
    "RESERVED_EDGE_FEATURE_DIM",
    "neural_state_to_tensors",
    "verify_edge_alignment",
    "edge_to_ports",
    # gat
    "GATStack",
    "GATLayer",
    "EdgeFeatureAttention",
    # transformer
    "TransformerEncoderStack",
    "EncoderOutput",
    "joint_encoder_forward",
    # masks
    "MASK_FILL_VALUE",
    "all_available",
    "none_available",
    "to_pytorch_padding_mask",
    "apply_mask_to_logits",
    "masked_log_softmax",
    "encoder_only_port_mask",
    "draft_feasible_port_mask",
    "already_selected_port_mask",
    "global_node_mask",
    "edge_padding_mask",
    "apply_edge_mask",
    "DecoderPhaseMask",
    "decoder_phase_mask",
    "describe_mask",
    # backbone
    "NeuralBackbone",
    "BackboneOutput",
    "save_backbone",
    "load_backbone",
]
