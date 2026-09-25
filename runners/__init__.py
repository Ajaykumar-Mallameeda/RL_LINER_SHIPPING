"""
RL Liner Shipping — Shared Runner Layer.

Provides configuration validation, terminal reporting, and result
serialization used by the three user-facing entry points:

    run_pipeline.py   – end-to-end pipeline
    tune_pipeline.py  – tunable experiment script
    train_rl.py       – dedicated training interface

Internal engine modules are NOT modified by this layer.
"""
