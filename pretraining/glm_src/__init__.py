"""Standalone pretraining stack for scDNAm-GPM.

Bundled from the internal scDNAm_GLM project so that pretraining works
without touching the repository's existing src/ package (used by
fine-tuning/inference). Only the internal imports are rewritten to
`glm_src.*`; the code is otherwise identical.
"""
