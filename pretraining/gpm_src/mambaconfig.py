from transformers import PretrainedConfig
# Define MambaConfig
class MambaConfig(PretrainedConfig):
    model_type = "mamba"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)