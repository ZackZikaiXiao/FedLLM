from torch import nn
from transformers import Trainer

def compute_model_difference_loss(original_model, updated_model):
    # define how to compute difference loss
    pass
class FedProxTrainer(Trainer):
    def get_previous_global_model(self, model):
        self.previous_global_model = model
        
    def compute_loss(self, model, inputs, return_outputs=False):
        (loss, outputs) = super().compute_loss(model, inputs, return_outputs=True)
        # outputs = model(**inputs)
        # loss = outputs.get("loss")
        loss += compute_model_difference_loss(original_model=self.previous_global_model, updated_model=model)
        return (loss, outputs) if return_outputs else loss