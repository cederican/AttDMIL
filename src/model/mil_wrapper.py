import torch as th
from src.modules.ModelWrapperABC import ModelWrapper
from src.model.mil import MILModel

class AttDMILWrapper(ModelWrapper):
    def __init__(
            self,
            *,
            model: MILModel,
            config: dict,
            epochs: int,

    ):
        super().__init__()
        self.model = model
        self.config = config
        self.epochs = epochs

        if config.ckpt_path is not None:
            self.load_model_checkpoint(config.ckpt_path)
    
    def init_val_metrics(self):
        self.val_metrics = {
            "val/precision": ,
            "val/recall": ,
            "val/f1": ,
            "val/accuracy": ,
            "val/loss": ,
        }

    def configure_optimizers(self):
        print(f"Using base learning rate: {self.config.lr}")
        optimizer = th.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay
        )
        # optional: add lr_scheduler
        lr_scheduler = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config.T_0,
            T_mult=self.config.T_mult,
            eta_min=self.config.eta_min
        )
        return [optimizer], [lr_scheduler]
    
    def training_step(
            self,
            model: MILModel,
            batch: tuple,
            optimizer: list[th.optim.Optimizer],
            lr_scheduler: list[th.optim.lr_scheduler._LRScheduler],
            global_step: int,
    ):  
        # adjust for special case with basgs
        bag, label = batch
        output = model(bag)
        loss = model.loss(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        return {"loss": loss} ############## check if this is correct
    
    def validation_step(
            self,
            model: MILModel,
            batch: tuple,
            global_step: int,
    ):
        bag, label = batch
        output = model(bag)
        loss = model.loss(output, label)
        return {"loss": loss}
    
    def predict_step(
            self,
            model: MILModel,
            batch: tuple,
            global_step: int,
    ):
        bag, label = batch
        output = model(bag)
        loss = model.loss(output, label)
        return {"loss": loss}
        

        

        





    def load_model_checkpoint(self, ckpt_path):
        try:
            model_state = th.load(ckpt_path) 
            self.model.load_state_dict(model_state["model"])
            print(f"Model loaded from {ckpt_path}")
        except Exception as e:
            print(f"Error loading model from {ckpt_path}: {e}")
