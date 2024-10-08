from typing import overload

import torch
from src.models.ae_module import AELitModule


class DiffusionDemoLitModule(AELitModule):
    def __init__(
        self,
        net: torch.nn.Module,
        num_classes: int,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        average: str = "macro",
        compile: bool = False,
        experiment_name="experiment"
    ):
        super().__init__(
            net=net,
            num_classes=num_classes,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            average=average,
            compile=compile,
            experiment_name=experiment_name
        )
        self.example_input_array = None
    
    def forward(self, x:torch.Tensor):
        t = torch.randint(0, self.net.num_steps, (x.size(0),)).long().to(x.device)
        return self.net(x, t)
    