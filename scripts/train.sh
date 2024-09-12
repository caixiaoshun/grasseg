python src/train.py experiment=farseg_resnet18 trainer.devices=[1] logger=wandb
python src/train.py experiment=farseg_resnet34 trainer.devices=[1] logger=wandb
python src/train.py experiment=farseg_resnet50 trainer.devices=[1] logger=wandb
python src/train.py experiment=farseg_resnet101 trainer.devices=[1] logger=wandb
python src/train.py experiment=fcn trainer.devices=[1] logger=wandb
python src/train.py experiment=unetmobv2 trainer.devices=[1] logger=wandb