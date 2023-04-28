import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = [torch.rand(512) for _ in range(10000)]
    def __getitem__(self,idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)

def collate_fn(samples):
    return {
        "input":torch.stack(samples,dim=0),
        'labels':torch.rand(len(samples))
    }


class Model(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(512,1024),
            nn.Linear(1024,2048),
            nn.Linear(2048,4096),
            nn.Linear(4096,512),
            nn.Linear(512,1),
        )
    
    def training_step(self,batch,batch_idx):
        output = self.model(batch['input'])
        loss = torch.mean(batch['labels']-output)
        return loss
    
    def training_step_end(self,output):
        print(self.local_rank,output)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(),lr=0.01)
    
    def validate_step(self,batch,batch_idx):
        output = self.model(batch['input'])
        loss = torch.mean(batch['output']-output)
        self.log('valid_loss',loss)
    
    def setup(self,stage):
        if stage == 'fit':
            self.train_dataset = RandomDataset()
            self.valid_dataet = RandomDataset()
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,collate_fn=collate_fn,batch_size=64)
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset,collate_fn=collate_fn,batch_size=64)
    

if __name__ == '__main__':
    if torch.cuda.device_count()>1:
        strategy = DDPStrategy(find_unused_parameters=False)
    trainer = pl.Trainer(
        default_root_dir='/tmp',
        accelerator='gpu',
        max_epochs=10,
        strategy=strategy,
    )
    model = Model()
    trainer.fit(model)