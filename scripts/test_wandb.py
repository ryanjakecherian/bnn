import random

import wandb

config = {
    'learning_rate': 0.2,
    'architecture': 'CNN',
    'dataset': 'CIFAR-100',
    'epochs': 10,
}


wandb.init(
    project='helloworld',
    config=config,
)

epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    metrics = {'acc': acc, 'loss': loss}

    wandb.log(metrics)

wandb.finish()
