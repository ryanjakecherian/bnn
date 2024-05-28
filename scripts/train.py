import bnn
import bnn.config
import hydra
import omegaconf
import torch
import tqdm

import wandb

omegaconf.OmegaConf.register_new_resolver(
    name='sandwich_list',
    resolver=bnn.config.sandwich_list,
)


def train(
    TBNN: bnn.network.TernBinNetwork,
    loss_func: bnn.loss.LossFunction,
    DL: bnn.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    log_rate: int,
    train_epochs: int,
):
    zero_loss_count = 0
    zero_loss_count_for_early_stop = 10

    for epoch in tqdm.trange(train_epochs):
        metrics = {}
        assert bnn.network.network_params_al_ternary(TBNN)

        epoch_loss = 0
        for batch in DL:
            # forward pass and loss
            output = TBNN.forward(batch.input)
            loss = loss_func.forward(output=output, target=batch.target)

            # backward pass
            grad = loss_func.backward(output=output, target=batch.target)
            TBNN.backward(grad)

            # optimizer step
            optimizer.step(metrics)

            # sum loss
            epoch_loss += loss

        if epoch_loss == 0:
            zero_loss_count += 1
        else:
            zero_loss_count = 0

        # metrics
        metrics['loss'] = epoch_loss
        if (epoch % log_rate) == 0:
            wandb.log(metrics)

        early_exit = zero_loss_count >= zero_loss_count_for_early_stop

        if early_exit or (epoch % log_rate) == 0:
            print(f'epoch: {epoch}\tloss: {loss}')

        if early_exit:
            break


@hydra.main(config_path='../config', config_name='main', version_base=None)
def main(cfg: omegaconf.DictConfig):
    # resolve config
    omegaconf.OmegaConf.resolve(cfg)

    wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)

    network: bnn.network.TernBinNetwork = hydra.utils.instantiate(cfg.network.model)
    data_loader: bnn.data.DataLoader = hydra.utils.instantiate(cfg.data.data_loader)
    loss_func: bnn.loss.LossFunction = hydra.utils.instantiate(cfg.loss)
    optim: torch.optim.Optimizer = hydra.utils.instantiate(
        config=cfg.optimizer,
        params=network.parameters(),
    )

    network._initialise(W_mean=0, W_zero_prob=0.5)

    train(
        TBNN=network,
        loss_func=loss_func,
        DL=data_loader,
        optimizer=optim,
        train_epochs=cfg.train.epochs,
        log_rate=cfg.train.log_rate,
    )


if __name__ == '__main__':
    main()
