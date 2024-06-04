import bnn
import bnn.config
import bnn.metrics
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
        epoch_proportion_flipped = 0
        for batch in DL:
            # forward pass and loss
            output = TBNN.forward(batch.input)
            loss = loss_func.forward(output=output, target=batch.target)

            # backward pass
            grad = loss_func.backward(output=output, target=batch.target)
            TBNN.backward(grad)

            # optimizer step
            epoch_proportion_flipped += optimizer.step()

            # sum loss
            epoch_loss += loss

        if epoch_loss == 0:
            zero_loss_count += 1
        else:
            zero_loss_count = 0

        early_exit = zero_loss_count >= zero_loss_count_for_early_stop

        if early_exit or (epoch % log_rate) == 0:
            # metrics
            metrics['train/epoch'] = epoch
            metrics['train/loss'] = epoch_loss
            metrics['train/proportion_flipped'] = epoch_proportion_flipped

            total_w_g = 0
            total_w_g_0 = 0
            for name, param in TBNN.named_parameters():
                if 'W' not in name:
                    continue
                total_w_g += torch.numel(param.grad)
                total_w_g_0 += torch.sum(param.grad == 0).item()

            total_a_g = 0
            total_a_g_0 = 0
            for grad in TBNN.grad.values():
                total_a_g += torch.numel(grad)
                total_a_g_0 += torch.sum(grad == 0).item()

            metrics['train/prop_w_g_nonzero'] = 1 - total_w_g_0 / total_w_g
            metrics['train/prop_a_g_nonzero'] = 1 - total_a_g_0 / total_a_g

            # images
            w_ds = []
            w_g_ds = []
            for name, param in TBNN.named_parameters():
                if 'W' not in name:
                    continue
                w_ds.append(bnn.metrics.distribution(param.data))
                w_g_ds.append(bnn.metrics.distribution(param.grad))

            i_ds = []
            for input in TBNN.input.values():
                i_ds.append(bnn.metrics.distribution(input))

            g_ds = []
            for grad in TBNN.grad.values():
                g_ds.append(bnn.metrics.distribution(grad))

            w_ds_im = bnn.metrics.distribution_plot(w_ds)
            w_g_ds_im = bnn.metrics.distribution_plot(w_g_ds)
            i_ds_im = bnn.metrics.distribution_plot(i_ds)
            g_ds_im = bnn.metrics.distribution_plot(g_ds)

            metrics['image/weights'] = wandb.Image(w_ds_im)
            metrics['image/weight_grads'] = wandb.Image(w_g_ds_im)
            metrics['image/inputs'] = wandb.Image(i_ds_im)
            metrics['image/grads'] = wandb.Image(g_ds_im)

            # log
            wandb.log(metrics)

            log_str = f'epoch: {epoch}\t'
            log_str += '\t'.join([f'{key}: {value:.4g}' for key, value in metrics.items() if 'image' not in key])

            print(log_str)

        if early_exit:
            break


def setup_wandb(cfg: omegaconf.DictConfig):
    wandb_config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(project='train_network', config=wandb_config)

    wandb.define_metric('train/epoch')
    wandb.define_metric('train/*', step_metric='train/epoch')

    return


@hydra.main(config_path='../config', config_name='main', version_base=None)
def main(cfg: omegaconf.DictConfig):
    # resolve config
    omegaconf.OmegaConf.resolve(cfg)

    network: bnn.network.TernBinNetwork = hydra.utils.instantiate(cfg.network.model)
    data_loader: bnn.data.DataLoader = hydra.utils.instantiate(cfg.data.data_loader)
    loss_func: bnn.loss.LossFunction = hydra.utils.instantiate(cfg.loss)
    optim: torch.optim.Optimizer = hydra.utils.instantiate(
        config=cfg.optimizer,
        params=network.parameters(),
    )

    network._initialise(W_mean=0, W_zero_prob=0.5)

    setup_wandb(cfg=cfg)

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
