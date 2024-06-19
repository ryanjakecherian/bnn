import logging
import os
import pathlib

import bnn
import bnn.config
import bnn.metrics
import bnn.save
import hydra
import omegaconf
import torch

import wandb

logger = logging.getLogger(__name__)

omegaconf.OmegaConf.register_new_resolver(
    name='sandwich_list',
    resolver=bnn.config.sandwich_list,
)
omegaconf.OmegaConf.register_new_resolver(
    name='pow',
    resolver=bnn.config.pow,
)


def train_epoch(
    TBNN: bnn.network.TernBinNetwork,
    loss_func: bnn.loss.LossFunction,
    DL: bnn.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    log: bool,
    metrics: dict,
) -> dict:
    assert bnn.network.network_params_al_ternary(TBNN)

    epoch_loss = 0
    datapoints = 0
    epoch_proportion_flipped = 0
    num_correct = 0

    for batch_id, batch in enumerate(DL):
        # number of dps seen...
        datapoints += len(batch.input)

        # forward pass and loss
        output = TBNN.forward(batch.input)
        loss = loss_func.forward(output=output, target=batch.target)

        # backward pass
        grad = loss_func.backward(output=output, target=batch.target)
        TBNN.backward(grad)

        # optimizer step
        batch_proportion_flipped = optimizer.step()
        epoch_proportion_flipped += batch_proportion_flipped

        # sum loss
        epoch_loss += loss

        if log:
            # acc
            output_argmax = torch.argmax(output, dim=-1)
            target_argmax = torch.argmax(batch.target, dim=-1)
            num_correct += torch.sum(target_argmax == output_argmax)

    num_batches = batch_id + 1

    if log:
        # metrics
        metrics['train/accuracy'] = num_correct / datapoints

        # train health
        metrics['train/mean_loss'] = epoch_loss / num_batches
        metrics['train/mean_proportion_flipped'] = epoch_proportion_flipped / num_batches

        total_w_g = 0
        total_w_g_0 = 0
        total_w = 0
        total_w_0 = 0
        for name, param in TBNN.named_parameters():
            if 'W' not in name:
                continue
            total_w_g += torch.numel(param.grad)
            total_w_g_0 += torch.sum(param.grad == 0).item()
            total_w += torch.numel(param)
            total_w_0 += torch.sum(param == 0).item()

        total_a_g = 0
        total_a_g_0 = 0
        for grad in TBNN.grad.values():
            total_a_g += torch.numel(grad)
            total_a_g_0 += torch.sum(grad == 0).item()

        total_a = 0
        total_a_0 = 0
        for input in TBNN.input.values():
            total_a += torch.numel(input)
            total_a_0 += torch.sum(input == 0).item()

        metrics['train/prop_w_g_nonzero'] = 1 - total_w_g_0 / total_w_g
        metrics['train/prop_w_nonzero'] = 1 - total_w_0 / total_w
        metrics['train/prop_a_g_nonzero'] = 1 - total_a_g_0 / total_a_g
        metrics['train/prop_a_nonzero'] = 1 - total_a_0 / total_a

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

        epoch = metrics['train/epoch']
        metrics['image/weights'] = wandb.Image(w_ds_im, caption=f'epoch_{epoch}')
        metrics['image/weight_grads'] = wandb.Image(w_g_ds_im, caption=f'epoch_{epoch}')
        metrics['image/inputs'] = wandb.Image(i_ds_im, caption=f'epoch_{epoch}')
        metrics['image/grads'] = wandb.Image(g_ds_im, caption=f'epoch_{epoch}')

        # log
        wandb.log(metrics)

    return epoch_loss


def test_epoch(
    TBNN: bnn.network.TernBinNetwork,
    loss_func: bnn.loss.LossFunction,
    DL: bnn.data.DataLoader,
    metrics: dict,
) -> None:
    epoch_loss = 0
    num_correct = 0
    total = DL._datapoints

    for batch_id, batch in enumerate(DL):
        output = TBNN.forward(batch.input)
        output_argmax = torch.argmax(output, dim=-1)
        target_argmax = torch.argmax(batch.target, dim=-1)

        epoch_loss += loss_func.forward(output=output, target=batch.target)
        num_correct += torch.sum(target_argmax == output_argmax)

    num_batches = batch_id + 1

    metrics['test/accuracy'] = num_correct / total
    metrics['test/loss'] = epoch_loss / num_batches

    # log
    wandb.log(metrics)

    return


def train(
    TBNN: bnn.network.TernBinNetwork,
    loss_func: bnn.loss.LossFunction,
    train_DL: bnn.data.DataLoader,
    test_DL: bnn.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    log_rate: int,
    train_epochs: int,
    checkpoint_rate: int,
    save_dir: pathlib.Path,
):
    zero_loss_count = 0
    zero_loss_count_for_early_stop = 10
    early_exit = False

    for epoch in range(train_epochs):
        log = early_exit or (epoch % log_rate) == 0 or (epoch + 1 == train_epochs)
        checkpoint = early_exit or (epoch % checkpoint_rate) == 0 or (epoch + 1 == train_epochs)

        epoch_loss = train_epoch(
            TBNN=TBNN,
            loss_func=loss_func,
            DL=train_DL,
            optimizer=optimizer,
            log=log,
            metrics={'train/epoch': epoch},
        )

        if log:
            logger.info(f'epoch {epoch}: logging')
            test_epoch(
                TBNN=TBNN,
                loss_func=loss_func,
                DL=test_DL,
                metrics={'test/epoch': epoch},
            )

        if checkpoint:
            logger.info(f'epoch {epoch}: checkpointing')
            fname = save_dir / f'chkpt_epoch_{epoch:06d}.pckl'
            bnn.save.save_network(network=TBNN, filename=fname)

        # NOTE early exit before recalculating so that it another loop is run and logged before exit!
        if early_exit:
            break

        early_exit = zero_loss_count >= zero_loss_count_for_early_stop

        # increment zero loss count
        if epoch_loss == 0:
            zero_loss_count += 1
        else:
            zero_loss_count = 0

    return


def setup_wandb(cfg: omegaconf.DictConfig):
    wandb_config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run = wandb.init(project='train_classifier', config=wandb_config)

    wandb.define_metric('train/epoch')
    wandb.define_metric('train/*', step_metric='train/epoch')
    wandb.define_metric('test/epoch')
    wandb.define_metric('test/*', step_metric='test/epoch')

    return run


@hydra.main(config_path='../config', config_name='main', version_base=None)
def main(cfg: omegaconf.DictConfig):
    # resolve config
    omegaconf.OmegaConf.resolve(cfg)

    # instantiate
    network: bnn.network.TernBinNetwork = hydra.utils.instantiate(cfg.network.model)
    train_data_loader: bnn.data.DataLoader = hydra.utils.instantiate(cfg.dataset.train_data_loader)
    test_data_loader: bnn.data.DataLoader = hydra.utils.instantiate(cfg.dataset.test_data_loader)
    loss_func: bnn.loss.LossFunction = hydra.utils.instantiate(cfg.loss)
    optim: torch.optim.Optimizer = hydra.utils.instantiate(
        config=cfg.optimizer,
        params=network.parameters(),
    )
    network._initialise(W_mean=0, W_zero_prob=0.5)

    run = setup_wandb(cfg=cfg)
    # convert to path
    save_dir = pathlib.Path(os.path.expanduser(cfg.train.save_dir)) / run.name
    logger.info(f'save_dir: {save_dir}')

    if cfg.train.gpu is not None:
        if not torch.cuda.is_available():
            raise RuntimeError('No GPU available!')
        device = torch.device(f'cuda:{cfg.train.gpu}')
        network.to(device)
        train_data_loader.to(device)
        test_data_loader.to(device)

    train(
        TBNN=network,
        loss_func=loss_func,
        train_DL=train_data_loader,
        test_DL=test_data_loader,
        optimizer=optim,
        log_rate=cfg.train.log_rate,
        checkpoint_rate=cfg.train.checkpoint_rate,
        train_epochs=cfg.train.epochs,
        save_dir=save_dir,
    )


if __name__ == '__main__':
    main()
