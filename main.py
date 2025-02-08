import torch
import skeletonkey as sk

from pathlib import Path
import random

from dataset import MixedDataset, DebugDataset
from models.gan import GAN
from runner import GANRunner, RunnerProtocol
from logger import get_logger, LoggerProtocol
from viz import plot_fake_vs_real


def get_datasets(config: sk.Config, logger: LoggerProtocol, split: str = "train"):
    logger.info(f"loading fashion mnist {split} data...")
    fashion_mnist = sk.instantiate(config.dataset.fashion_mnist, split=split)

    logger.info(f"loading cifar10 {split} data...")
    cifar10 = sk.instantiate(
        config.dataset.cifar10,
        normalizer=lambda x: (x - x.min()) / (x.max() - x.min()),
        augs=[
            # interpolate to 28x28 (same as fashion mnist)
            lambda x: torch.nn.functional.interpolate(
                x.unsqueeze(0), size=(28, 28), mode="bilinear", align_corners=False
            ).squeeze(0),
            # greyscale
            lambda x: x.mean(dim=0),
        ],
        split=split,
    )

    logger.info("combining datasets...")
    mixed_dataset = MixedDataset(
        datasets=[fashion_mnist, cifar10],
        transforms=[],
        split=split,
    )
    return mixed_dataset


def get_gan_runner(
    config: sk.Config,
    gan: GAN,
    logger: LoggerProtocol,
    dataloader: torch.utils.data.DataLoader,
) -> RunnerProtocol:
    def d_objective_fn(gen_preds, gen_targets, real_preds, real_targets):
        return torch.nn.functional.binary_cross_entropy(
            gen_preds.squeeze(), gen_targets.squeeze()
        ) + torch.nn.functional.binary_cross_entropy(
            real_preds.squeeze(), real_targets.squeeze()
        )

    def g_objective_fn(preds, targets):
        return 1 - torch.nn.functional.binary_cross_entropy(
            preds.squeeze(), targets.squeeze()
        )

    runner = GANRunner(
        dataloader=dataloader,
        gan=gan,
        g_optimizer=sk.instantiate(config.optimizer, params=gan.generator.parameters()),
        d_optimizer=sk.instantiate(
            config.optimizer, params=gan.discriminator.parameters()
        ),
        d_objective_fn=d_objective_fn,
        g_objective_fn=g_objective_fn,
        device=torch.device(
            "cuda" if torch.cuda.is_available() and not config.debug else "cpu"
        ),
    )

    return runner


@sk.unlock(config_name="configs/config")
def main(config: sk.Config):
    logger = get_logger(
        config=config, run_name=config.run_name, experiment=config.experiment
    )

    if config.debug:
        logger.warning("Running in debug mode")
        logger.warning("Creating DEBUG datasets...")
        train_dataset, dev_dataset = (DebugDataset() for _ in range(2))
    else:
        # WARN: We are using "test set" as the dev set
        train_dataset, dev_dataset = (
            get_datasets(config, logger, split=split) for split in ["train", "test"]
        )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=config.batch_size, shuffle=False
    )

    if config.arch == "gan":
        model = sk.instantiate(config.model.gan)
        train_runner, dev_runner = (
            get_gan_runner(config, model, logger, dataloader)
            for dataloader in [train_dataloader, dev_dataloader]
        )
    else:
        # TODO: Add support for diffusion
        raise ValueError(f"Unsupported architecture: {config.arch}")

    best_dev_metric = float("inf")
    for epoch in range(config.epochs):
        logger.info(f"Training epoch {epoch}")

        train_metrics = train_runner.run_epoch()
        train_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
        logger.log_metrics(train_metrics, step=epoch)

        logger.info(f"Dev epoch {epoch}")
        dev_metrics = dev_runner.run_epoch(train=False)
        dev_metrics = {f"dev_{k}": v for k, v in dev_metrics.items()}
        logger.log_metrics(dev_metrics, step=epoch)

        if dev_metrics[config.dev_metric] < best_dev_metric:
            logger.info(
                f"New best dev metric found: {dev_metrics[config.dev_metric]} < {best_dev_metric}"
            )
            best_dev_metric = dev_metrics[config.dev_metric]
            logger.info("Saving model...")
            try:
                torch.save(
                    model.state_dict(),
                    Path(config.checkpoint_dir)
                    / f"{config.arch}_{config.run_name}.pth",
                )
            except Exception as e:
                logger.error(f"Failed to save model: {e}")
        logger.info("Inferencing the model for visualization...")

        generated_sample = dev_runner.inference()
        random_true_sample, _ = dev_dataset[random.randint(0, len(dev_dataset))]

        fig = plot_fake_vs_real(
            fake_image=generated_sample,
            real_image=random_true_sample,
            show=False,
            style="ggplot",
        )

        logger.log_figure(fig, f"fake_vs_real_epoch_{epoch}")

        logger.info(f"Epoch {epoch} completed")


if __name__ == "__main__":
    main()
