import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm

from typing import Callable, Dict, Protocol, Union


class RunnerProtocol(Protocol):
    def run_epoch(self, train: bool = True) -> Dict: ...


class Runner(RunnerProtocol):
    def __init__(
        self,
        dataloader: DataLoader,
        model: nn.Module,
        optimizer: Optimizer,
        objective_fn: Callable,
    ) -> None:
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.objective_fn = objective_fn

    def run_epoch(
        self,
        train: bool = True,
    ) -> Dict:
        self.model.train(train)
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(self.dataloader):
            self.optimizer.zero_grad()
            x, y = batch
            y_pred = self.model(x)
            loss = self.objective_fn(y_pred, y)
            if train:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(y_pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        return {
            "loss": total_loss / len(self.dataloader),
            "accuracy": correct / total,
        }


class GANRunner(RunnerProtocol):
    def __init__(
        self,
        dataloader: DataLoader,
        gan: nn.Module,
        g_optimizer: Optimizer,
        d_optimizer: Optimizer,
        d_objective_fn: Callable,
        g_objective_fn: Callable,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.dataloader = dataloader
        self.gan = gan
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.d_objective_fn = d_objective_fn
        self.g_objective_fn = g_objective_fn

    def train_discriminator(
        self, real_imgs: torch.Tensor, noise: torch.Tensor, train: bool = True
    ) -> float:
        real_labels = torch.ones(len(real_imgs))
        fake_labels = torch.zeros(len(noise))

        real_imgs = real_imgs.to(self.gan.device)
        noise = noise.to(self.gan.device)

        if noise.shape[0] == 0:
            noise = None

        gen_preds, real_preds = self.gan(noise=noise, real_imgs=real_imgs)

        if noise is not None:
            d_loss = self.d_objective_fn(
                gen_preds, fake_labels, real_preds, real_labels
            )
        else:
            d_loss = self.d_objective_fn(None, None, real_preds, real_labels)

        if train:
            d_loss.backward()
            self.d_optimizer.step()

        return d_loss.item()

    def train_generator(self, noise: torch.Tensor, train: bool = True) -> float:
        fake_labels = torch.zeros(len(noise))
        noise = noise.to(self.gan.device)
        gen_preds, _ = self.gan(noise=noise)
        g_loss = self.g_objective_fn(gen_preds, fake_labels)

        if train:
            g_loss.backward()
            self.g_optimizer.step()

        return g_loss.item()

    def run_epoch(
        self,
        train: bool = True,
        percent_real: float = 0.5,
    ) -> Dict:
        self.gan.train(train)
        self.gan.to(self.gan.device)

        generator_loss = 0.0
        discriminator_loss = 0.0

        num_real = 0
        num_fake = 0
        with tqdm(total=len(self.dataloader), desc="g_L: - | d_L: - ") as pbar:
            for batch in self.dataloader:
                self.g_optimizer.zero_grad()
                self.d_optimizer.zero_grad()
                real_imgs, conditioning_variable = batch
                noise = torch.randn(
                    int(len(real_imgs) * (1 - percent_real)),
                    *real_imgs.shape[1:],
                )
                _num_real = len(real_imgs)
                _num_fake = len(noise)

                num_real += _num_real
                num_fake += _num_fake

                discriminator_loss += self.train_discriminator(
                    real_imgs, noise, train=train
                )
                generator_loss += self.train_generator(noise, train=train)
                pbar.set_description(
                    f"g_L: {generator_loss / num_fake:.4f} | d_L: {discriminator_loss / num_real:.4f}"
                )
                pbar.update(1)

        return {
            "discriminator_loss": discriminator_loss / num_real,
            "generator_loss": generator_loss / num_fake,
            "loss": (discriminator_loss + generator_loss) / (num_real + num_fake),
        }

    def inference(self, noise: Union[torch.Tensor, None] = None) -> torch.Tensor:
        self.gan.eval()
        self.gan.to(self.gan.device)
        if noise is not None:
            noise = noise.to(self.gan.device)
        else:
            in_out_size = self.dataloader.dataset.get_in_out_size
            noise = torch.randn(1, *in_out_size[0]).to(self.gan.device)

        with torch.no_grad():
            return self.gan.generator(noise)
