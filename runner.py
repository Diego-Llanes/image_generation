import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm

from typing import Callable, Dict, Protocol, Union, Tuple


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
    ) -> Tuple[float, float]:
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

        correct = (real_preds > 0.5).sum().item() + (gen_preds < 0.5).sum().item()
        overall_acc = correct / (len(real_preds) + len(gen_preds))

        if train:
            d_loss.backward()
            self.d_optimizer.step()

        return d_loss.item(), overall_acc

    def train_generator(self, noise: torch.Tensor, train: bool = True) -> float:
        fake_labels = torch.ones(len(noise))
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
        discriminator_acc = 0.0

        with tqdm(total=len(self.dataloader), desc="g_L: - | d_A: - ") as pbar:
            for batch_count, batch in enumerate(self.dataloader, start=1):
                self.g_optimizer.zero_grad()
                self.d_optimizer.zero_grad()
                real_imgs, conditioning_variable = batch
                noise = torch.randn(
                    int(len(real_imgs) * (1 - percent_real)),
                    *real_imgs.shape[1:],
                )
                _discriminator_loss, _discriminator_acc = self.train_discriminator(
                    real_imgs, noise, train=train
                )
                discriminator_loss += _discriminator_loss
                discriminator_acc += _discriminator_acc
                generator_loss += self.train_generator(noise, train=train)
                pbar.set_description(
                    f"g_L: {generator_loss / batch_count:.4f} | d_A: {discriminator_acc / batch_count:.4f}"
                )
                pbar.update(1)

        return {
            "discriminator_loss": discriminator_loss / batch_count,
            "generator_loss": generator_loss / batch_count,
            "discriminator_acc": discriminator_acc / batch_count,
            "loss": (discriminator_loss + generator_loss) / batch_count,
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


class DiffusionRunner(RunnerProtocol):
    def __init__(
        self,
        dataloader: DataLoader,
        model: nn.Module,
        optimizer: Optimizer,
        objective_fn: Callable,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(dataloader, model, optimizer, objective_fn)
        self.device = device
        self.timesteps = timesteps

        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alpha = (1 - self.beta).to(device)
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(device)

        self.model = model
        self.optimizer = optimizer
        self.objective_fn = objective_fn
        self.dataloader = dataloader

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: Union[None, torch.Tensor] = None
    ) -> torch.Tensor:
        """Adds noise to the input image x0 at a given timestep t."""
        if noise is None:
            noise = torch.randn(x0.shape, device=self.device)
        sqrt_alpha_bar = self.alpha_bar[t].view(-1, 1, 1, 1).sqrt()
        sqrt_one_minus_alpha_bar = (1 - self.alpha_bar[t]).view(-1, 1, 1, 1).sqrt()
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

    def run_epoch(self, train: bool = True) -> Dict:
        self.model.train(train)
        self.model.to(self.device)
        total_loss = 0.0

        for batch in tqdm(self.dataloader):
            self.optimizer.zero_grad()
            x0, cond = batch
            t = torch.randint(0, self.timesteps, (x0.size(0),), device=self.device)
            x0, cond, t = x0.to(self.device), cond.to(self.device), t.to(self.device)

            x_t = self.q_sample(x0, t)
            preds = self.model(x_t, t)
            loss = self.objective_fn(preds, x0)

            if train:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

        return {"loss": total_loss / len(self.dataloader)}

    def inference(
        self,
        noise: Union[torch.Tensor, None] = None,
        condition: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """Performs the full denoising process, starting from noise."""

        if noise is None:
            # attempt to infer the input size from the dataset
            in_out_size = self.dataloader.dataset.get_in_out_size
            noise = torch.randn(1, *in_out_size[0]).to(self.device)

        self.model.eval()
        noise = noise.to(self.device)

        with torch.no_grad():
            for i in range(self.timesteps - 1, -1, -1):
                pred_noise = self.model(noise, torch.tensor([i], device=self.device))
                noise = self.q_sample(
                    noise, torch.tensor([i], device=self.device), pred_noise
                )

        return noise
