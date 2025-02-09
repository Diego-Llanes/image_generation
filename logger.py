import mlflow
from torch.utils.tensorboard import SummaryWriter
from rich.logging import RichHandler
import skeletonkey as sk
import pandas as pd
import yaml
import wandb
import json

from typing import Protocol
import logging
from pathlib import Path


class LoggerProtocol(Protocol):
    def log_metrics(self, metrics: dict, step: int = None) -> None: ...
    def log_params(self, params: dict) -> None: ...
    def log_artifact(self, artifact: str) -> None: ...
    def log_figure(self, figure, name: str) -> None: ...
    def clean_up(self) -> None: ...
    def __getattr__(self, name) -> logging.Logger: ...


class WandbWrapper(LoggerProtocol):
    def __init__(self, logger: logging.Logger, run_name: str, experiment: str):
        self._logger = logger
        wandb.init(name=run_name, project=experiment)

    def log_metrics(self, metrics: dict, step: int = None) -> None:
        wandb.log(metrics, step=step)
        self._logger.info("Metrics logged: %s", ", ".join(k for k in metrics))

    def log_params(self, params: dict) -> None:
        wandb.config.update(params)
        self._logger.info("Parameters logged")

    def log_artifact(self, artifact: str) -> None:
        wandb.log_artifact(artifact)
        self._logger.info("Artifact logged: %s", type(artifact))

    def log_figure(self, figure, name: str) -> None:
        wandb.log({name: figure})
        self._logger.info("Figure logged: %s", name)

    def clean_up(self) -> None:
        self._logger.info("Cleaning up")
        wandb.finish()
        self._logger.info("Run ended")

    def __getattr__(self, name) -> logging.Logger:
        return getattr(self._logger, name)


class ConsoleWrapper(LoggerProtocol):
    def __init__(
        self,
        logger: logging.Logger,
        log_dir: str,
        run_name: str,
        save_frequently: bool = True,
    ):
        self._logger = logger
        self.run_name = run_name
        _log_dir = Path(log_dir) / run_name
        if _log_dir.exists():
            i = 0
            while (Path(log_dir) / (run_name + f"_{i}")).exists():
                i += 1
            self.log_dir = Path(log_dir) / (run_name + f"_{i}")
        else:
            self.log_dir = _log_dir
        self.log_dir.mkdir(parents=True)
        self.metrics = pd.DataFrame()
        self.save_frequently = save_frequently

    def log_metrics(self, metrics: dict, step: int = None) -> None:
        self.metrics = self.metrics._append(metrics, ignore_index=True)
        if self.save_frequently:
            self.metrics.to_csv(f"{self.log_dir}/metrics.csv")
        self._logger.info("Metrics logged: %s", ", ".join(k for k in metrics))

    def log_params(self, params: dict) -> None:
        with open(f"{self.log_dir}/params.yaml", "w") as f:
            yaml.dump(params, f)
        self._logger.info("Parameters logged")

    def log_artifact(self, artifact: str) -> None:
        with open(f"{self.log_dir}/artifact.txt", "w") as f:
            f.write(artifact)
        self._logger.info("Artifact logged: %s", type(artifact))

    def log_figure(self, figure, name: str) -> None:
        if not (self.log_dir / "figs").exists():
            (self.log_dir / "figs").mkdir()
        figure.savefig(f"{self.log_dir}/figs/{name}.png")
        self._logger.info("Figure logged: %s", name)

    def clean_up(self) -> None:
        self._logger.info("Cleaning up")
        self.metrics.to_csv(f"{self.log_dir}/metrics.csv")
        self._logger.info("Run ended")

    def __getattr__(self, name) -> logging.Logger:
        return getattr(self._logger, name)


class TensorboardWrapper(LoggerProtocol):
    def __init__(self, logger: logging.Logger, log_dir: str):
        self._logger = logger
        self.writer = SummaryWriter(log_dir)

    def log_metrics(self, metrics: dict, step: int = None) -> None:
        self.writer.add_scalars("metrics", metrics, step)
        self._logger.info("Metrics logged: %s", ", ".join(k for k in metrics))

    def log_params(self, params: dict) -> None:
        self.writer.add_hparams(params, {})
        self._logger.info("Parameters logged")

    def log_artifact(self, artifact: str) -> None:
        self.writer.add_text("artifact", artifact)
        self._logger.info("Artifact logged: %s", type(artifact))

    def log_figure(self, figure, name: str) -> None:
        self.writer.add_figure(name, figure)
        self._logger.info("Figure logged: %s", name)

    def clean_up(self) -> None:
        self._logger.info("Cleaning up")
        self.writer.close()
        self._logger.info("Run ended")

    def __getattr__(self, name) -> logging.Logger:
        return getattr(self._logger, name)


class MLFlowWrapper(LoggerProtocol):
    def __init__(self, logger: logging.Logger, run_name: str, experiment_id: str):
        self._logger = logger
        mlflow.start_run(run_name=run_name, experiment_id=experiment_id)

    def log_metrics(self, metrics: dict, step: int = None) -> None:
        mlflow.log_metrics(metrics, step=step)
        self._logger.info("Metrics logged: %s", ", ".join(k for k in metrics))

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)
        self._logger.info("Parameters logged")

    def log_artifact(self, artifact: str) -> None:
        mlflow.log_artifact(artifact)
        self._logger.info("Artifact logged: %s", type(artifact))

    def log_figure(self, figure, name: str) -> None:
        mlflow.log_figure(figure, name)
        self._logger.info("Figure logged: %s", name)

    def clean_up(self) -> None:
        self._logger.info("Cleaning up")
        mlflow.end_run()
        self._logger.info("Run ended")

    def __getattr__(self, name) -> logging.Logger:
        return getattr(self._logger, name)


class DebugWrapper(LoggerProtocol):
    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def _pretty_print(self, obj: dict) -> str:
        return json.dumps(obj, indent=4)

    def log_metrics(self, metrics: dict, step: int = None) -> None:
        self._logger.info(f"Metrics: {self._pretty_print(metrics)}")
        self._logger.warning("Metrics not logged in debug mode")

    def log_params(self, params: dict) -> None:
        self._logger.warning("Parameters not logged in debug mode")

    def log_artifact(self, artifact: str) -> None:
        self._logger.warning("Artifacts not logged in debug mode")

    def log_figure(self, figure, name: str) -> None:
        self._logger.warning("Figures not logged in debug mode")

    def clean_up(self) -> None:
        self._logger.info("Run ended")

    def __getattr__(self, name) -> logging.Logger:
        return getattr(self._logger, name)


def get_logger(
    config: sk.Config,
    run_name: str = None,
    experiment: str = None,
) -> LoggerProtocol:
    fmt = "[%(levelname)-8s %(asctime)s %(filename)15s:%(lineno)-4d] %(message)s"
    datefmt = "%d-%b-%y %H:%M:%S"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    c_handler = RichHandler()
    c_handler.setLevel(logging.DEBUG)
    logger.addHandler(c_handler)

    if config.debug:
        return DebugWrapper(logger)

    match config.logger:
        case "tensorboard":
            return TensorboardWrapper(logger, config.logdir)
        case "mlflow":
            return MLFlowWrapper(logger, run_name, experiment)
        case "console":
            return ConsoleWrapper(logger, config.logdir, run_name)
        case "wandb":
            return WandbWrapper(logger, run_name, experiment)
        case "debug":
            return DebugWrapper(logger)
        case _:
            raise ValueError(f"Unsupported logger: {config.logger}")
