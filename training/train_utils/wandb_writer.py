"""Log Plann3r training scalars and images to both TensorBoard and Weights & Biases.

WandbTensorBoardLogger extends the upstream TensorBoardLogger and mirrors each
log call to a W&B run on rank zero. The released nav_costmap.yaml uses the plain
TensorBoardLogger, so this is used only when logging.tensorboard_writer._target_
is overridden to train_utils.wandb_writer.WandbTensorBoardLogger.
"""

from pathlib import Path
from typing import Any

import numpy as np

from .tb_writer import TensorBoardLogger


class WandbTensorBoardLogger(TensorBoardLogger):
    """Mirror trainer scalars and images to TensorBoard and W&B on rank zero."""

    def __init__(
        self,
        path: str,
        project: str,
        name: str,
        wandb_dir: str,
        entity: str | None = None,
        mode: str = "online",
        **kwargs: Any,
    ) -> None:
        self._wandb_run = None
        super().__init__(path=path, **kwargs)
        if self.writer is not None:
            import wandb

            Path(wandb_dir).mkdir(parents=True, exist_ok=True)
            self._wandb_run = wandb.init(
                project=project,
                entity=entity,
                name=name,
                dir=wandb_dir,
                mode=mode,
                resume="allow",
            )

    def log(self, name: str, data: Any, step: int) -> None:
        super().log(name, data, step)
        if self._wandb_run is not None:
            if hasattr(data, "detach"):
                data = data.detach().float().cpu().item()
            self._wandb_run.log({name: data}, step=int(step))

    def log_visuals(self, name: str, data: Any, step: int, fps: int = 4) -> None:
        super().log_visuals(name, data, step, fps)
        if self._wandb_run is None:
            return
        import wandb

        array = np.asarray(data)
        if array.ndim == 3:
            if array.shape[0] in (1, 3, 4):
                array = np.moveaxis(array, 0, -1)
            self._wandb_run.log({name: wandb.Image(array)}, step=int(step))
        elif array.ndim == 5:
            self._wandb_run.log(
                {name: wandb.Video(array, fps=fps, format="mp4")}, step=int(step)
            )

    def close(self) -> None:
        # wandb.init registers its own process-exit handler. Calling finish here would
        # race that handler during torchrun shutdown and can produce a double-finish error.
        self._wandb_run = None
        super().close()
