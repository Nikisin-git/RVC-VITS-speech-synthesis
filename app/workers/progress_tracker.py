"""Aggregate ProgressEvents into a state suitable for progress bars."""

from __future__ import annotations

from dataclasses import dataclass

from app.workers.log_parser import ProgressEvent


@dataclass
class ProgressState:
    current_epoch: int = 0
    total_epochs: int = 0
    stage: str = ""
    last_message: str = ""
    last_loss: float | None = None
    checkpoints_saved: int = 0

    @property
    def percent(self) -> int:
        if self.total_epochs > 0:
            return min(100, int(round(100 * self.current_epoch / self.total_epochs)))
        return 0

    def apply(self, event: ProgressEvent) -> None:
        self.last_message = event.message
        if event.kind == "epoch":
            if event.epoch is not None:
                self.current_epoch = event.epoch
            if event.total_epochs is not None:
                self.total_epochs = event.total_epochs
            if event.loss is not None:
                self.last_loss = event.loss
        elif event.kind == "checkpoint":
            self.checkpoints_saved += 1
        elif event.kind == "stage":
            self.stage = event.message
