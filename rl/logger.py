import csv
import time
from pathlib import Path

import numpy as np

class Logger:
    '''Logs in TensorBoard, CSV, or Stdout'''
    def __init__(self, run_dir: Path, use_tb: bool = True, use_csv: bool = True, print_every: int = 1):
        self.run_dir = run_dir
        self.print_every = print_every
        self._buf: dict[str, float] = {}

        # Tensorboard
        self.tb_writer = None
        if use_tb:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(run_dir)
        # CSV
        self._rows: list[dict[str, float]] = []
        self._fields: list[str] = ["step"]
        self.csv_path = run_dir / "metrics.csv" if use_csv else None

        self._dumps = 0
        self._t0 = time.time()

    def log(self, key: str, value) -> None:
        self._buf[key] = float(value)

    def log_dict(self, prefix: str, d: dict) -> None:
        '''log_dict('reward', {'alive': 1.0, ...}) -> reward/alive, ...'''
        for k, v in d.items():
            self.log(f"{prefix}/{k}", v)

    def histogram(self, key: str, values, step: int) -> None:
        if self.tb_writer is None: return
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        self.tb_writer.add_histogram(key, values, step)

    def dump(self, step: int) -> None:
        '''Dump all logged values to TensorBoard, CSV, stdout.'''
        if not self._buf: return
        self._buf["time/elapsed_s"] = time.time() - self._t0
        # Tensorboard
        if self.tb_writer is not None:
            for k, v in self._buf.items():
                self.tb_writer.add_scalar(k, v, step)
        # CSV
        if self.csv_path is not None:
            self._write_csv({"step": step, **self._buf})
        # STDOUT
        if self.print_every and self._dumps % self.print_every == 0:
            self._print(step)

        self._buf.clear()
        self._dumps += 1

    def _write_csv(self, row: dict) -> None:
        self._rows.append(row)
        new = [k for k in row if k not in self._fields]
        if new:
            self._fields.extend(new)
            with open(self.csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, self._fields, restval="")
                w.writeheader()
                w.writerows(self._rows)

        else:
            with open(self.csv_path, "a", newline="") as f:
                csv.DictWriter(f, self._fields, restval="").writerow(row)

    def _print(self, step: int) -> None:
        width = max((len(k) for k in self._buf), default=0)
        print(f"\n-- step {step:,} " + "-" * max(0, 46 - len(f"{step:,}")))
        for k in sorted(self._buf):
            print(f"  {k:<{width}}  {self._buf[k]:>12.4g}")

    def close(self) -> None:
        if self.tb_writer is not None:
            self.tb_writer.close()


class EpisodeTracker:
    '''Rolling window of finished-episode metrics from info dicts.'''
    def __init__(self, window: int = 100):
        self.window = window
        self.returns: list[float] = []
        self.lengths: list[int] = []
        self.total_episodes = 0

    def update(self, infos: dict) -> int:
        '''Call once per env step with info dict. Returns episodes finished.'''
        if "episode" not in infos:
            return 0
        mask = np.asarray(infos["_episode"], dtype=bool)
        if not mask.any():  # if no episodes finished
            return 0
        r = np.asarray(infos["episode"]["r"])[mask]
        length = np.asarray(infos["episode"]["l"])[mask]
        self.returns.extend(r.tolist())
        self.lengths.extend(length.tolist())
        del self.returns[:-self.window]
        del self.lengths[:-self.window]
        self.total_episodes += int(mask.sum())
        return int(mask.sum())

    def log_to(self, logger: Logger) -> None:
        if not self.returns:
            return
        logger.log("charts/episodic_return", float(np.mean(self.returns)))
        logger.log("charts/episodic_return_std", float(np.std(self.returns)))
        logger.log("charts/episodic_length", float(np.mean(self.lengths)))
        logger.log("charts/episodes_total", self.total_episodes)
