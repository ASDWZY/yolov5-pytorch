import shutil

from torch.utils.tensorboard import SummaryWriter

from datasets.YoloTransforms import *
from datasets.YoloDataset import YoloDataset
from utils.train import *
import os

from utils.optimize import YoloOptimizer
from yolov5.utils import Timer, check_dir

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from YOLO import YOLO
import subprocess


class YoloLogWriter:

    def __init__(self, log_dir, resume, ):
        if not resume and os.path.exists(log_dir):
            shutil.rmtree(log_dir)
            LOGGER.info(f"Removed tensorboard logs in {log_dir}")
        self.writer = SummaryWriter(log_dir=log_dir)

    @staticmethod
    def _run_commands(commands):
        output = None
        for command in commands:
            result = subprocess.run(command, input=output, capture_output=True, text=True)
            output = result.stdout
            print(output, result.stderr)

    def _write_loss(self, epoch, loss, name):
        loss_sum = loss.sum()
        self.writer.add_scalars(name, {
            "sum": loss_sum,
            "box": loss[0],
            "obj": loss[1],
            "cls": loss[2],
        }, epoch + 1)
        return loss_sum

    def _write_lrs(self, epoch, lrs):
        lr_dict = {str(i): lr for i, lr in enumerate(lrs)}
        self.writer.add_scalars("lr", lr_dict, epoch + 1)

    def write(self, epoch, train_loss, val_loss, lrs):
        self._write_lrs(epoch, lrs)

        train_l = self._write_loss(epoch, train_loss, "train loss")
        val_l = self._write_loss(epoch, val_loss, "val loss")
        self.writer.add_scalars("loss", {
            "train": train_l,
            "val": val_l
        }, epoch + 1)


class FitMonitor:
    def __init__(self, epochs, save_dir, save_period, patience, train_proportion=1.0):

        self.epochs = epochs
        self.save_dir = save_dir
        self.save_period = save_period

        check_dir(save_dir)

        self.train_proportion = train_proportion

        self.patience = patience
        self.min_loss = float("inf")
        self.best_epoch = -1

        self.frozen = False
        self.mosaic = True
        self.timer = Timer()

    def _check_resume(self, checkpoint, resume):
        if resume:
            if "epoch" in checkpoint:
                return checkpoint["epoch"]
            LOGGER.error("Cannot find epoch from checkpoint file")
        elif "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
            if 0 < start_epoch < self.epochs:
                LOGGER.warning(
                    f"Starting a new training with epoch={start_epoch}. Did you forgot setting the param resume=True?")
        return 0

    def start(self, checkpoint, freeze_epochs, resume):
        start_epoch = self._check_resume(checkpoint, resume)
        if start_epoch < 0 or start_epoch > self.epochs:
            LOGGER.Error("Resuming epoch must be less than num_epochs and greater than 0.")
        if start_epoch >= freeze_epochs:
            self.frozen = True
        self.timer.reset()
        return start_epoch

    def check_epoch(self, epoch, freeze_epochs, close_mosaic):
        if epoch < freeze_epochs and not self.frozen:
            self.frozen = True
            return "freeze"
        if epoch >= freeze_epochs and self.frozen:
            self.frozen = False
            return "unfreeze"
        num_mosaic_epochs = self.epochs - close_mosaic
        if epoch >= num_mosaic_epochs and self.mosaic:
            self.mosaic = False
            return "close mosaic"
        return "none"

    def save_model(self, f, model, epoch, optimizer):
        torch.save({
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer.state_dict(),
        }, os.path.join(self.save_dir, f))

    def display_progress(self, i, num_batches, epoch):
        train_time = self.timer.get_interval_formats()
        print(f"\repoch {epoch + 1}/{self.epochs}   train_iter {i + 1}/{num_batches}   trained for {train_time}",
              end="")

    def step(self, model, optimizer, epoch, train_loss, val_loss):
        train_l, val_l = train_loss.sum(), val_loss.sum()

        if epoch - self.best_epoch > self.patience:
            LOGGER.warning(f"Reach early stopping patience{self.patience}.")
            LOGGER.step(f"Early stopping with train_loss={float(train_l)}, val_loss={float(val_l)}")
            self.finish()

        if check_loss(train_loss, "total_train_loss") or check_loss(val_loss, "total_val_loss"):
            return
        fitness = train_l * self.train_proportion + val_l * (1 - self.train_proportion)
        self.save_model("last.pt", model, epoch, optimizer)
        if fitness < self.min_loss:
            self.best_epoch = epoch
            self.min_loss = fitness
            self.save_model("best.pt", model, epoch, optimizer)

        if self.save_period:
            if (epoch % self.save_period) == (self.save_period - 1):
                self.save_model(f"epoch{epoch}.pt", model, epoch, optimizer)

        return fitness

    def finish(self):
        train_time = self.timer.get_interval_formats()
        LOGGER.step(f"\nFinished training. Trained for {train_time}")
        exit(0)


class BatchTrainer:
    def __init__(self, model: YOLO, loss: YoloLoss, epochs, amp):
        self.model = model
        self.loss = loss
        self.epochs = epochs
        self.amp = amp

        self.timer = Timer()
        self.train_loss = None
        self.epoch = -1

    def start(self):
        if self.train_loss is None:
            self.timer.reset()
        self.epoch += 1
        self.train_loss = 0.0

    def _display_progress(self, i, num_batches, epoch):
        train_time = self.timer.get_interval_formats()
        print(f"\repoch {epoch + 1}/{self.epochs}   train_iter {i + 1}/{num_batches}   trained for {train_time}",
              end="")

    def __call__(self, data_loader):
        for i, (imgs, targets) in enumerate(data_loader):
            batch_size=len(targets)
            self._display_progress(i, len(data_loader), self.epoch)
            with torch.cuda.amp.autocast(self.amp):
                predictions, targets = self.model.train_process(imgs, targets)
                for layer_idx, prediction in enumerate(predictions):
                    check_value(prediction, f"prediction_layer{layer_idx}")
                loss = self.loss(predictions, targets, "batch_train_loss")
                self.train_loss += loss.data
                yield loss.sum()*batch_size


class Trainer:
    default_cfg = {
        "train_dataset": "datasets/train.txt",
        "val_dataset": "datasets/val.txt",
        "names": ["fire", "smoke"],
        "transforms": Transforms([RandomHorizontalFlip(), RandomRotate(10.0), RandomHSV()]),

        "epochs": 100,
        "close_mosaic": 10,
        "batch_size": 16,
        "patience": 50,
        "save_period": -1,
        "save_dir": "models/train/",
        "log_dir": "logs",
        "mosaic": 1.0,
        "mixup": 0.1,

        "momentum": 0.937,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "weight_decay": 5e-4,
        "cos_lr": True,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "nbs": 64,

        "amp": False,
        "ema": False,
        "benchmark": False,
    }

    def __init__(self, model: YOLO, cfg: dict):
        self.model = model

        cfg = self._check_cfg(cfg)
        self.cfg = cfg
        self.names = cfg["names"]

        val_dataset = YoloDataset(cfg["val_dataset"], model.imgsz, names=self.names)
        train_dataset = YoloDataset(cfg["train_dataset"], model.imgsz, mosaic_prob=cfg["mosaic"],
                                    mixup_prob=cfg["mixup"], names=self.names, augments=eval(cfg["transforms"]))
        model.names = self.names
        if train_dataset.names != val_dataset.names or train_dataset.names != self.names:
            LOGGER.warning(
                f"Names of train_dataset: {train_dataset.names}, val_dataset: {val_dataset.names} and the cfg: {self.names} would better be equal.\n")
        self.datasets = {"train": train_dataset, "val": val_dataset}

    def _setup_train(self, resume, freeze_epochs):
        if self.model.device != torch.device("cpu") and self.cfg["benchmark"]:
            torch.backends.cudnn.benchmark = True
        if self.cfg["amp"]:
            LOGGER.warning("Enabling amp may cause problems in division or log. Be careful to enable it.")

        num_classes = len(self.names)

        fit_monitor = FitMonitor(self.cfg["epochs"], self.cfg["save_dir"], self.cfg["save_period"],
                                 self.cfg["patience"])
        start_epoch = fit_monitor.start(self.model.model.ckpt, freeze_epochs, resume)
        self.model.model.attempt_reset_nc(num_classes)

        train_iter = self.datasets["train"].get_dataloader(self.cfg["batch_size"])
        val_iter = self.datasets["val"].get_dataloader(self.cfg["batch_size"])

        optimizer = YoloOptimizer(self.base_model, self.cfg, len(train_iter))
        if resume:
            optimizer.check_resume(self.model.model.ckpt)

        yolo_loss = YoloLoss(self.model)
        return fit_monitor, train_iter, val_iter, optimizer, yolo_loss, start_epoch

    def train(self, freeze_epochs=0, resume=False):
        model = self.base_model
        fit_monitor, train_iter, val_iter, optimizer, yolo_loss, start_epoch = self._setup_train(resume,
                                                                                                 freeze_epochs)
        log_writer = YoloLogWriter(self.cfg["log_dir"], resume)
        batch_trainer = BatchTrainer(self.model, yolo_loss, self.cfg["epochs"], self.cfg["amp"])

        for epoch in range(start_epoch, self.cfg["epochs"]):
            state = fit_monitor.check_epoch(epoch, freeze_epochs, self.cfg["close_mosaic"])
            if state == "freeze":
                LOGGER.step("Freezing model")
                self.model.model.freeze()

            elif state == "unfreeze":
                LOGGER.step("Unfreezing model")
                self.model.model.unfreeze()

            elif state == "close mosaic":
                LOGGER.step("Closing Mosaic")
                train_iter = self._close_mosaic()

            batch_trainer.start()
            optimizer.optimize(batch_trainer(train_iter), model, self.cfg["batch_size"])

            val_loss = self.dataset_loss(yolo_loss, val_iter)
            train_loss = batch_trainer.train_loss / len(train_iter)

            log_writer.write(epoch, train_loss, val_loss, optimizer.lrs)
            fit_monitor.step(model, optimizer.optimizer, epoch, train_loss, val_loss)
        fit_monitor.finish()

    def dataset_loss(self, loss, data_loader):
        total_loss = 0.0
        for i, (imgs, targets) in enumerate(data_loader):
            with torch.no_grad():
                predictions, targets = self.model.train_process(imgs, targets)
                total_loss += loss(predictions, targets, "batch_val_loss")
        return total_loss / len(data_loader)

    def _close_mosaic(self):
        self.datasets["train"].close_mosaic()
        return self.datasets["train"].get_dataloader(self.cfg["batch_size"])

    @classmethod
    def _check_cfg(cls, cfg):
        for k, v in cls.default_cfg.items():
            if k not in cfg:
                cfg[k] = v
        for k, v in cfg.items():
            if k not in cfg:
                LOGGER.warning(f"Unexpected key: {k}")
        return cfg

    @property
    def base_model(self):
        return self.model.model.model


if __name__ == '__main__':
    model = YOLO("models/yolov5m.pt", device_id=0)
    model.train("train_fire.yaml")
