import glob
import os
import pytorch_lightning as pl
from pytorch_lightning.core.saving import load_hparams_from_yaml
from bert_token_classifier import BertForToken, LoggingCallback
from argparse import Namespace
from pathlib import Path
from sys import argv
# pylint: disable=attribute-defined-outside-init

def generic_trainer(model: BertForToken, args, gpus=None):
    Path(model.hparams.output_dir).mkdir(exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=1
    )
    gpus = args.gpus if gpus is None else gpus
    distributed_backend = "ddp" if gpus > 1 else None

    return pl.Trainer(
        logger=True,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gpus=args.gpus if gpus is None else gpus,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
        checkpoint_callback=checkpoint_callback,
        callbacks=[LoggingCallback(), pl.callbacks.LearningRateLogger()],
        fast_dev_run=args.fast_dev_run,
        val_check_interval=args.val_check_interval,
        weights_summary=None,
        resume_from_checkpoint=args.resume_from_checkpoint,
        distributed_backend=distributed_backend,
    )

def load_config(name):
    configs_dir = Path(os.path.dirname(os.path.realpath(__file__))) / 'configs'
    config = Namespace(**load_hparams_from_yaml(configs_dir / (name + '.yaml')))
    assert config
    return config

# pylint: disable=no-member
def main(config_yaml):
    config = load_config(config_yaml)
    pl.seed_everything(config.seed)

    model = BertForToken(config)
    trainer = generic_trainer(model, config)

    if config.do_train:
        trainer.fit(model)
    trainer.logger.log_hyperparams(config)
    trainer.logger.save()

    if config.do_predict:        
        # Bug in pl==0.85 -> testing only works with num gpus=1
        trainer = generic_trainer(model, config, gpus=1)

        checkpoints = list(sorted(glob.glob(os.path.join(config.output_dir, "checkpointepoch=*.ckpt"), recursive=True)))
        model = model.load_from_checkpoint(checkpoints[-1])
        model.freeze()
        trainer.test(model)

if __name__ == "__main__":
    argv = ['', 'example']
    main(argv[1])
