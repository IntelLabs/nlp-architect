import glob
import os
import pytorch_lightning as pl
from bert_token_classifier import BertForToken, load_config, get_trainer
from sys import argv
# pylint: disable=attribute-defined-outside-init

# pylint: disable=no-member
def main(config_yaml):
    config = load_config(config_yaml)
    pl.seed_everything(config.seed)

    model = BertForToken(config)
    trainer = get_trainer(model, config)

    if config.do_train:
        trainer.fit(model)
    trainer.logger.log_hyperparams(config)
    trainer.logger.save()

    if config.do_predict:        
        # Bug in pl==0.85 -> testing only works with num gpus=1
        trainer = get_trainer(model, config, gpus=1)

        checkpoints = list(sorted(glob.glob(os.path.join(config.output_dir, "checkpointepoch=*.ckpt"), recursive=True)))
        model = model.load_from_checkpoint(checkpoints[-1])
        model.freeze()
        trainer.test(model)

if __name__ == "__main__":
    argv = ['', 'example']
    main(argv[1])
