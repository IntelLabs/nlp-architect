import pytorch_lightning as pl
from bert_token_classifier import BertForToken, load_config, load_last_ckpt
from sys import argv

# pylint: disable=no-member
def main(config_yaml):
    config = load_config(config_yaml)
    pl.seed_everything(config.seed)

    model = BertForToken(config)
    trainer = model.get_trainer()

    if config.do_train:
        trainer.fit(model)

    trainer.logger.log_hyperparams(config)
    trainer.logger.save()

    if config.do_predict:        
        # Bug in pytorch_lightning==0.85 -> testing only works with num gpus=1
        trainer = model.get_trainer(gpus_override=1)
        trainer.test(load_last_ckpt(model, config))

if __name__ == "__main__":
    argv = ['', 'example']
    main(argv[1])
