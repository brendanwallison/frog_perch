# main.py
from frog_perch.training.train import train
import frog_perch.config as config

if __name__ == '__main__':
    model, val_ds = train(
        label_mode=config.LABEL_MODE,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        pool_method=config.POOL_METHOD
    )
