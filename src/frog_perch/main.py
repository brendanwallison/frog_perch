# main.py
from training.train import train
import frog_perch.config as config

if __name__ == '__main__':
    train(label_mode=config.LABEL_MODE, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
          perch_savedmodel_path=config.PERCH_SAVEDMODEL_PATH)
