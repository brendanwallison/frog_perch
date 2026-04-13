from src.frog_perch.nn_training.train import train
import configs.nn_config as config

if __name__ == '__main__':
    # The updated train function now returns the model, 
    # the validation dataset (for calibration), 
    # and the test results (for final reporting).
    model, val_ds = train(
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE
    )
    
    # Optional: If you want to use the model immediately for 
    # validation-based calibration, you have val_ds ready here.
    print("Pipeline complete. Model is ready for inference or calibration.")