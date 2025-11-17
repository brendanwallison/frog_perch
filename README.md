FrogCall + Perch v2 training pipeline (modular)

1) Install deps
   pip install tensorflow librosa soundfile pandas numpy

2) Download Perch v2 SavedModel from Kaggle (google/bird-vocalization-classifier)
   and set PERCH_SAVEDMODEL_PATH in config.py to the directory containing saved_model.

3) Set AUDIO_DIR and ANNOTATION_DIR in config.py.

4) Run:
   python main.py

Notes:
 - Dataset returns (spatial_embedding, label, audio_file, start_sample).
 - spatial_embedding expected shape (16,4,1536).
 - All windows are 5s at 32 kHz for Perch. Metadata is computed using original sample rate (DATASET_SAMPLE_RATE).
