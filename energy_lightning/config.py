# Module
EMBED_DIM = 9
C_DIM = 30
D_DIM = 100
INTERACTION_DIM = 60
NUM_INTERACTION = 3
READOUT_DIM = 15
LEARNING_RATE = 0.003

# DataModule
ROOT = "./datasets/"
NAME = "revised aspirin"
BATCH_SIZE = 8
TRAIN_SIZE = 1000
VAL_SIZE = 200
TEST_SIZE = 2000
PRED_SIZE = 1
NUM_WORKERS = 4

# Trainer
NUM_EPOCHS = 1000
ACCELERATOR = "gpu"
