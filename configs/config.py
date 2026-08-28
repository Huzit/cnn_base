class Config:

    # Random Seed
    SEED = 42

    # Device
    DEVICE = "cuda"

    # Dataset
    TRAIN_DIR = "./data/train"
    VALID_DIR = "./data/valid"
    TEST_DIR = "./data/test"

    # Image
    IMAGE_SIZE = (512, 512) or 512
    NUM_CHANNELS = 3

    # DataLoader
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    PIN_MEMORY = True
    SHUFFLE = True

    # Model
    MODEL_NAME = "UNet"
    NUM_CLASSES = 1

    # Training
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Optimizer
    OPTIMIZER = "AdamW"

    # Scheduler
    SCHEDULER = "CosineAnnealingLR"

    # Loss Function
    LOSS = "DiceBCELoss"

    # Checkpoint
    SAVE_DIR = "./checkpoints"

    # Output
    RESULT_DIR = "./results"

cfg = Config()