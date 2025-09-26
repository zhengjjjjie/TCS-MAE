from easydict import EasyDict as edict
cfg = edict()

# data settings
cfg.TRAIN_DATA_FILE = r'/x32001107/data'
cfg.EVAL_DATA_FILE = r''
cfg.TEST_DATA_FILE = r'D:\data\chest_reconstruction\penu\covid19'

# model settings
cfg.BATCH_SIZE = 5
cfg.INPUT_SHAPE = [256, 256]
cfg.SPATIALA_MASK_RATIO = 0
cfg.SPATIALA_MASK_SIZE = 16
cfg.INTENSITY_MASK_RATIO = 0.7
cfg.INTENSITY_MASK_SIZE = 5
cfg.EPOCHS = 100
cfg.LR = 0.00001
cfg.DECAY_STEPS = 100000
cfg.DECAY_RATE = 0.96
cfg.STEPS_PER_EPOCH = 100
cfg.CHECKPOINTS_ROOT = 'checkpoints'
cfg.MAX_KEEPS_CHECKPOINTS = 1
cfg.AMP = False


