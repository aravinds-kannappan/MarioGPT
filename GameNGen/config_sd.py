# config_sd.py
HEIGHT = 240
WIDTH = 320
BUFFER_SIZE = 3
ZERO_OUT_ACTION_CONDITIONING_PROB = 0.1
CFG_GUIDANCE_SCALE = 1.5
DEFAULT_NUM_INFERENCE_STEPS = 50
NUM_BUCKETS = 10
PRETRAINED_MODEL_NAME_OR_PATH = "runwayml/stable-diffusion-v1-5"

TRAINING_DATASET_DICT = {
    "small": "Flaaaande/mario-small",
    "full": "Flaaaande/mario-full"
}
