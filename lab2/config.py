
from pathlib import Path

LAB2_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LAB2_DIR.parent

BOARD_SIZE = 9
NUM_CELLS = BOARD_SIZE * BOARD_SIZE
ACTION_SPACE = NUM_CELLS * NUM_CELLS
INPUT_CHANNELS = 7  # white reg/queen/vozhd, black reg/queen/vozhd, side-to-move
INPUT_DIM = NUM_CELLS * INPUT_CHANNELS

HOST = "127.0.0.1"
TRAIN_PORT = 5555
INFERENCE_PORT = 5556

# PPO hyperparameters (tuned for 9x9 Scythian checkers)
GAMMA = 0.995
LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 5e-4
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01

UPDATE_EPOCHS = 10
MINIBATCH_SIZE = 128
ROLLOUT_SIZE = 256

# Reward shaping — вождь (king) ценнее обычной шашки
CAPTURE_REWARD = 0.5
QUEEN_REWARD = 0.8
VOZHD_CAPTURE_REWARD = 3.0
WIN_REWARD = 10.0
LOSE_PENALTY = -10.0
DRAW_REWARD = 0.0
STEP_PENALTY = -0.005

# Curriculum training
SUPERVISED_POSITIONS = 2000
SUPERVISED_EPOCHS = 20
SUPERVISED_BATCH = 256
TEACHER_DEPTH = 4
CURRICULUM_DEPTHS = [1, 2, 3, 4]
GAMES_PER_DEPTH = 150
EVAL_GAMES = 20
TARGET_WIN_RATE = 0.95

CHECKPOINT_DIR = LAB2_DIR / "checkpoints"
RUN_DIR = LAB2_DIR / "runs" / "scythian_checkers"
BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pth"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Comparison / eval vs Minimax d=4
DEFAULT_OPPONENT = "random"
MINIMAX_DEPTH = 4
RL_SEARCH_DEPTH = 6
RL_ROOT_WIDTH = 20
COMPARISON_GAMES = 20
COMPARISON_USE_GREEDY = False

# GUI: быстрый 2-ply (policy top-K + value), лимит времени на ход
GUI_SEARCH_DEPTH = 2
GUI_ROOT_WIDTH = 8
GUI_OPP_WIDTH = 4
GUI_MOVE_TIME_SEC = 0.35

# BC от Minimax (только обучение, не в GUI)
BC_POSITIONS = 20000
BC_EPOCHS = 12
TEACHER_DEPTH = 4

# Self-play / training speed
LOG_EVERY_N_GAMES = 100
SAVE_EVERY_N_GAMES = 500
SELF_PLAY_MAX_PLIES = 500
