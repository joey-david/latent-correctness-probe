import json
import math
import random
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

# Central experiment configuration shared across modules.
CHECKPOINT_STEPS = [4, 8, 16, 32, 64, 128, 192, 256, 384, 512]
MAX_PREFIX_TOKENS = 1024  # allow analysing longer reasoning prefixes when needed
MAX_NEW_TOKENS = 512  # limit chain-of-thought length before forcing a conclusion
DEVICE = "cuda"
SEED = 356

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
