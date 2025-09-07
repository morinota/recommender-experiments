import itertools
import random
from pathlib import Path
from typing import Callable, Literal, Optional, TypedDict

import numpy as np
import polars as pl
import pydantic
from joblib import Parallel, delayed
from obp.dataset import SyntheticBanditDataset, logistic_reward_function
from obp.ope import BaseOffPolicyEstimator, InverseProbabilityWeighting, ReplayMethod
from obp.policy import BernoulliTS, IPWLearner, LogisticTS, NNPolicyLearner, Random
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from recommender_experiments.service.opl.two_tower_nn_model import (
    PolicyByTwoTowerModel,
)
from recommender_experiments.service.utils.expected_reward_functions import (
    ContextAwareBinary,
    ContextFreeBinary,
)
from recommender_experiments.service.utils.logging_policies import random_policy
