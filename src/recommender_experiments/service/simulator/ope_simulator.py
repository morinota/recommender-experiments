import itertools
from pathlib import Path
import random
from typing import Callable, Literal, Optional, TypedDict
from joblib import Parallel, delayed
import numpy as np
from obp.dataset import SyntheticBanditDataset, logistic_reward_function
from obp.ope import ReplayMethod, InverseProbabilityWeighting, BaseOffPolicyEstimator
import polars as pl
from obp.policy import IPWLearner, NNPolicyLearner, Random, LogisticTS, BernoulliTS
import pydantic
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from recommender_experiments.service.opl.two_tower_nn_model import (
    PolicyByTwoTowerModel,
)
from recommender_experiments.service.utils.expected_reward_functions import (
    ContextFreeBinary,
    ContextAwareBinary,
)
from recommender_experiments.service.utils.logging_policies import random_policy
