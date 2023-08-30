import abc
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Type, TypeVar

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from rl_recsys.user_modeling.choice_model import AbstractChoiceModel
from rl_recsys.user_modeling.features_gen import AbstractFeaturesGenerator
from rl_recsys.user_modeling.response_model import AbstractResponseModel
from rl_recsys.user_modeling.user_state import AbstractUserState

user_state_model_type = TypeVar("user_state_model_type", bound=AbstractUserState)
user_choice_model_type = TypeVar("user_choice_model_type", bound=AbstractChoiceModel)
user_response_model_type = TypeVar(
    "user_response_model_type", bound=AbstractResponseModel
)
feature_gen_type = TypeVar("feature_gen_type", bound=AbstractFeaturesGenerator)

randomList = []


class UserModel(nn.Module):
    def __init__(
        self,
        user_features: torch.Tensor,
        user_state_model: user_state_model_type,
        user_choice_model: user_choice_model_type,
        user_response_model: user_response_model_type,
        sess_budget: int = 30,
    ) -> None:
        super().__init__()

        self.state_model = user_state_model
        self.choice_model = user_choice_model
        self.response_model = user_response_model
        self.register_buffer("_features", user_features)
        self.sess_budget = sess_budget

        # initialized by init budget
        self.budget = self.init_budget()

    @property
    def features(self) -> torch.Tensor:
        return self._features  # type: ignore

    def get_state(self):
        return self.state_model.user_state

    def is_terminal(self) -> bool:
        return self.budget <= 0

    def init_budget(self) -> float:
        return self.sess_budget

    def update_budget(self, response: torch.Tensor, doc_length: int) -> None:
        _response = response.item()
        depreciation = doc_length - (9 / 34) * doc_length * _response
        self.budget -= depreciation

    def update_budget_noselection(self) -> None:
        self.budget -= 0.5

    def get_boredom(self):
        return self.state_model.boredom


class UserSampler:
    # has to call user features generator to initialize a user
    def __init__(
        self,
        user_feature_gen: feature_gen_type,
        state_model_cls: type[user_state_model_type],
        choice_model_cls: type[user_choice_model_type],
        response_model_cls: type[user_response_model_type],
        state_model_kwargs: dict[str, Any] = {},
        choice_model_kwargs: dict[str, Any] = {},
        response_model_kwargs: dict[str, Any] = {},
        sess_budget: int = 30,
        num_user_features: int = 14,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.device = device

        self.sess_budget = sess_budget
        self.state_model_cls = state_model_cls
        self.choice_model_cls = choice_model_cls
        self.response_model_cls = response_model_cls
        self.feature_gen = user_feature_gen
        self.num_user_features = num_user_features

        self.state_model_kwargs = state_model_kwargs
        self.choice_model_kwargs = choice_model_kwargs
        self.response_model_kwargs = response_model_kwargs

        self.users: List[UserModel] = []

    def _generate_user(self) -> UserModel:
        # generate a user
        user_features = self.feature_gen(num_features=self.num_user_features)
        user_features = torch.Tensor(user_features)

        # initialize models
        state_model = self.state_model_cls(
            user_features=user_features, **self.state_model_kwargs
        )
        choice_model = self.choice_model_cls(**self.choice_model_kwargs)
        response_model = self.response_model_cls(**self.response_model_kwargs)

        user = UserModel(
            user_features=user_features,
            user_state_model=state_model,
            user_choice_model=choice_model,
            user_response_model=response_model,
            sess_budget=self.sess_budget,
        ).to(self.device)

        return user

    def generate_users(self, num_users: int = 100) -> List[UserModel]:
        self.users = [self._generate_user() for _ in range(num_users)]
        return self.users

    def sample_user(self) -> UserModel:
        assert (
            len(self.users) > 0
        ), "No users generated yet. call generate_user_batch() first.)"

        i = np.random.randint(0, len(self.users))
        # while True:
        #     i = np.random.randint(0, len(self.users))
        #     if i not in randomList:
        #         break
        # randomList.append(i)

        print(f"sampled user {i}")
        return self.users[i]

    def test_sample_user(self) -> UserModel:
        assert (
            len(self.users) > 0
        ), "No users generated yet. call generate_user_batch() first.)"

        # i = np.random.randint(0, len(self.users))
        while True:
            i = np.random.randint(0, len(self.users))
            if i not in randomList:
                break
        randomList.append(i)

        print(f"sampled user {i}")
        return self.users[i]
