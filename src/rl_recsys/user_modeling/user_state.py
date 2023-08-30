import abc
from typing import Any, Type, TypeVar

import torch
import torch.nn as nn

from rl_recsys.user_modeling.features_gen import AbstractFeaturesGenerator

feature_gen_type = TypeVar("feature_gen_type", bound=AbstractFeaturesGenerator)


class AbstractUserState(nn.Module, metaclass=abc.ABCMeta):
    # hidden state of the user
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        user_state = self._generate_state(**kwargs)
        self.register_buffer("user_state", user_state)
        # used to reset the intent to the initial create one at the end of an episode
        self.register_buffer("user_state_init", user_state)

    @abc.abstractmethod
    def _generate_state(self, **kwargs: Any) -> torch.Tensor:
        """Generate the user hidden state"""
        pass

    @abc.abstractmethod
    def update_state(self, selected_doc_feature: torch.Tensor, **kwargs) -> None:
        """Update the user hidden state"""
        pass


class ObservableUserState(AbstractUserState):
    def __init__(
        self,
        user_features: torch.Tensor,
        interest_update_rate: float = 0.3,  # y in the paper
        **kwargs: Any,
    ) -> None:
        self.user_features = user_features
        self.interest_update_rate = interest_update_rate
        self.index1 = -1
        self.boredom = 0
        self.boredom_list = []
        super().__init__(**kwargs)

    def _generate_state(self) -> torch.Tensor:
        return self.user_features

    def reset_state(self) -> None:
        self.user_state = self.user_state_init

    def update_state(self, selected_doc_feature: torch.Tensor) -> None:
        index = torch.argmax(selected_doc_feature)
        delta_t = (
            -self.interest_update_rate * torch.abs(self.user_state[index])  # type: ignore
            + self.interest_update_rate
        ) * -self.user_state[
            index
        ]  # type: ignore
        I = torch.dot(self.user_state, selected_doc_feature)  # type: ignore
        p_positive = (I + 1) / 2
        p_negative = (1 - I) / 2

        random = torch.rand(1)
        if random < p_positive:
            self.user_state[index] += delta_t  # type: ignore
        # if random < p_negative:
        #     self.user_state[index] -= delta_t  # type: ignore
        elif random > p_positive:
            self.user_state[index] -= delta_t  # type: ignore

        self.user_state = torch.clamp(self.user_state, -1, 1)


class BoredomObservableUserState(ObservableUserState):
    def update_state(self, selected_doc_feature: torch.Tensor) -> None:
        # Perform additional update logic here
        # You can access the parent class methods and attributes using the `super()` function
        super().update_state(selected_doc_feature)
        index = torch.argmax(selected_doc_feature)
        delta_t = (
            -self.interest_update_rate * torch.abs(self.user_state[index])  # type: ignore
            + self.interest_update_rate
        ) * -self.user_state[
            index
        ]  # type: ignore
        # print(f"Selected_Index: {self.index1}")
        # print(f"Boredom:{self.boredom}")
        if self.index1 == index:
            self.boredom += 1
        else:
            self.boredom = 0
        I = torch.dot(self.user_state, selected_doc_feature)  # type: ignore
        p_positive = (I + 1) / 2
        p_negative = (1 - I) / 2

        random = torch.rand(1)
        if (random < p_positive) & (self.boredom < 3):
            self.user_state[index] += delta_t  # type: ignore
        # if random < p_negative:
        #     self.user_state[index] -= delta_t  # type: ignore
        elif (random > p_positive) & (self.boredom < 3):
            self.user_state[index] -= delta_t  # type: ignore
        elif self.boredom >= 3:
            self.user_state[index] = -1

        self.index1 = index
        # print(f"User State:{self.user_state}")
        self.user_state = torch.clamp(self.user_state, -1, 1)


# class BoredomObservableUserState(ObservableUserState):
#     def update_state(self, selected_doc_feature: torch.Tensor) -> None:
#         # Perform additional update logic here
#         # You can access the parent class methods and attributes using the `super()` function
#         super().update_state(selected_doc_feature)
#         index = torch.argmax(selected_doc_feature)
#         delta_t = (
#             -self.interest_update_rate * torch.abs(self.user_state[index])  # type: ignore
#             + self.interest_update_rate
#         ) * -self.user_state[
#             index
#         ]  # type: ignore

#         if len(self.boredom_list) < 10:
#             self.boredom_list.append(index)
#         else:
#             self.boredom_list.append(index)
#             self.boredom_list.pop(0)
#         count = self.boredom_list.count(index)
#         I = torch.dot(self.user_state, selected_doc_feature)  # type: ignore
#         p_positive = (I + 1) / 2
#         p_negative = (1 - I) / 2

#         random = torch.rand(1)
#         if (random < p_positive) & (count < 5):
#             self.user_state[index] += delta_t  # type: ignore
#         # if random < p_negative:
#         #     self.user_state[index] -= delta_t  # type: ignore
#         elif (random > p_positive) & (count < 5):
#             self.user_state[index] -= delta_t  # type: ignore
#         elif count >= 5:
#             self.user_state[index] = -1


#         # print(f"User State:{self.user_state}")
#         self.user_state = torch.clamp(self.user_state, -1, 1)
