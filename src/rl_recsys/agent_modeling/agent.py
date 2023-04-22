import abc
from typing import TypeVar

import torch
import torch.nn as nn


torch_model = TypeVar("torch_model", bound=torch.nn.Module)


class AbstractSlateAgent(metaclass=abc.ABCMeta):
    # model an abstract agent recommending slates of documents
    def __init__(self, slate_gen) -> None:
        self.slate_gen = slate_gen

    def get_slate(
        self, state: torch.Tensor, candidate_docs: torch.Tensor
    ) -> torch.Tensor:
        """Get the action (slate) of the agent"""
        scores, ids = self.slate_gen(state, candidate_docs)
        return ids


class BeliefAgent(
    nn.Module,
):
    # model an abstract agent with a belief state
    def __init__(self, agent: torch_model) -> None:
        nn.Module.__init__(self)

        self.agent = agent
        # self.belief_model = belief_model

    def update_belief(self, *args, **kwargs) -> torch.Tensor:
        """Update the belief state of the agent"""

        return self.belief_model(*args, **kwargs)

    def get_action(
        self, docs_scores: torch.Tensor, docs_qvalues: torch.Tensor
    ) -> torch.Tensor:
        # agent is extending AbstractSlateAgent
        return self.agent.get_slate(docs_scores, docs_qvalues)  # type: ignore
