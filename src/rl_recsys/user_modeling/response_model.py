import abc
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from sklearn.metrics.pairwise import cosine_similarity


class AbstractResponseModel(metaclass=abc.ABCMeta):
    def __init__(self, null_response: float = -1.0) -> None:
        self.null_response = null_response

    @abc.abstractmethod
    def generate_response(
        self,
        estimated_user_state: torch.Tensor,
        doc_repr: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate the user response (reward) to a slate,
        is a function of the user state and the chosen document in the slate.

        Args:
            estimated_user_state (np.array): estimated user state
            doc_repr (np.array): document representation

        Returns:
            float: user response
        """
        pass

    def generate_null_response(self) -> torch.Tensor:
        return torch.tensor(self.null_response)


class AmplifiedResponseModel(AbstractResponseModel):
    def __init__(self, amp_factor: int = 1, **kwds: Any) -> None:
        super().__init__(**kwds)
        self.amp_factor = amp_factor

    @abc.abstractmethod
    def _generate_response(
        self,
        estimated_user_state: torch.Tensor,
        doc_repr: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        pass

    def generate_response(
        self,
        estimated_user_state: torch.Tensor,
        doc_repr: torch.Tensor,
        slate: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return (
            self._generate_response(estimated_user_state, doc_repr, slate, **kwargs)
            * self.amp_factor
        )

    def generate_null_response(self) -> float:
        return super().generate_null_response() * self.amp_factor


class WeightedDotProductResponseModel(AmplifiedResponseModel):
    def __init__(self, amp_factor: int = 1, alpha: float = 1.0, **kwds: Any) -> None:
        super().__init__(amp_factor, **kwds)
        self.alpha = alpha

    def diversity_score(self, slate: torch.Tensor):
        slate1 = slate[1:]

        # Convert the rest of the tensors in the list to a tensor matrix
        slate_items = slate1[:-1]
        if slate_items.numel() == 0:
            return 0
        else:
            tensor_tuple = tuple(slate[:-1])
            tensor_matrix = torch.stack(tensor_tuple, dim=0)
            last_tensor = slate[-1]

            # Convert the last tensor to (1, 20) shape
            last_tensor = last_tensor.view(1, -1)

            # Compute the cosine similarity between the last tensor and the rest of the tensors
            similarities = cosine_similarity(last_tensor, tensor_matrix)

            # Calculate the diversity score as the average dissimilarity
            d_score = 1 - similarities.mean()
            return d_score

    def _generate_response(
        self,
        estimated_user_state: torch.Tensor,
        doc_repr: torch.Tensor,
        slate: torch.Tensor,
        doc_quality: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        diversity = self.diversity_score(slate)
        satisfaction = torch.dot(estimated_user_state, doc_repr)
        response = (1 - self.alpha) * diversity + self.alpha * doc_quality
        return response


class CosineResponseModel(AmplifiedResponseModel):
    def _generate_response(
        self,
        estimated_user_state: torch.Tensor,
        doc_repr: torch.Tensor,
    ) -> torch.Tensor:
        satisfaction = torch.nn.functional.cosine_similarity(
            estimated_user_state, doc_repr, dim=0
        )
        return satisfaction


class DotProductResponseModel(AmplifiedResponseModel):
    def _generate_response(
        self,
        estimated_user_state: torch.Tensor,
        doc_repr: torch.Tensor,
    ) -> torch.Tensor:
        satisfaction = torch.dot(estimated_user_state, doc_repr)
        return satisfaction
