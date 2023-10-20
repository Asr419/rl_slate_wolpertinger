from ast import Tuple
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.distributions as dist


@dataclass
class DocumentSampler:
    # default values as in SlateQ paper
    num_topics: int = 20
    frac_low_quality_topics: float = 0.7
    low_quality_interval: tuple[float, float] = (-3.0, 0.0)
    low_quality_variance: float = 0.1
    high_quality_interval: tuple[float, float] = (0.0, 3.0)
    high_quality_variance: float = 0.1
    doc_length: int = 4
    seed: int = None
    device: str = "cpu"

    def __post_init__(self):
        torch.manual_seed(self.seed)
        num_low_quality_topics = int(self.frac_low_quality_topics * self.num_topics)
        num_high_quality_topics = self.num_topics - num_low_quality_topics
        # Initialize the topic distribution over documents
        low_quality_topic_probs = torch.ones(num_low_quality_topics) / self.num_topics
        high_quality_topic_probs = torch.ones(num_high_quality_topics) / self.num_topics

        self.topic_dist = dist.Categorical(
            torch.cat([low_quality_topic_probs, high_quality_topic_probs])
        )

        lq_s, lq_e = self.low_quality_interval
        hq_s, hq_e = self.high_quality_interval
        # Initialize the quality distribution for each document
        low_quality_topic_means = torch.linspace(lq_s, lq_e, num_low_quality_topics)
        high_quality_topic_means = torch.linspace(hq_s, hq_e, num_high_quality_topics)

        self.topic_means = torch.cat(
            [low_quality_topic_means, high_quality_topic_means]
        )

        self.quality_dists: list[dist.Normal] = []
        for t in range(self.num_topics):
            if t < num_low_quality_topics:
                variance = self.low_quality_variance
            else:
                variance = self.high_quality_variance

            mean_quality = self.topic_means[t]
            self.quality_dists.append(dist.Normal(mean_quality, variance))

    def sample_document(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        # Sample a topic

        topic = self.topic_dist.sample()
        doc = torch.zeros(self.num_topics)
        doc[topic] = 1
        # doc = torch.rand(self.num_topics, device=self.device)

        # Sample the quality of the document
        quality = self.quality_dists[topic].sample()
        # note in this case the doc_length is fixed
        return doc, quality, self.doc_length

    def sample_documents(
        self, num_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # sampling the candidate documents
        docs = torch.zeros(num_samples, self.num_topics)
        qualities = torch.zeros(num_samples)
        doc_length = torch.zeros(num_samples)
        torch.manual_seed(self.seed)
        for i in range(num_samples):
            doc, quality, doc_len = self.sample_document()
            docs[i] = doc
            qualities[i] = quality
            doc_length[i] = doc_len

        return docs, qualities, doc_length
