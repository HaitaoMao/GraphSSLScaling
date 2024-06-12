from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple, NamedTuple, List


class Graph(NamedTuple):
    x: torch.FloatTensor
    edge_index: torch.LongTensor
    edge_weights: Optional[torch.FloatTensor]

    def unfold(self) -> Tuple[torch.FloatTensor, torch.LongTensor, Optional[torch.FloatTensor]]:
        return self.x, self.edge_index, self.edge_weights


class Augmentor(ABC):
    """Base class for graph augmentors."""
    def __init__(self):
        pass

    @abstractmethod
    def augment(self, g: Graph) -> Graph:
        raise NotImplementedError(f"GraphAug.augment should be implemented.")

    def __call__(
            self, x: torch.FloatTensor,
            edge_index: torch.LongTensor, edge_weight: Optional[torch.FloatTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.augment(Graph(x, edge_index, edge_weight)).unfold()


class Compose(Augmentor):
    def __init__(self, augmentors: List[Augmentor]):
        super(Compose, self).__init__()
        self.augmentors = augmentors

    def augment(self, g: Graph) -> Graph:
        for aug in self.augmentors:
            g = aug.augment(g)
        return g


class RandomChoice(Augmentor):
    def __init__(self, augmentors: List[Augmentor], num_choices: int):
        super(RandomChoice, self).__init__()
        assert num_choices <= len(augmentors)
        self.augmentors = augmentors
        self.num_choices = num_choices

    def augment(self, g: Graph) -> Graph:
        num_augmentors = len(self.augmentors)
        perm = torch.randperm(num_augmentors)
        idx = perm[:self.num_choices]
        for i in idx:
            aug = self.augmentors[i]
            g = aug.augment(g)
        return g

class RandomChoiceWithProb(Augmentor):
    def __init__(self, augmentors: List[Augmentor], num_choices: int, probabilities: List[float]):
        super(RandomChoiceWithProb, self).__init__()
        assert num_choices <= len(augmentors)
        assert len(probabilities) == len(augmentors)
        self.augmentors = augmentors
        self.num_choices = num_choices
        self.probabilities = probabilities

    def augment(self, g: Graph) -> Graph:
        # Sample indices according to the specified probabilities
        idx = torch.multinomial(torch.tensor(self.probabilities), self.num_choices, replacement=False)
        for i in idx:
            aug = self.augmentors[i]
            g = aug.augment(g)
        return g