import torch

from typing import List, Union


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        conditional_size: int = 0,
        activation: torch.nn.Module = torch.nn.ReLU(),
        classification: bool = False,
        generator: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.generator = generator

        self.layers = torch.nn.ModuleList()

        # input and hidden layers
        prev_size = input_size + conditional_size
        for size in hidden_sizes:
            self.layers.append(torch.nn.Linear(prev_size, size))
            self.layers.append(activation)
            prev_size = size

        # output layer
        self.layers.append(torch.nn.Linear(prev_size, output_size))

        if classification and output_size > 1:
            self.layers.append(torch.nn.LogSoftmax(dim=-1))
        elif classification:
            self.layers.append(torch.nn.Sigmoid())

    def forward(
        self, x: torch.Tensor, cond: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        if len(x.shape) > 2:
            original_shape = x.shape
            x = x.view(x.size(0), -1)

        if cond is not None:
            x = torch.cat([x, cond], dim=-1)

        for layer in self.layers:
            x = layer(x)

        if self.generator:
            return x.view(original_shape)

        return x
