import numpy as np
from collections import OrderedDict
import torch
import warnings
import re
from torch import nn


class MetaModule(nn.Module):
    """
    Base class for PyTorch meta-learning modules. These modules accept an
    additional argument `params` in their `forward` method.

    Notes
    -----
    Objects inherited from `MetaModule` are fully compatible with PyTorch
    modules from `torch.nn.Module`. The argument `params` is a dictionary of
    tensors, with full support of the computation graph (for differentiation).
    """

    def __init__(self):
        super(MetaModule, self).__init__()
        self._children_modules_parameters_cache = dict()

    def meta_named_parameters(self, prefix="", recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items()
            if isinstance(module, MetaModule)
            else [],
            prefix=prefix,
            recurse=recurse,
        )
        for elem in gen:
            yield elem

    def meta_parameters(self, recurse=True):
        for _, param in self.meta_named_parameters(recurse=recurse):
            yield param

    def get_subdict(self, params, key=None):
        if params is None:
            return None

        all_names = tuple(params.keys())
        if (key, all_names) not in self._children_modules_parameters_cache:
            if key is None:
                self._children_modules_parameters_cache[(key, all_names)] = all_names

            else:
                key_escape = re.escape(key)
                key_re = re.compile(r"^{0}\.(.+)".format(key_escape))

                self._children_modules_parameters_cache[(key, all_names)] = [
                    key_re.sub(r"\1", k)
                    for k in all_names
                    if key_re.match(k) is not None
                ]

        names = self._children_modules_parameters_cache[(key, all_names)]
        if not names:
            warnings.warn(
                "Module `{0}` has no parameter corresponding to the "
                "submodule named `{1}` in the dictionary `params` "
                "provided as an argument to `forward()`. Using the "
                "default parameters for this submodule. The list of "
                "the parameters in `params`: [{2}].".format(
                    self.__class__.__name__, key, ", ".join(all_names)
                ),
                stacklevel=2,
            )
            return None

        return OrderedDict([(name, params[f"{key}.{name}"]) for name in names])


class MetaSequential(nn.Sequential, MetaModule):
    __doc__ = nn.Sequential.__doc__

    def forward(self, input, params=None):
        for name, module in self._modules.items():
            if isinstance(module, MetaModule):
                input = module(input, params=self.get_subdict(params, name))
            elif isinstance(module, nn.Module):
                input = module(input)
            else:
                raise TypeError(
                    "The module must be either a torch module "
                    "(inheriting from `nn.Module`), or a `MetaModule`. "
                    "Got type: `{0}`".format(type(module))
                )
        return input


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, "weight"):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity="relu", mode="fan_in")


class BatchLinear(nn.Linear, MetaModule):
    """A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork."""

    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get("bias", None)
        weight = params["weight"]

        output = input.matmul(
            weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2)
        )
        output += bias.unsqueeze(-2)
        return output


class FCLayer(MetaModule):
    def __init__(self, in_features, out_features, nonlinearity="relu", norm=None):
        super().__init__()
        self.net = [BatchLinear(in_features, out_features)]

        if norm == "layernorm":
            self.net.append(
                nn.LayerNorm([out_features], elementwise_affine=True),
            )
        elif norm == "layernorm_na":
            self.net.append(
                nn.LayerNorm([out_features], elementwise_affine=False),
            )

        if nonlinearity == "relu":
            self.net.append(nn.ReLU(inplace=True))
        elif nonlinearity == "leaky_relu":
            self.net.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = MetaSequential(*self.net)
        self.net.apply(init_weights_normal)

    def forward(self, input, params=None):
        return self.net(input, params=self.get_subdict(params, "net"))


class FCBlock(MetaModule):
    def __init__(
        self,
        hidden_ch,
        num_hidden_layers,
        in_features,
        out_features,
        outermost_linear=False,
        norm=None,
        activation="relu",
        nonlinearity="relu",
    ):
        super().__init__()

        self.net = []
        self.net.append(
            FCLayer(
                in_features=in_features,
                out_features=hidden_ch,
                nonlinearity=nonlinearity,
                norm=norm,
            )
        )

        for i in range(num_hidden_layers):
            self.net.append(
                FCLayer(
                    in_features=hidden_ch,
                    out_features=hidden_ch,
                    nonlinearity=nonlinearity,
                    norm=norm,
                )
            )

        if outermost_linear:
            self.net.append(
                BatchLinear(in_features=hidden_ch, out_features=out_features)
            )
        else:
            self.net.append(
                FCLayer(
                    in_features=hidden_ch,
                    out_features=out_features,
                    nonlinearity=nonlinearity,
                    norm=norm,
                )
            )

        self.net = MetaSequential(*self.net)
        self.net.apply(init_weights_normal)

    def forward(self, input, params=None):
        return self.net(input, params=self.get_subdict(params, "net"))


class SineLayer(MetaModule):
    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = float(omega_0)

        self.is_first = is_first

        self.in_features = in_features
        self.linear = BatchLinear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward_with_film(self, input, gamma, beta):
        intermed = self.linear(input)
        return torch.sin(gamma * self.omega_0 * intermed + beta)

    def forward(self, input, params=None):
        intermed = self.linear(input, params=self.get_subdict(params, "linear"))
        return torch.sin(self.omega_0 * intermed)


class Siren(MetaModule):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=False,
        first_omega_0=30,
        hidden_omega_0=30.0,
        special_first=True,
    ):
        super().__init__()
        self.hidden_omega_0 = hidden_omega_0

        layer = SineLayer

        self.net = []
        self.net.append(
            layer(
                in_features,
                hidden_features,
                is_first=special_first,
                omega_0=first_omega_0,
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                layer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if outermost_linear:
            final_linear = BatchLinear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / 30.0,
                    np.sqrt(6 / hidden_features) / 30.0,
                )
            self.net.append(final_linear)
        else:
            self.net.append(
                layer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.ModuleList(self.net)

    def forward(self, coords, params=None):
        x = coords

        for i, layer in enumerate(self.net):
            x = layer(x, params=self.get_subdict(params, f"net.{i}"))

        return x

    def forward_with_film(self, coords, film):
        x = coords

        for i, (layer, layer_film) in enumerate(zip(self.net, film)):
            if i < len(self.net) - 1:
                x = layer.forward_with_film(x, layer_film["gamma"], layer_film["beta"])
            else:
                x = layer.forward(x)

        return x
