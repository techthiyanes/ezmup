import math
from copy import copy
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam


def spectral_sigma(fan_in, fan_out, init_std):
    """Spectral parameterization from the [paper](https://arxiv.org/abs/2310.17813)."""
    return (init_std / math.sqrt(fan_in)) * min(1, math.sqrt(fan_out / fan_in))


def spectral_lr(fan_in, fan_out):
    """Spectral parameterization from the [paper](https://arxiv.org/abs/2310.17813)."""
    return fan_out / fan_in


SPECTRAL_DEFAULT = (spectral_sigma, spectral_lr)


LAYER_REGISTRY = {
    "Conv1d.weight": SPECTRAL_DEFAULT,
    "Conv1d.bias": SPECTRAL_DEFAULT,
    "Conv2d.weight": SPECTRAL_DEFAULT,
    "Conv2d.bias": SPECTRAL_DEFAULT,
    "Conv3d.weight": SPECTRAL_DEFAULT,
    "Conv3d.bias": SPECTRAL_DEFAULT,
    "ConvTranspose1d.weight": SPECTRAL_DEFAULT,
    "ConvTranspose1d.bias": SPECTRAL_DEFAULT,
    "ConvTranspose2d.weight": SPECTRAL_DEFAULT,
    "ConvTranspose2d.bias": SPECTRAL_DEFAULT,
    "ConvTranspose3d.weight": SPECTRAL_DEFAULT,
    "ConvTranspose3d.bias": SPECTRAL_DEFAULT,
    "Linear.weight": SPECTRAL_DEFAULT,
    "Linear.bias": SPECTRAL_DEFAULT,
    "Embedding.weight": (1.0, 1.0),
    "Embedding.bias": (1.0, 1.0),
    "BatchNorm2d.weight": (1.0, 1.0),
    "BatchNorm2d.bias": (0.0, 1.0),
    "LayerNorm.weight": (1.0, 1.0),
    "LayerNorm.bias": (0.0, 1.0),
    "GroupNorm.weight": (1.0, 1.0),
    "GroupNorm.bias": (0.0, 1.0),
}


class Ezmup:
    """Easier maximal update parametrization(muP)."""

    def __init__(self, width_basis: int, model: nn.Module, init_std: float = 1.0):
        """Initialize Ezmup by specifying the width basis, the model, and the init_std.

        Args:
            width_basis (int): A Base dimension.
            model (nn.Module): A model to be changed.
            init_std (float, optional): The initial standard deviation of the model parameters. Defaults to 1.0.
        """
        self.width_basis = width_basis
        self.model = model
        self.init_std = init_std  # Can be a float or a dict
        self.model_param_shape_dict = {
            name: param.shape for name, param in self.model.named_parameters()
        }
        self.lr_scaling_dict = {}

    @torch.no_grad()
    def change_width_as(self, new_width: int):
        """Update model parameters with new width multiplier.

        Args:
            new_width (int): Width multiplier used for calculating μP scaling.

        Raises:
            ValueError: When the module with the name is not found.
            NotImplementedError: When the parameter class is not found in the LAYER_REGISTRY.
        """
        new_param_dict = {}
        dtype, device = (
            self.model.parameters().__next__().dtype,
            self.model.parameters().__next__().device,
        )

        for name, param in self.model.named_parameters():
            shape = self.model_param_shape_dict[name]
            new_shape = [
                new_width * (dim // self.width_basis)
                if dim % self.width_basis == 0
                else dim
                for dim in shape
            ]

            print(f"Now changing {name} from {shape} to {new_shape}")

            # if this is not a new layer, we want to skip it.
            if all(dim == new_shape[i] for i, dim in enumerate(shape)):
                new_param_dict[name] = param
                continue

            weight_shape = new_shape
            init_std, lr_scaling = SPECTRAL_DEFAULT
            # check where the parameter is in the registry. See if the parameter's class is in the registry.
            # this difficulty arises due to the fact that bias does not itself have an implication of the fan_in just from the parameter.

            if name.endswith("bias") or name.endswith("weight"):
                # remove the last part of the name.
                oname = (
                    name[: -len(".bias")]
                    if name.endswith("bias")
                    else name[: -len(".weight")]
                )
                # print(name)
                module_with_name = self.model.get_submodule(oname)

                if module_with_name is None:
                    # Exceptions must not be used with f-strings
                    msg = f"Could not find module with name {name}"
                    raise ValueError(msg)

                module_class = module_with_name.__class__.__name__
                param_classname = module_class + ".bias"
                # print(param_classname)

                if param_classname in LAYER_REGISTRY:
                    init_std, lr_scaling = LAYER_REGISTRY[param_classname]

                    if name.endswith("bias"):
                        fan_in_of_weight = self.model_param_shape_dict[
                            oname + ".weight"
                        ][-1]
                        if fan_in_of_weight % self.width_basis == 0:
                            fan_in_of_weight = fan_in_of_weight * (
                                new_width // self.width_basis
                            )
                        fan_in = fan_in_of_weight
                        fan_out = 1
                    else:
                        weight_shape = new_shape
                        fan_in = weight_shape[-1]
                        fan_out = np.prod(weight_shape[:-1])

                else:
                    # Exceptions must not be used with f-strings
                    msg = f"Could not find {param_classname} in LAYER_REGISTRY"
                    raise NotImplementedError(msg)

            else:
                # we don't recognize this parameter : it is not a bias or a weight.
                # so we will assume fan_in and fan_out are simply the product of the dimensions.
                fan_in = weight_shape[-1]
                fan_out = np.prod(weight_shape[:-1])
                init_std, lr_scaling = SPECTRAL_DEFAULT

            # assert fan_in * fan_out == np.prod(new_shape), f"fan_in * fan_out != np.prod(new_shape) for {name}, {fan_in * fan_out} != {np.prod(new_shape)}"
            print(f"{name} fan_in: {fan_in}, fan_out: {fan_out}")

            if isinstance(init_std, float):
                init_std = init_std * self._get_init_std(name)
            else:
                init_std = init_std(fan_in, fan_out, self._get_init_std(name))

            if isinstance(lr_scaling, float):
                lr_scaling = lr_scaling / 64
            else:
                lr_scaling = lr_scaling(fan_in, fan_out) / 64

            self.lr_scaling_dict[name] = lr_scaling

            new_param = torch.randn(new_shape) * init_std
            new_param_dict[name] = new_param

            # print(f"Changing {name} from {shape} to {new_shape}")

        for name, named_module in self.model.named_modules():
            if hasattr(named_module, "weight"):
                named_module.weight = torch.nn.Parameter(
                    new_param_dict[name + ".weight"],
                    requires_grad=True,
                ).to(dtype=named_module.weight.dtype)

            if hasattr(named_module, "bias"):
                named_module.bias = torch.nn.Parameter(
                    new_param_dict[name + ".bias"],
                    requires_grad=True,
                ).to(dtype=named_module.bias.dtype)

        self.model.to(dtype=dtype, device=device)

    def get_optimizer(self, optimizer_class: Any, lr: float, **kwargs):
        """Get an optimizer for the model.

        Args:
            optimizer_class (Any): Optimizer class.
            lr (float): Learning rate.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: Updated optimizer.
        """
        mup_scaling = self.lr_scaling_dict

        optimizer_groups = [
            {"params": [p], "lr": lr * mup_scaling.get(name, 1.0)}
            for name, p in self.model.named_parameters()
        ]

        return optimizer_class(optimizer_groups, **kwargs)

    def _get_init_std(self, name):
        if isinstance(self.init_std, dict):
            return self.init_std.get(name, 1.0)
        return self.init_std

    def forward(self, *args, **kwargs):
        """Forward pass of the model."""
        pass


def cov(x: torch.Tensor) -> torch.Tensor:
    """Treat `x` as a collection of vectors and its Gram matrix.

    Args:
        x (torch.Tensor): If it has shape [..., d], then it's treated as
            a collection of d-dimensional vectors

    Returns:
        torch.Tensor: a matrix of size N x N where N is the product of
            the non-last dimensions of `x`.
    """
    if x.nelement() == 1:
        width = 1
        xx = x.reshape(1, 1)
    else:
        width = x.shape[-1]
        xx = x.reshape(-1, x.shape[-1])
    return xx @ xx.T / width


def covoffdiag(x: torch.Tensor) -> torch.Tensor:
    """Get off-diagonal entries of `cov(x)` in a vector.

    Args:
        x (torch.Tensor): If it has shape [..., d], then it's treated as
            a collection of d-dimensional vectors

    Returns:
        torch.Tensor: Off-diagonal entries of `cov(x)` in a vector.
    """
    c = cov(x)
    return c[~torch.eye(c.shape[0], dtype=bool)]


#: dict of provided functions for use in coord check
FDICT = {
    "l1": lambda x: torch.abs(x).mean(),
    "l2": lambda x: (x**2).mean() ** 0.5,
    "mean": lambda x: x.mean(),
    "std": lambda x: x.std(),
    "covl1": lambda x: torch.abs(cov(x)).mean(),
    "covl2": lambda x: (cov(x) ** 2).mean() ** 0.5,
    "covoffdiagl1": lambda x: torch.abs(covoffdiag(x)).mean(),
    "covoffdiagl2": lambda x: (covoffdiag(x) ** 2).mean() ** 0.5,
}


def convert_fdict(d: dict[Any, Any]) -> dict[Any, Any]:
    """Convert a dict `d` with string values to function values.

    Args:
        d (dict[Any, Any]): a dict whose values are either strings or functions

    Returns:
        dict[Any, Any]: a new dict, with the same keys as `d`, but the string values are
        converted to functions using `FDICT`.
    """
    return dict(
        [((k, FDICT[v]) if isinstance(v, str) else (k, v)) for k, v in d.items()],
    )


def _record_coords(
    records,
    width,
    modulename,
    t,
    output_fdict=None,
    input_fdict=None,
    param_fdict=None,
):
    """Returns a forward hook that records coordinate statistics.

    Returns a forward hook that records statistics regarding the output, input,
    and/or parameters of a `nn.Module`. This hook is intended to run only once,
    on the timestep specified by `t`.

    On forward pass, the returned hook calculates statistics specified in
    `output_fdict`, `input_fdict`, and `param_fdict`, such as the normalized l1
    norm, of output, input, and/or parameters of the module. The statistics are
    recorded along with the `width`, `modulename`, and `t` (the time step) as a
    dict and inserted into `records` (which should be a list). More precisely,
    for each output, input, and/or parameter, the inserted dict is of the form

        {
            'width': width, 'module': modified_modulename, 't': t,
            # keys are keys in fdict
            'l1': 0.241, 'l2': 0.420, 'mean': 0.0, ...
        }

    where `modified_modulename` is a string that combines the `modulename` with
    an indicator of which output, input, or parameter tensor is the statistics
    computed over.

    The `*_fdict` inputs should be dictionaries with string keys and whose
    values can either be functions or strings. The string values are converted
    to functions via `convert_fdict`. The default values of `*_dict` inputs are
    converted to `output_fdict = dict(l1=FDICT['l1'])`, `input_fdict = {}`,
    `param_fdict = {}`, i.e., only the average coordinate size (`l1`) of the
    output activations are recorded.

    Args:
        records: list to append coordinate data to.
        width: width of the model. This is used only for plotting coord check later
            on, so it can be any notion of width.
        modulename: string name of the module. This is used only for plotting coord check.
        t: timestep of training. This is used only for plotting coord check.
        output_fdict: dict with string keys and whose values can either be functions or strings.
            The string values are converted to functions via `convert_fdict`.
        input_fdict: dict with string keys and whose values can either be functions or strings.
            The string values are converted to functions via `convert_fdict`.
        param_fdict: dict with string keys and whose values can either be functions or strings.
            The string values are converted to functions via `convert_fdict`.

    Returns:
        Any: a forward hook that records statistics regarding the output, input,
        and/or parameters of a `nn.Module`, as discussed above.
    """
    if output_fdict is None:
        output_fdict = dict(l1=FDICT["l1"])
    else:
        output_fdict = convert_fdict(output_fdict)
    # SIM108: Use the ternary operator if it's reasonable
    input_fdict = {} if input_fdict is None else convert_fdict(input_fdict)
    param_fdict = {} if param_fdict is None else convert_fdict(param_fdict)

    def f(module, input, output):
        def get_stat(d, x, fdict):
            if isinstance(x, tuple | list):
                for i, _x in enumerate(x):
                    _d = copy(d)
                    _d["module"] += f"[{i}]"
                    get_stat(_d, _x, fdict)
            elif isinstance(x, dict):
                for name, _x in x.items():
                    _d = copy(d)
                    _d["module"] += f"[{name}]"
                    get_stat(_d, _x, fdict)
            elif isinstance(x, torch.Tensor):
                _d = copy(d)
                for fname, f in fdict.items():
                    _d[fname] = f(x).item()
                records.append(_d)
            else:
                msg = f"Unexpected output type: {type(x)}"
                raise NotImplementedError(msg)

        with torch.no_grad():
            ret = {"width": width, "module": modulename, "t": t}

            # output stats
            if isinstance(output, tuple | list):
                for i, out in enumerate(output):
                    _ret = copy(ret)
                    _ret["module"] += f":out[{i}]"
                    get_stat(_ret, out, output_fdict)
            elif isinstance(output, dict):
                for name, out in output.items():
                    _ret = copy(ret)
                    _ret["module"] += f":out[{name}]"
                    get_stat(_ret, out, output_fdict)
            elif isinstance(output, torch.Tensor):
                _ret = copy(ret)
                for fname, f in output_fdict.items():
                    _ret[fname] = f(output).item()
                records.append(_ret)
            else:
                msg = f"Unexpected output type: {type(output)}"
                raise NotImplementedError(msg)

            # input stats
            if input_fdict:
                if isinstance(input, tuple | list):
                    for i, out in enumerate(input):
                        _ret = copy(ret)
                        _ret["module"] += f":in[{i}]"
                        get_stat(_ret, out, input_fdict)
                elif isinstance(input, dict):
                    for name, out in input.items():
                        _ret = copy(ret)
                        _ret["module"] += f":in[{name}]"
                        get_stat(_ret, out, input_fdict)
                elif isinstance(input, torch.Tensor):
                    _ret = copy(ret)
                    for fname, f in input_fdict.items():
                        _ret[fname] = f(input).item()
                    records.append(_ret)
                else:
                    msg = f"Unexpected output type: {type(input)}"
                    raise NotImplementedError(msg)

            # param stats
            if param_fdict:
                for name, p in module.named_parameters():
                    _ret = copy(ret)
                    _ret["module"] += f":param[{name}]"
                    for fname, f in param_fdict.items():
                        _ret[fname] = f(p).item()
                    records.append(_ret)
                    
        assert 1 == 1

    return f


def get_coord_data(
    model_engine,
    datapoint,
    width_list=None,
    optim_cls=Adam,
    optim_kwargs=None,
    n_seeds=1,
    n_steps=3,
) -> pd.DataFrame:
    """Get coordinate data for coord check.

    Args:
        model_engine (_type_): Ezmup model engine.
        datapoint (_type_): A datapoint to be used for forward pass.
        width_list (list, optional): _description_. Defaults to [64, 128, 256, 512, 1024, 2048, 4096, 8192].
        optim_cls (_type_, optional): _description_. Defaults to Adam.
        optim_kwargs (_type_, optional): _description_. Defaults to None.
        n_seeds (int, optional): _description_. Defaults to 1.
        n_steps (int, optional): _description_. Defaults to 3.

    Returns:
        pd.DataFrame: A dataframe containing the coordinate data.
    """
    df = []

    # Mutable default arguments are dangerous. Use None instead.
    if width_list is None:
        width_list = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

    for i in range(n_seeds):
        torch.manual_seed(i)
        for width in width_list:
            model_engine.change_width_as(width)
            optim = (
                model_engine.get_optimizer(optim_cls, lr=1e-3)
                if optim_kwargs is None
                else model_engine.get_optimizer(optim_cls, lr=1e-3, **optim_kwargs)
            )

            for j in range(n_steps):
                remove_hooks = []
                for name, module in model_engine.model.named_modules():
                    remove_hooks.append(
                        module.register_forward_hook(
                            _record_coords(
                                df,
                                width,
                                name,
                                j,
                                output_fdict=None,
                                input_fdict=None,
                                param_fdict=None,
                            ),
                        ),
                    )

                model_engine.model.train()

                loss = model_engine.forward(datapoint, model_engine.model)
                                
                loss.backward()
                optim.step()
                optim.zero_grad()

                for handle in remove_hooks:
                    handle.remove()

    return pd.DataFrame(df)



def get_coord_data(
    model_engine,
    datapoint,
    width_list=None,
    optim_cls=Adam,
    optim_kwargs=None,
    n_seeds=1,
    n_steps=3,
) -> pd.DataFrame:
    """Get coordinate data for coord check.

    Args:
        model_engine (_type_): Ezmup model engine.
        datapoint (_type_): A datapoint to be used for forward pass.
        width_list (list, optional): _description_. Defaults to [64, 128, 256, 512, 1024, 2048, 4096, 8192].
        optim_cls (_type_, optional): _description_. Defaults to Adam.
        optim_kwargs (_type_, optional): _description_. Defaults to None.
        n_seeds (int, optional): _description_. Defaults to 1.
        n_steps (int, optional): _description_. Defaults to 3.

    Returns:
        pd.DataFrame: A dataframe containing the coordinate data.
    """
    df = []

    # Mutable default arguments are dangerous. Use None instead.
    if width_list is None:
        width_list = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

    for i in range(n_seeds):
        torch.manual_seed(i)
        for width in width_list:
            model_engine.change_width_as(width)
            optim = (
                model_engine.get_optimizer(optim_cls, lr=1e-3)
                if optim_kwargs is None
                else model_engine.get_optimizer(optim_cls, lr=1e-3, **optim_kwargs)
            )

            for j in range(n_steps):
                remove_hooks = []
                for name, module in model_engine.model.named_modules():
                    remove_hooks.append(
                        module.register_forward_hook(
                            _record_coords(
                                df,
                                width,
                                name,
                                j,
                                output_fdict=None,
                                input_fdict=None,
                                param_fdict=None,
                            ),
                        ),
                    )

                model_engine.model.train()

                loss = model_engine.forward(datapoint, model_engine.model)
                                
                loss.backward()
                optim.step()
                optim.zero_grad()

                for handle in remove_hooks:
                    handle.remove()

    return pd.DataFrame(df)


def plot_coord_data(
    df,
    y="l1",
    save_to=None,
    suptitle=None,
    x="width",
    hue="module",
    legend=True,
    name_contains=None,
    name_not_contains=None,
    loglog=True,
    logbase=2,
    face_color=None,
    jitter=True,
    jitter_strength=0.1,
):
    """Plot coord check data `df`.

    Args:
        df: pandas DataFrame
        y: column for y-axis. Default: 'l1'
        save_to: path to save the figure, or None. Default: None.
        suptitle: The title of the entire figure.
        x: column for x-axis. Default: 'width'
        hue: column for color. Default: 'module'
        legend: whether to show legend. Default: True
        name_contains: filter modules by name inclusion
        name_not_contains: filter modules by name exclusion
        loglog: use loglog scale. Default: True
        logbase: log base if using loglog. Default: 2
        face_color: background color of the plot. Default: None
        jitter: Whether to apply jitter to the y-axis. Default: True
        jitter_strength: The strength of the jitter. Default: 0.1

    Returns:
        the matplotlib figure object
    """

    def apply_jitter(values, jitter_strength):
        # Apply a random multiplicative shift to each data point
        rng = np.random.default_rng()
        jitter = rng.uniform(-jitter_strength, jitter_strength, size=len(values))
        return values * np.exp(jitter)

    # Preprocessing
    df = df.copy()
    df = df[df.module != ""]
    if name_contains is not None:
        df = df[df["module"].str.contains(name_contains)]
    elif name_not_contains is not None:
        df = df[~(df["module"].str.contains(name_not_contains))]

    ts = df.t.unique()

    # Plot
    fig, axes = plt.subplots(1, len(ts), figsize=(5 * len(ts), 4))
    if face_color:
        fig.patch.set_facecolor(face_color)
    if suptitle:
        plt.suptitle(suptitle)

    for idx, t in enumerate(ts):
        ax = axes[idx] if len(ts) > 1 else axes
        subset = df[df.t == t]
        groups = subset.groupby(hue)

        for name, group in groups:
            x_values = group[x]
            y_values = group[y]

            if jitter:
                y_values = apply_jitter(y_values, jitter_strength)

            ax.plot(x_values, y_values, label=name)

        ax.set_title(f"t={t}")
        if loglog:
            ax.set_xscale("log", base=logbase)
            ax.set_yscale("log", base=logbase)

        if legend and idx == len(ts) - 1:
            ax.legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_to:
        plt.savefig(save_to)
        print(f"Plot saved to {save_to}")

    return fig
