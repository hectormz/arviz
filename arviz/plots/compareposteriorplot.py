# import warnings
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import NullFormatter
# from mpl_toolkits.axes_grid1 import make_axes_locatable
#
# from ..data import convert_to_dataset, convert_to_inference_data
# from .kdeplot import plot_kde
# from .plot_utils import _scale_fig_size, xarray_to_ndarray, get_coords
# from ..utils import _var_names

# from kdeplot import plot_kde, _fast_kde
# from posteriorplot import plot_posterior

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import NullFormatter

from ..data import convert_to_dataset, convert_to_inference_data
from ..plots.plot_utils import _scale_fig_size, get_coords, purge_duplicates
from ..utils import _var_names
from .pairplot import plot_pair
from .plot_utils import _scale_fig_size, get_coords, xarray_to_ndarray
from .posteriorplot import plot_posterior


def gen_var_dims_list(data, var_names, coords):
    """Generate list of valid var_names and coords pairings
    """

    # If value for key in coords is string and not list of string(s),
    # parameter_list does not build properly
    for j in coords:
        if isinstance(coords[j], str):
            coords[j] = [coords[j]]

    posterior_data = convert_to_dataset(data, group="posterior")
    posterior_data = get_coords(posterior_data, coords)
    skip_dims = set()
    skip_dims = skip_dims.union({"chain", "draw"})
    var_dims_list = []
    for var_name in var_names:
        if var_name in posterior_data:
            new_dims = [dim for dim in posterior_data[var_name].dims if dim not in skip_dims]
            vals = [purge_duplicates(posterior_data[var_name][dim].values) for dim in new_dims]
            dims = [{k: v for k, v in zip(new_dims, prod)} for prod in product(*vals)]
            var_dims = [[var_name, d] for d in dims]

            var_dims_list += var_dims

    return var_dims_list


def plot_dist_diff(data, var_names, coords, textsize=None, figsize=None, ax=None, **kwargs):
    var_dims_list = gen_var_dims_list(data, var_names, coords)

    assert len(var_dims_list) == 2, "Too many parameters provided"

    (figsize, ax_labelsize, _, xt_labelsize, _, _) = _scale_fig_size(
        figsize, textsize, 1, 1
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    var_name_0 = var_dims_list[0][0]
    var_name_1 = var_dims_list[1][0]
    coord_0 = var_dims_list[0][1]
    coord_1 = var_dims_list[1][1]

    if len(coord_0) == 1:
        coord_0_suff = "_" + list(coord_0.values())[0]
    else:
        coord_0_suff = ""
    if len(coord_1) == 1:
        coord_1_suff = "_" + list(coord_1.values())[0]
    else:
        coord_1_suff = ""

    diff_DataArray = data.posterior[var_name_0].sel(coord_0) - data.posterior[
        var_name_1
    ].sel(coord_1)

    diff_var_name = f"{var_name_0}{coord_0_suff} - {var_name_1}{coord_1_suff}"

    data.posterior[diff_var_name] = diff_DataArray
    plot_posterior(data, var_names=[diff_var_name], ax=ax, **kwargs)


def pair_plot_extended_Q(data, var_names, coords=None, lower_fun=plot_pair, upper_fun=None, diag_fun=plot_posterior,
                       lower_kwargs=None, upper_kwargs=None, diag_kwargs=None):
    if coords is None:
        coords = {}

    if lower_kwargs is None:
        lower_kwargs = {}

    if upper_kwargs is None:
        upper_kwargs = {}

    if diag_kwargs is None:
        diag_kwargs = {}

    var_dims_list = gen_var_dims_list(data, var_names, coords)
    fig, axes = plt.subplots(len(var_dims_list), len(var_dims_list), figsize=(10, 10))

    for i in range(len(var_dims_list)):
        for j in range(len(var_dims_list)):
            print(i, j)
            sub_var_name = purge_duplicates([var_dims_list[i][0], var_dims_list[j][0]])
            if len(var_dims_list[i][1]) > 0 and len(var_dims_list[j][1]) > 0:
                if list(var_dims_list[i][1].keys())[0] in var_dims_list[j][1]:
                    dict1 = var_dims_list[i][1]
                    dict2 = var_dims_list[j][1]
                    sub_coords = {
                        i: purge_duplicates(list(j))
                        for i in dict1.keys()
                        for j in zip(dict1.values(), dict2.values())
                    }
                else:
                    sub_coords = {**var_dims_list[i][1], **var_dims_list[j][1]}
            else:
                sub_coords = {**var_dims_list[i][1], **var_dims_list[j][1]}

            ax = axes[j, i]
            if i < j:
                if lower_fun is not None:
                    lower_fun(data, var_names=sub_var_name, coords=sub_coords, ax=ax, **lower_kwargs)
                else:
                    ax.axis("off")
            elif i > j:
                if upper_fun is not None:
                    upper_fun(data, var_names=sub_var_name, coords=sub_coords, ax=ax, **upper_kwargs)
                else:
                    ax.axis("off")
            elif i == j:
                if diag_fun is not None:
                    diag_fun(data, var_names=sub_var_name, coords=sub_coords, ax=ax, **diag_kwargs)
                else:
                    ax.axis("off")
    plt.tight_layout()


def plot_func_posterior(data, ax, np_fun=np.diff, **kwargs):
    if isinstance(data, xr.core.dataarray.DataArray):
        var_names, data = xarray_to_ndarray(data)
        data = np_fun(data, axis=0).squeeze()
    else:
        data = convert_to_dataset(data)
        var_names = data.data_vars
        data = np_fun(data.to_array(), axis=0).squeeze()

    var_names = [name.replace("\n", "_") for name in var_names]
    func_name = np_fun.__name__
    xlabel = "{}({},\n{})".format(func_name, *var_names)
    plot_posterior({xlabel: data}, ax=ax, **kwargs)


def plot_func_posterior_0(data, ax, np_fun=np.diff, **kwargs):
    data = convert_to_dataset(data)
    var_names = [name.replace("\n", "_") for name in data.data_vars]
    func_name = np_fun.__name__
    xlabel = "{}({},\n{})".format(func_name, *var_names)
    data = np_fun(data.to_array(), axis=0).squeeze()
    plot_posterior({xlabel: data}, ax=ax, **kwargs)


def plot_pair_extended(
    data,
    var_names,
    coords=None,
    combined=True,
    lower_fun=plot_pair,
    upper_fun=None,
    diag_fun=plot_posterior,
    lower_kwargs=None,
    upper_kwargs=None,
    diag_kwargs=None,
    figsize=None,
    labels='edges',
    ax=None,
):
    if coords is None:
        coords = {}

    if lower_kwargs is None:
        lower_kwargs = {}

    if upper_kwargs is None:
        upper_kwargs = {}

    if diag_kwargs is None:
        diag_kwargs = {}

    if labels not in ("edges", "all", "none"):
        raise ValueError("labels must be one of (edges, all, none)")

    # Get posterior draws and combine chains
    data = convert_to_inference_data(data)
    posterior_data = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, posterior_data)
    flat_var_names, _posterior = xarray_to_ndarray(
        get_coords(posterior_data, coords), var_names=var_names, combined=combined
    )
    flat_var_names = np.array(flat_var_names)
    numvars = len(flat_var_names)

    (figsize, _, _, _, _, _) = _scale_fig_size(figsize, None, numvars, numvars)
    if ax is None:
        _, ax = plt.subplots(numvars, numvars, figsize=figsize, constrained_layout=True, sharex='col', sharey='row')

    for i in range(numvars):
        for j in range(numvars):
            index = np.array([i, j], dtype=int)
            if i > j:
                if lower_fun is not None:
                    lower_fun(
                        {flat_var_names[j]: _posterior[j], flat_var_names[i]: _posterior[i]},
                        ax=ax[i, j],
                        **lower_kwargs
                    )
                else:
                    ax[i, j].axis("off")
            elif i < j:
                if upper_fun is not None:
                    upper_fun(
                        {flat_var_names[j]: _posterior[j], flat_var_names[i]: _posterior[i]},
                        ax=ax[i, j],
                        **upper_kwargs
                    )
                else:
                    ax[i, j].axis("off")
            elif i == j:
                if diag_fun is not None:
                    diag_fun({flat_var_names[i]: _posterior[i]}, ax=ax[i, j], **diag_kwargs)
                else:
                    ax[i, j].axis("off")

            if (i + 1 != numvars and labels=="edges") or labels=="none":
                ax[i, j].axes.get_xaxis().set_major_formatter(NullFormatter())
                ax[i, j].set_xlabel("")
            if (j != 0 and labels=="edges") or labels=="none":
                ax[i, j].axes.get_yaxis().set_major_formatter(NullFormatter())
                ax[i, j].set_ylabel("")
