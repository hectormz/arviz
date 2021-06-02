"""Plot a flexible comparison of sampled parameters"""
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import NullFormatter

from ..data import convert_to_dataset, convert_to_inference_data
from ..sel_utils import xarray_to_ndarray
from ..utils import _var_names, get_coords
from .pairplot import plot_pair
from .plot_utils import _scale_fig_size
from .posteriorplot import plot_posterior


def plot_func_posterior(data, ax=None, np_fun=np.diff, **kwargs):
    """Transform multiple posteriors and plot densities

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    ax: axes
        Matplotlib axes
    np_fun : Callable
        Valid numpy-style function that can combine posteriors. Defaults to np.diff
    Returns
    -------
    ax : matplotlib axes

    Examples
    --------
    Posterior Difference Plot

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> centered = az.load_arviz_data('centered_eight')
        >>> coords = {'school': ['Choate', 'Deerfield']}
        >>> az.plot_func_posterior(centered,
        >>>                     var_names=['theta'],
        >>>                     coords=coords)

    Posterior Difference Plot with ROPE

    .. plot::
        :context: close-figs

        >>> az.plot_func_posterior(centered,
        >>>                     var_names=['theta'],
        >>>                     coords=coords,
        >>>                     rope=(-3, 3))

    """
    if isinstance(data, xr.DataArray):
        var_names, data = xarray_to_ndarray(data)
        data = np_fun(data, axis=0).squeeze()
    else:
        data = convert_to_dataset(data)
        var_names = data.data_vars
        data = np_fun(data.to_array(), axis=0).squeeze()

    var_names = [name.replace("\n", "_") for name in var_names]
    func_name = np_fun.__name__
    xlabel = "{}({},\n{})".format(func_name, *var_names)
    if ax is None:
        ax = plt.gca()
    plot_posterior({xlabel: data}, ax=ax, **kwargs)
    return ax


def plot_pair_extended(
    data,
    var_names=None,
    coords=None,
    figsize=None,
    textsize=None,
    ax=None,
    combined=True,
    lower_fun=plot_pair,
    upper_fun=None,
    diag_fun=plot_posterior,
    lower_kwargs=None,
    upper_kwargs=None,
    diag_kwargs=None,
    labels="edges",
):
    """Plot flexible pairplot with custom functions on triables and diagonal

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names : list of variable names
        Variables to be plotted, if None all variable are plotted
    coords : mapping, optional
        Coordinates of var_names to be plotted. Passed to `Dataset.sel`
    figsize : figure size tuple
        If None, size is (8 + numvars, 8 + numvars)
    textsize: int
        Text size for labels. If None it will be autoscaled based on figsize.
    ax: axes
        Matplotlib axes
    combined : bool, optional
        Whether to combine chains in plot
    lower_fun : Callable
        Plot function on lower triangle. Defaults to plot_pair
    upper_fun : Callable
        Plot function on upper triangle. Defaults to None
    diag_fun : Callable
        Plot function on diagonal. Defaults to plot_posterior
    lower_kwargs : dicts, optional
        Additional keywords passed to plots in lower triangle
    upper_kwargs : dicts, optional
        Additional keywords passed to plots in upper triangle
    diag_kwargs : dicts, optional
        Additional keywords passed to plots on diagonal
    labels : str
        Determines which subplot labels should be shown
    Returns
    -------
    ax : matplotlib axes

    Examples
    --------
    Pair plot with posterior plot

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> centered = az.load_arviz_data('centered_eight')
        >>> coords = {'school': ['Choate', 'Deerfield', 'Hotchkiss']}
        >>> az.plot_pair_extended(centered,
        >>>                     var_names=['theta'],
        >>>                     coords=coords)

    Pair plot with posterior plot on diagonal and posterior difference on upper triangle

    .. plot::
        :context: close-figs

        >>> az.plot_pair_extended(centered,
        >>>                     var_names=['theta'],
        >>>                     coords=coords,
        >>>                     upper_fun=az.plot_func_posterior,
        >>>                     upper_kwargs={"textsize": 12, "kind": "hist"})

        Pair plot with scatter and kde, with labels on axis

    .. plot::
        :context: close-figs

        >>> def plot_name(data, ax, **kwargs):
        >>>     name = list(data.keys())[0]
        >>>     ax.text(0.5, 0.5, name, verticalalignment="center", horizontalalignment="center", **kwargs)
        >>>     ax.axis("off")
        >>>     return ax
        >>> az.plot_pair_extended(centered,
        >>>                     var_names=['tau'],
        >>>                     combined=False,
        >>>                     lower_kwargs={"plot_kwargs":
        >>>                         {"marker": "+",
        >>>                          "color": "darkblue",
        >>>                          "alpha": 0.6}},
        >>>                     diag_fun=plot_name,
        >>>                     upper_fun=az.plot_pair,
        >>>                     upper_kwargs={"kind": "kde", "fill_last": False})
    """

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

    if numvars < 2:
        raise Exception("Number of variables to be plotted must be 2 or greater.")

    (figsize, __, __, __, __, _) = _scale_fig_size(figsize, textsize, numvars, numvars)
    if ax is None:
        __, ax = plt.subplots(
            numvars,
            numvars,
            figsize=figsize,
            constrained_layout=True,
            #  sharex="col", sharey="row"
        )
    # x_offset = y_offset = 0
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

            if (i + 1 != numvars and labels == "edges") or labels == "none":
                ax[i, j].axes.get_xaxis().set_major_formatter(NullFormatter())
                ax[i, j].set_xlabel("")
            if (j != 0 and labels == "edges") or labels == "none":
                ax[i, j].axes.get_yaxis().set_major_formatter(NullFormatter())
                ax[i, j].set_ylabel("")
