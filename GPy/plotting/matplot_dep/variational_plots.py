import pylab as pb, numpy as np

def plot(parameterized, fignum=None, ax=None, colors=None):
    """
    Plot latent space X in 1D:

        - if fig is given, create input_dim subplots in fig and plot in these
        - if ax is given plot input_dim 1D latent space plots of X into each `axis`
        - if neither fig nor ax is given create a figure with fignum and plot in there

    colors:
        colors of different latent space dimensions input_dim

    """
    if ax is None:
        fig = pb.figure(num=fignum, figsize=(8, min(12, (2 * parameterized.mean.shape[1]))))
    if colors is None:
        colors = pb.gca()._get_lines.color_cycle
        pb.clf()
    else:
        colors = iter(colors)
    plots = []
    means, variances = parameterized.mean, parameterized.variance
    x = np.arange(means.shape[0])
    for i in range(means.shape[1]):
        if ax is None:
            a = fig.add_subplot(means.shape[1], 1, i + 1)
        elif isinstance(ax, (tuple, list)):
            a = ax[i]
        else:
            raise ValueError("Need one ax per latent dimension input_dim")
        a.plot(means, c='k', alpha=.3)
        plots.extend(a.plot(x, means.T[i], c=colors.next(), label=r"$\mathbf{{X_{{{}}}}}$".format(i)))
        a.fill_between(x,
                        means.T[i] - 2 * np.sqrt(variances.T[i]),
                        means.T[i] + 2 * np.sqrt(variances.T[i]),
                        facecolor=plots[-1].get_color(),
                        alpha=.3)
        a.legend(borderaxespad=0.)
        a.set_xlim(x.min(), x.max())
        if i < means.shape[1] - 1:
            a.set_xticklabels('')
    pb.draw()
    fig.tight_layout(h_pad=.01) # , rect=(0, 0, 1, .95))
    return fig

def plot_SpikeSlab(parameterized, fignum=None, ax=None, colors=None, side_by_side=True):
    """
    Plot latent space X in 1D:

        - if fig is given, create input_dim subplots in fig and plot in these
        - if ax is given plot input_dim 1D latent space plots of X into each `axis`
        - if neither fig nor ax is given create a figure with fignum and plot in there

    colors:
        colors of different latent space dimensions input_dim

    """
    if ax is None:
        if side_by_side:
            fig = pb.figure(num=fignum, figsize=(16, min(12, (2 * parameterized.mean.shape[1]))))
        else:
            fig = pb.figure(num=fignum, figsize=(8, min(12, (2 * parameterized.mean.shape[1]))))
    if colors is None:
        colors = pb.gca()._get_lines.color_cycle
        pb.clf()
    else:
        colors = iter(colors)
    plots = []
    means, variances, gamma = parameterized.mean, parameterized.variance, parameterized.binary_prob
    x = np.arange(means.shape[0])
    for i in range(means.shape[1]):
        if side_by_side:
            sub1 = (means.shape[1],2,2*i+1)
            sub2 = (means.shape[1],2,2*i+2)
        else:
            sub1 = (means.shape[1]*2,1,2*i+1)
            sub2 = (means.shape[1]*2,1,2*i+2)

        # mean and variance plot
        a = fig.add_subplot(*sub1)
        a.plot(means, c='k', alpha=.3)
        plots.extend(a.plot(x, means.T[i], c=colors.next(), label=r"$\mathbf{{X_{{{}}}}}$".format(i)))
        a.fill_between(x,
                        means.T[i] - 2 * np.sqrt(variances.T[i]),
                        means.T[i] + 2 * np.sqrt(variances.T[i]),
                        facecolor=plots[-1].get_color(),
                        alpha=.3)
        a.legend(borderaxespad=0.)
        a.set_xlim(x.min(), x.max())
        if i < means.shape[1] - 1:
            a.set_xticklabels('')
        # binary prob plot
        a = fig.add_subplot(*sub2)
        a.bar(x,gamma[:,i],bottom=0.,linewidth=0,width=1.0,align='center')
        a.set_xlim(x.min(), x.max())
        a.set_ylim([0.,1.])
    pb.draw()
    fig.tight_layout(h_pad=.01) # , rect=(0, 0, 1, .95))
    return fig
