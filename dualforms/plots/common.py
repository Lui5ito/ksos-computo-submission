import matplotlib

matplotlib.rcParams["savefig.dpi"] = 300

SCATTER_TRAIN = dict(s=2, c="gray", alpha=0.2, zorder=3, rasterized=True)
SCATTER_TEST  = dict(s=1,  c="gray",  alpha=0.1, zorder=0, rasterized=True)
