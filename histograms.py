from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class Conf:
    # base_out_dir = None
    # compute = False
    xlabel_fs = "15"
    ylabel_fs = "15"
    title_fs = "xx-large"
    x_ticksize = "13"
    y_ticksize = "13"
    nonsense = None


def generate(out_fn, data_map, show=True):

    plt.rc("xtick", labelsize=Conf.x_ticksize)
    plt.rc("ytick", labelsize=Conf.y_ticksize)

    plt.figure()
    plt.title("Maximal distance of 3D centers of the same object", fontsize=Conf.title_fs)
    plt.xlabel("Distance [m.]", fontsize=Conf.xlabel_fs)
    if cumulative:
        plt.ylabel("cumulative probability", fontsize=Conf.ylabel_fs)
    else:
        plt.ylabel("# data points", fontsize=Conf.ylabel_fs)

    for dir, data in data_map.items():
        max_dist_value = 8
        # bins_c = max(50, int(data.shape[0] / 10))
        bins_c = 100
        bins = np.linspace(0, max_dist_value, bins_c)
        label = dir[8:]
        data = np.clip(data, a_min=0.0, a_max=max_dist_value - 0.01)
        plt.hist(data, bins, alpha=0.5, label=label)

    plt.legend(loc='upper right')
    plt.savefig(out_fn, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def get_dirs():
    glob = Path("./").glob("object--*")
    dirs = [i.name for i in glob if i.name != "object--vehicle--other-vehicle"]
    print(f"detected dirs: {dirs}")
    return dirs


def compute_for_dir(dirs):

    m = {}
    for dir in dirs:
        print(f"computing dir: {dir}")
        glob = Path(dir).glob("*.png")
        data = []
        for i, png in enumerate(glob):
            name = png.name
            i = name.index("_")
            distance = float(name[:i])
            data.append(distance)
        data = np.array(data)
        m[dir] = data
        print(f"computed dir: {dir}")

    generate(out_fn=f"histograms/histogram_all.png", data_map=m, show=True)

    for dir in dirs:
        mm = {dir: m[dir]}
        generate(out_fn=f"histograms/histogram_{dir[8:]}.png", data_map=mm, show=True)


def run():
    dirs = get_dirs()
    compute_for_dir(dirs)


if __name__ == "__main__":
    run()

