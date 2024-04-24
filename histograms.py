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


def read_map(file_path):
    def read_np(s):
        r = np.fromstring(s, dtype=float, sep=" ")
        return r

    data = {}
    curr_key = None
    with open(file_path, "r") as f:

        buffer = ""
        for line in f.readlines():
            if line.startswith("###"):
                if curr_key:
                    data[curr_key] = read_np(buffer)
                curr_key = line[4:-4]
                buffer = ""
            else:
                buffer += line

        data[curr_key] = read_np(buffer)

    return data
def generate(out_fn,
             data_map,
             max_dist_value,
             show=True,
             cumulative=False,
             x_label="Distance [m.]",
             title=f"Maximal distance of 3D centers of the same object"):

    plt.rc("xtick", labelsize=Conf.x_ticksize)
    plt.rc("ytick", labelsize=Conf.y_ticksize)

    plt.figure()

    plot_type = "cdf" if cumulative else "histogram"
    title = f"{title} - {plot_type}"
    plt.title(title, fontsize=Conf.title_fs)
    plt.xlabel(x_label, fontsize=Conf.xlabel_fs)
    if cumulative:
        plt.ylabel("cumulative probability", fontsize=Conf.ylabel_fs)
    else:
        plt.ylabel("# data points", fontsize=Conf.ylabel_fs)
    colors = ["g", "b", "m", "r", "y", "k"]

    for ind, (dir, data) in enumerate(data_map.items()):
        # max_dist_value = 1
        # bins_c = max(50, int(data.shape[0] / 10))
        label = dir[8:]
        data = np.clip(data, a_min=0.0, a_max=max_dist_value - 0.0001)
        if cumulative:
            data_aug = np.zeros(data.shape[0] + 1)
            data_aug[1:] = data
            # data_aug[0] = 0.0
            count, bins_count = np.histogram(data, bins=1000)
            # AUG
            # bins_count[0] = 0

            count_aug = np.zeros(count.shape[0] + 1)
            count_aug[1:] = count
            pdf_aug = (count_aug) / sum(count_aug)
            cdf_aug = np.cumsum(pdf_aug)
            #plt.plot(bins_count[:-1], cdf_aug[:-1], label=label, color=colors[ind])

            pdf = (count) / sum(count)
            cdf = np.cumsum(pdf)
            plt.plot(bins_count[1:-1], cdf[:-1], label=label, color=colors[ind])
        else:
            bins_c = 50 if max_dist_value < 4.0 else 100
            bins = np.linspace(0, max_dist_value, bins_c)
            plt.hist(data, bins, alpha=0.5, label=label)

    plt.legend(loc='upper right')
    parent = Path(out_fn).parent
    Path.mkdir(parent, parents=True, exist_ok=True)
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

    generate(out_fn=f"histograms/density_1/histogram_all.png", max_dist_value=1.0, data_map=m, show=False, cumulative=False)
    generate(out_fn=f"histograms/density_8/histogram_all.png", max_dist_value=8.0, data_map=m, show=False, cumulative=False)
    generate(out_fn=f"histograms/cumulative/histogram_all.png", max_dist_value=1.1, data_map=m, show=False, cumulative=True)

    for dir in dirs:
        mm = {dir: m[dir]}
        generate(out_fn=f"histograms/density_1/histogram_{dir[8:]}.png", max_dist_value=1.0, data_map=mm, show=False, cumulative=False)
        generate(out_fn=f"histograms/density_8/histogram_{dir[8:]}.png", max_dist_value=8.0, data_map=mm, show=False, cumulative=False)
        generate(out_fn=f"histograms/cumulative/histogram_{dir[8:]}.png", max_dist_value=1.0002, data_map=mm, show=False, cumulative=True)


def compute_projection_errors(fp="data/projection_errs.txt"):

    m = read_map(fp)
    k = "object--vehicle--other-vehicle "
    if m.__contains__(k):
        m.__delitem__(k)

    generate(out_fn=f"histograms/reprojection_error/density_100/histogram_all.png",
             max_dist_value=100.0,
             data_map=m,
             show=False,
             cumulative=False,
             x_label="[px.]",
             title=f"Projection errors")
    generate(out_fn=f"histograms/reprojection_error/cumulative/histogram_all.png",
             max_dist_value=100.0,
             data_map=m,
             show=False,
             cumulative=True,
             x_label="[px.]",
             title=f"Projection errors")

    for k, v in m.items():
        mm = {k: v}
        generate(out_fn=f"histograms/reprojection_error/density_100/histogram_{k}.png",
                 max_dist_value=100.0,
                 data_map=mm,
                 show=False,
                 cumulative=False,
                 x_label="[px.]",
                 title=f"Projection errors")
        generate(out_fn=f"histograms/reprojection_error/cumulative/histogram_{k}.png",
                 max_dist_value=100.0,
                 data_map=mm,
                 show=False,
                 cumulative=True,
                 x_label = "[px.]",
                 title = f"Projection errors")


def compute_distances():
    dirs = get_dirs()
    compute_for_dir(dirs)


if __name__ == "__main__":
    #compute_distances()
    compute_projection_errors()
