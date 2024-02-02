import argparse
import logging
from pathlib import Path

import laspy
import numpy as np
import smrf


def handle_args(args):
    parser = argparse.ArgumentParser(
        prog="smrf",
        description="smrf",
    )
    parser.add_argument("las_path", help="path/to/las", type=Path)
    parser.add_argument("--out_path", help="path/to/las", type=Path)
    parsed_args = parser.parse_args(args)
    return parsed_args


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    if "debugpy" in sys.modules:
        cdir = Path(__file__).parent
        # lasf = cdir / "data" / "samp11.las"
        lasf = cdir / "data" / "DK22_partial.las"
        argv = [str(lasf)]
    else:
        argv = sys.argv[1:]
    args = handle_args(argv)

    lasf = args.las_path
    groundf = lasf.with_name(lasf.stem + "_smrf.las") if args.out_path is None else args.out_path
    print("# loading las...###")
    with laspy.open(lasf) as f:
        las = f.read()
    print("%d pts loaded" % len(las.points))
    x = np.array(list(las.x))
    y = np.array(list(las.y))
    z = np.array(list(las.z))

    Zsmrf, Tsmrf, obj_cells, obj_array = smrf.smrf(
        x,
        y,
        z,
        cellsize=1,
        windows=10,
        slope_threshold=0.25,
        elevation_threshold=0.05,
        elevation_scaler=1,
        low_filter_slope=5,
        low_outlier_fill=False,
    )

    outlas = laspy.LasData(las.header)
    outlas.points = las.points[~obj_array].copy()
    outlas.write(groundf)
