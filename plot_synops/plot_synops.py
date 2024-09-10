# retrieve SYNOP messages and then plot them.
from __future__ import annotations
import matplotlib
matplotlib.use("Agg") # headless matplotlib
import pathlib
import typing

import matplotlib.pyplot
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


from . import metlib
import pymetdecoder
cache_dir = metlib.cache_dir
pd.options.mode.copy_on_write = True

import datetime
typ_flt_int = typing.Union[float, int]




if __name__ == '__main__':
    # code is running as a script
    # setup parser and deal with arguments
    import argparse


    parser = argparse.ArgumentParser(description='Plot SYNOP data for Great Britain and Ireland. Could be extended to other regions.')
    parser.add_argument('--date', type=pd.Timestamp, help='UTC date/time to plot data for. ',
                        default=pd.Timestamp.utcnow() - pd.Timedelta(1, 'h'))
    parser.add_argument('--use_cache', type=bool, help='Use the cache for the data.', default=True)
    parser.add_argument('--thin', type=float, help='Thin distance in km', default=40.)
    parser.add_argument('--output', type=pathlib.Path,
                        help='Output file to save the plot to. If not specified with be constructed from date and be pdf',
                        default=None)
    parser.add_argument('--figsize', nargs=2, type=float, help='Page size for the plot. Default is A3.',
                        default=(11.69, 16.53))
    parser.add_argument('--region', nargs=4, type=float,
                        help='Region (long0,long1,lat0,lat1) to plot in degrees. Default is GB + Ireland. ',
                        default=(-11., 2., 49.0, 61.5))  # from Guernsey to Shetland, W-Ireland to E-England
    parser.add_argument('--use_midas_csv', action='store_true',
                        help='Use open-midas csv files. They should already have been downloaded from BADC. ')
    parser.add_argument('--nocache', action='store_true', help='Do not use the cache.')
    parser.add_argument('--plot_pressure', action='store_true',
                        help='Plot the pressure on the map. Will try and retrieve ERA5 data.')
    parser.add_argument('--nointeractive', action='store_true', help='Do not use an interactive backend.')
    args = parser.parse_args()

    save_file = args.output
    if save_file is None:
        save_file = args.date.strftime("station_plot_%Y%m%d_%H.pdf")

    if not args.nointeractive:
        backends_to_try = ['QtAgg','TkAgg']
        for backend in backends_to_try:
            try:
                matplotlib.use(backend)
                print("Using backend ", backend)
                break
            except ImportError:
                print(f'Failed to use backend {backend}')


    synops_to_plot, pressure = metlib.read_synops(args.date, region=args.region, nocache=args.nocache,
                                                  use_midas_csv=args.use_midas_csv,
                                                  cachier__skip_cache=args.nocache, # need to not use this routines cach
                                 get_pressure=args.plot_pressure)
    fig_map_synop,ax = metlib.plot_synops(synops_to_plot, pressure, thin=args.thin, figsize=args.figsize,
                                          region=args.region)
    fig_map_synop.show()
    fig_map_synop.savefig(save_file, dpi=300, bbox_inches="tight")
