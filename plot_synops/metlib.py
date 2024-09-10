# common functions to support Met:A&E
# see https://library.wmo.int/viewer/35713 for SYNOP code
# https://cloudatlas.wmo.int/en/home.html cloud atlas

from __future__ import annotations

import typing
import pathlib
import datetime

import pandas as pd
import metpy.calc
import cachier
from metpy.units import units
import pymetdecoder.synop
import requests
import xarray
from metpy.calc import reduce_point_density
from io import StringIO
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
pd.options.mode.copy_on_write = True

# setup cachier
root_dir = pathlib.Path(__file__).resolve().parent
cache_dir = root_dir / 'data' / 'cache'
cache_dir.mkdir(exist_ok=True, parents=True)
cachier.set_default_params(cache_dir=cache_dir, stale_after=datetime.timedelta(weeks=2), next_time=True)
typ_flt_int = typing.Union[float, int]
default_region = (-11., 2., 49.0, 61.5)

@cachier.cachier()
def fix_midas_data(input: pd.DataFrame,
                   date_range: typing.Optional[(pd.Timestamp, pd.Timestamp)] = None) -> pd.DataFrame:
    hdr = input.columns
    # fix strings by removing leading and  trailing space.
    cols_to_fix = input.select_dtypes(include='object').columns
    data = input.copy(deep=True)
    data[cols_to_fix] = data[cols_to_fix].apply(lambda x: x.str.strip())
    # select time range.
    if date_range is not None:
        data = data[data.ob_time.between(*date_range)]
    # Some cols should have int values with Null for missing.
    cols_ignore = list(hdr[hdr.str.contains('time', case=False)]) + ['id', 'id_type', 'met_domain_name']
    cols_int = list(set(hdr) - set(cols_ignore))
    cols_id = {col: 'Int32' for col in hdr if (col.endswith('_id') or '_id_' in col)}
    data[cols_int] = data[cols_int].apply(pd.to_numeric, errors='coerce')
    data = data.astype(cols_id)  # convert all id data to Int32
    data.src_opr_type = data.src_opr_type.fillna(1)
    # check all int data is >=0
    for c in cols_int:
        if c.endswith('id') and (data[c].notnull().sum() > 0) and (data[c].min() < 0):
            print(f'Warning: {c} has negative values')
            data[c] = data[c].clip(lower=0)
    return data


def read_midas_wh_file(path: typing.Union[pathlib.Path],
                       qc: bool = True,
                       obs_type: typing.Optional[typing.Literal['SYNOP', 'METAR', 'AWSHRLY', 'DLY3208']] = None,
                       oper_type: typing.Optional[typing.Literal['AUTOMATIC', 'MANUAL']] = None,
                       date_range: typing.Optional[typing.Tuple[pd.Timestamp, pd.Timestamp]] = None,
                       nrows: typing.Optional[int] = None,
                       read_midas_open: bool = False,
                       hdr: typing.Optional[pd.Series] = None
                       ) -> pd.DataFrame:
    """
    Read a MIDAS weather file
     See https://artefacts.ceda.ac.uk/badc_datadocs/ukmo-midas/WH_Table.html for details of headers.
     Same appies to MIDAS open data.
    :param path: path to file to be read in,
    :param qc: If True (default) only read in QC'ed data
    :param nrows: Number of rows. If None (default) all rows are read in
    :return: dataframe of data.
    """

    if hdr is None:
        hdr_file = root_dir / 'data/WH_Column_headers.txt'
        hdr = pd.read_csv(hdr_file, header=None)
        hdr = hdr.iloc[0].str.strip().str.lower()  # lowercase for consistency with html doc
    time_cols = hdr[hdr.str.contains('_time', case=False)]
    time_indices = list(time_cols.index)
    if read_midas_open:
        # work out skiprows. Probably changed at some point. Will guess (for now 2022)
        year = int(path.stem.split('_')[-1])
        if year <= 2022:
            skiprows = 280
        else:
            skiprows = 281
        data = pd.read_csv(path, nrows=nrows, skiprows=skiprows,  # might need to change...
                           index_col=False, parse_dates=time_indices, date_format='ISO8601',
                           skipfooter=1, engine='python')

    else:
        data = pd.read_csv(path, header=None, names=hdr, index_col=False, nrows=nrows, parse_dates=time_indices,
                           date_format='ISO8601')
    # times are utc so fix that.
    for c in time_cols:
        data[c] = pd.to_datetime(data[c], utc=True)

    data = fix_midas_data(data, date_range=date_range)
    # Set src_opr_type when nan to 1
    # add in plain text observation_type to Automatic or Manual to make it easier to work with data.
    data['operator_type'] = data.src_opr_type.apply(lambda x: 'AUTOMATIC' if x in [4, 5, 6, 7] else 'MANUAL')
    if qc:  # restrict to QC'ed data
        data = data[data.version_num >= 1]
    if obs_type is not None:  # restrict to obs_type
        data = data[data.met_domain_name == obs_type]
    if oper_type is not None:  # and oper type
        if oper_type in ['AUTOMATIC', 'MANUAL']:
            data = data[data.operator_type == oper_type]
        else:
            raise ValueError(f'oper_type {oper_type} not recognised')
    # rename columns to more useful names:
    rename_cols = dict(cld_base_ht_id_1='cloud_base_height',
                       cld_ttl_amt_id='total_cloud_amount',
                       prst_wx_id='present_weather',
                       past_wx_id_1='past_weather',
                       low_cld_type_id='low_cloud_type',
                       med_cld_type_id='medium_cloud_type',
                       hi_cld_type_id='high_cloud_type',
                       cld_amt_id_1='cloud_base_amount',
                       )
    data = data.rename(columns=rename_cols)
    scales = dict(cld_base_height=10., visibility=10)  # convert from decameters to meters.
    for col, scale in scales.items():
        if col in data.columns:
            data[col] = data[col] * scale
    # add some derived values
    data['visibility_code'] = vis_code(data['visibility'])  # vis code
    data['cloud_base_height_code'] = cld_base_code(data['cloud_base_height'])  # cld base ht code.
    # convert anything in m/s to knots -- doing so as Met Office pressure charts have geostrophic scale that converts pressure grads to knots
    L = data.wind_speed_unit_id.isin([1, 2])
    wind = data.wind_speed
    data['wind_speed'] = wind.where(~L, wind / 0.51444)  # convert any m/s to knots
    wind_unit = data.wind_speed_unit_id
    data['wind_speed_unit_id'] = wind_unit.where(~L, wind_unit + 2)  # convert to knots
    # and fix missing data. Variable dependent
    fill_na = dict(total_cloud_amount=10)
    for key, fill_value in fill_na.items():
        data[key] = data[key].fillna(fill_value)

    return data


def read_midas_srce(path: pathlib.Path) -> pd.DataFrame:
    """
    Read in MIDAS source file giving metadata on MIDAS stations
    :param path: path to SRCE table
    :return: dataframe with following columns:
    src_id
    src_name
    high_prcn_lat
    high_prcn_lon
    src_bgn_date
    src_end_date
    east_grid_ref
    north_grid_ref
    See https://dap.ceda.ac.uk/badc/ukmo-midas/metadata/00README?download=1 for detailed description.
    """
    cols = {"src_id": (0, 10), "src_name": (11, 51), "high_prcn_lat": (53, 65), "high_prcn_lon": (67, 79),
            "src_bgn_date": (96, 106), 'src_end_date': (234, 244), "east_grid_ref": (129, 141),
            "north_grid_ref": (142, 156)}
    srce = pd.read_fwf(path, header=None, colspecs=list(cols.values()), names=list(cols.keys()))
    # drop unnecessary rows
    rows_to_drop = srce.iloc[:, 0].str.strip().str.startswith('SDO_GEOM') | srce.iloc[:, 0].str.strip().str.startswith(
        'UKMO_SURFA')
    srce = srce[~rows_to_drop]

    # make times
    for c in ['src_bgn_date', 'src_end_date']:
        srce[c] = pd.to_datetime(srce[c], errors='coerce', format='ISO8601')
    # convert to numbers where we can
    srce = srce.apply(pd.to_numeric, errors='ignore')
    # rename some things
    srce = srce.rename(columns=dict(high_prcn_lat='latitude', high_prcn_lon='longitude'))
    return srce


import numpy as np

p_patterns = [np.array([0., 1.5, 3.0, 2.]),  # rising increase then fall
              np.array([0., 1.0, 2., 2.]),  # rising increase then steady
              np.array([0, 1., 2., 3.]),  # increasing
              np.array([0, -1, 0., 1]),  # increasing then decreasing
              ]
p_patterns = [p / np.sqrt(p.dot(p)) for p in p_patterns]
p_patterns = np.array(p_patterns)  # make into a 2D array


def pressure_change(pressure: np.ndarray, patterns: np.ndarray) -> (float, typing.Optional[int, None]):
    """
    Compute the pressure change and pressure change pattern that best matches the data
    :param pressure: 4 hours of pressure (t-3h to t)
    :param patterns: matrix of patterns
    :return:dp and indx into symbol table for best match.
    """

    def min_p_tendancy(dp: np.ndarray) -> typing.Optional[int]:
        """
        Compute minimalist p tendency from start and end values.
        """
        if not np.isfinite(dp):
            idx = None
        elif np.abs(dp) < 0.05:
            idx = 4
        elif dp > 0:
            idx = 2
        else:
            idx = 7
        return idx

    if len(pressure) == 2:  # only two values. Assume value at start and end.
        dp = pressure[-1] - pressure[0]
        return dp, min_p_tendancy(dp)
    if len(pressure) != 4:
        return np.nan, None

    p0 = pressure[0]
    dp = pressure[-1] - p0  # pressure change over 3 hours
    if np.isnan(pressure).any():
        return dp, min_p_tendancy(dp)
    pattern_dot = patterns.dot(pressure - p0)
    if np.abs(dp) < 0.05:
        idx = 4
    elif dp > 0:
        idx = np.argmax(pattern_dot)
    else:
        idx = np.argmin(pattern_dot) + 5

    return dp, idx


def fill_fn(x: pd.Series, rv, type: str = 'Int32', miss: str = '') -> pd.Series:
    """
    Round a pandas series to the nearest integer, convert to a string and then fill
    :param x: Values
    :param rv: how many 0 to include.
    :param miss: What to fill missing values with
    :return:  Pandas series
    """
    return x.round().astype(type).astype(str).str.zfill(rv).where(x.notnull(), miss).astype(str)


def vis_code(x: pd.Series) -> pd.Series:
    """
    Convert visibility in meters to a code
    :param x: Visibility in meters
    :return: Code (as a string) for visibility.
    """
    result = x.copy().rename('visibility_code').astype('Int32')
    result[x < 5e3] = (result[x < 5e3] // 100.)  # 0-5 km
    result[result.round(-3) == 5e3] = 50  # 5 km
    L = (x >= 5500) & (x < 30e3)
    result[L] = (x[L] - 5e3) // 1e3 + 55  # 5.5-30 km
    result[result.round(-3) == 30e3] = 80  # 30km
    L = (x > 30e3) & (x <= 70e3)
    result[L] = (x[L] - 30e3) // 5e3 + 80  # 30 to 70km
    result[x > 70e3] = 89  # > 70km
    #result = fill_fn(result, 2)  # convert to a string.
    return result


def cld_base_code(ht: pd.Series) -> pd.Series:
    """
    Compute cld base height code from ht in meters. https://www.nodc.noaa.gov/archive/arc0001/0001334/1.1/data/0-data/html/wmo-code/WMO1600.HTM for vertial levels. Can't find on WMO site..
    :param ht:Height in meters
    :return: code for cloud base height.
    """
    cld_hts = [50, 100, 200, 300, 600, 1000, 1500, 2000, 2500]
    result = pd.Series(np.searchsorted(cld_hts, ht, side='right'), index=ht.index, dtype='Int32')
    # set missing data missing!
    result = result.where(ht.notnull(), pd.NA).rename('cld_base_code')
    return result


def df_p_change(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the pressure change for a given dataframe
    :param df_in: DataFrame with columns 'ob_time','msl_pressure'
    :return: DataFrame with columns 'dp' & 'dp_pattern'
    """
    df = df_in.sort_values('ob_time').reset_index(drop=True)
    cols = ['dp', 'dp_pattern']
    dtype = dict(dp='float64', dp_pattern='Int32')
    time = df.ob_time.iloc[-1]
    if len(df) == 1:
        return pd.DataFrame([[np.nan, None]], columns=cols).astype(dtype)
    time_range = df.ob_time.iloc[-1] - df.ob_time.iloc[0]
    if time_range != pd.Timedelta(3, 'h'):
        return pd.DataFrame([[np.nan, None]], columns=cols).astype(dtype)

    v = [list(pressure_change(df.msl_pressure.values, p_patterns))]
    return pd.DataFrame(v, columns=cols).astype(dtype)


# imports from metpy + a monkey patch
from metpy.plots import current_weather, sky_cover, StationPlot, low_clouds, mid_clouds, \
    high_clouds, pressure_tendency, current_weather_auto


def past_weather(code: typing.Optional[int]) -> str:
    """
    mapper function for past weather codes
    :param code: code to be decoded (0-9)
    :return: Symbol
    """
    if code is None:
        return current_weather(103)  # missing data
    lookup = [0, 0, 0, 38, 45, 50, 60, 70, -80, -90]  # using alt symbol for code 3 'driving snow'
    if (np.abs(code) >= len(lookup)) or (np.abs(code) < 0):
        raise ValueError(f'past weather code {code} not recognised')

    code = lookup[code]
    if code >= 0:
        return current_weather(code)
    else:
        return current_weather_auto(-code)


def current_weather_both(code: typing.Optional[int]) -> str:
    """
    mapper function for present weather codes. manual are  0 -99, automatic 100-199.
    :param code: code to be decoded.
    :return: symbol to be plotted
    """
    if code is None:
        return current_weather(103)  # missing data shown with a //
    if code < 100:
        return current_weather(code)
    else:
        return current_weather_auto(code - 100)


def auto_stations(code: typing.Optional[int]) -> str:
    """
    Mapper function for automatic stations
    :param code: code to be decoded
    :return: symbol to be plotted
    """
    if pd.isna(code):
        return ''  #plot empty char
    return sky_cover(code)


class SynopPlot(StationPlot):
    """
    Class to plot SYNOP data. This is a specialised ubclass of StationPlot
    """
    text_size_type = typing.Union[int, str, None]

    def __init__(self, ax, x, y, data_values, transform=None, fontsize=10, spacing=None, clip_on=True,
                 text_fontsize: text_size_type = None, small_text_fontsize: text_size_type = None,
                 sym_fontsize: text_size_type = None, **kwargs):
        if spacing is None:
            spacing = fontsize * 1.2

        super().__init__(ax, x, y, transform=transform, fontsize=fontsize, spacing=spacing, **kwargs)

        if text_fontsize is None:
            text_fontsize = fontsize
        if sym_fontsize is None:
            sym_fontsize = text_fontsize
        if small_text_fontsize is None:
            small_text_fontsize = text_fontsize * 0.8
        # make sure input vales are appropriate
        dtypes = dict(cloud_base_amount='Int32', cloud_base_height_code='Int32', dp=float, msl_pressure=float,
                      air_temperature=float, dewpoint=float, wind_speed=float, wind_direction=float,
                      present_weather='Int32',
                      past_weather='Int32',
                      visibility_code='Int32', total_cloud_amount='Int32', low_cloud_type='Int32',
                      medium_cloud_type='Int32',
                      high_cloud_type='Int32',
                      dp_pattern='Int32', precipitation=float, precipitation_time_code='Int32', gust=float,
                      operator_type=str)
        if not isinstance(data_values, pd.DataFrame):
            data_values = pd.DataFrame(data_values)
        self.data_values = data_values.astype(dtypes)
        # Set  any zero precip values to missing so they are not plotted.
        L = self.data_values.precipitation == 0.0
        self.data_values.loc[L, 'precipitation'] = np.nan
        self.data_values.loc[L, 'precipitation_time_code'] = pd.NA
        self.text_fontsize = text_fontsize
        self.small_text_fontsize = small_text_fontsize
        self.sym_fontsize = sym_fontsize

        # add in the derived data
        # derived vars
        derived = dict()
        cld_text = (fill_fn(self.data_values['cloud_base_amount'], 1, miss='/') + '/' +
                    fill_fn(self.data_values['cloud_base_height_code'], 1, miss='/'))
        cld_text = cld_text.where(cld_text != '///', '')  # all missing. Do not plot anything.
        cld_text = cld_text.where(cld_text != '//9', '')  # all missing. Do not plot anything.
        derived['cld_text'] = cld_text
        dp_text = fill_fn((self.data_values['dp'] * 10).abs(), 2)
        derived['dp_text'] = dp_text
        current_weather = np.where(self.data_values.operator_type == 'MANUAL', self.data_values.present_weather,
                                   self.data_values.present_weather + 100)

        msk = np.isfinite(current_weather)
        current_weather[~msk] = -1
        current_weather = np.ma.array(current_weather.astype(int), mask=~msk)
        derived['current_weather_both'] = current_weather
        self.derived = derived

    # @staticmethod
    # def _to_string_list(vals, fmt): # overwrite the default _to_string_list
    #     """Convert a sequence of values to a list of strings."""
    #
    #     # if fmt is None then reset it to default value. Not sure why this is needed.
    #     if fmt is None:
    #         fmt='.0f'
    #
    #
    #     if not callable(fmt):
    #         def formatter(s):
    #             """Turn a format string into a callable."""
    #             if pd.isna(s):
    #                 return ''
    #             else:
    #                 return format(s, fmt)
    #     else:
    #         formatter = fmt
    #     import copy
    #     result = [formatter(copy.copy(val)) for val in vals] # run the formatter
    #
    #     return result
    @staticmethod
    def _to_string_list(vals, fmt):
        """Convert a sequence of values to a list of strings."""
        if fmt is None:
            import sys
            if sys.version_info >= (3, 11):
                fmt = 'z.0f'
            else:
                def fmt(s):
                    """Perform default formatting with no decimal places and no negative 0."""
                    return format(round(s, 0) + 0., '.0f')
        if not callable(fmt):
            def formatter(s):
                """Turn a format string into a callable."""
                return format(s, fmt)
        else:
            formatter = fmt

        return [formatter(v) if pd.notna(v) else '' for v in vals]

    @staticmethod
    def Int_to_list(vals: pd.Series) -> list:
        """
        Convert a series of dtype Int32 to a list.
        :param vals: series to convert
        :return: list of strings
        """
        result = [np.nan if pd.isna(v) else v for v in vals.to_list()]
        return result

    def extract_data(self, parameter,
                     fill_value=None) -> typing.Optional[pd.Series | list]:

        try:
            values = self.data_values[parameter]
        except KeyError:
            try:
                values = self.derived[parameter]
            except KeyError:
                print('No data for ', parameter)
                return None

        # specials for values being a pandas series
        if isinstance(values, pd.Series):
            if fill_value is not None:
                values = values.fillna(fill_value)
            #if isinstance(values.dtype, (pd.Int8Dtype, pd.Int32Dtype, pd.Int16Dtype, pd.Int64Dtype,dtype('O'))):
            values = self.Int_to_list(values)
        return values

    def plot(self, annotate: typing.Union[dict, bool] = False, **kwargs):

        if annotate is True:
            annotate = dict()  # make it an empty directory/
        text_dict = dict(fontsize=self.text_fontsize, fontweight='bold', color='black')
        sym_dict = dict(fontsize=self.sym_fontsize, fontweight='bold', color='black')
        null_format = lambda x: x  # dummy formatter to avoid formatting strings.

        def mdi_format(v: float | int | str | None, fmt: str | int = 'z0.f', mdi: str = '') -> str | int:
            """
            Formatter that can cope with missing data
            :param v: Value
            :param fmt: format to use
            :param mdi: What to return if missing
            :return: string or int
            """
            if isinstance(v, str):
                return v
            if pd.isna(v):
                return mdi
            else:
                return format(v, fmt)

        def gust_formatter(v: float | None) -> str:
            if pd.isna(v) or v < 25.0:
                return ''
            else:
                return f'G{v:.0f}'

        fill_values = dict(current_weather=103, total_cloud_amount=10)

        # default values for text elements.
        text_elements = dict(
            msl_pressure=dict(fontsize=self.small_text_fontsize, formatter=lambda v: mdi_format(10 * v, fmt='.0f')[-3:],
                              color='black', location='NE'),
            air_temperature=dict(**text_dict, location='NW'),
            dewpoint=dict(**text_dict, location='SW'),
            precipitation=dict(**text_dict, location=(1, -2)),
            visibility_code=dict(color='black', fontsize=self.small_text_fontsize,
                                 formatter=lambda v: mdi_format(v, fmt='02d', mdi='/'), ha='right', location=(-1.1, 0)),
            cld_text=dict(fontsize=self.small_text_fontsize, formatter=null_format, va='top', location=(0, -1.5)),
            dp_text=dict(fontsize=self.small_text_fontsize, ha='right', formatter=null_format, location=(1.25, 0)),
            gust=dict(fontsize=self.small_text_fontsize, formatter=gust_formatter, location=(1.0, 2.)),
            precipitation_time_code=dict(fontsize=self.small_text_fontsize, ha='left',
                                         formatter=lambda x: mdi_format(x, fmt='1d', mdi='/'), location=(1.5, -1)),
        )
        for k in text_elements.keys():
            text_elements[k].update(kwargs.get(k, {}))

        # symbol plotting
        sym_elements = dict(
            total_cloud_amount=dict(location='C', ha='center', symbol_mapper=sky_cover, fontsize=self.sym_fontsize,
                                    color='black'),
            low_cloud_type=dict(location=(0, -0.5), symbol_mapper=low_clouds, **sym_dict, va='top'),
            dp_pattern=dict(location=(1.255, 0), symbol_mapper=pressure_tendency, fontsize=self.small_text_fontsize,
                            ha='left'),
            current_weather_both=dict(location=(-0.5, 0), symbol_mapper=current_weather_both, **sym_dict, ha='right'),
            medium_cloud_type=dict(location=(0, 0.75), symbol_mapper=mid_clouds, **sym_dict, va='bottom'),
            high_cloud_type=dict(location=(0, 1.75), symbol_mapper=high_clouds, **sym_dict, va='bottom'),
            past_weather=dict(location='SE', symbol_mapper=past_weather, **sym_dict)
        )
        # and potentially override them
        for key in sym_elements.keys():
            sym_elements[key].update(kwargs.get(key, {}))
        barb = dict(zorder=-100, color='black')
        barb.update(kwargs.get('barb', {}))

        # Plot all the text elements. See plot_text_elements if you want to modify.
        for parameter, keywrds in text_elements.items():
            loc = keywrds.pop('location')
            values = self.extract_data(parameter, fill_value=fill_values.get(parameter))
            if values is not None:
                self.plot_parameter(loc, values, **keywrds)

        for parameter, keywrds in sym_elements.items():
            loc = keywrds.pop('location')
            values = self.extract_data(parameter, fill_value=fill_values.get(parameter))
            if values is not None:
                self.plot_symbol(loc, values, **keywrds)

        # handle automatic stations -- a bit of a hack!

        mask = self.data_values.operator_type != 'AUTOMATIC'
        values = [pd.NA if x else 11 for x in mask]
        tt = sym_elements['total_cloud_amount'].copy()
        tt.update(fontsize=tt['fontsize'] * 1.6)
        tt.update(location=(0.00, 0.27))  # empirically defined
        tt.update(fontweight='black')
        tt.update(symbol_mapper=auto_stations)

        self.plot_symbol(codes=values, **tt)  # triangles for automatic stations.
        u, v = metpy.calc.wind_components(speed=self.data_values.wind_speed.values * units.knots,
                                          wind_direction=self.data_values.wind_direction.values * units.degrees)
        north = np.cos(np.deg2rad(self.data_values['wind_direction'])) * self.data_values['wind_speed']
        east = np.sin(np.deg2rad(self.data_values['wind_direction'])) * self.data_values['wind_speed']

        self.plot_barb(u, v, plot_units=units.knots, **barb)  # direction from which wind is coming.


@cachier.cachier()
def get_era5_pressure(date: pd.Timestamp,
                      region: typing.Optional[tuple[float, float, float, float]] = default_region) -> typing.Optional[
    xarray.Dataset]:
    import cdsapi
    import tempfile

    print('Getting SLP data from ERA-5')
    # data only available upto 5 days from now
    if (pd.Timestamp.utcnow() - date) < pd.Timedelta(5, 'D'):
        raise ValueError(
            'ERA-5 data only available upto 5 days from now')  # raise a valuer error -- trap it. But hopefully stops caching.
    dataset = "reanalysis-era5-single-levels"
    area = [region[indx] for indx in [3, 0, 2, 1]]  # ERA-5 area select is NWSE or indices is 3,0,2,1
    request = {
        'product_type': ['reanalysis'],
        'variable': ['mean_sea_level_pressure'],
        'year': [date.year],
        'month': [date.month],
        'day': [date.day],
        'time': [date.hour],
        'data_format': 'netcdf',
        'download_format': 'unarchived',
        'area': area,
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='nc') as f:  # used a temp file. Will be deleted when context done.
        filename = f.name
        client = cdsapi.Client()  # will need to set up .cdsapirc file. See https://cds-beta.climate.copernicus.eu/how-to-api
        client.retrieve(dataset, request, filename)
        pressure = xarray.load_dataset(filename)  # now load it
    return pressure  #


@cachier.cachier()
def retrieve_synops(date_range: tuple[pd.Timestamp, pd.Timestamp],
                    use_cache: bool = True,
                    block: typing.Optional[str] = '03',
                    state: typing.Optional[str] = None) -> pd.DataFrame:
    """
    Retrieve SYNOP messages from the OGIMET website for a given date range, WMO block and state.
     block and state seem to be mutually exclusive.
    :param date_range:data range to retrieve data for. 0 is min; 1 is max.
    :param use_cache: If True use the cache if it exists. If False, retrieve the data from the website.
    :param block: block to use. Default is 33 -- GB & NI
    :param state: state to use. Note state='UK' will retrieve all UK data. (GB+NI + overseas territories)
    :return: dataframe of SYNOP messages.
    """

    # URL to retrieve the data
    url = 'https://www.ogimet.com/cgi-bin/getsynop'
    if block is not None and state is not None:
        raise ValueError('block and state are mutually exclusive')
    # Define query parameters
    params = {
        'begin': date_range[0].strftime('%Y%m%d%H%M'),
        'end': date_range[1].strftime('%Y%m%d%H%M'),
        'block': block,  # UK stations
        'state': state,
        'header': 'yes',
        'lang': 'eng',

    }

    # Define headers
    headers = {
        'User-Agent': 'UniEdinburgh_retr_synop/0.0.1'
    }

    file = 'SYNOP'
    for p in ['begin', 'end', 'block', 'state']:
        file += f'_{params[p]}'
    file += '.csv'
    save_file = cache_dir / file
    if save_file.exists() and use_cache:
        print(f'loading data from {save_file}')
        synop_messages = pd.read_csv(save_file, index_col=[0], parse_dates=['ob_time'])
    else:  # retrieve it
        print(f'Retrieving data from {url}')
        # Send an HTTP GET request to the URL with additional keyword arguments
        response = requests.get(url, params=params, headers=headers, timeout=10)

        # Check if the request was successful
        if response.status_code == 200:
            # Read the CSV data from the response content
            synop_messages = pd.read_csv(StringIO(response.text))
            # Fix the columns names so they are in English
            translate = dict(
                zip(synop_messages.columns, ['Station', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Synop']))
            synop_messages = synop_messages.rename(columns=translate)
            time_cols = ['Year', 'Month', 'Day', 'Hour', 'Minute']
            synop_messages['ob_time'] = pd.to_datetime(synop_messages[time_cols], utc=True)
            synop_messages = synop_messages.drop(columns=time_cols)
            print(synop_messages.head())  # Display the first few rows of the dataframe
            synop_messages.to_csv(save_file)  # save the file for later use.
        else:
            raise ValueError(f'Failed to retrieve data: {response.status_code}')
    return synop_messages


def decode_synop_messages(synop_messages: pd.DataFrame) -> pd.DataFrame:
    """
    Decode the SYNOP messages and return a dataframe with the decoded data.
    :param synop_messages: dataframe containing the SYNOP messages -- must have a column called 'Synop'
    :return: dataframe with the decoded data + initial data
    """

    specials = dict(cloud_base_height_code='_code',
                    visibility_code='_code')  # special cases where we want something else than the value
    # for each element we want to provide a path to the element in the decoded message.
    decode = dict(cloud_base_amount=['cloud_layer', 0, 'cloud_cover'],
                  cloud_base_height_code=['lowest_cloud_base'],
                  msl_pressure=['sea_level_pressure'],
                  air_temperature=['air_temperature'], dewpoint=['dewpoint_temperature'],
                  wind_speed=['surface_wind', 'speed'],
                  wind_direction=['surface_wind', 'direction'], present_weather=['present_weather'],
                  past_weather=['past_weather', 0],
                  visibility_code=['visibility'], total_cloud_amount=['cloud_cover'],
                  low_cloud_type=['cloud_types', 'low_cloud_type'],
                  medium_cloud_type=['cloud_types', 'middle_cloud_type'],
                  high_cloud_type=['cloud_types', 'high_cloud_type'],
                  dp_pattern=['pressure_tendency', 'tendency'], dp=['pressure_tendency', 'change'],
                  src_oper_id=['weather_indicator'],
                  wmo_station_id=['station_id'],
                  gust=['highest_gust', 0, 'speed'])
    convert_to_knots = ['wind_speed', 'gust']  # variables to convert to knots.
    synops = (synop_messages['Synop'].str.replace('==', '').str.replace('=', ''))
    # remove the == and = that are sometimes in the messages.
    decoded_synops = []
    for name, message in synops.items():

        decode_raw = pymetdecoder.synop.SYNOP().decode(message.strip())
        decode_trans = dict()
        # Extract what we need from the decoded message
        for key, value in decode.items():
            v = decode_raw
            try:
                for item in value:
                    v = v[item]
                final_key = specials.get(key, 'value')
                decode_trans[key] = v[final_key]
                if key in convert_to_knots:  # need to convert to knots
                    unit = v.get('unit', 'KT')  # assume knots if not specified.
                    if unit == 'm/s':
                        decode_trans[key] *= 1.943844  # convert to knots
                        print(f'converted {key} from {unit} to knots')
            except (KeyError, TypeError):
                decode_trans[key] = None
        # precipitation is tricky! As comes in form precipitation_sx where x is the time
        # find the key that is precipitation.
        for key in decode_raw.keys():
            if (key.startswith('precipitation_s') and (decode_raw[key] is not None) and
                    (decode_raw[key].get('amount') is not None)):
                decode_trans['precipitation'] = decode_raw[key]['amount']['value']
                decode_trans['precipitation_time_code'] = decode_raw[key]['time_before_obs']['_code']
                continue

        # deal with automatic/manual stations
        wx = decode_raw.get('weather_indicator', None)
        if wx is None:
            print(f'Warning: no weather indicator for {message.strip()}')
            decode_trans['operator_type'] = 'UNKNOWN'  # no weather indicator
        else:
            decode_trans['operator_type'] = 'AUTOMATIC' \
                if decode_raw['weather_indicator'].get('automatic', False) else 'MANUAL'
        decode_trans = pd.Series(decode_trans).rename(name)
        decoded_synops.append(decode_trans)
    decoded_synops = pd.DataFrame(decoded_synops)
    float_values = ['air_temperature', 'dewpoint', 'wind_speed', 'msl_pressure', 'dp',
                    'wind_direction', 'gust', 'precipitation']
    non_int_values = float_values + ['operator_type']
    types = {key: 'Int32' for key in decoded_synops.columns if key not in non_int_values}
    types.update({key: 'float' for key in float_values if key in decoded_synops.columns})
    decoded_synops = decoded_synops.astype(types)
    all_data = synop_messages.join(decoded_synops)
    return all_data


@cachier.cachier()
def read_isd_metadata(file: typing.Optional[pathlib.Path] = None,
                      country: typing.Optional[str | tuple[str]] = None,
                      use_cache: bool = True) -> pd.DataFrame:
    """
    Read the ISD metadata file and return a dataframe with the data.
    :param country: Country to extract. If None, will extract all countries.
    :param use_cache: If True, use the cache if it exists. If False, retrieve the data from the NOAA website.
    :param file: file to read. If None, will read the default file.
    :return: dataframe with the metadata.
    """

    if file is None:
        file = cache_dir / 'isd-history.txt'
    if isinstance(country, str):
        country = (country,)
    if not (file.exists() and use_cache):

        import urllib.request
        import urllib.error

        # FTP URL to retrieve the data
        url = f'ftp://ftp.ncdc.noaa.gov/pub/data/noaa/{file.name}'

        # retrieve the data
        try:
            print(f'Retrieving ISD metadata to {file}')
            f, h = urllib.request.urlretrieve(url, filename=file)
            print(f'Downloaded {url} to {file} successfully.')
        except urllib.error.URLError as e:
            raise ValueError(f'Failed to retrieve data: {e.reason}')

    isd_data = pd.read_fwf(file, skiprows=20, parse_dates=['BEGIN', 'END'])  # letting pandas do the work. :-)
    if country is not None:
        isd_data = isd_data[isd_data.CTRY.isin(country)]
        #isd_data = isd_data[isd_data.CTRY == country]
    # convert dates.
    return isd_data


@cachier.cachier()
def load_open_midas_synop(date_range: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    """
    Load the open midas data for the given date range.
    :param date_range: date range to load data for. 0 -- min value, 1 max value to load.
    :return: dataframe of SYNOP data from open_midas files.
    :rtype:
    """

    print('loading data')
    srce = read_midas_srce(root_dir / 'data/SRCE.DATA')  # read the meta data
    srce = srce.rename(columns=dict(high_prcn_lat='latitude', high_prcn_lon='longitude'))
    years = [d.year for d in date_range]  # years to load
    years = range(years[0], years[1] + 1)
    all_stations = []
    for year in years:
        print('Loading year ', year)
        files = list(
            pathlib.Path(f'Synop_data/badc/ukmo-midas-open/data/uk-hourly-weather-obs/dataset-version-202308').glob(
                f'**/midas-open_uk-hourly-weather-obs_*qcv-1*{year}.csv'))
        for file in files:
            all_stations += [read_midas_wh_file(file, read_midas_open=True,
                                                obs_type='SYNOP', date_range=date_range)]
    all_stations = pd.concat(all_stations, axis=0)
    all_stations = all_stations.merge(srce, left_on='src_id', right_on='src_id')
    return all_stations


@cachier.cachier()
def read_synops(date: pd.Timestamp = pd.Timestamp.utcnow(),
                region: tuple[typ_flt_int, typ_flt_int, typ_flt_int, typ_flt_int] = default_region,
                nocache: bool = False,
                use_midas_csv: bool = False,
                get_pressure: bool = False) -> tuple[pd.DataFrame, typing.Optional[xarray.Dataset]]:
    """
    Read the SYNOP data for the given date and region.
    :param date: date to read the data for.
    :param region: region to read the data for. (long0, long1, lat0, lat1)
    :param nocache: If True, do not use the cache.
    :param use_midas_csv: If True, use the open-midas csv files. If False, use the OGIMET website.
    :param get_pressure: If True, try and retrieve the pressure data from ERA5.
    :return: tuple of the SYNOP data and the pressure data. pressure data will be None if get_pressure is False.
    """
    try:
        date = date.tz_localize('UTC')
    except TypeError:  # already in UTC
        pass
    date = date.round('h')  # round to nearest hour
    if date > pd.Timestamp.utcnow() + pd.Timedelta(1, 'h'):
        raise ValueError('Date is in the future. Cannot plot data for the future.')

    if use_midas_csv:  # use the downloaded open-midas data
        date_range = (date - pd.Timedelta(4, 'h'), date + pd.Timedelta(1, 'h'))  # data range to retrieve.
        midas_stations = load_open_midas_synop(date_range, cachier__skip_cache=nocache)
        synops = midas_stations[midas_stations.ob_time == date].groupby(by='src_id', as_index=True).head(
            1).set_index('src_id')

        # add in derived data for pressure changes.
        p_data = midas_stations.loc[:, ['src_id', 'ob_time', 'msl_pressure']]
        p_data = p_data[p_data.ob_time.between(date - pd.Timedelta(3, 'h'), date)]
        pp = p_data.groupby('src_id').apply(df_p_change, include_groups=False).droplevel(1)
        synops = synops.merge(pp, left_index=True, right_index=True)
        synops['cloud_base_height_code'] = cld_base_code(synops.cloud_base_height)  # temp hac

    else:  # get the raw synop messages and decode them
        date_range = (date, date)  # date range to retrieve -- just the date!
        isd = read_isd_metadata(country=('UK', 'EI'), cachier__skip_cache=nocache)  # extract the UK data
        # now convert USAF locations to WMO locations. -- first five values.
        isd['wmo_station_id'] = isd.USAF.str[0:5].astype('Int32')
        isd = isd.rename(columns=dict(LON='longitude', LAT='latitude'))
        synops = retrieve_synops(date_range, state=None, cachier__skip_cache=nocache)
        decoded_synops = decode_synop_messages(synops)
        # add on meta-data
        decoded_synops = decoded_synops.merge(isd, left_on='wmo_station_id', right_on='wmo_station_id')
        synops = decoded_synops[decoded_synops.ob_time == date]

    # restrict to region..
    synops = synops[
        synops.latitude.between(region[2], region[3]) &
        synops.longitude.between(region[0], region[1])]
    # fix current weather
    synops['present_weather'] = synops['present_weather'].fillna(0)  # should be 103??
    # extract pressure if requested.
    pressure = None
    if get_pressure:
        try:
            pressure = get_era5_pressure(date, region=region, cachier__skip_cache=nocache)
        except ValueError:
            print('Failed to retrieve ERA5 data. ')
            pass
    return synops, pressure


def plot_synops(synops: pd.DataFrame,
                pressure: typing.Optional[xarray.Dataset] = None,
                thin: float = 100.,
                figsize: tuple[float, float] = (10, 10),  #
                region: tuple[float, float, float, float] = (-11., 2., 49.0, 61.5)
                ) -> tuple[plt.Figure, plt.Axes]:
    """"
    Plot the SYNOP data on a map.
    :param synops: dataframe of SYNOP data.
    :param pressure: xarray dataset of pressure data. (If None not plotted)
    :param thin: thinning distance in km.
    :param figsize: size of the figure.

    :returns: matplotlib figure and ax.
    """
    # thin the data
    proj = ccrs.AlbersEqualArea(central_longitude=0, central_latitude=54, false_easting=400000,
                                false_northing=-100000)  # UK projection
    # should be a function of lon/lat co-ords.
    point_locs = proj.transform_points(ccrs.PlateCarree(), synops.longitude, synops.latitude)
    priority = np.where(synops.operator_type == 'MANUAL', 10, 0)
    # increase priority where have present_weather.
    priority[synops.present_weather.notnull()] = priority[synops.present_weather.notnull()] + 5
    synops_to_plot = synops[reduce_point_density(point_locs, thin * 1e3, priority=priority)]
    # plot the data
    # default tweaked plotting arguments
    cloud_type = dict(color='black')
    weather = dict(color='black', fontsize=11)
    kwrds = dict(
        low_cloud_type=cloud_type,
        medium_cloud_type=cloud_type,
        high_cloud_type=cloud_type,
        current_weather_both=weather,
        past_weather=weather,
        air_temperature=dict(color='red'),
        precipitation=dict(color='blue'),
        precipitation_time_code=dict(color='blue'),
        dewpoint=dict(color='green'),

    )

    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw=dict(projection=proj), clear=True,
                           layout='tight', num='synop_circ')

    ax.set_extent(region, crs=ccrs.PlateCarree())
    ax.coastlines(color='grey', linewidth=1)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, color='grey')
    # plot the pressure (if we have it)
    if pressure is not None:
        p = pressure.msl.squeeze(drop=True) / 100  # convert to hPa
        cs = p.plot.contour(ax=ax, transform=ccrs.PlateCarree(), colors='grey', levels=range(960, 1040, 4), zorder=-200)
        ax.clabel(cs, inline=True, fontsize=10, zorder=-200, colors='grey')
    stns = SynopPlot(ax, synops_to_plot.longitude, synops_to_plot.latitude, synops_to_plot,
                     transform=ccrs.PlateCarree(),
                     text_fontsize=10, small_text_fontsize=8, clip_on=True)
    stns.plot(**kwrds)  # plot them.  Override anything that needs overwritten here
    ax.set_title(f'{synops_to_plot.ob_time.unique()[0]}')
    return fig, ax

def clear_cache(module):
    """

    :return: Nothing
    Clear cache for all functions in a module.
    Does by finding all callables that have a clear_cache method and calling it.
    """
    import inspect
    for (name,thing) in inspect.getmembers(module):
        if callable(thing) and hasattr(thing, 'clear_cache'):
            print(name)
            thing.clear_cache()
