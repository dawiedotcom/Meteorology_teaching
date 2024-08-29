# retrieve SYNOP messages and then plot them.
from __future__ import annotations

import argparse
import pathlib
import typing
from io import StringIO

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use("Agg") # headless matplotlib
import numpy as np
import pandas as pd
import pymetdecoder.synop
import requests
from metpy.calc import reduce_point_density

import metlib
import pymetdecoder

cache_dir = metlib.cache_dir
pd.options.mode.copy_on_write = True
from cachier import cachier  # provides a cache for functions
import datetime


@cachier(cache_dir=cache_dir, separate_files=True)
@cachier(stale_after=datetime.timedelta(weeks=2), next_time=True)
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


@cachier(stale_after=datetime.timedelta(weeks=2), next_time=True)
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


@cachier(stale_after=datetime.timedelta(weeks=2), next_time=True)
def load_open_midas_synop(date_range: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    """
    Load the open midas data for the given date range.
    :param date_range: date range to load data for. 0 -- min value, 1 max value to load.
    :return: dataframe of SYNOP data from open_midas files.
    :rtype:
    """

    print('loading data')
    srce = metlib.read_midas_srce(metlib.root_dir / 'data/SRCE.DATA')  # read the meta data
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
            all_stations += [metlib.read_midas_wh_file(file, read_midas_open=True,
                                                       obs_type='SYNOP', date_range=date_range)]
    all_stations = pd.concat(all_stations, axis=0)
    all_stations = all_stations.merge(srce, left_on='src_id', right_on='src_id')
    return all_stations


# setup parser and deal with arguments
typ_flt_int = typing.Union[float, int]
parser = argparse.ArgumentParser(description='Plot SYNOP data')
parser.add_argument('--date', type=pd.Timestamp, help='UTC date/time to plot data for. ',
                    default=pd.Timestamp.utcnow() - pd.Timedelta(2, 'h'))
parser.add_argument('--use_cache', type=bool, help='Use the cache for the data.', default=True)
parser.add_argument('--thin', type=float, help='Thin distance in km', default=40.)
#parser.add_argument('--block', type=str, help='WMO block to use.', default='33')
#parser.add_argument('--state', type=str, help='State to use. Do not specify block and state', default=None)
parser.add_argument('--output', type=pathlib.Path,
                    help='Output file to save the plot to. If not specified with be constructed from date and be pdf',
                    default=None)
parser.add_argument('--figsize', nargs=2, type=float, help='Page size for the plot. Default is A3.',
                    default=(11.69, 16.53))
parser.add_argument('--region', nargs=4, type=float,
                    help='Region (long0,long1,lat0,lat1) to plot in degrees. Default is UK.',
                    default=(-11., 2., 49.0, 61.5))  # from Guernsey to Shetland, W-Ireland to E-England
parser.add_argument('--use_midas_csv', action='store_true',
                    help='Use open-midas csv files. They should already have been downloaded from BADC. ')
parser.add_argument('--nocache', action='store_true', help='Do not use the cache.')
parser.add_argument('--plot_pressure', action='store_true',
                    help='Plot the pressure on the map. Will try and retrieve ERA5 data.')
args = parser.parse_args()

date = args.date.round('h')  # round to nearest hour
try:
    date = date.tz_localize('UTC')
except TypeError:  # already in UTC
    pass

if date > pd.Timestamp.utcnow() + pd.Timedelta(1, 'h'):
    raise ValueError('Date is in the future. Cannot plot data for the future.')

save_file = args.output
if save_file is None:
    save_file = date.strftime("station_plot_%Y%m%d_%H.pdf")
date_range = (date - pd.Timedelta(4, 'h'), date + pd.Timedelta(1, 'h'))  # data range to retrieve.
#proj = ccrs.OSGB()
proj = ccrs.AlbersEqualArea(central_longitude=0, central_latitude=54, false_easting=400000, false_northing=-100000)
# add in meta-data
if args.use_midas_csv:  # use the downloaded open-midas data
    midas_stations = load_open_midas_synop(date_range, cachier__skip_cache=args.nocache)
    synops_to_plot = midas_stations[midas_stations.ob_time == date].groupby(by='src_id', as_index=True).head(
        1).set_index('src_id')

    # add in derived data for pressure changes.
    p_data = midas_stations.loc[:, ['src_id', 'ob_time', 'msl_pressure']]
    p_data = p_data[p_data.ob_time.between(date - pd.Timedelta(3, 'h'), date)]
    pp = p_data.groupby('src_id').apply(metlib.df_p_change, include_groups=False).droplevel(1)
    synops_to_plot = synops_to_plot.merge(pp, left_index=True, right_index=True)
    synops_to_plot['cloud_base_height_code'] = metlib.cld_base_code(synops_to_plot.cloud_base_height)  # temp hac

else:  # get the raw synop messages and decode them
    isd = read_isd_metadata(country=('UK', 'EI'), cachier__skip_cache=args.nocache)  # extract the UK data
    # now convert USAF locations to WMO locations. -- first five values.
    isd['wmo_station_id'] = isd.USAF.str[0:5].astype('Int32')
    isd = isd.rename(columns=dict(LON='longitude', LAT='latitude'))
    synops = retrieve_synops(date_range, state=None, cachier__skip_cache=args.nocache)
    decoded_synops = decode_synop_messages(synops)
    # add on meta-data
    decoded_synops = decoded_synops.merge(isd, left_on='wmo_station_id', right_on='wmo_station_id')
    synops_to_plot = decoded_synops[decoded_synops.ob_time == date]

# restrict to region..
synops_to_plot = synops_to_plot[
    synops_to_plot.latitude.between(args.region[2], args.region[3]) &
    synops_to_plot.longitude.between(args.region[0], args.region[1])]

# thin
point_locs = proj.transform_points(ccrs.PlateCarree(), synops_to_plot.longitude, synops_to_plot.latitude)
priority = np.where(synops_to_plot.operator_type == 'MANUAL', 10, 0)
# increase priority where have present_weather.
priority[synops_to_plot.present_weather.notnull()] = priority[synops_to_plot.present_weather.notnull()] + 5
synops_to_plot['present_weather'] = synops_to_plot['present_weather'].fillna(0)
synops_to_plot = synops_to_plot[reduce_point_density(point_locs, args.thin * 1e3, priority=priority)]
pressure = None
if args.plot_pressure:
    try:
        pressure = metlib.get_era5_pressure(date, region=args.region, cachier__skip_cache=args.nocache)
    except ValueError:
        print('Failed to retrieve ERA5 data. Will not plot pressure')
        pass
## data fixes
#set 0 precip missing.

## plot the data
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
    #visibility_code = dict(color='springgreen'),
    #cld_text = dict(color='purple'),
)

fig_map_synop, ax = plt.subplots(1, 1, figsize=args.figsize, subplot_kw=dict(projection=proj), clear=True,
                                 layout='tight', num='synop_circ')

ax.set_extent(args.region, crs=ccrs.PlateCarree())
ax.coastlines(color='grey', linewidth=1)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, color='grey')
# plot the pressure (if we have it)
if pressure is not None:
    p = pressure.sel(valid_time=date.tz_convert(None)).msl.squeeze(drop=True) / 100  # convert to hPa
    cs = p.plot.contour(ax=ax, transform=ccrs.PlateCarree(), colors='grey', levels=range(960, 1040, 4), zorder=-200)
    ax.clabel(cs, inline=True, fontsize=10, zorder=-200, colors='grey')
synop = metlib.SynopPlot(ax, synops_to_plot.longitude, synops_to_plot.latitude, synops_to_plot,
                         transform=ccrs.PlateCarree(),
                         text_fontsize=10, small_text_fontsize=8, clip_on=True)
synop.plot(**kwrds)  # plot them.  Override anything that needs overwritten here
ax.set_title(f'{date}')
fig_map_synop.show()
fig_map_synop.savefig(save_file, dpi=300, bbox_inches="tight")
