import os
import shutil
import zipfile
import numpy as np
import xarray as xr
import pandas as pd


def open_zarr(path, variables=None, preprocess=None):
    """ Open variables from a zarr collection. Varaibles that don't exist are ignored
        without warning
    """
    def _get_variables(ds):
        """Return variables that are in dataset"""
        if variables is None:
            return ds
        else:
            return ds[list(set(ds.data_vars) & set(variables))]
    ds = _get_variables(
        xr.open_zarr(path, consolidated=True, use_cftime=True))
    if preprocess:
        ds = preprocess(ds)       
    return ds


def open_mfzarr(paths, variables=None, preprocess=None):
    """ Open a variables from a zarr collections and combine"""
    ds = [open_zarr(
        path, variable, preprocess) for path in paths]
    return xr.combine_by_coords(ds)


def open_zarr_forecasts(paths, variables=None, preprocess=None, time_name='time'):
    """ Open multiple forecast zarr collections and stack by initial date and lead time"""
    datasets = []; times = []
    for path in paths:
        ds = open_zarr(path, variables, preprocess)
        lead_time = range(len(ds[time_name]))
        init_date = ds[time_name].values[0]
        times.append(
            ds[time_name].rename({time_name: 'lead_time'}
            ).assign_coords({'lead_time': lead_time}))
        datasets.append(
            ds.rename({time_name: 'lead_time'}
            ).assign_coords({'lead_time': lead_time,
                             'init_date': init_date}))
        
    dataset = xr.concat(datasets, dim='init_date')
    time = xr.concat(times, dim='init_date')
    return dataset.assign_coords({time_name: time})


def to_zarr(ds, filename, zip=True):
    """ Write to zarr file"""
    
    def _zip_zarr(zarr_filename):
        """ Zip a zarr collection"""
        filename = f'{zarr_filename}{os.path.extsep}zip'
        with zipfile.ZipFile(
            filename, "w", 
            compression=zipfile.ZIP_STORED, 
            allowZip64=True) as fh:
            for root, _, filenames in os.walk(zarr_filename):
                for each_filename in filenames:
                    each_filename = os.path.join(root, each_filename)
                    fh.write(
                        each_filename,
                        os.path.relpath(each_filename, zarr_filename))
                
    for var in ds.variables:
        ds[var].encoding = {}
    ds.to_zarr(filename, mode='w', consolidated=True)
    if zip:
        _zip_zarr(filename)
        shutil.rmtree(filename)


def get_region(ds, region, wrap_lons=True, lat_name='lat', lon_name='lon'):
    """ Return a region from a provided DataArray or Dataset
        
        Parameters
        ----------
        region: xarray DataArray or list
            The region to extract. Can be provided in three formats:
              - if DataArray, a mask with True for points to include and False elsewhere
              - if list of length 4, a region specified by latitudinal and longitudinal 
                bounds as follows [lat_min, lat_max, lon_min, lon_max]
              - if list of length 2, a point location as follows [lat, lon]. The nearest 
                grid cell is returned in this case
        wrap_lons : boolean, optional
            Wrap longitude values of da into the range between 0 and 360
    """
    if wrap_lons:
        ds = ds.assign_coords({lon_name: (ds[lon_name] + 360)  % 360})
        ds = ds.sortby(lat_name).sortby(lon_name)
        
    if isinstance(region, xr.DataArray):
        return ds.where(region)
    elif isinstance(region, (tuple, list)):
        if len(region) == 2:
            return ds.sel({lat_name: region[0], 
                           lon_name: region[1]},
                          method='nearest')
        elif len(region) == 4:
            return ds.sel({lat_name: slice(region[0], region[1]),
                           lon_name: slice(region[2], region[3])})
    else:
        raise InputError('Unrecognised format for input region')
        
        
def stack_by_init_date(da, init_dates, N_lead_steps, init_date_name='init_date', 
                       lead_time_name='lead_time', time_name='time',
                      tolerance='D'):
    """ Stacks provided timeseries array in an inital date / lead time format. Note this process \
        replicates data and can substantially increase memory usage. Lead time frequency will match \
       frequency of input data. Returns nans if requested times lie outside of the available range.
    """

    # Convert cftimes to nptimes
    if xr.core.common.contains_cftime_datetimes(da[time_name]):
        times_np = xr.coding.times.cftime_to_nptime(da[time_name])
    else:
        times_np = da[time_name]
    
    if xr.core.common.contains_cftime_datetimes(init_dates):
        init_dates_np = xr.coding.times.cftime_to_nptime(init_dates)
    else:
        init_dates_np = init_dates
        
    freq = pd.infer_freq(times_np)
    if freq is None:
        raise ValueError('Cannot determine frequency of input timeseries')
    
    init_list = []
    for i in range(len(init_dates)):
        start_index = np.where(times_np == init_dates_np[i])[0]

        if start_index.size == 0:
            end_time = np.datetime64(pd.DatetimeIndex([init_dates_np[i]]).shift(N_lead_steps, freq=freq)[0])
            end_index = np.where(times_np.astype(f'datetime64[{tolerance}]') == 
                                 end_time.astype(f'datetime64[{tolerance}]'))[0]
            if end_index.size == 0:
                # Try step back from the end time - this can help with strange frequencies eg Q-DEC
                start_time = np.datetime64(pd.DatetimeIndex([end_time]).shift(-N_lead_steps, freq=freq)[0])
                start_index = np.where(times_np.astype(f'datetime64[{tolerance}]') == 
                                       start_time.astype(f'datetime64[{tolerance}]'))[0]
                
                if start_index.size == 0:
                    da_slice = da.isel({time_name:range(0, N_lead_steps)}).where(False, np.nan)
                    time_slice = da_slice[time_name].where(False, np.nan) \
                                                    .expand_dims({init_date_name: [init_dates[i].item()]})
                    init_list.append(da_slice.expand_dims({init_date_name: [init_dates[i].item()]}) \
                                             .assign_coords({f'{time_name}_new': time_slice}) \
                                             .assign_coords({time_name: np.arange(0, N_lead_steps)}) \
                                             .rename({time_name: lead_time_name}) \
                                             .rename({f'{time_name}_new': time_name}))
                else:
                    start_index = start_index.item()
                    da_slice = da.isel({time_name:slice(start_index, None)})
                    time_slice = da_slice[time_name].expand_dims({init_date_name: [init_dates[i].item()]})
                    init_list.append(da_slice.expand_dims({init_date_name: [init_dates[i].item()]}) \
                                             .assign_coords({f'{time_name}_new': time_slice}) \
                                             .assign_coords({time_name: np.arange(0, len(time_slice[time_name]))}) \
                                             .rename({time_name: lead_time_name}) \
                                             .rename({f'{time_name}_new': time_name}))
            else:
                end_index = end_index.item()
                start_index = end_index - N_lead_steps
                if start_index < 0:
                    hang = -start_index
                    start_index = 0
                else: hang = 0
                    
                da_slice = da.isel({time_name:range(start_index, end_index)})
                time_slice = da_slice[time_name].expand_dims({init_date_name: [init_dates[i].item()]})
                init_list.append(da_slice.expand_dims({init_date_name: [init_dates[i].item()]}) \
                                         .assign_coords({f'{time_name}_new': time_slice}) \
                                         .assign_coords({time_name: np.arange(0, end_index-start_index)+hang}) \
                                         .rename({time_name: lead_time_name}) \
                                         .rename({f'{time_name}_new': time_name}))
        else:
            start_index = start_index.item()
            end_index = min([start_index + N_lead_steps, len(times_np)])
            
            da_slice = da.isel({time_name:range(start_index, end_index)})
            time_slice = da_slice[time_name].expand_dims({init_date_name: [init_dates[i].item()]})
            init_list.append(da_slice.expand_dims({init_date_name: [init_dates[i].item()]}) \
                                     .assign_coords({f'{time_name}_new': time_slice}) \
                                     .assign_coords({time_name: np.arange(0, end_index-start_index)}) \
                                     .rename({time_name: lead_time_name}) \
                                     .rename({f'{time_name}_new': time_name}))
            
    stacked = xr.concat(init_list, dim=init_date_name)
    stacked[lead_time_name].attrs['units'] = freq
    
    return stacked