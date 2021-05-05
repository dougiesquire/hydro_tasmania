import os
import dask
import shutil
import zipfile
import itertools
import numpy as np
import xarray as xr
import pandas as pd
import dask.bag as db
import xskillscore as xs
import matplotlib.pyplot as plt


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
    try:
        ds = xr.open_zarr(path, consolidated=True, use_cftime=True)
    except KeyError:
        # Try zip file
        ds = xr.open_zarr(
            f'{path}{os.path.extsep}zip', consolidated=True, use_cftime=True)
    ds = _get_variables(ds)
    if preprocess:
        ds = preprocess(ds)       
    return ds


def open_mfzarr(paths, variables=None, preprocess=None):
    """ Open a variables from a zarr collections and combine"""
    ds = [open_zarr(
        path, variables, preprocess) for path in paths]
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


def mask_time_period(ds, period, dim='time'):
    """ Mask a period of time. Only works for cftime objects"""
    if dim in ds.dims:
        masked = ds.sel({dim: period})
    elif dim in ds.coords:
        time_bounds = xr.cftime_range(start=period.start, end=period.stop, periods=2, 
                                      freq=None, calendar=ds.time.calendar_type.lower())
        mask = (ds[dim].compute() >= time_bounds[0]) & (ds[dim].compute() <= time_bounds[1])
        masked = ds.where(mask)
    masked.attrs = ds.attrs
    return masked
    
    
def get_monthly_clim(ds, dim, period=None):
    """ Return the climatology over a period of time"""
    if period is not None:
        ds = mask_time_period(ds.copy(), period)
        ds.attrs['climatological_period'] = str(period)
    return ds.groupby(f'{dim}.month').mean(dim, keep_attrs=True)


def get_monthly_anom(ds, dim, clim=None):
    """ Return the monthly anomalies"""
    if clim is not None:
        anom = (ds.groupby(f'{dim}.month') - clim).drop('month')
        anom.attrs['climatological_period'] = clim.attrs['climatological_period']
    else:
        anom = (ds.groupby(f'{dim}.month').map(
            lambda x: x - x.mean(dim, keep_attrs=True))).drop('month')
    anom.attrs = {**anom.attrs, **ds.attrs}
    return anom


def get_bias(fcst, obsv, period, method):
    """ Return the forecast bias over a period"""
    fcst_clim = get_monthly_clim(
        mask_time_period(fcst.mean('ensemble', keep_attrs=True), period), 
        dim='init_date')
    obsv_clim = get_monthly_clim(
        mask_time_period(obsv, period),
        dim='init_date')
    if method == 'additive':
        bias = fcst_clim - obsv_clim
    elif method == 'multiplicative':
        bias = fcst_clim / obsv_clim
    else: 
        raise ValueError(f'Unrecognised mode {mode}')
    bias.attrs['bias_correction_method'] = method
    bias.attrs['bias_correction_period'] = str(period)
    return bias


def remove_bias(fcst, bias):
    """ Remove the forecast bias"""
    if bias.attrs['bias_correction_method'] == 'additive':
        fcst_bc = (fcst.groupby('init_date.month') - bias).drop('month')
    elif bias.attrs['bias_correction_method'] == 'multiplicative':
        fcst_bc = (fcst.groupby('init_date.month') / bias).drop('month')
    else: 
        raise ValueError(f'Unrecognised mode {mode}')
    fcst_bc.attrs['bias_correction_method'] = bias.attrs['bias_correction_method']
    fcst_bc.attrs['bias_correction_period'] = bias.attrs['bias_correction_period']
    return fcst_bc
        
        
def get_metric(fcst, obsv, metric,
               metric_kwargs=None, period=None):
    """ Return an xskillscore metric over a given period. If metric is
        a string, looks for it in xskillscore. Otherwise, metric should
        be a function that receives obsc, fcst and any other kwargs"""
    if period:
        obsv = mask_time_period(obsv.copy(), period=period)
        fcst = mask_time_period(fcst.copy(), period=period)
    if isinstance(metric, str):
        return getattr(xs, metric)(obsv, fcst, **metric_kwargs)
    else:
        return metric(obsv, fcst, **metric_kwargs)


def get_skill_score(fcst, obsv, fcst_baseline, metric, 
                    metric_kwargs=None, period=None):
    """ Return a skill score for a given xskillscore metric over a given period"""
    numerator = get_metric(fcst, obsv, metric, metric_kwargs, period)
    denominator = get_metric(fcst_baseline, obsv, metric, metric_kwargs, period)
    return 1 - (numerator / denominator)


def pearson_r_maybe_ensemble_mean(obsv, fcst, **kwargs):
    if 'ensemble' in fcst.dims:
        fcst = fcst.copy().mean('ensemble')
    return xs.pearson_r(obsv, fcst, **kwargs)

def mse_maybe_ensemble_mean(obsv, fcst, **kwargs):
    if 'ensemble' in fcst.dims:
        fcst = fcst.copy().mean('ensemble')
    return xs.mse(obsv, fcst, **kwargs)


def random_resample(*args, samples,
                    function=None, function_kwargs=None, bundle_args=True,
                    replace=True):
    """
        Randomly resample from provided xarray args and return the results of the subsampled dataset passed through \
        a provided function
                
        Parameters
        ----------
        *args : xarray DataArray or Dataset
            Objects containing data to be resampled. The coordinates of the first object are used for resampling and the \
            same resampling is applied to all objects
        samples : dictionary
            Dictionary containing the dimensions to subsample, the number of samples and the continuous block size \
            within the sample. Of the form {'dim1': (n_samples, block_size), 'dim2': (n_samples, block_size)}. The first \
            object in args must contain all dimensions listed in samples, but subsequent objects need not.
        function : function object, optional
            Function to reduced the subsampled data
        function_kwargs : dictionary, optional
            Keyword arguments to provide to function
        bundle_args : boolean, optional
            If True, pass all resampled objects to function together, otherwise pass each object through function \
            separately
        replace : boolean, optional
            Whether the sample is with or without replacement
                
        Returns
        -------
        sample : xarray DataArray or Dataset
            Array containing the results of passing the subsampled data through function
    """
    samples_spec = samples.copy() # copy because use pop below
    args_sub = [obj.copy() for obj in args]
    dim_block_1 = [d for d, s in samples_spec.items() if s[1] == 1]

    # Do all dimensions with block_size = 1 together
    samples_block_1 = { dim: samples_spec.pop(dim) for dim in dim_block_1 }
    random_samples = {dim: 
                      np.random.choice(
                          len(args_sub[0][dim]),
                          size=n,
                          replace=replace)
                      for dim, (n, _) in samples_block_1.items()}
    args_sub = [obj.isel(
        {dim: random_samples[dim] 
         for dim in (set(random_samples.keys()) & set(obj.dims))}) for obj in args_sub]

    # Do any remaining dimensions
    for dim, (n, block_size) in samples_spec.items():
        n_blocks = int(n / block_size)
        random_samples = [slice(x,x+block_size) 
                          for x in np.random.choice(
                              len(args_sub[0][dim])-block_size+1, 
                              size=n_blocks,
                              replace=replace)]
        args_sub = [xr.concat([obj.isel({dim: random_sample}) 
                               for random_sample in random_samples],
                              dim=dim) 
                       if dim in obj.dims else obj 
                       for obj in args_sub]

    if function:
        if bundle_args:
            res = function(*args_sub, **function_kwargs)
        else:
            res = tuple([function(obj, **function_kwargs) for obj in args_sub])
    else:
        res = tuple(args_sub,)

    if isinstance(res, tuple) & len(res) == 1:
        return res[0]
    else:
        return res
    
    
def n_random_resamples(*args, samples, n_repeats, 
                       function=None, function_kwargs=None, bundle_args=True, 
                       replace=True, with_dask=True):
    """
        Repeatedly randomly resample from provided xarray objects and return the results of the subsampled dataset passed \
        through a provided function
                
        Parameters
        ----------
        args : xarray DataArray or Dataset
            Objects containing data to be resampled. The coordinates of the first object are used for resampling and the \
            same resampling is applied to all objects
        samples : dictionary
            Dictionary containing the dimensions to subsample, the number of samples and the continuous block size \
            within the sample. Of the form {'dim1': (n_samples, block_size), 'dim2': (n_samples, block_size)}
        n_repeats : int
            Number of times to repeat the resampling process
        function : function object, optional
            Function to reduced the subsampled data
        function_kwargs : dictionary, optional
            Keyword arguments to provide to function
        replace : boolean, optional
            Whether the sample is with or without replacement
        bundle_args : boolean, optional
            If True, pass all resampled objects to function together, otherwise pass each object through function \
            separately
        with_dask : boolean, optional
            If True, use dask to parallelize across n_repeats using dask.delayed
                
        Returns
        -------
        sample : xarray DataArray or Dataset
            Array containing the results of passing the subsampled data through function
    """

    if with_dask & (n_repeats > 500):
        n_args = itertools.repeat(args[0], times=n_repeats)
        b = db.from_sequence(n_args, npartitions=100)
        rs_list = b.map(random_resample, *(args[1:]), 
                        **{'samples':samples, 'function':function, 
                           'function_kwargs':function_kwargs, 'replace':replace}).compute()
    else:              
        resample_ = dask.delayed(random_resample) if with_dask else random_resample
        rs_list = [resample_(*args,
                             samples=samples,
                             function=function,
                             function_kwargs=function_kwargs,
                             bundle_args=bundle_args,
                             replace=replace) for _ in range(n_repeats)] 
        if with_dask:
            rs_list = dask.compute(rs_list)[0]
            
    if all(isinstance(r, tuple) for r in rs_list):
        return tuple([xr.concat([r.unify_chunks() for r in rs], dim='k') for rs in zip(*rs_list)])
    else:
        return xr.concat([r.unify_chunks() for r in rs_list], dim='k')
    
    
def get_significance(sample, bootstrap, no_skill_value, alpha, transform=None):
    """ Return points of statistical significance. Statistical significance at 1-alpha is 
        identified at all points where the sample skill metric is positive (negative) and 
        the fraction of transformed values in the bootstrapped distribution below (above) 
        no_skill_value--defining the p-values--is less than or equal to alpha.)
    """
        
    if transform:
        bootstrap = transform(bootstrap.copy())
        no_skill_value = transform(no_skill_value)
        
    pos_signif = xr.where(bootstrap < no_skill_value, 1, 0).mean('k')  <= alpha
    neg_signif = xr.where(bootstrap > no_skill_value, 1, 0).mean('k')  <= alpha
    pos_signif = pos_signif & (sample > no_skill_value)
    neg_signif = neg_signif & (sample < no_skill_value)
    
    return pos_signif | neg_signif


def Fisher_z(ds):
    """ Return the Fisher-z transformation of ds """
    return np.arctanh(ds)


def get_climatological_probabilities(obsv, period):
    def _init_date_to_ensemble(ds):
        """ Collapse all available init_dates at a given lead into a new ensemble dimension"""
        ds_drop = ds.copy().dropna('init_date', how='all')
        ds_drop = ds_drop.rename({'init_date': 'ensemble'})
        return ds_drop.assign_coords({'ensemble': range(len(ds_drop['ensemble']))})
    obsv_period = mask_time_period(obsv, period=period)
    obsv_ensemble = obsv_period.groupby('lead_time').map(_init_date_to_ensemble)
    clim_prob = obsv_ensemble.broadcast_like(obsv)
    return clim_prob.assign_coords({'time': obsv['time']})


def get_rolling_leadtime_averages(ds, list_of_months_to_average):
    """ Return rolling averages along lead time stacked along new dimension"""
    def _rolling_leadtime_average(ds, n_points, lead_time_name='lead_time'):
        return ds.rolling({lead_time_name: n_points}, center=True).mean()
    dst_dict = {'1 month': ds}
    for avg in list_of_months_to_average:
        dst_dict[str(avg)+' months'] = _rolling_leadtime_average(dst_dict['1 month'], 
                                                                 n_points=avg)
    dst = xr.concat(dst_dict.values(), dim='time_scale')
    dst = dst.assign_coords(time_scale=[int(time_scale.split(' ')[0]) 
                                        for time_scale in dst_dict.keys()])
    dst = dst.assign_coords(time_scale_name=('time_scale', list(dst_dict.keys())))
    return dst


def jelly_plot(skill, stipple=None, stipple_type='//', stipple_color='k',
               title=None, cmap=None, vlims=None, figsize=None):
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.patches as mpatches
    
    def _stipple(stipple, stipple_type):
        if '/' in stipple_type:
            for j, i in np.column_stack(
                np.where(stipple)):
                ax.add_patch(
                    mpatches.Rectangle((i, j-0.5), 1, 1,    
                                       fill=False, linewidth=0,
                                       snap=False, hatch=stipple_type))
        else:
            xOrg = range(len(stipple[stipple.dims[1]]))
            yOrg = range(len(stipple[stipple.dims[0]]))
            nx = len(xOrg)
            ny = len(yOrg)
            xData = np.reshape( np.tile(xOrg, ny), stipple.shape )
            yData = np.reshape( np.repeat(yOrg, nx), stipple.shape )
            sigPoints = stipple > 0
            xPoints = xData[sigPoints.values]
            yPoints = yData[sigPoints.values]
            ax.scatter(xPoints+0.5, yPoints, s=3, 
                       c=stipple_color, marker=stipple_type, alpha=1)
        
    fig, ax = plt.subplots(figsize=figsize)
    
    if vlims is None:
        vlims = [None, None]
               
    x = np.arange(len(skill.lead_time)+1)
    y = np.arange(len(skill.time_scale)+1)-0.5
    c = skill.transpose('time_scale','lead_time')
    im = ax.pcolor(x, y, c,
                    vmin=vlims[0],
                    vmax=vlims[1],
                    cmap=cmap)
    
    if stipple is not None:
        _stipple(stipple.transpose('time_scale','lead_time'), stipple_type)
    
    # Plot annotations
    for y, ts in enumerate(skill.time_scale.values):
        if y != 0:
            ends = 0.15
            line_color = 'grey'
            ax.plot([0.5, ts-0.5], [y, y], 
                    color=line_color, linewidth=1)
            ax.plot([0.5, 0.5], [y-ends, y+ends], 
                    color=line_color, linewidth=1)
            ax.plot([ts-0.5, ts-0.5], [y-ends, y+ends], 
                    color=line_color, linewidth=1)
            ax.plot([int(ts/2)+0.5], y,
                    marker='x', color=line_color, markersize=4, linewidth=1)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='8%', pad=0.55)
    fig.colorbar(im, cax=cax, orientation='horizontal')
    
    ax.set_xticks(range(0,120,6))
    ax.set_yticks(range(len(skill.time_scale)))
    ax.set_yticklabels(skill.time_scale_name.values)
    ax.set_xlabel('Lead month')
    ax.set_ylabel('Averaging period')
    
    if title:
        ax.set_title(title)