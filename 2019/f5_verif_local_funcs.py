import doppyo
import random
import operator
import functools
import geopandas
import regionmask
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
from calendar import monthrange
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

from __main__ import tmp_dir

# Define some analysis parameters -----
test_period = slice('1965','2012')
clim_period = slice('1980','2012')
stacked_init_dates = pd.date_range(start='1960-01-01', 
                                   end='2018-12-01', 
                                   freq='MS')
N_lead_steps = 120

monthly_avgs = [3, 6, 12, 24, 36, 60]
yranges = [(1,1), (2,2), (5,5), (9,9), 
           (1,4), (2,5), (5,8), (7,10), 
           (1,9), (2,10)]

n_bootstraps = 100 # For significance testing
alpha = 0.05 # For significance testing


# skill_cmap = sns.diverging_palette(13, 240, sep=20, n=20)
skill_cmap = sns.diverging_palette(240, 13, sep=20, n=20)
# sns.palplot(skill_cmap)


def add_target_time_coord(ds, init_date_name='init_date', lead_time_name='lead_time', 
                                target_time_name='time', lead_time_freq=None):
    
    if lead_time_freq is None:
            try:
                lead_time_freq = ds[lead_time_name].attrs['units']
            except KeyError:
                raise KeyError('Lead time units are not specified in dataset, please specify manually')

    target_times = xr.DataArray([ds.get_index(init_date_name).shift(int(lead), lead_time_freq) 
                                 for lead in ds[lead_time_name]],
                                dims={lead_time_name: ds[lead_time_name],
                                      init_date_name: ds[init_date_name]})
    ds[lead_time_name].attrs['units'] = lead_time_freq
    
    return ds.assign_coords({target_time_name: target_times})


def similarize(ds, 
               transform_names=None, scale_by=None, shift_by=None,
               time_dim_freq=None, time_name='time', 
               add_target_times=False, lead_time_name='lead_time', lead_time_freq=None, target_time_name='time',
               fix_longitudes=False):
    """ 
        Make the various datasets look similar
        Assumes input data time axis has been decoded using cftime
    """
        
    if scale_by is not None:
        for variable, factor in scale_by.items():
            ds[variable] = factor * ds[variable]
            
    if shift_by is not None:
        for variable, addition in shift_by.items():
            ds[variable] = ds[variable] + addition
            
    if transform_names is not None:
        ds = ds.rename(transform_names)
            
    if time_dim_freq == 'MS':
        ds = ds.assign_coords({time_name: 
                               np.array([time.replace(day=1, hour=0, minute=0, second=0, microsecond=0) 
                                         for time in ds[time_name].values])})
    elif time_dim_freq == 'M':
        ds = ds.assign_coords({time_name: 
                               np.array([time.replace(day=calendar.monthrange(time.year,
                                                                              time.month)[1], 
                                                      hour=0, minute=0, second=0, microsecond=0) 
                                         for time in ds[time_name].values])})
    elif time_dim_freq == 'D':
        ds = ds.assign_coords({time_name: 
                               np.array([time.replace(hour=0, minute=0, second=0, microsecond=0) 
                                         for time in ds[time_name].values])})
    elif time_dim_freq == 'h':
        ds = ds.assign_coords({time_name: 
                               np.array([time.replace(minute=0, second=0, microsecond=0) 
                                         for time in ds[time_name].values])})
    elif time_dim_freq:
        raise ValueError('Unrecognised/unsuported input for "time_dim_freq"')
        
    if add_target_times:
        ds = add_target_time_coord(ds, init_date_name=time_name, lead_time_name=lead_time_name, 
                                   target_time_name=target_time_name, lead_time_freq=lead_time_freq)
            
    if fix_longitudes:
        lon_names = doppyo.utils.get_lon_name(ds)
        if isinstance(lon_names, list):
            for lon_name in lon_names:
                ds = ds.assign_coords({lon_name: (ds[lon_name] + 360)  % 360}).sortby(lon_name)
        else:
            lon_name = lon_names
            ds = ds.assign_coords({lon_name: (ds[lon_name] + 360)  % 360}).sortby(lon_name)

    return ds


def as_datetime64(ds, time_dim='time', fmt='%Y-%m-%d'):
    return ds.assign_coords({time_dim:
                            [np.datetime64(time.strftime(fmt)) for time in ds[time_dim].values]})


def get_month_anomalies(full_field, clim_fields, save_name, compute=False, save_chunks=False):
    
    variable = full_field.name
    
    if compute: 
        anom = []
        for clim_name, clim_field in clim_fields.items():
            anom.append(month_anomaly(full_field, clim_field).rename(clim_name))
        anoms = xr.merge(anom)
        
        # Add full_field attrs -----
        for coord in full_field.coords:
            anoms[coord].attrs = full_field[coord].attrs
        
        if save_chunks:
            anoms = anoms.chunk(save_chunks)
        anoms.to_zarr(tmp_dir + save_name + '.zarr', mode='w', consolidated=True) 
        
    return xr.open_zarr(tmp_dir + save_name + '.zarr', consolidated=True)


def month_climatology(ds, time_name='time'):
    return ds.groupby(time_name+'.month', squeeze=False).mean(dim=time_name)


def month_anomaly(ds, ds_clim):
    if 'lead_time' in ds_clim.dims: 
        return (ds.groupby('init_date.month') - ds_clim).drop('month')
    else:
        return (ds.groupby('time.month') - ds_clim).drop('month')
    
    
def month_std_ratio(ds_cmp_anom, ds_ref_anom, time_name='time'):
    ds_cmp_std = ds_cmp_anom.groupby(time_name+'.month', squeeze=False).std(dim=time_name)
    ds_ref_std = ds_ref_anom.groupby(time_name+'.month', squeeze=False).std(dim=time_name)
    
    return ds_ref_std / ds_cmp_std


def month_scale_std(ds_anom, std_ratio):
    if 'init_date' in ds_anom.dims: 
        return (ds_anom.groupby('init_date.month') * std_ratio).drop('month')
    else:
        return (ds_anom.groupby('time.month') * std_ratio).drop('month')
    
    
def monthly_resample(da):
    attrs = da.attrs
    da = da.resample(time='MS').sum('time', keep_attrs=True) 
    da.attrs = attrs

    return da


def get_diagnostic(diagnostic_function, args, kwargs, save_name, save_type='zarr', mean_and_std_only=False, 
                   compute=False, convert_to_dataset=False, save_chunks=False):
    """ 
        If mean_and_std_only = True, returns two outputs (the mean and std)
    """
    if compute:
        if mean_and_std_only:
            diagnostic = diagnostic_function(args, **kwargs) if kwargs else diagnostic_function(args)
            diagnostic_ensmean = diagnostic.mean('ensemble')
            if save_type == 'zarr':
                if save_chunks:
                    diagnostic_ensmean = diagnostic_ensmean.chunk(save_chunks)
                if convert_to_dataset:
                    diagnostic_ensmean.to_dataset(name=convert_to_dataset).to_zarr(tmp_dir + save_name + '_ensmean.zarr', 
                                                                                   mode='w', 
                                                                                   consolidated=True)
                else:
                    diagnostic_ensmean.to_zarr(tmp_dir + save_name + '_ensmean.zarr', mode='w', consolidated=True) 
            elif save_type == 'nc':
                if convert_to_dataset:
                    diagnostic_ensmean.to_dataset(name=convert_to_dataset).to_netcdf(tmp_dir + save_name + '_ensmean.nc') 
                else:
                    diagnostic_ensmean.to_netcdf(tmp_dir + save_name + '_ensmean.nc') 
            else:
                raise ValueError('Unrecognised file type')
                
            diagnostic_ensstd = diagnostic.std('ensemble')
            if save_type == 'zarr':
                if save_chunks:
                    diagnostic_ensstd = diagnostic_ensstd.chunk(save_chunks) 
                if convert_to_dataset:
                    diagnostic_ensstd.to_dataset(name=convert_to_dataset).to_zarr(tmp_dir + save_name+ '_ensstd.zarr', 
                                                                                  mode='w', 
                                                                                  consolidated=True)
                else:
                    diagnostic_ensstd.to_zarr(tmp_dir + save_name+ '_ensstd.zarr', mode='w', consolidated=True)
            elif save_type == 'nc':
                if convert_to_dataset:
                    diagnostic_ensstd.to_dataset(name=convert_to_dataset).to_netcdf(tmp_dir + save_name+ '_ensstd.nc')
                else:
                    diagnostic_ensstd.to_netcdf(tmp_dir + save_name+ '_ensstd.nc')
            else:
                raise ValueError('Unrecognised file type')
        else:
            diagnostic = diagnostic_function(args, **kwargs) if kwargs else diagnostic_function(args)
            if save_type == 'zarr':
                if save_chunks:
                    diagnostic = diagnostic.chunk(save_chunks)
                if convert_to_dataset:
                    diagnostic.to_dataset(name=convert_to_dataset).to_zarr(tmp_dir + save_name + '.zarr', 
                                                                           mode='w', 
                                                                           consolidated=True) 
                else:
                    diagnostic.to_zarr(tmp_dir + save_name + '.zarr', mode='w', consolidated=True) 
            elif save_type == 'nc':
                if convert_to_dataset:
                    diagnostic.to_dataset(name=convert_to_dataset).to_netcdf(tmp_dir + save_name + '.nc')
                else:
                    diagnostic.to_netcdf(tmp_dir + save_name + '.nc')
            else:
                raise ValueError('Unrecognised file type')
   
    if mean_and_std_only:
        if save_type == 'zarr':
            return xr.open_zarr(tmp_dir + save_name + '_ensmean.zarr', consolidated=True), \
                   xr.open_zarr(tmp_dir + save_name + '_ensstd.zarr', consolidated=True)
        elif save_type == 'nc':
            return xr.open_dataset(tmp_dir + save_name + '_ensmean.nc', chunks=save_chunks), \
                   xr.open_dataset(tmp_dir + save_name + '_ensstd.nc', chunks=save_chunks)
        else:
            raise ValueError('Unrecognised file type')
    else:
        if save_type == 'zarr':
            return xr.open_zarr(tmp_dir + save_name + '.zarr', consolidated=True)
        elif save_type == 'nc':
            return xr.open_dataset(tmp_dir + save_name + '.nc', chunks=save_chunks)
        else:
            raise ValueError('Unrecognised file type')
            
            
# def _resample_axis(da, n_points):
#     """
#         Aggregate and resample groups of n_points along axis=-1
#     """
#     leads = np.arange(da.shape[-1])
#     da_clipped = da[...,:len(leads)-(len(leads)%n_points)]
    
#     da_resampled = np.mean(da_clipped.reshape(da.shape[:-1] + (-1,n_points)), axis=-1)
#     # da_replicated = np.repeat(da_resampled, n_points, axis=-1)
    
#     return da_resampled


# def resample_lead_time(ds, n_points, lead_time_name='lead_time'):
    
#     labels = np.arange((n_points-1)/2, 
#                        len(ds[lead_time_name])-(len(ds[lead_time_name])%n_points), 
#                        n_points)
    
#     ds_resampled =  xr.apply_ufunc(_resample_axis,
#                                    ds.chunk({lead_time_name:-1}), n_points,
#                                    input_core_dims=[[lead_time_name],[]],
#                                    exclude_dims=set((lead_time_name,)),
#                                    output_core_dims=[[lead_time_name]],
#                                    dask='allowed').assign_coords({lead_time_name: labels})
    
#     ds_resampled[lead_time_name].attrs['units'] = str(n_points) + ds[lead_time_name].attrs['units']
    
#     if len(ds_resampled[lead_time_name]) == 1:
#         return ds_resampled.squeeze() + (0*ds)
#     else:
#         return ds_resampled.chunk({lead_time_name: -1}) \
#                            .interp({lead_time_name: np.arange(0,len(ds[lead_time_name]))}, 
#                                     method='nearest', 
#                                     kwargs={'fill_value': 'extrapolate'})
    
    
# def resample_lead_time(ds, n_points, lead_time_name='lead_time'):
    
#     return ds.rolling({lead_time_name: n_points}, center=True).mean() #.shift({'lead_time': -int(n_points/2)})
    
    
# def resample_lead_time(ds, n_points, lead_time_name='lead_time'):
#     """
#         Resample along lead time dimension
#     """
#     bins = np.arange(ds[lead_time_name].min(), 
#                      ds[lead_time_name].max()+2, 
#                      n_points).astype(int)
#     labels = np.arange((n_points-1)/2, 
#                        ds[lead_time_name].max()+2, 
#                        n_points)[:len(bins)-1]
    
#     if bins[-1] != 120:
#         bins = np.append(bins, 120)
#         labels = np.append(labels, bins[-2] + (bins[-2] - labels[-1]) - 0.5)

#     ds_resampled = ds.groupby_bins(lead_time_name, 
#                                    bins, 
#                                    right=False, 
#                                    labels=labels,
#                                    include_lowest=True) \
#                      .mean(lead_time_name) \
#                      .rename({lead_time_name+'_bins': lead_time_name})
#     ds_resampled[lead_time_name].attrs['units'] = str(n_points) + ds[lead_time_name].attrs['units']
    
#     if len(ds_resampled[lead_time_name]) == 1:
#         return ds_resampled.squeeze() + (0*ds)
#     else:
#         return ds_resampled.chunk({lead_time_name: -1}) \
#                            .interp({lead_time_name: np.arange(0,len(ds[lead_time_name]))}, 
#                                     method='nearest', 
#                                     kwargs={'fill_value': 'extrapolate'})
    

def define_spatial_masks(da):
    """ Return masks on input grid. Add points/regions here to add to analysis """
    
    # Cells to mask -----
    grid_cells = {'Melbourne grid box': point_mask(da, (144.96, -37.81)),
                  'Tasmania grid box':  point_mask(da, (146.5,  -42)),
                  'Charters Towers grid box':  point_mask(da, (146.2614,  -20.0781)),
                  'Georgetown grid box':  point_mask(da, (143.5483,  -18.2922)),
                  'Mataranka grid box':  point_mask(da, (133.1320,  -14.9230)),
                  'Avon Downs grid box':  point_mask(da, (137.4907,  -20.0297)),
                  'Fitzroy Crossing grid box':  point_mask(da, (125.5644,  -18.1919))}
 
    # Regions -----
    NRM = geopandas.read_file("NRM_clusters/NRM_clusters.shp")
    NRM_regions = {number: name for number, name in zip(list(NRM.OBJECTID), list(NRM.label))}
    NRM_mask = regionmask.Regions_cls(name='NRM_regions', 
                                      numbers=list(NRM.OBJECTID), 
                                      names=list(NRM.label), 
                                      abbrevs=list(NRM.code), 
                                      outlines=list(NRM.geometry))
    NRM_mask = NRM_mask.mask(da, lon_name=doppyo.utils.get_lon_name(da), 
                             lat_name=doppyo.utils.get_lat_name(da))
    
    NRM_sub = geopandas.read_file("NRM_sub_clusters/NRM_sub_clusters.shp")
    NRM_sub_regions = {number: name for number, name in zip(list(NRM_sub.OBJECTID), list(NRM_sub.label))}
    NRM_sub_mask = regionmask.Regions_cls(name='NRM_sub_regions', 
                                          numbers=list(NRM_sub.index), 
                                          names=list(NRM_sub.label), 
                                          abbrevs=list(NRM_sub.code), 
                                          outlines=list(NRM_sub.geometry))
    NRM_sub_mask = NRM_sub_mask.mask(da, lon_name=doppyo.utils.get_lon_name(da), 
                                     lat_name=doppyo.utils.get_lat_name(da))

    region_masks = {'South-East Australia NRM region': xr.where((NRM_mask == 4) | (NRM_mask == 7), 1, np.nan),
                    'Australia NRM region':            xr.where(NRM_mask.notnull(), 1, np.nan),
                    'Western Tasmania NRM region':     xr.where(NRM_sub_mask == 12, 1, np.nan),
                    'Eastern Tasmania NRM region':     xr.where(NRM_sub_mask == 11, 1, np.nan),}
    
    return {**grid_cells, **region_masks}


def get_rolling_leadtime_averages(ds, list_of_months_to_average):
    
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
    
    return dst.chunk({'lead_time':1})


def get_yrange_leadtime_averages(ds, list_of_ranges_to_average):
    """
        Ranges are yearly and in the format, e.g., (2,5) for the 2-5 year average
        Single year averages can be specified, e.g., (1,1)
    """
    
    def _yrange_leadtime_average(ds, span, lead_time_name='lead_time'):
        if 'M' in ds[lead_time_name].units:
            start = (span[0]-1)*12
            end = (span[1]*12)-1
            if (start in ds.lead_time) & (end in ds.lead_time):
                return ds.sel({lead_time_name: slice(start, end)}).mean(lead_time_name)
            else:
                return np.nan * ds.isel({lead_time_name: 0}, drop=True)
        else:
            raise ValueError('get_yrange_leadtime_average() not yet set up to do lead time'+\
                             'frequencies other than monthly')
    
    ds = ds.chunk({'lead_time': 1})
    
    dst_dict = {}
    for span in list_of_ranges_to_average:
        dst_dict['year '+str(span[0]) if span[0] == span[1] else
                 'years '+str(span[0])+'-'+str(span[1])] = _yrange_leadtime_average(ds,
                                                                                    span)
    dst = xr.concat(dst_dict.values(), dim='time_scale')
    dst = dst.assign_coords(time_scale=range(len(dst.time_scale)))
    dst = dst.assign_coords(time_scale_name=('time_scale', list(dst_dict.keys())))
    
    return dst


def regrid(da, ref_grid):
    
    da_lat_name = doppyo.utils.get_lat_name(da)
    da_lon_name = doppyo.utils.get_lon_name(da)
    ref_grid_lat_name = doppyo.utils.get_lat_name(ref_grid)
    ref_grid_lon_name = doppyo.utils.get_lon_name(ref_grid)
    
    da_regrid = da.interp({da_lat_name: ref_grid[ref_grid_lat_name],
                           da_lon_name: ref_grid[ref_grid_lon_name]}) \
                  .drop([da_lat_name, da_lon_name])
    
    return da_regrid


def spatial_avg(ds, mask):
    lon_name = doppyo.utils.get_lon_name(ds)
    lat_name = doppyo.utils.get_lat_name(ds)

#     ds.where(mask.notnull())['unfair bias corrected'].isel(time=0).plot()
#     plt.xlim([112, 155])
#     plt.ylim([-46, -9])
#     plt.show()
    
    return ds.where(mask.notnull()).mean([lon_name, lat_name])


def get_regions(ds, dict_of_spatial_masks=None):
    if dict_of_spatial_masks is None:
        dict_of_spatial_masks = define_spatial_masks(ds)
    
    dss_dict = {spatial_scale: spatial_avg(ds, mask) for spatial_scale, mask in dict_of_spatial_masks.items()}
    dss = xr.concat(dss_dict.values(), dim='spatial_scale').compute()
    dss = dss.assign_coords(spatial_scale=range(len(dss_dict.keys())))
    dss = dss.assign_coords(spatial_scale_name=('spatial_scale', list(dss_dict.keys())))
    
    return dss


def regrid_and_get_regions(da, ref_grid):
    
    return get_regions(regrid(da, ref_grid))


def prepare_obsv(obsv_full, hcst_name='', 
                 diagnostic_function=None, diagnostic_kwargs=None, 
                 leadtime_resample_function=None, leadtime_resample_kwargs=None,
                 anomalize=None, compute=False):
    """ anomalize = 'first', 'last' or None """
    
    if anomalize == 'first':
        obsv_clim = month_climatology(obsv_full.sel({'time' : clim_period}))
        obsv1 = get_month_anomalies(
            full_field=obsv_full,
            clim_fields={'biased': obsv_clim},
            save_name=hcst_name+'_obsv1',
            compute=compute) #.compute()
        obsv1['unfair bias corrected'] = obsv1['biased']
        obsv1['unfair bias and std corrected'] = obsv1['biased']
    else: 
        obsv_clim = None
        obsv1 = obsv_full
        
    # Get diagnostic -----
    if diagnostic_function is not None:
        name = diagnostic_function.__name__
        obsv2 = get_diagnostic(
            diagnostic_function=diagnostic_function,
            args=obsv1,
            kwargs=diagnostic_kwargs,
            save_name=hcst_name+'_obsv2',
            save_type='zarr',
            compute=compute,
            convert_to_dataset=name if anomalize != 'first' else None,
            save_chunks={'time': -1}) #.compute()
        obsv2 = obsv2[name] if anomalize != 'first' else obsv2
    else: 
        obsv2 = obsv1
        
    if anomalize == 'last':
        obsv_clim = month_climatology(obsv2.sel({'time' : clim_period}))
        obsv3 = get_month_anomalies(
            full_field=obsv2,
            clim_fields={'biased': obsv_clim},
            save_name=hcst_name+'_obsv3',
            compute=compute) #.compute()
        obsv3['unfair bias corrected'] = obsv3['biased']
        obsv3['unfair bias and std corrected'] = obsv3['biased']
    else: 
        obsv3 = obsv2
    
    # Stack into initial dat - lead time format ----
    if obsv3.nbytes < 0.05e9:
        obsv3_stacked = doppyo.utils.stack_by_init_date(obsv3.compute(),
                                                        init_dates=stacked_init_dates,
                                                        N_lead_steps=120).sel({'init_date': test_period})
    else:
        name = diagnostic_function.__name__
        obsv3_stacked = get_diagnostic(
            diagnostic_function=doppyo.utils.stack_by_init_date,
            args=obsv3,
            kwargs={'init_dates': stacked_init_dates,
                    'N_lead_steps':120},
            save_name=hcst_name+'_obsv3_stacked',
            convert_to_dataset=name if anomalize != 'last' else None,
            compute=compute).sel({'init_date': test_period})
        obsv3_stacked = obsv3_stacked[name] if anomalize != 'last' else obsv3_stacked
    
    # Resample lead time dimension -----
    if leadtime_resample_function is not None:
        name = leadtime_resample_function.__name__
        if obsv3_stacked.nbytes < 0.5e9:
            obsv = get_diagnostic(
                diagnostic_function=leadtime_resample_function,
                args=obsv3_stacked.compute(),
                kwargs=leadtime_resample_kwargs,
                save_name=hcst_name+'_obsv',
                convert_to_dataset=name if anomalize != 'last' else None,
                compute=compute).compute()
        else:
            obsv = get_diagnostic(
                diagnostic_function=leadtime_resample_function,
                args=obsv3_stacked,
                kwargs=leadtime_resample_kwargs,
                save_name=hcst_name+'_obsv',
                convert_to_dataset=name if anomalize != 'last' else None,
                compute=compute)
        obsv = obsv[name] if anomalize != 'last' else obsv
    else: 
        obsv = obsv3_stacked
    
    if obsv.nbytes < 0.1e9:
        return obsv.compute(), obsv_clim
    else:
        return obsv, obsv_clim 


def prepare_hcst(hcst_full, hcst_name, obsv_clim=None, obsv=None, 
                 diagnostic_function=None, diagnostic_kwargs=None, 
                 leadtime_resample_function=None, leadtime_resample_kwargs=None,
                 anomalize=None, compute=False):
    
    hcst_lat_name = doppyo.utils.get_lat_name(hcst_full)
    hcst_lon_name = doppyo.utils.get_lon_name(hcst_full)
        
    if anomalize == 'first':
        if obsv_clim is not None:
            obsv_lat_name = doppyo.utils.get_lat_name(obsv_clim)
            obsv_lon_name = doppyo.utils.get_lon_name(obsv_clim)
            hcst1 = get_month_anomalies(
                full_field=hcst_full,
                clim_fields={'biased':                obsv_clim.interp({obsv_lat_name: hcst_full[hcst_lat_name],
                                                                        obsv_lon_name: hcst_full[hcst_lon_name]}) \
                                                               .drop([obsv_lat_name, obsv_lon_name]),
                             'unfair bias corrected': month_climatology(hcst_full.mean('ensemble') \
                                                                                    .sel({'init_date' : clim_period}),
                                                                           time_name='init_date')},
                save_name=hcst_name+'_hcst1',
                compute=compute).sel({'init_date': test_period}).drop('time')
        else:
            hcst1 = get_month_anomalies(
                full_field=hcst_full,
                clim_fields={'unfair bias corrected': month_climatology(hcst_full.mean('ensemble') \
                                                                                 .sel({'init_date' : clim_period}),
                                                                           time_name='init_date')},
                save_name=hcst_name+'_hcst1',
                compute=compute).sel({'init_date': test_period}).drop('time')
    else: hcst1 = hcst_full
    
    # Get diagnostic -----
    if diagnostic_function is not None:
        name = diagnostic_function.__name__
        hcst2 = get_diagnostic(
            diagnostic_function=diagnostic_function,
            args=hcst1,
            kwargs=diagnostic_kwargs,
            save_name=hcst_name+'_hcst2',
            save_type='zarr',
            compute=compute,
            convert_to_dataset=name if anomalize != 'first' else None,
            save_chunks={'init_date':-1, 'lead_time':-1, 'ensemble':-1}) #.compute()
        hcst2 = hcst2[name] if anomalize != 'first' else hcst2
    else: 
        hcst2 = hcst1
    
    if anomalize == 'last':
        if obsv_clim is not None:
            try:
                obsv_lat_name = doppyo.utils.get_lat_name(obsv_clim)
                obsv_clim = obsv_clim.interp({obsv_lat_name: hcst_full[hcst_lat_name]}).drop(obsv_lat_name)
            except KeyError:
                pass
            try:
                obsv_lon_name = doppyo.utils.get_lon_name(obsv_clim)
                obsv_clim = obsv_clim.interp({obsv_lon_name: hcst_full[hcst_lon_name]}).drop(obsv_lon_name)
            except KeyError:
                pass

            hcst3 = get_month_anomalies(
                full_field=hcst2,
                clim_fields={'biased':                obsv_clim,
                             'unfair bias corrected': month_climatology(hcst2.mean('ensemble') \
                                                                             .sel({'init_date' : clim_period}),
                                                                        time_name='init_date')},
                save_name=hcst_name+'_hcst3',
                compute=compute).sel({'init_date': test_period}).drop('time')
        else:
            hcst3 = get_month_anomalies(
                full_field=hcst2,
                clim_fields={'unfair bias corrected': month_climatology(hcst2.mean('ensemble') \
                                                                             .sel({'init_date' : clim_period}),
                                                                        time_name='init_date')},
                save_name=hcst_name+'_hcst3',
                compute=compute).sel({'init_date': test_period}).drop('time')
    else: hcst3 = hcst2
    
    # Resample lead time dimension -----
    if leadtime_resample_function is not None:
        name = leadtime_resample_function.__name__
        if hcst3.nbytes < 0.5e9:
            hcst = get_diagnostic(
                diagnostic_function=leadtime_resample_function,
                args=hcst3.compute(),
                kwargs=leadtime_resample_kwargs,
                save_name=hcst_name+'_hcst',
                save_chunks={'init_date':1},
                convert_to_dataset=name if anomalize != 'last' else None,
                compute=compute).compute()
        else:
            hcst = get_diagnostic(
                diagnostic_function=leadtime_resample_function,
                args=hcst3,
                kwargs=leadtime_resample_kwargs,
                save_name=hcst_name+'_hcst',
                save_chunks={'init_date':1},
                convert_to_dataset=name if anomalize != 'last' else None,
                compute=compute)
        hcst = hcst[name] if anomalize != 'last' else hcst
    else:
        hcst = hcst3
    
    if anomalize:
        if obsv is not None:
            hcst['unfair bias and std corrected'] = month_scale_std(
                hcst['unfair bias corrected'],
                month_std_ratio(hcst['unfair bias corrected'],
                                obsv['unfair bias corrected'],
                                time_name='init_date'))
    
    if hcst.nbytes < 0.1e9:
        return hcst.compute()
    else:
        return hcst


def get_Pearson_corrcoeff(da_cmp, da_ref, over_dims):
    
    da_cmp = da_cmp.mean('ensemble')
    
    return doppyo.skill.Pearson_corrcoeff(da_cmp, da_ref, over_dims)


def get_RMSE(da_cmp, da_ref, over_dims):
    
    da_cmp = da_cmp.mean('ensemble')
    da_cmp_overlap = da_cmp.where(xr.ufuncs.isfinite(da_ref))
    da_ref_overlap = da_ref.where(xr.ufuncs.isfinite(da_cmp))

    return doppyo.skill.rms_error(da_cmp_overlap, da_ref_overlap, over_dims=over_dims)


def get_MSSS(da_cmp, da_ref, over_dims, reference_MSE=None):
    
    da_cmp = da_cmp.mean('ensemble')
    da_cmp_overlap = da_cmp.where(xr.ufuncs.isfinite(da_ref))
    da_ref_overlap = da_ref.where(xr.ufuncs.isfinite(da_cmp))
    
    MSE = doppyo.skill.mean_squared_error(da_cmp_overlap, da_ref_overlap, over_dims=over_dims)
    
#     xr.ufuncs.square(da_cmp_overlap).isel(spatial_scale=0, time_scale=0)['unfair bias corrected'].plot()
#     plt.show()
#     xr.ufuncs.square(da_ref_overlap).isel(spatial_scale=0, time_scale=0)['unfair bias corrected'].plot()
#     plt.show()
    
    if reference_MSE is None:
        reference_MSE = xr.ufuncs.square(da_ref_overlap).mean(over_dims)
    
    return 1 - ( MSE / reference_MSE )


def get_amplitude_bias(da_cmp, da_ref, over_dims):
    
    da_cmp = da_cmp.mean('ensemble')
    da_cmp_overlap = da_cmp.where(xr.ufuncs.isfinite(da_ref))
    da_ref_overlap = da_ref.where(xr.ufuncs.isfinite(da_cmp))
    
    cor = doppyo.skill.Pearson_corrcoeff(da_cmp_overlap, da_ref_overlap, over_dims=over_dims)
    stds = da_cmp_overlap.std(over_dims) / da_ref_overlap.std(over_dims)
    
    return cor - stds


def get_mean_bias(da_cmp, da_ref, over_dims):
    
    da_cmp = da_cmp.mean('ensemble')
    da_cmp_overlap = da_cmp.where(xr.ufuncs.isfinite(da_ref))
    da_ref_overlap = da_ref.where(xr.ufuncs.isfinite(da_cmp))
    
    return (da_cmp_overlap.mean(over_dims) - 
            da_ref_overlap.mean(over_dims)) / da_ref_overlap.std(over_dims)


def get_deterministic_ROC_score(ds_cmp, ds_ref, over_dims):
    
    ds_cmp, ds_ref = xr.align(ds_cmp, ds_ref, join='left')
    
    
    contingency = doppyo.skill.contingency_table(ds_cmp, ds_ref,
                                                 [-np.inf, 0.5, np.inf],
                                                 [-np.inf, 0.5, np.inf],
                                                 over_dims)
    return xr.merge([doppyo.skill.false_alarm_rate(contingency).to_dataset(name='false_alarm_rate'),
                     doppyo.skill.hit_rate(contingency).to_dataset(name='hit_rate')])


def get_ROC_score(cmp_events, ref_events, over_dims, probability_bin_edges):
    
    cmp_likelihood, ref_logical = xr.align(cmp_events.mean('ensemble'), 
                                           ref_events, 
                                           join='left')
    
    return doppyo.skill.roc(cmp_likelihood, ref_logical, 
                            over_dims=over_dims, probability_bin_edges=probability_bin_edges)


def get_Gerrity_score(da_cmp, da_ref, category_edges_cmp, category_edges_ref, over_dims):
    contingency = doppyo.skill.contingency_table(da_cmp.mean('ensemble'),
                                                 da_ref,
                                                 category_edges_cmp,
                                                 category_edges_ref,
                                                 over_dims)
    return doppyo.skill.Gerrity_score(contingency).to_dataset()


def restack_by_region(skill_metric, regions):
    return xr.merge([skill_metric.sel(spatial_scale=s) \
                                 .to_dataset(name=skill_metric.spatial_scale_name.sel(spatial_scale=s).item()) \
                                 .drop(['spatial_scale_name', 'spatial_scale'])
                     for s in skill_metric.spatial_scale 
                     if skill_metric.spatial_scale_name.sel(spatial_scale=s).item() in regions])


def plot_skill_score_avgs(skill_list, title, headings=None, stipple=None, 
                          vlims=(-1,1), cmap=ListedColormap(skill_cmap), xlims=None, ylims=None, 
                          group_rows_by=1):
    fig = plt.figure()
    nrow = len(skill_list)
    ncol = len(skill_list[0])
    gridspec_outers = gridspec.GridSpec(int(nrow/group_rows_by), 1, hspace = 0.3 
                                        if group_rows_by == 1 else 0.15) 
    SubplotSpecs = [gridspec.GridSpecFromSubplotSpec(group_rows_by, ncol, 
                                                     subplot_spec=gridspec_outers, 
                                                     hspace=0) 
                    for gridspec_outers in gridspec_outers]
    axes = [plt.subplot(cell) for SubplotSpec in SubplotSpecs for cell in SubplotSpec]
    
    ax_count = 0
    for idy, skill in enumerate(skill_list):
        if stipple:
            stippled = stipple[idy]
        else:
            stippled = None
        for idx, variable in enumerate(skill):
            if hasattr(axes, '__getitem__'):
                axc = axes[ax_count]
            else:
                axc = axes

            norm = None
            x = np.arange(len(skill[variable].lead_time)+1)
            y = np.arange(len(skill[variable].time_scale)+1)-0.5
            c = skill[variable].transpose('time_scale','lead_time')
            im = axc.pcolor(x, y, c,
                            norm=norm,
                            vmin=vlims[0],
                            vmax=vlims[1],
                            cmap=cmap)

#             x_nogo = [0]
#             y_nogo = [-0.5]
#             for yi, xi in enumerate(skill[variable].time_scale.values):
#                 x_nogo.extend([xi, xi])
#                 y_nogo.extend([yi-0.5, yi+0.5])
#             x_nogo.append(0)
#             y_nogo.append(yi+0.5)
#             axc.plot(x_nogo, y_nogo, color='w', linewidth=2)
#             axc.plot(x_nogo, y_nogo, color='grey', linewidth=1)
            
#             time_scales = skill[variable].time_scale.values
#             for y in np.arange(len(time_scales),0,-1):
#                 xs = [0, 1, 1]
#                 ys = [y-0.5, y-0.5, y-1.5]
#                 ts = time_scales[y-1]
#                 for y2 in np.arange(y, 0, -1):
#                     xs.extend([ts-time_scales[y2-1]+1, ts-time_scales[y2-1]+1])
#                     ys.extend([y2-0.5, y2-1.5])
#                 axc.plot(xs, ys, color='w', linewidth=2)
#                 axc.plot(xs, ys, color='grey', linewidth=1)
                
            time_scales = skill[variable].time_scale.values
            for y, ts in enumerate(time_scales):
                if y != 0:
                    ends = 0.1
#                     axc.plot([0.5, ts-0.5], [y, y], color='w', linewidth=2)
#                     axc.plot([0.5, 0.5], [y-ends, y+ends], color='w', linewidth=2)
#                     axc.plot([ts-0.5, ts-0.5], [y-ends, y+ends], color='w', linewidth=2)
#                     axc.plot([int(ts/2)+0.5], y, marker='x', color='w', markersize=4, linewidth=2)

                    axc.plot([0.5, ts-0.5], [y, y], color='grey', linewidth=1)
                    axc.plot([0.5, 0.5], [y-ends, y+ends], color='grey', linewidth=1)
                    axc.plot([ts-0.5, ts-0.5], [y-ends, y+ends], color='grey', linewidth=1)
                    axc.plot([int(ts/2)+0.5], y, marker='x', color='grey', markersize=4, linewidth=1)


            if stippled:
                #stippled = xr.where(skill[variable] > stipple, 1, 0).transpose('time_scale','lead_time')
                for j, i in np.column_stack(np.where(stippled[variable].transpose('time_scale','lead_time'))):
                      axc.add_patch(
                          mpatches.Rectangle((i, j-0.5),
                                             1,         
                                             1,    
                                             fill=False, 
                                             linewidth=0,
                                             snap=False,
                                             hatch='//'))

            if headings:
                if (idy == 0) & (ncol != 1):
                    axc.set_title(variable)

            axc.set_ylabel('')
            axc.set_yticks(range(len(skill['time_scale'])))
            axc.set_yticklabels(skill['time_scale_name'].values)
            if ncol == 1:
                axc.yaxis.tick_left()
                axc.set_ylabel('Averaging \nperiod')
            elif (ax_count+1) % ncol == 0:
                axc.yaxis.tick_right()
            elif (ax_count+ncol) % ncol == 0: 
                axc.yaxis.tick_left()
                axc.set_ylabel('Averaging \nperiod')
            else:
                axc.set_yticklabels('')
                
            if ylims:
                axc.set_ylim(ylims)

            axc.set_xticks(range(0,120,12))
            if ax_count / ncol < nrow - 1:
                axc.set_xticklabels('')
                axc.set_xlabel('')
            else:
                axc.set_xlabel('Lead time') # [months]')
            if xlims:
                axc.set_xlim(xlims)

            ax_count += 1
            
    if headings:
        axz = axes[::ncol]
#         if axes.ndim == 2:
#             axz = axes[:,0]
#         else:
#             axz = axes[:ncol:]
        for ax, row in zip(axz, headings):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        ha='right', va='center')
                        
    vmin, vmax = vlims
    fig.subplots_adjust(bottom=0.4*np.sqrt(1/nrow))#0.001*((13-nrow)**(1/2)))
    cbar_ax = fig.add_axes([0.16, 0.01*(nrow), 0.7, 0.06*(1/nrow)])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal');
    # cbar_ax.set_xlabel('skill', rotation=0, labelpad=15);
    if norm is None:
        cbar.set_ticks(np.linspace(vmin,vmax,5))
    else:
        cbar.set_ticks(np.unique(np.concatenate((np.linspace(vmin,0,3),
                                                 np.linspace(0,vmax,3)))))
        
    fig.suptitle(title, y=1.1-0.02*nrow)
    
    plt.subplots_adjust(wspace=0.04)
    
    
def plot_skill_score_ts(skill_list, conf_list, title, headings=None, stipple_above=None, 
                        vlims=(-1,1), cmap=ListedColormap(skill_cmap), xlims=None, ylims=None):
    fig = plt.figure()
    nrow = len(skill_list)
    ncol = len(skill_list[0])
    axes = fig.subplots(nrows=nrow, ncols=ncol, sharex=False, sharey=False)

    ax_count = 0
    for idy, skill in enumerate(skill_list):
        for idx, variable in enumerate(skill):
            if hasattr(axes, '__getitem__'):
                if axes.ndim == 2:
                    axc = axes[idy, idx]
                else:
                    axc = axes[idx]
            else:
                axc = axes

            axc.fill_between(conf_list[idy][0][variable].lead_time, 
                             0,
                             100, color=skill_cmap[-1], alpha=0.2)
            axc.fill_between(conf_list[idy][0][variable].lead_time, 
                             -100,
                             0, color=skill_cmap[0], alpha=0.2)
            axc.fill_between(conf_list[idy][0][variable].lead_time, 
                             conf_list[idy][0][variable],
                             conf_list[idy][1][variable], color='k', alpha=0.2)
            
            x = skill[variable].lead_time
            y = skill[variable]
            if np.count_nonzero(~np.isnan(y)) == 1:
                axc.plot(x, y, marker='o', color='k')
            else:
                axc.plot(x, y, color='k')
#             points = np.array([x, y]).T.reshape(-1, 1, 2)
#             segments = np.concatenate([points[:-1], points[1:]], axis=1)
#             norm = plt.Normalize(-1, 1)
#             lc = LineCollection(segments, cmap=colors.ListedColormap(skill_cmap), norm=norm)
#             lc.set_array(y)
#             lc.set_linewidth(2)
#             line = axc.add_collection(lc)

            if headings:
                if idy == 0:
                    axc.set_title(variable)

            axc.set_ylabel('')
#             axc.set_yticks(range(len(skill['time_scale'])))
#             axc.set_yticklabels(skill['time_scale_name'].values)
            if ncol == 1:
                axc.yaxis.tick_left()
            elif (ax_count+1) % ncol == 0:
                axc.yaxis.tick_right()
            elif (ax_count+ncol) % ncol == 0: 
                axc.yaxis.tick_left()
            else:
                axc.set_yticklabels('')
                
            if ylims:
                axc.set_ylim(ylims[idy])

            axc.set_xticks(range(0,120,24))
            if ax_count / ncol < nrow - 1:
                axc.set_xticklabels('')
                axc.set_xlabel('')
            else:
                axc.set_xlabel('Lead time\n[months]')
            if xlims:
                axc.set_xlim(xlims)

            ax_count += 1
            
    if headings:
        if axes.ndim == 2:
            axz = axes[:,0]
        else:
            axz = axes[:]
        for ax, row in zip(axz, headings):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        ha='right', va='center')
        
    fig.suptitle(title, y=1+0.05*(3-nrow))
    
    plt.subplots_adjust(wspace=0.04, hspace=0.3)
    
    
def plot_ROC_scores(roc_dict, title, Droc_dict=None, expand_axis='time_scale', cmap='gist_ncar_r'):
    
    fig = plt.figure()
    nrow = len(roc_dict)
    ncol = len(roc_dict[list(roc_dict.keys())[0]][expand_axis])
    axes = fig.subplots(nrows=nrow, ncols=ncol, sharex=False, sharey=False)
    
    events = []
    for row_count, (event, roc) in enumerate(roc_dict.items()):
        events.append(event)
        for col_count, expand_value in enumerate(roc[expand_axis].values):
            axc = axes[row_count,col_count]
            
            axc.plot((0,1),(0,1),'k--')
            
            curr_roc = roc.sel({expand_axis: expand_value}) \
                          .isel(lead_time=slice(0,len(roc.lead_time)+1,
                                                roc.sel({expand_axis: expand_value}).time_scale.values))
                
            segments = [np.column_stack([x, y]) for x, y in zip(np.flipud(curr_roc['false_alarm_rate'].transpose('lead_time', 'probability_bin_edge')), 
                                                                np.flipud(curr_roc['hit_rate'].transpose('lead_time', 'probability_bin_edge')))]
            lc = LineCollection(segments, cmap=cmap, clip_on=False, alpha=0.8)
            lc.set_array(np.linspace(0, 1, len(segments)))
            axc.add_collection(lc)
            
            x = [i[0] for j in segments for i in j]
            y = [i[1] for j in segments for i in j]
            cmap_rgb = mpl.cm.get_cmap(cmap)
            c = [[cmap_rgb(i)]*len(curr_roc.probability_bin_edge) for i in np.linspace(0,1,int(len(x)/len(curr_roc.probability_bin_edge)))]
            im = axc.scatter(x, y, c=[item for sublist in c for item in sublist], clip_on=False, cmap=cmap, marker='x', s=15)
            
            if Droc_dict is not None:
                Droc = Droc_dict[event]
                curr_Droc = Droc.sel({expand_axis: expand_value}) \
                                .isel(lead_time=slice(0,len(Droc.lead_time)+1,
                                                            Droc.sel({expand_axis: expand_value}).time_scale.values))
                axc.scatter(curr_Droc['false_alarm_rate'],
                            curr_Droc['hit_rate'], 30, c=np.flipud(np.array(c)[:,1]), edgecolors='k', linewidth=0.75,
                            clip_on=False, cmap=cmap, zorder=10)
                
            
            axc.set_xlim((0,1))
            axc.set_ylim((0,1))
            if col_count == 0: 
                axc.set_ylabel('Hit rate')
            else:
                axc.set_yticklabels('')
            if row_count == (nrow-1): 
                axc.set_xlabel('False-alarm rate')
            else:
                axc.set_xticklabels('')

            if row_count == 0:
                axc.set_title(str(curr_roc[expand_axis+'_name'].values) + ' average')
                
    for ax, row in zip(axes[:,0], events):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    ha='right', va='center')
            
    vmin, vmax = (0, len(roc.lead_time))
    fig.subplots_adjust(bottom=0.25)
    cbar_ax = fig.add_axes([0.16, 0.13, 0.7, 0.020])

    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=mpl.cm.get_cmap(cmap),
                                     orientation='horizontal')
    # cbar_ax.set_xlabel(title, rotation=0, labelpad=15);
    cbar.set_ticks([1,0])
    cbar.set_ticklabels(['forecast start','forecast end'])
    cbar.ax.invert_xaxis()
    
    fig.suptitle(title)
    
    
def get_events(ds, threshold, gl):
    if gl == '>=':
        return xr.where(ds >= threshold, 1, 0).where(ds.notnull())
    elif gl == '<':
        return xr.where(ds < threshold, 1, 0).where(ds.notnull())
    elif gl == '>=<':
        return xr.where((ds >= threshold[0]) & (ds < threshold[1]), 1, 0).where(ds.notnull())
    else:
        raise ValueError('Unrecognised input for gl')
        
        
def get_events_monthly(ds, monthly_threshold, gl):
    results = []
    for month, group in ds.groupby('init_date.month'):
        results.append(get_events(group, monthly_threshold.sel(month=month), gl=gl))
    return xr.concat(results, 'init_date').sortby('init_date').drop('month')


def bundle_into_event_logical(t1, t2, t3):

    return -1*t1 + 0*t2 + 1*t3


def common_dict_entries(*dcts):
    for i in set(dcts[0]).intersection(*dcts[1:]):
        yield (i,) + tuple(d[i] for d in dcts)
        
        
def sort_dict(d, order):
    return  {k: d[k] for k in order.keys()}


def get_centred_box(ds, centre_lon_lat):

    lon_name = doppyo.utils.get_lon_name(ds)
    lat_name = doppyo.utils.get_lat_name(ds)
    
    lons = ds[lon_name]
    lats = ds[lat_name]
    centre = (lons.sel({lon_name: centre_lon_lat[0]}, method='nearest').item(),
              lats.sel({lat_name: centre_lon_lat[1]}, method='nearest').item())
    dlons = lons.diff(lon_name)
    dlats = lats.diff(lat_name)
    ilon = np.argmin(abs(lons - centre[0]))
    ilat = np.argmin(abs(lats - centre[1]))
    bounds = (centre[0]-dlons[ilon].item()/2, centre[0]+dlons[ilon].item()/2,
              centre[1]-dlats[ilat].item()/2, centre[1]+dlats[ilat].item()/2)

    return centre, bounds  


def point_mask(ds, point):
    
    lon_name = doppyo.utils.get_lon_name(ds)
    lat_name = doppyo.utils.get_lat_name(ds)
    lons = ds[lon_name]
    lats = ds[lat_name]
    
    mask = xr.DataArray(np.ones((len(lats), len(lons))),
                        coords=(lats, lons))
    return mask.where(mask[lat_name] == mask[lat_name].sel({lat_name:point[1]}, method='nearest')) \
               .where(mask[lon_name] == mask[lon_name].sel({lon_name:point[0]}, method='nearest'))


def aggregate_grid(ds, lat_des, lon_des):
    
    lat_name = doppyo.utils.get_lat_name(ds)
    lon_name = doppyo.utils.get_lon_name(ds)
    
    lat_edges = doppyo.utils.get_bin_edges(lat_des.values)
    lon_edges = doppyo.utils.get_bin_edges(lon_des.values)
    
    ds_cpy = ds.copy()
    ds_cpy['N'] = (0*ds_cpy[list(ds_cpy.data_vars)[0]]+1)
    ds_sum = ds_cpy.groupby_bins(lon_name, lon_edges, labels=lon_des.values).sum(lon_name) \
                   .groupby_bins(lat_name, lat_edges, labels=lat_des.values).sum(lat_name)

    return (ds_sum.drop('N') / ds_sum['N']).rename({lon_name+'_bins': lon_name,
                                                    lat_name+'_bins': lat_name})


def scale_by_ndays_in_month(ds):
    
    @np.vectorize
    def _ndays_in_month(cftime_object):
        return monthrange(cftime_object.year, cftime_object.month)[1]
    
    n_days = xr.DataArray(np.array(list(map(_ndays_in_month, ds.time.values))),
                          coords=ds.time.coords)
    return ds * n_days


# def bootstrap_skill_metric(da_cmp, da_ref, skill_metric, k, n, j, no_skill_value, alpha, 
#                            block=1, transform=None, skill_metric_kwargs=None, with_dask=True):
#     """
#         Bootstrap a skill metric and return all points where the sample skill metric is positive \
#             (negative) and the fraction of transformed values in the bootstrapped distribution \
#             below (above) no_skill_value is less (greater) than or equal to alpha
#             k = number of bootstraps
#             n = number of initial dates to sample
#             j = number of ensembles to sample. If None, don't bootstrap ensemble
#     """
#     if with_dask:
#         random_sample_skill_metric_ = dask.delayed(random_sample_skill_metric)
#         bs_list = [random_sample_skill_metric_(
#                       da_cmp=da_cmp,
#                       da_ref=da_ref,
#                       skill_metric=skill_metric,
#                       n=n,
#                       j=j,
#                       block=block,
#                       skill_metric_kwargs=skill_metric_kwargs) for _ in range(k)]
#         skill_bs = xr.concat(dask.compute(bs_list)[0], dim='k')
#     else:
#         bs_list = []
#         for _ in range(k):
#             bs_list.append(random_sample_skill_metric(
#                 da_cmp=da_cmp,
#                 da_ref=da_ref,
#                 skill_metric=skill_metric,
#                 n=n,
#                 j=j,
#                 block=block,
#                 skill_metric_kwargs=skill_metric_kwargs))
#         skill_bs = xr.concat(bs_list, dim='k')
#     return xr.where(skill_bs < no_skill_value, 1, 0).mean('k')  <= alpha


def bootstrap_skill_metric(da_cmp, da_ref, skill_metric, 
                           n_dates=None, n_block_dates=1, n_ensembles=None, 
                           skill_metric_kwargs=None,
                           date_name='init_date', ensemble_name='ensemble'):
    """
        Compute a skill metric using data that is randomly sampled (with replacement) \
                from da_cmp and da_ref
                
        Parameters
        ----------
        da_cmp : xarray DataArray or Dataset
            Array containing data to be compared to reference dataset (usually forecasts)
        da_ref : xarray DataArray or Dataset
            Array containing reference data (usually observations)
        skill_metric : function object
            Function for computing skill metric. The first two expected inputs to this function \
            should be da_cmp and da_ref
        n_dates : int, optional
            Number of points to bootstrap along the date_name dimension
        n_block_dates : int, optional
            Number of dates per continuous block when bootstrapping along the date_name dimension. \
            Increase this (above 1) to account for temporal autocorrelation.
        n_ensembles : int, optional
            Number of points to bootstrap along the ensemble_name dimension
        skill_metric_kwargs : dictionary, optional
            Additional arguments to hand to skill_metric function
                
        Returns
        -------
        skill : xarray DataArray or Dataset
            Array containing the skill computed using the randomly sampled data
    """

    def _select_random_ensembles(da, n_ensembles, ensemble_name='ensemble'):
        random_ensembles = random.choices(da[ensemble_name].values, k=n_ensembles)
        return da.sel({ensemble_name: random_ensembles}) \
                 .assign_coords({ensemble_name: range(n_ensembles)})

    def _select_random_dates(da, n_dates, block, date_name='init_date'):
        n_blocks = int(n_dates / block)
        n_dates = n_blocks * block
        random_dates = [list(da[date_name].isel({date_name:slice(x,x+block)}).values)
                        for x in np.random.randint(len(da[date_name])-block, 
                                                   size=n_blocks)]
        random_dates = functools.reduce(operator.iconcat, random_dates, [])
        return random_dates, da.sel({date_name: random_dates}) \
                               .assign_coords({date_name: range(n_dates)})
    
    # Randomly sample dates with replacement -----
    if n_dates:
        random_dates, da_cmp_random_dates = _select_random_dates(da_cmp, n_dates, n_block_dates)
        da_ref_random_dates = da_ref.copy().sel({date_name: random_dates}) \
                                           .assign_coords({date_name: range(len(random_dates))})
    else:
        random_dates = da_cmp[date_name].values
        da_cmp_random_dates = da_cmp
        da_ref_random_dates = da_ref

    # For each date block, randomly sample ensembles with replacement -----
    if n_ensembles:
        da_cmp_random_dates_ensembles = xr.concat([
            _select_random_ensembles(da_cmp_random_dates.isel({date_name: slice(i,
                                                                                i+n_block_dates)}), 
                                     n_ensembles=n_ensembles)
           for i in range(0, n_dates-n_block_dates+1, n_block_dates)], dim=date_name)

    else:
        da_cmp_random_dates_ensembles = da_cmp_random_dates

    return skill_metric(da_cmp_random_dates_ensembles, 
                        da_ref_random_dates, 
                        **skill_metric_kwargs)

def n_bootstrap_skill_metric(da_cmp, da_ref, skill_metric, 
                             n_resamples, n_dates, n_block_dates, n_ensembles,
                             skill_metric_kwargs=None, with_dask=False,
                             date_name='init_date', ensemble_name='ensemble'):
    """
        Compute a skill metric using data that is randomly sampled (with replacement) \
                from da_cmp and da_ref
                
        Parameters
        ----------
        da_cmp : xarray DataArray or Dataset
            Array containing data to be compared to reference dataset (usually forecasts)
        da_ref : xarray DataArray or Dataset
            Array containing reference data (usually observations)
        skill_metric : function object
            Function for computing skill metric. The first two expected inputs to this function \
            should be da_cmp and da_ref
        n_resamples : int, optional
            Number of times to repeat the bootstrapping to build the distribution
        n_dates : int, optional
            Number of points to bootstrap along the date_name dimension
        n_block_dates : int, optional
            Number of dates per continuous block when bootstrapping along the date_name dimension. \
            Increase this (above 1) to account for temporal autocorrelation.
        n_ensembles : int, optional
            Number of points to bootstrap along the ensemble_name dimension
        skill_metric_kwargs : dictionary, optional
            Additional arguments to hand to skill_metric function
        with_dask : boolean, optional
            If True, use dask to parallelize across n_resamples using dask.delayed
                
        Returns
        -------
        bs_skill : xarray DataArray or Dataset
            Array containing the bootstrapped skill score distributions along added 'k' dimension
    """
    bootstrap_skill_metric_ = dask.delayed(bootstrap_skill_metric) if with_dask \
                                  else bootstrap_skill_metric
    bs_list = [bootstrap_skill_metric_(
                   da_cmp=da_cmp,
                   da_ref=da_ref,
                   skill_metric=skill_metric,
                   n_dates=n_dates,
                   n_block_dates=n_block_dates,
                   n_ensembles=n_ensembles,
                   skill_metric_kwargs=skill_metric_kwargs,
                   date_name=date_name,
                   ensemble_name=ensemble_name) for _ in range(n_resamples)] 
    return xr.concat(dask.compute(bs_list)[0], dim='k') if with_dask \
               else xr.concat(bs_list, dim='k')


def get_skill_and_signif(da_cmp, da_ref, skill_metric, 
                         no_skill_value, alpha, transform=None, return_bs_distributions=False,
                         bootstrap_kwargs=None, skill_metric_kwargs=None):
    """
        Compute the sample skill scores and estimate their statistical significance. For the \
            latter, a bootstrapped distribution of the skill metric is constructed. Statistical \
            significance at 1-alpha is identified at all points where the sample skill metric \
            is positive (negative) and the fraction of transformed values in the bootstrapped \
            distribution below (above) no_skill_value--defining the p-values--is less than or \
            equal to alpha.
        
        Parameters
        ----------
        da_cmp : xarray DataArray or Dataset
            Array containing data to be compared to reference dataset (usually forecasts)
        da_ref : xarray DataArray or Dataset
            Array containing reference data (usually observations)
        skill_metric : function object
            Function for computing skill metric. The first two expected inputs to this function \
            should be da_cmp and da_ref
        no_skill_value : float
            No skill value for the given skill metric
        alpha : float
            alpha value for significance testing 
        transform : function, optional
            transformation to apply to the bootstrapped distribution prior to computing p-values.
        return_bs_distributions : boolean, optionial
            if True, return full bootstrapped distrubtions. Otherwise run significance test and return \
            mask for significant points
        bootstrap_kwargs : dictionary, optional
            Arguments to hand to n_bootstrap_skill_metric function
        skill_metric_kwargs : dictionary, optional
            Additional arguments to hand to skill_metric function
                
        Returns
        -------
        skill : xarray DataArray or Dataset
            Sample skill scores
        signif_mask/skill_bs : xarray DataArray or Dataset
            If return_bs_distributions=True this is the full bootstrapped skill score distributions, \
            otherwise it is a mask containing ones where skill is deemed significant
    """
    sample_skill = skill_metric(da_cmp, da_ref,
                                **skill_metric_kwargs)

    if bootstrap_kwargs:
        bootstrap_kwargs['skill_metric_kwargs'] = skill_metric_kwargs
        da_cmp_bs = bootstrap_kwargs.pop('da_cmp') if 'da_cmp' in bootstrap_kwargs else da_cmp
        da_ref_bs = bootstrap_kwargs.pop('da_ref') if 'da_ref' in bootstrap_kwargs else da_ref
        bs_skill = n_bootstrap_skill_metric(da_cmp_bs, da_ref_bs, skill_metric, 
                                            **bootstrap_kwargs)
    
    if return_bs_distributions:
        return sample_skill, bs_skill
    else:
        if transform:
            pos_signif = xr.where(transform(bs_skill) < 
                                  transform(no_skill_value), 1, 0).mean('k')  <= alpha
            neg_signif = xr.where(transform(bs_skill) > 
                                  transform(no_skill_value), 1, 0).mean('k')  <= alpha
        else:
            pos_signif = xr.where(bs_skill < 
                                  no_skill_value, 1, 0).mean('k')  <= alpha
            neg_signif = xr.where(bs_skill > 
                                  no_skill_value, 1, 0).mean('k')  <= alpha

        return sample_skill, \
               ((sample_skill > no_skill_value) & pos_signif) | \
               ((sample_skill < no_skill_value) & neg_signif)


def Fisher_z(ds):
    " Return the Fisher-z transformation of ds "
    return np.arctanh(ds)


def remove_co2_trend(ds, co2):
    def _regress(y, x, dim):
        ya, xa = xr.align(y, x, join='inner')
        yd = ya - ya.mean(dim)
        xd = xa - xa.mean(dim)
        return (yd*xd).mean(dim) / (xd*xd).mean(dim)
    
    def _remove_trend(ds, co2, dim):
        if ('lead_time' in ds.coords) & ('lead_time' not in ds.dims):
            ds_a, co2_a = xr.align(ds, co2.sel(lead_time=ds.lead_time.values), join='inner')
        else:
            ds_a, co2_a = xr.align(ds, co2, join='inner')

        trend = _regress(ds_a, co2_a, dim=dim) * co2_a
        
        return ds_a - (trend-trend.mean(dim))
    
    if co2.nbytes < 0.1e9:
        co2 = co2.compute()
        
    if 'lead_time' in ds:
        return ds.groupby('lead_time').apply(_remove_trend, co2=co2, dim='init_date')
    else:
        return _remove_trend(ds, co2=co2, dim='init_date')


def get_diagnostic_from_txt(file_name, single_column=False):
    if single_column:
        data = np.loadtxt(fname=file_name, skiprows=1)
        years = data[:,0]
        months = data[:,1]
        data = data[:,single_column]
        dates = np.array([np.datetime64(str(int(date[0]))+'-'+str(int(date[1])).zfill(2)+'-01') 
                             for date in zip(years, months)])
    else:
        data = np.loadtxt(fname=file_name, skiprows=1)
        years = data[:,0]
        data = data[:,1:].flatten()
        dates = np.array([np.datetime64(str(int(year))+'-'+str(int(month)).zfill(2)+'-01')
                               for year in years 
                               for month in range(1,13)])
    return xr.DataArray(data, dims='time', coords={'time':dates}).to_dataset(name=file_name[:-4])