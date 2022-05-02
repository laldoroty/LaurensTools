import numpy as np
import pandas as pd

def get_bands(path):
    
    """
    Lauren Aldoroty, 20200903
    
    Takes a file in conventional SNooPy format and identifies the bands in the data file. 
    Rows containing data must be indented with one (1) space (except the 0th row), or 
    you will need to modify the indices in the function. 
    
    """
    
    bands = []
    data = pd.read_csv(path, sep = ' ', index_col=False, names = ['no_data','date','mag','emag'])
    
    for row in data.itertuples():
        if row[1] == 'filter':
            bands.append(row[2])
        
    return bands


def get_band_data(path):
    
    """
    Lauren Aldoroty, 20210511
    
    Takes a file in conventional SNooPy format and returns one dictionary with date,
    magnitude, and error on magnitude for each photometric band. Returns redshift 
    and the supernova name as separate values. Rows containing data must be indented 
    with one (1) space (except the 0th row), or you will need to modify the indices 
    in the function. 
    
    """
    
    data = pd.read_csv(path, sep = ' ', index_col=False, names = ['no_data','date','mag','emag'])
    bands = get_bands(path)
    
    redshift = data['date'][0]
    sn_name = data['no_data'][0]
    ra = data['mag'][0]
    dec = data['emag'][0]
    
    data_dict = {band: {'date': [], 'mag': [], 'emag': []} for band in bands}

    write_condition = False
    write_to_band = np.nan

    for band in bands:
        for row in data.itertuples():
            if write_condition == True:
                try:
                    if write_to_band == band:
                        data_dict[band]['date'].append(float(row[2]))
                        data_dict[band]['mag'].append(float(row[3]))
                        data_dict[band]['emag'].append(float(row[4]))
                except:
                    pass

            if row[2] == band:
                write_condition = True
                write_to_band = band

            elif row[2] != band and str(row[1]) != str('nan'):
                write_condition = False

        data_dict[band]['date'] = np.array(data_dict[band]['date'])
        data_dict[band]['mag'] = np.array(data_dict[band]['mag'])
        data_dict[band]['emag'] = np.array(data_dict[band]['emag'])

    return sn_name, float(redshift), ra, dec, data_dict
