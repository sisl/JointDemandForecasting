import pandas as pd
import numpy as np

# LOAD FIELDS

FEATURE_MAP_TMY3 = {}
FEATURE_MAP_TMY3["DT"] = "Date/Time"
FEATURE_MAP_TMY3["ELEC"] = "Electricity:Facility [kW](Hourly)"
FEATURE_MAP_TMY3["GAS"] = "Gas:Facility [kW](Hourly)"
FEATURE_MAP_TMY3["HELEC"] = "Heating:Electricity [kW](Hourly)"
FEATURE_MAP_TMY3["HGAS"] = "Heating:Gas [kW](Hourly)"
FEATURE_MAP_TMY3["CELEC"] = "Cooling:Electricity [kW](Hourly)"
FEATURE_MAP_TMY3["HVACFANS"] = "HVACFan:Fans:Electricity [kW](Hourly)"
FEATURE_MAP_TMY3["HVACELEC"] = "Electricity:HVAC [kW](Hourly)"
FEATURE_MAP_TMY3["FANSELEC"] = "Fans:Electricity [kW](Hourly)"
FEATURE_MAP_TMY3["INTLIGHTSELEC"] = "General:InteriorLights:Electricity [kW](Hourly)"
FEATURE_MAP_TMY3["EXtLIGHTSELEC"] = "General:ExteriorLights:Electricity [kW](Hourly)"
FEATURE_MAP_TMY3["APPLELEC"] = "Appl:InteriorEquipment:Electricity [kW](Hourly)"
FEATURE_MAP_TMY3["MISCELEC"] = "Misc:InteriorEquipment:Electricity [kW](Hourly)"
FEATURE_MAP_TMY3["WATERGAS"] = "Water Heater:WaterSystems:Gas [kW](Hourly)" 

FEATURE_MAP_TMY2 = {}
for key, value in FEATURE_MAP_TMY3.items():
    FEATURE_MAP_TMY2[key] = value.replace("[kW]","[kWh]")

def read_load_csv(path, load_features):
    """
    This method reads the load file from highD data.
    :param path: the input path for the tracks csv file.
    :return: a pandas df containing only relevant features.
    """
    # Read the csv file, convert it into a useful data structure
    df = pd.read_csv(path)
    
    if path.find("TMY2") > -1:
        map = FEATURE_MAP_TMY2
    else:
        map = FEATURE_MAP_TMY3
    
    df = df.loc[:,[map[f] for f in load_features]]
    
    if "DT" in load_features:
        df["Date/Time"] = df["Date/Time"].apply(my_to_datetime)
        df['Hour'] = df["Date/Time"].dt.hour +1
        df['Weekday'] = df["Date/Time"].dt.weekday + 1
        df['Month'] = df["Date/Time"].dt.month
    return df

def my_to_datetime(date_str):
    year = '2005/'
    if ' ' not in date_str[8:10] and ':' not in date_str[8:10]: 
        date_str = year + date_str[1:8] + str(int(date_str[8:10])-1) + date_str[10:]
    else:
        try:
            date_str = year + date_str[1:7] + str(int(date_str[7:9])-1) + date_str[9:]
        except:
            raise Exception("Bad String: "+date_str)
        
    return pd.to_datetime(date_str, format='%Y/%m/%d  %H:%M:%S')

    #if date_str[8:10] != '24':
    #    return pd.to_datetime(date_str, format=' %m/%d  %H:%M:%S'')
    #
    #date_str = date_str[0:8] + '00' + date_str[10:]
    #return pd.to_datetime(date_str, format='%Y%m%d%H%M') + \
    #       dt.timedelta(days=1)
    
def df_to_numpy(df):
    return df.iloc[:,1:].to_numpy(), list(df.columns)[1:]

def encode_features(data, options):
    
    n, m = data.shape
    # Assuming data ordering has Hour, Weekday, Month as last 3 
    
    # Want to encode T:-1:T-12, T-23:-1:T-25, T-47:-1:T-49, one-hot month, one-hot weekday, one-hot hour
    # Want to Predict T+{1,2,6,12,24}, dT+{1,2,6,12,24}, T_max{0:23}, dT_max{0:23

    x,y = [],[]
    for i in range(n):
        
        # if not enough data to make full input/output, continue
        if i<24 or (n-i-1)<24:
            continue
        
        x_idxs = np.concatenate((np.arange(i,i-24,-1),
                                 #np.arange(i-23,i-26,-1),
                                 #np.arange(i-47,i-50,-1)
                                ))
        new_x = np.concatenate((data[x_idxs,0], 
                                one_hot(data[i,-1],12),one_hot(data[i,-2],7), one_hot(data[i,-3],24), 
                                np.array([
                                    data[i-24:i,0].max(),
                                    # data[i-48:i-24,0].max(),
                                    int(data[i,-1])-1, int(data[i,-2])-1, int(data[i,-3])-1,
                                    np.sin((data[i,-1]-1)*2*np.pi/12), np.cos((data[i,-1]-1)*2*np.pi/12),
                                    np.sin((data[i,-2]-1)*2*np.pi/7),  np.cos((data[i,-2]-1)*2*np.pi/7),
                                    np.sin((data[i,-3]-1)*2*np.pi/24), np.cos((data[i,-3]-1)*2*np.pi/24)
                                ])))
        
        y_idxs = i+np.arange(1,13) #np.array([1,2,6,12,24])
        fut_y = data[y_idxs,0]
        fut_dy = fut_y - data[i,0]
        fut_max = np.array([data[i:i+24,0].max()])
        fut_dmax = fut_max - data[i,0]
        new_y = np.concatenate((fut_y, 
                                #fut_dy, 
                                #fut_max, 
                                #fut_dmax
                               ))
        x.append(new_x)
        y.append(new_y)
        
    x = np.vstack(x)
    y = np.vstack(y)
    return x,y

def one_hot(input, length):
    if input>length:
        raise Exception("Can't one-hot encode correctly since input %f >length" %(input))
    elif input < 1:
        raise Exception("Can't one-hot encode correctly since input %f <1" %(input))
    oh = np.zeros(length)
    oh[int(input)-1] = 1
    return oh
    
    
