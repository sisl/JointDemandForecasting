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

    
    
