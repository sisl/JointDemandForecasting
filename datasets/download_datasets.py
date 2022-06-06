import os
import requests, zipfile, io
import urllib.request, csv
opj =  os.path.join

basepath = './raw/'

# OpenEI
openei = 'https://openei.org/datasets/files/961/pub/EPLUS_TMY2_RESIDENTIAL_BASE.zip'
if not os.path.exists(opj(basepath,'EPLUS_TMY2_RESIDENTIAL_BASE/')):
    print('Downloading OpenEI TMY2')
    r = requests.get(openei)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(basepath)

# North-American Utility
nau = 'http://www.ee.washington.edu/class/555/el-sharkawi/datafiles/forecasting.zip'
if not os.path.exists(opj(basepath,'nau/')):
    print('Downloading North American Utility')
    r = requests.get(nau)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(opj(basepath,'nau'))

# ISO NE
iso_ne = 'https://www.iso-ne.com/transform/csv/hourlysystemdemand?start=20160101&end=20191231'
if not os.path.exists(opj(basepath,'iso-ne.csv')):
    print('Download New England ISO dataset manually at:')
    print(iso_ne)
    print('and save as ./raw/iso-ne.csv')
    #r = requests.get(iso_ne)
    #with open(opj(basepath,'iso-ne.csv'), 'wb') as csv_file:
    #    csv_file.write(r.content)

# NSW 400
nsw = 'https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/zm4f727vvr-1.zip'
if not os.path.exists(opj(basepath,'nsw/')):
    print('Downloading New South Wales 400')
    r = requests.get(nsw)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(opj(basepath,'nsw/'))

#GEFCom2014
gef = 'https://www.dropbox.com/s/pqenrr2mcvl0hk9/GEFCom2014.zip?dl=1'
if not os.path.exists(opj(basepath,'GEFCom2014/')):
    print('Downloading GEF Competition 2014')
    r = requests.get(gef)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(opj(basepath,'GEFCom2014/'))