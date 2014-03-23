#-*-encoding:utf-8-*-
import urllib
import json
import subprocess
import os
import math
import time
from httplib import BadStatusLine
##Some time conversion functions

def gmt_to_local(_time,make_string=False,format='%d %b %H:%M:%S %Y'):

	_time =  time.localtime(time.mktime(_time)+time.mktime(time.localtime())-time.mktime(time.gmtime()))
	if not make_string:
		return _time
	else:
		return time.strftime(format,_time)

def local_to_gmt(_time,make_string=False,TZ='EST'):
		_time  = time.gmtime(time.mktime(time.strptime(_time.replace('+0000',TZ),"%b %d %H:%M:%S %Z %Y")))

		if not make_string:
			return _time
		else:
			return time.strftime('%b %d %b %H:%M:%S %Z %Y',_time)

###Some common locations and their geocode details###
###Source :  "http://maps.googleapis.com/maps/api/geocode/json?address=%s&sensor=false"%place_name

location = {'Delhi':dict(latitude=28.6353080,longitude= 77.2249600,radius=25),'Mumbai':dict(latitude=19.0759837,longitude=72.8776559,radius=25),'NYC':dict(latitude=40.7143528,longitude=-74.00597309999999,radius=25),
            'Boston':dict(latitude=42.3606249,longitude=-71.0591155,radius=25),'SanFrancisco':dict(latitude=37.7749295,longitude=-122.4194155,radius=25),'London':dict(latitude=51.6723432,longitude=-0.1998244,radius=25),'LosAngeles':dict(latitude=34.337306,longitude=-118.155289,radius=25),'Toronto':dict(latitude=43.5810847000001,longitude=-79.639219,radius=25),'SanDiego':dict(latitude=32.7153292,longitude=-117.1572551,radius=25),'Houston':dict(latitude=29.7601927,longitude=-95.36938959999999,radius=25)}

locationbox = {'Delhi':[76.8396999,28.4010669,77.3418145999999,28.88981589],'Mumbai':[72.775908899,18.8928676,72.9864994,19.2716338],'NYC':[-74.2590879,40.495996,-73.700272,40.9152555],
            'Boston':[ -71.191113,42.22788,-70.92320099999999,42.3988669],'SanFrancisco':[-122.75,36.8,-121.75,37.8],'London':[-0.3514684,51.38494009999999,0.148271,51.6723432],'LosAngeles':[-118.668176,33.7036918,-118.155289,34.337306],'Toronto':[-79.639219,43.58108470000001,-79.1161932,43.855458],'SanDiego':[-117.2821665,32.534856,-116.90816,33.114249],'Houston':[-95.78808690000001,29.523624,-95.01449599999999,30.110732]}

Placenames = { 'Boston':['Boston','Cambridge','Brookline','Somerville','Quincy'] }

GOOGLE_GEOCODE_BASE_URL = 'https://maps.googleapis.com/maps/api/geocode/json'
OSM_GEOCODE_BASE_URL = 'http://nominatim.openstreetmap.org/'

def GetGeocode(address,sensor='false'):
	"""Makes call to GoogleMaps API and returns
	   longitude,latitude for place name"""

	argsGOOGLE = {}
	argsOSM = {}
	argsGOOGLE.update({
			'address': address,
			'sensor' : sensor,

		})
	argsOSM.update({
			'addressdetails': 0,
			'polygon':0,
			'q':address.replace(',','')

		})

	urlGOOGLE = GOOGLE_GEOCODE_BASE_URL + '?' + urllib.urlencode(argsGOOGLE)
	urlOSM = OSM_GEOCODE_BASE_URL + 'search?format=json'+'&' + urllib.urlencode(argsOSM)

	try:
		out = json.load(urllib.urlopen(urlGOOGLE))['results'][0]['geometry']['location']
		lon,lat = (out['lng'],out['lat'])
		return [lon,lat]
	except (IndexError,IOError,BadStatusLine) as e:
		pass

	try:
		out = json.load(urllib.urlopen(urlOSM))[0]
		lon,lat = (float(out['lon']),float(out['lat']))
		return [lon,lat]
	except KeyError:
		raise IndexError


def GetPlaceName(lat,lon,zoom=16,sensor='false'):
	"""Makes call to GoogleMaps API and returns
	   human readable address for lat,lon"""
	argsGOOGLE = {}
	argsOSM = {}
	argsGOOGLE.update({
			'latlng': '%s,%s'%(lat,lon),
			'sensor' : sensor,
		})
	argsOSM.update({
			'lat': lat,
			'lon': lon,
			'zoom'  : zoom,
			'addressdetails': 0
		})

	urlGOOGLE = GOOGLE_GEOCODE_BASE_URL + '?' + urllib.urlencode(argsGOOGLE)
	urlOSM = OSM_GEOCODE_BASE_URL + 'reverse?format=json'+'&' + urllib.urlencode(argsOSM)

	ct=0;count = 2;
	while ct<count:
		try:
			out = json.loads(subprocess.check_output("curl --request GET '%s'"%urlGOOGLE,stderr=open(os.devnull, 'w'),shell=True))['results'][0]['formatted_address']
			break
		except (CalledProcessError,IndexError) as e:
			pass
		try:
			out = json.loads(subprocess.check_output("curl --request GET '%s'"%urlOSM,stderr=open(os.devnull, 'w'),shell=True))['display_name']
			out = ' ,'.join(out.split(',')[0:2])
			break
		except KeyError:
			ct+=1
			continue

	if ct<count:
		return out
	else:
		raise IndexError

def latlon2km((lat1,lon1),(lat2,lon2)):
	"""
	Reference
	---------
	longitude :
	The equator is divided into 360 degrees of longitude, so each degree at the equator represents 111,319.9 metres or
    approximately 111.32 km. As one moves away from the equator towards a pole, however, one degree of longitude is multiplied by
    the cosine of the latitude, decreasing the distance, approaching zero at the pole

    latitude:
    Each degree of latitude is approximately 69 miles (111 kilometers) apart. The range varies (due to the earth's slightly ellipsoid shape)
    from 68.703 miles (110.567 km) at the equator to 69.407 (111.699 km) at the poles.
    """

	rad_km  = ((lat1-lat2)*(110.567+((111.699-110.567)*(abs(lat1)/90.0))),(lon1-lon2)*(math.cos(lat1*math.pi/180.0))*111.32)
	return rad_km[0]**2 + rad_km[1]**2
