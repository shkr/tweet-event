# -*- coding: utf-8 -*-

import subprocess
import sys
import os
import time
import json
import MySQLdb
import urllib
import math
import numpy as np
from t_auth import *
from geocode import locationbox
from tweetokenize import Tokenizer
from sklearn import cluster

class Cluster:
	def __init__(self,created_at,EventWord,SpatialSigature):
		self.created_at = created_at
		self.lifetime   = 1
		self.Y 			= [EventWord]
		self.SC         = SpatialSigature


def Tokenize(output,tags,regex=None):
	"""tags  : POS Tagged words to return
	   regex : Matching regex expressions to return
	"""
	WORDS  = output[0].split()
	TAGS   = output[1].split()
	KeyWords  = []

	for i,tag in enumerate(TAGS):
		KeyWords+=[WORDS[i]] if tag in tags else []

	return KeyWords

class TweetIterator:
	def __init__(self,place,useall=False):
		self.place = place
		self.useall = useall
		

		conn = MySQLdb.connect(**mysql_auth)
		self.curr = conn.cursor()
		Grid          = locationbox[self.place]
		self.collect_items = ['user_id','place','lat','lon','created_at','text']
		if self.useall:
			Streamingq  = "SELECT %s FROM streamerallTable WHERE ((lon >= %d AND lon <= %d) AND (lat>=%d AND lat<=%d))"%(','.join(self.collect_items),Grid[0],Grid[2],Grid[1],Grid[3])
		else:
			Streamingq  = "SELECT %s FROM streamer%sTable"%(','.join(self.collect_items),self.place)
		self.curr.execute(Streamingq)

		self.tokenize = Tokenizer(lowercase=False,normalize=2,ignorestopwords=True).tokenize

	def __iter__(self):
		for tweet in self.curr:
			yield self.preprocessing(tweet[5]).split()
			#yield [tweet[0],tweet[1],tweet[2],tweet[3],tweet[4],self.preprocessing(tweet[5])]
	
	def preprocessing(self,text):
		"""
		Process tweet here 
		"""
		return ' '.join(filter(lambda x: x is not None,[ i if (len(i)>1 and i[0]!='@' and i.isalnum() and i not in ['USERNAME','URL','PHONENUMBER','TIME','NUMBER']) else None for i in self.tokenize(text)]))

	def next(self):
		tweet = self.curr.fetchone()
		return self.preprocessing(tweet[5]).split()
		#return [tweet[0],tweet[1],tweet[2],tweet[3],tweet[4],self.preprocessing(tweet[5])]

def TweetTrain(timeWindow=60):
	"""
		timeWindow : units are Seconds
	"""
	conn = MySQLdb.connect(**mysql_auth)
	
	curr = conn.cursor()
	collect_items =['created_at','text','user_id']
	StreamingallBoston = "SELECT %s FROM streamerallTable WHERE (user_location LIKE ', MA') OR ((lon < -69 AND lon > -73) AND (lat>41 AND lat<43))"%(','.join(collect_items))	
	curr.execute(StreamingallBoston)
	
	ListOfTweets = curr.fetchall()
	print len(ListOfTweets)

	KeyTags    = ['Z','$','G','M','^','#','@']  #Removed V
	KeyRegex   = ['A.M.','AM','digit am','P.M.','PM','digit pm']
	time_start = time.gmtime()
	timeHashed = time.strftime("%x %X",time_start)
	Vocabulary = {}
	Vocabulary[timeHashed] = {}
	LegacyVocabulary = {}
	UniqueUids = []

	for item in ListOfTweets:
		item = dict(zip(collect_items,item))
		# print "State @ %s"%timeHashed
		# print item['text']
		# print 'TweetTime : %s'%item['created_at']
		# print Vocabulary[timeHashed]
		# print '-'*50
		
		try:
			#WORDS  = Tokenize(output= subprocess.check_output(['ark-tweet-nlp-0.3.2/./runtagger.sh --no-confidence <<< "%s"'%item['text'].replace('"','\\"').replace('`','')],stderr=open(os.devnull, 'w'),shell=True).split('\t'),tags=KeyTags,regex=KeyRegex)
			WORDS  = subprocess.check_output(['ark-tweet-nlp-0.3.2/./twokenize.sh --just-tokenize --no-confidence <<< "%s"'%item['text'].replace('"','\\"').replace('`','')],stderr=open(os.devnull, 'w'),shell=True).split("\t")[0].split()
		except subprocess.CalledProcessError as e:
			print (e)
			continue

		UID   = item['user_id']
		TIME  = time.strptime(item['created_at'],"%a %b %d %H:%M:%S +0000 %Y")

		#TimeWindow update
		shiftWindow = (TIME<time_start) or (time.mktime(TIME)-time.mktime(time_start)>timeWindow)
		if shiftWindow:
			#Vocabulary update
			for key,value in Vocabulary[timeHashed].items(): Vocabulary[timeHashed][key] = float(value)/len(UniqueUids)
			#Legacy Vocabulary update
			for word,frequency in Vocabulary[timeHashed].items():
				try:
					LegacyVocabulary[word]+=[frequency]
				except KeyError:
					LegacyVocabulary[word]=[frequency]
			#TimeWindow update
			time_start = TIME
			timeHashed = time.strftime("%x %X",time_start)
			Vocabulary[timeHashed]={}
			UniqueUids = []

		#Vocabulary update
		newUser = 	UID not in UniqueUids
		if newUser:
			UniqueUids+=[UID]
			for word in set(WORDS):
				if word in Vocabulary[timeHashed].keys():
					Vocabulary[timeHashed][word]+=1
				else:
					Vocabulary[timeHashed][word]=1

	with open('LegacyVocabulary.json','wb') as f:
		json.dump(LegacyVocabulary,f,ensure_ascii=False,indent=5,separators=(',',':'))

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
	except IndexError:
		pass
	try:
		out = json.load(urllib.urlopen(urlOSM))[0]
		lon,lat = (float(out['lon']),float(out['lat']))
	except IndexError:
		raise IndexError

	return [lon,lat] 
	
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
			'zoom'  : 16,
			'addressdetails': 0
		})
	
	urlGOOGLE = GOOGLE_GEOCODE_BASE_URL + '?' + urllib.urlencode(argsGOOGLE)
	urlOSM = OSM_GEOCODE_BASE_URL + 'reverse?format=json'+'&' + urllib.urlencode(argsOSM)
	
	ct=0;count = 2;
	while ct<count:
		try:
			out = json.loads(subprocess.check_output("curl --request GET '%s'"%urlGOOGLE,stderr=open(os.devnull, 'w'),shell=True))['results'][0]['formatted_address']
			break		
		except IndexError:
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
	


def TweetEvent(place,time_start=None,timeWindow=60,time_end=None,collect_items=None,useall=False):
	"""
	   time_start : TimeStruct
	   timeWindow : Units are Seconds
	   time_end   : TimeStruct

	   From time_start to time_end send list of tweets at time steps timeWindow,
	   in the form of a tuple (tweetid(tid),KeyWords(W),user_id(uid),[latitude,longitude](l),Time(t)).
	"""

	conn = MySQLdb.connect(**mysql_auth)
	collect_items = ['tid','lat','lon','created_at','text','user_id','user_location'] if collect_items==None else collect_items
	curr = conn.cursor()
	Grid          = locationbox[place]
	if useall:
		Streamingq  = "SELECT %s FROM streamerallTable WHERE ((lon >= %d AND lon <= %d) AND (lat>=%d AND lat<=%d))"%(','.join(collect_items),Grid[0],Grid[2],Grid[1],Grid[3])
	else:
		Streamingq  = "SELECT %s FROM streamer%sTable"%(','.join(collect_items),place)
	curr.execute(Streamingq)
	ListOfTweets = curr.fetchall()
    
	KeyTags    = ['Z','$','G','M','^','#','@']  #Removed V
	KeyRegex   = ['A.M.','AM','digit am','P.M.','PM','digit pm']
	time_start = time.gmtime() if time_start==None else time_start
	timeHashed = time.strftime("%x %X",time_start)
	Vocabulary = {}
	Vocabulary[timeHashed] = {}

	tw  = {}
	tw[timeHashed] = {}
	UniqueUids = []
	
	LegacyVocabulary = TweetTrain(timeWindow=60)

	for item in ListOfTweets:

		#Tweet Dictionary
		item = dict(zip(collect_items,item))

		#Data points
		try:
			WORDS = Tokenize(output= subprocess.check_output(['ark-tweet-nlp-0.3.2/./runtagger.sh --no-confidence <<< "%s"'%item['text'].replace('"','\\"').replace('`','')],stderr=open(os.devnull, 'w'),shell=True).split('\t'),tags=KeyTags,regex=KeyRegex)
			WORDS = item['text'].split(' ')

		except subprocess.CalledProcessError as e:
			print (e)
			continue
		UID   = item['user_id']
		LOC   = [float(item['lon']),float(item['lat'])] if (item['lon']!=0) else GetGeocode(item['user_location'])
		TIME  = time.strptime(item['created_at'],"%a %b %d %H:%M:%S +0000 %Y")
		
		#TimeWindow update
		shiftWindow = (TIME<time_start) or (time.mktime(TIME)-time.mktime(time_start)>timeWindow)
		if shiftWindow:
			for key,value in Vocabulary[timeHashed].items(): Vocabulary[timeHashed][key] = float(value)/len(UniqueUids)

			for word,frequency in Vocabulary[timeHashed].items():
				if word in LegacyVocabulary.keys():
					frequencies = np.array(LegacyVocabulary[word])
					if ((frequency-np.mean(frequencies,axis=0))/np.std(frequencies))>2:
						UpdateEventWords(word,tw[timeHashed],Grid)
				else:
					UpdateEventWords(word,[(tid,value) if word in value['words'] else None for key,value in tw[timeHashed].items()])   
					        

			time_start = TIME
			timeHashed = time.strftime("%x %X",time_start)
			Vocabulary[timeHashed]={}
			UniqueUids = [] 
		
		if UID not in UniqueUids:
			UniqueUids+= [UID]

		#Vocabulary update
		for word in set(WORDS):
			if word in Vocabulary[timeHashed].keys():
				Vocabulary[timeHashed][word]+=1
			else:
				Vocabulary[timeHashed][word]=1
		
		#Store Tweet
		tw[timeHashed][item['tid']]          = {} 
		tw[timeHashed][item['tid']]['words'] = WORDS
		tw[timeHashed][item['tid']]['uid']   = UID
		tw[timeHashed][item['tid']]['loc']   = LOC
		tw[timeHashed][item['tid']]['time']  = TIME
		
		
	return [tw,Vocabulary]

def DataRetreiveForTopicDiscovery(place,time_start=None,timeWindow=60*30,time_end=None,useall=False):
	
	"""
	   time_start : TimeStruct
	   timeWindow : Units are Seconds
	   time_end   : TimeStruct

	   From time_start to time_end send list of tweets at time steps timeWindow,
	   in the form of a tuple (tweetid(tid),KeyWords(W),user_id(uid),[latitude,longitude](l),Time(t)).
	"""

	conn = MySQLdb.connect(**mysql_auth)
	collect_items = ['created_at','text']
	curr = conn.cursor()
	Grid          = locationbox[place]
	if useall:
		Streamingq  = "SELECT %s FROM streamerallTable WHERE ((lon >= %d AND lon <= %d) AND (lat>=%d AND lat<=%d))"%(','.join(collect_items),Grid[0],Grid[2],Grid[1],Grid[3])
	else:
		Streamingq  = "SELECT %s FROM streamer%sTable"%(','.join(collect_items),place)
	
	curr.execute(Streamingq)
	ListOfTweets = curr.fetchall()
	print len(ListOfTweets)
	KeyTags    = ['Z','$','G','M','^','#']  #Removed 'V','@'
	KeyRegex   = ['A.M.','AM','digit am','P.M.','PM','digit pm']
	time_start = time.gmtime() if time_start==None else time_start
	timeHashed = time.strftime("%x %X",time_start)
	
	tw  = {}
	tw[timeHashed]=[]	
	Vocabulary    =[]
	stpwrds = [ w.strip() for w in open('stopwords.txt').readlines()]

	for item in ListOfTweets:

		#Tweet Dictionary
		item = dict(zip(collect_items,item))

		#Data points
		try:
			#WORDS = Tokenize(output= subprocess.check_output(['ark-tweet-nlp-0.3.2/./runtagger.sh --no-confidence <<< "%s"'%item['text'].replace('"','\\"').replace('`','')],stderr=open(os.devnull, 'w'),shell=True).split('\t'),tags=KeyTags)
			WORDS = [ w.lower() if (w not in stpwrds and w[0]!='@') else None for w in item['text'].replace('.',' ').replace(',',' ').replace('\'',' ').replace('\"',' ').replace('!',' ').replace('?',' ').replace(';',' ').replace(':',' ').split()	]
			WORDS = filter(lambda x: x != None,WORDS)
				
		except subprocess.CalledProcessError as e:
			print (e)
			continue
		TIME  = time.strptime(item['created_at'],"%a %b %d %H:%M:%S +0000 %Y")
		#TimeWindow update
		shiftWindow = (TIME<time_start) or (time.mktime(TIME)-time.mktime(time_start)>timeWindow)
		if shiftWindow:
			time_start = TIME
			timeHashed = time.strftime("%x %X",time_start)
			tw[timeHashed]=[]

		#Store Tweet
		tw[timeHashed]+=WORDS
		
	#Create Global Vocabulary 13.323817 seconds
	start = time.time()
	for v in tw.values():
		Vocabulary = set(list(set(v))+list(Vocabulary))
	print 'Time taken to build Vocabulary is : %f seconds'%(time.time() - start)
	Vocabulary = list(Vocabulary)

	with open('retreiverData/TweetRetreiveVocab2.txt','wb') as f:
		for v in Vocabulary:
			f.write('\n'+v.encode('utf-8'))
	
	#Prepare data for LDA
	for doc in tw.values():
		with open('retreiverData/TweetRetreive2.dat','ab') as f:
			lineout = ''
			lineout+='%d'%len(set(doc))
			for word in set(doc):
				lineout  += ' %d'%Vocabulary.index(word)+':'+'%d'%doc.count(word)
			lineout+='\n'
			f.write(lineout)

	return True
def IsitCloseOnMap((lat1,lon1),(lat2,lon2),kms=2):
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
	if rad_km[0]**2 + rad_km[1]**2 < kms**2:
		return True
	else:
		return False


def ClusterGeoTag(place,n_clusters=5,radii=2,UsersUnique=False,timeWindow=60*15,useall=False,visualize=False,zoom=13,CatchImportantWord=False):
	"""
	   place        : Name of location which will be covered using heatmap
	   UsersUnique : Plot a tweet from a user only once
	   timeWindow   : The timecovered by one heatmap
	"""

	conn = MySQLdb.connect(**mysql_auth)
	
	#Fetch Tweets
	ListOfTweets = conn.cursor()
	collect_items =['text','user_id','screen_name','created_at','lat','lon','place']
	Grid          = locationbox[place]

	if useall:
		Streamingq  = "SELECT %s FROM streamerallTable WHERE ((lon >= %d AND lon <= %d) AND (lat>=%d AND lat<=%d))"%(','.join(collect_items),Grid[0],Grid[2],Grid[1],Grid[3])
	else:
		Streamingq  = "SELECT %s FROM streamer%sTable"%(','.join(collect_items),place)
	
	ListOfTweets.execute(Streamingq)
	
	#Time Vars
	time_start = time.gmtime()
	

	#TweetList and User List
	tw 			   = {}
	UniqueUids     = []
	

	#Conditional variable needs initialization
	UserUnique     = False
	
	if visualize:
		import smopy
		from PIL import ImageDraw
		ImageCount = 1

	
	if CatchImportantWord:
		from collections import Counter
		TOKEN = Tokenizer(lowercase=False,normalize=2,ignorestopwords=True).tokenize

	for item in ListOfTweets:

		#Tweet Dictionary
		item = dict(zip(collect_items,item))

		#Intialize with place name if GPS absent
		if item['lon']==0:
			continue
			try:
				item['lon'],item['lat'] = GetGeocode(item['place'])
			except IndexError:
				#print ('GoogleMap did not return for %s'%item['place'])
				continue

		#Block all tweets outside place grid
		if item['lon']==0 or not (item['lon']>=Grid[0] and item['lon']<=Grid[2] and item['lat']>=Grid[1] and item['lat']<=Grid[3]):
			continue

		#Intialize data point
		TEXT   = '@'+item['screen_name']+'\t _tweeted_ \t'+item['text']
		UID   = item['user_id']
		LOC   = (float(item['lat']),float(item['lon'])) 
		TIME  = time.strptime(item['created_at'],"%a %b %d %H:%M:%S +0000 %Y")
		del item

		#TimeWindow update
		shiftWindow = ((TIME<time_start) or (time.mktime(TIME)-time.mktime(time_start)>timeWindow)) and timeWindow!=-1
		
		if shiftWindow:
			#Write (long,lat) of Tweets collected to file
			localstart =  time.strftime('%d%b%HHR%MMN',time.localtime(time.mktime(time_start)+time.mktime(time.localtime())-time.mktime(time.gmtime())))
			localend   =  time.strftime('%d%b%HHR%MMN',time.localtime(time.mktime(TIME)+time.mktime(time.localtime())-time.mktime(time.gmtime())))

			if len(tw)!=0:
				#Generate Clusters Here with tw which contains list of LOCs
				km 		  = cluster.KMeans(n_clusters = n_clusters,n_jobs=-2)
				km.fit(np.vstack(tuple(tw['LOC'])))

				#Intialize TweetCluster dict to collect cluster tweets
				TweetCluster = {}
				for i in range(0,len(km.cluster_centers_)): TweetCluster[i]=[];
				
				#For visualize
				if visualize:
					TweetClusterLOC = {}
					for i in range(0,len(km.cluster_centers_)): TweetClusterLOC[i]=[];
				
				#Collect tweets for cluster
				for ct,cl in enumerate(km.labels_):
					if IsitCloseOnMap(tw['LOC'][ct],km.cluster_centers_[cl],kms=radii):
						TweetCluster[cl].append(tw['TEXT'][ct])
						if visualize:
							TweetClusterLOC[cl].append(tw['LOC'][ct])
					else:
						pass
				

				#Write cluster tweets to file 
				with open('retreiverData/ClusterCentersfrom%sto%s'%(localstart,localend),'wb') as f:
					for cl,cloc in enumerate(km.cluster_centers_):
						try:
							f.write('\n \nCluster %s : Location is %s'%(cl,GetPlaceName(cloc[0],cloc[1])))	
						except IndexError:
							f.write('\n \nCluster %s : Location (%f,%f)'%(cl,cloc[0],cloc[1]))
						
						"""Simple catching of common tweets which use TOP MOST USED/IMPORTANT WORD"""
						if CatchImportantWord:
							Cl_Vocab = []
							for item in TweetCluster[cl]: Cl_Vocab +=filter(lambda x: len(x)>2 and x.isalnum() and x not in ['USERNAME','URL','PHONENUMBER','TIME','NUMBER'],TOKEN(item.split('\t _tweeted_ \t')[-1]));
							if len(set(Cl_Vocab))==1:
									continue
							IMP_WORDS = [Counter(Cl_Vocab).most_common()[0][0],Counter(Cl_Vocab).most_common()[1][0]]
							f.write('\n Words which mention top used word in cluster : %s\n'%IMP_WORDS)
							counting = 0
							for item in TweetCluster[cl]:
								if (IMP_WORDS[0] in item.split('\t _tweeted_ \t')[-1]) or (IMP_WORDS[1] in item.split('\t _tweeted_ \t')[-1]):
									f.write('\n%d. %s'%(counting,item.encode('utf-8')))
									counting+=1

						f.write('\nPrinting tweets , Count = %s ..... \n%s'%(len(TweetCluster[cl]),'\n'.join(  [item.encode('utf-8') for item in TweetCluster[cl]])))

					#Remove TweetCluster from memory
					del TweetCluster

				#Plot on map
				if visualize:
					count = 0
					while count<10:
						try:
							Map = smopy.Map((Grid[1],Grid[0],Grid[3],Grid[2]),z=zoom)
							break
						except:
							count+=1
					del count

					draw = ImageDraw.Draw(Map.img)
					for cl,cloc in enumerate(km.cluster_centers_):
						if len(TweetClusterLOC[cl])==1:
							continue
						cx,cy = Map.to_pixels(cloc[0],cloc[1])	
						draw.polygon([Map.to_pixels(item[0],item[1]) for item in TweetClusterLOC[cl]],outline='#0000FF')
						try:
							draw.text((cx,cy),'%s. '%cl+GetPlaceName(cloc[0],cloc[1]).split(',')[0]+'(%s)'%len(TweetClusterLOC[cl]),fill='#000000')
						except IndexError:
							draw.text((cx,cy),"%s. [Cluster]"%cl+'(%s)'%len(TweetClusterLOC[cl]),fill='#000000')
					del draw
					Map.save_png('retreiverData/ClusterCentersfrom%sto%s.png'%(localstart,localend))
					Map.save_png('retreiverData/Img%s.png'%(((5-len(str(ImageCount)))*'0')+str(ImageCount)))
					ImageCount +=1
					del Map
					del TweetClusterLOC


			#Welcome new timeHashed
			time_start 		= TIME
			UniqueUids 		= [] 
			tw              = {}
			tw['LOC']		= []
			tw['TEXT']		= []


		if UID not in UniqueUids and UsersUnique:
			UniqueUids += [UID]
			UserUnique = True

		#Write LOC and TEXT values to tweet dictionary
		if (UsersUnique and UserUnique) or (not UsersUnique):
			tw['LOC'].append(LOC)
			tw['TEXT'].append(TEXT)
			UserUnique = False

	#Intialize TweetCluster dict to collect cluster tweets
	TweetCluster = {}
	for i in range(0,len(km.cluster_centers_)): TweetCluster[i]=[];
	
	#For visualize
	if visualize:
		TweetClusterLOC = {}
		for i in range(0,len(km.cluster_centers_)): TweetClusterLOC[i]=[];

	#Print last batch of clusters found
	localstart =  time.localtime(time.mktime(time_start)+time.mktime(time.localtime())-time.mktime(time.gmtime()))
	localend   =  time.localtime(time.mktime(TIME)+time.mktime(time.localtime())-time.mktime(time.gmtime()))

	#Generate Clusters Here with tw which contains list of LOCs
	km 		  = cluster.KMeans(n_clusters = n_clusters,n_jobs=-2)
	km.fit(np.vstack(tuple(tw['LOC'])))

	for ct,cl in enumerate(km.labels_):
		if IsitCloseOnMap(tw['LOC'][ct],km.clusters_centers_[cl]):
			TweetCluster[cl].append(tw['TEXT'][ct])
			if visualize:
				TweetClusterLOC[cl].append(tw['LOC'][ct])
		else:
			pass
		
	with open('retreiverData/ClusterCentersfrom%sto%s'%(localstart,localend),'wb') as f:
		for cl,cloc in enumerate(km.cluster_centers_):
			try:
				f.write('\n Cluster %s : Location is %s'%(cl,GetPlaceName(cloc[0],cloc[1])))	
			except IndexError:
				f.write('\n Cluster %s : Location (%f,%f)'%(cl,cloc[0],cloc[1]))

			"""Simple catching of common tweets which use TOP MOST USED/IMPORTANT WORD"""
			if CatchImportantWord:
				Cl_Vocab = []
				for item in TweetCluster[cl]: Cl_Vocab +=filter(lambda x: len(x)>2 and x.isalnum() and x not in ['USERNAME','URL','PHONENUMBER','TIME','NUMBER'],TOKEN(item.split('\t _tweeted_ \t')[-1]));
				if len(set(Cl_Vocab))==1:
					continue
				IMP_WORDS = [Counter(Cl_Vocab).most_common()[0][0],Counter(Cl_Vocab).most_common()[1][0]]
				f.write('\n Words which mention top used word in cluster : %s\n'%IMP_WORDS)
				counting = 0
				for item in TweetCluster[cl]:
					if (IMP_WORDS[0] in item.split('\t _tweeted_ \t')[-1]) or (IMP_WORDS[1] in item.split('\t _tweeted_ \t')[-1]):
						f.write('\n%d. %s'%(counting,item.encode('utf-8')))
						counting+=1

			f.write('\nPrinting tweets , Count = %s \n%s'%(len(TweetCluster[cl]),'\n'.join(  [item.encode('utf-8') for item in TweetCluster[cl]])))

		#Remove TweetCluster from memory
		del TweetCluster

	#Plot on map
	if visualize:
		Map = smopy.Map((Grid[1],Grid[0],Grid[3],Grid[2]),z=zoom)
		draw = ImageDraw.Draw(Map.img)
		for cl,cloc in enumerate(km.cluster_centers_):
			cx,cy = Map.to_pixels(cloc[0],cloc[1])	
			draw.polygon([Map.to_pixels(item[0],item[1]) for item in TweetClusterLOC[cl]],outline='#0000FF')
			try:
				draw.text((cx,cy),'%s. '%cl+GetPlaceName(cloc[0],cloc[1]).split(',')[0]+'(%s)'%len(TweetClusterLOC[cl]),fill='#000000')
			except IndexError:
				draw.text((cx,cy),"%s. [Cluster]"%cl+'(%s)'%len(TweetClusterLOC[cl]),fill='#000000')
		del draw
		Map.save_png('retreiverData/ClusterCentersfrom%sto%s.png'%(localstart,localend))
		Map.save_png('retreiverData/Img%s.png'%(((5-len(str(ImageCount)))*'0')+str(ImageCount)))
		del Map
		del TweetClusterLOC
			
	return True


def CreateHeatMap(place,fileid=True,UsersUnique=False,timeWindow=15*60,useall=False):
	"""
	   place        : Name of location which will be covered using heatmap
	   UsersUnique : Plot a tweet from a user only once
	   timeWindow   : The timecovered by one heatmap
	   begin_at_hour : Set the hour of starting the timeWindow , between (1-24)
	"""

	conn = MySQLdb.connect(**mysql_auth)
	
	#Fetch Tweets
	curr = conn.cursor()
	collect_items =['user_id','created_at','lat','lon','place']
	Grid          = locationbox[place]

	
	if useall:
		Streamingq  = "SELECT %s FROM streamerallTable WHERE ((lon >= %d AND lon <= %d) AND (lat>=%d AND lat<=%d))"%(','.join(collect_items),Grid[0],Grid[2],Grid[1],Grid[3])
	else:
		Streamingq  = "SELECT %s FROM streamer%sTable"%(','.join(collect_items),place)
	
	curr.execute(Streamingq)
	ListOfTweets = curr.fetchall()
	print len(ListOfTweets)
    #Time Vars
	time_start = time.gmtime()
	timeHashed = time.strftime("%x %X",time_start)

	#TweetList and User List
	tw 			   = {}		
	tw[timeHashed] = []
	UniqueUids     = []

	#HeatMap attributes
	scale = str(0.1)
	zoom  = str(14)
	if fileid:
		fid = 1

	for item in ListOfTweets:

		#Tweet Dictionary
		item = dict(zip(collect_items,item))

		if item['lon']==0:
			continue
		
		#Data points
		UID   = item['user_id']
		LOC   = (float(item['lat']),float(item['lon']))

		TIME  = time.strptime(item['created_at'],"%a %b %d %H:%M:%S +0000 %Y")


		#TimeWindow update
		shiftWindow = (TIME<time_start) or (time.mktime(TIME)-time.mktime(time_start)>timeWindow)
		if shiftWindow:
			#Write (long,lat) of Tweets collected to file
			localstart =  time.localtime(time.mktime(time_start)+time.mktime(time.localtime())-time.mktime(time.gmtime()))
			localend   =  time.localtime(time.mktime(TIME)+time.mktime(time.localtime())-time.mktime(time.gmtime()))
			if not fileid:
				filename = "HeatMapTweetsFROM%sTO%s"%(time.strftime('%d%b%HHR%MMN',localstart),time.strftime('%d%b%HHR%MMN',localend)) 
			else:
				filename = 'Img%s'%(((4-len(str(fid)))*'0')+str(fid))
				fid+=1
			with open('retreiverData/%s.coords'%filename,'wb') as f:
				for tweet in tw[timeHashed]:
					f.write('%f,%f \n'%tweet)

			#Check if picture file exists
			#try:
				#g = open('%s.png'%filename)
				
			#except IOError:
			subprocess.call(['python','heatmap.py','--csv=retreiverData/%s.coords'%filename,'-z',zoom,'-s', scale,'-W','6000','--extent=%f,%f,%f,%f'%(Grid[1],Grid[0],Grid[3],Grid[2]),'-R 25','--osm', '-o %s.png'%filename])
			
			#Welcome new timeHashed
			time_start 		= TIME
			timeHashed 		= time.strftime("%x %X",time_start)
			UniqueUids 		= [] 
			tw[timeHashed]  = []


		if UID not in UniqueUids:
			UniqueUids += [UID]
			UserUnique = True

		#Write Co-ordinates to tweet dictionary
		if (UsersUnique and UserUnique) or (not UsersUnique):
			tw[timeHashed].append(LOC)
	
	#Write last batch of tweets to file
	with open('retreiverData/HeatMapTweetsfrom%sto%s.coords'%(time.strftime('%d%b%HHR%MMN',time_start),time.strftime('%d%b%HHR%MMN',time.localtime())),'wb') as f:
				for tweet in tw[timeHashed]:
					f.write('%f,%f \n'%tweet)
	if not fileid:
		filename = 'HeatMapTweetsfrom%sto%s.coords'%(time.strftime('%d%b%HHR%MMN',time_start),time.strftime('%d%b%HHR%MMN',time.localtime()))
	else:
		filename = 'Img%s'%(((4-len(str(fid)))*'0')+str(fid))
		fid+=1
	
	subprocess.call(['python','heatmap.py','--csv=%s'%('retreiverData/%s'%filename),'-z',zoom,'-s', scale,'-H 3000','-W 3000','--extent=%f,%f,%f,%f'%(Grid[1],Grid[0],Grid[3],Grid[2]),'-R 25','--osm', '-o','%s.png'%(filename)])

	#Summarize retreiver action
	with open("retreiverData/HeatMapTweetRetreiveLOG.txt",'wb') as f:
		f.write("Simple report for tweets from %s\n , time in GMT"%place+"-"*10+"\n")
		for key in tw.keys():
			f.write("Timestamp : %s , No. of tweets collected =  %d;\n"%(key,len(tw[key])))

	return True


def UpdateEventWords(word,tweets):
	"""
		Receive tweets and cluster by geographical locations
	"""
	SpatialSigature = MakeSpatialSignature(word,tweets)


def MakeSpatialSignature(word,tweets,Grid):

	BANDWIDTH = 0.01           #Approximately Delta(1 lng or 1 lat) == 111 Km
	for key in tweets.keys(): tweets[key]['Grid'] = AssignGridID(tw['loc'])

	for g in G:
		pass

if __name__=='__main__':
	if sys.argv[1]=='lda':
		DataRetreiveForTopicDiscovery(sys.argv[2])
	elif sys.argv[1]=='heatmap':
		CreateHeatMap(sys.argv[2])
	elif sys.argv[1]=='cluster':
		ClusterGeoTag(sys.argv[2])
#ffmpeg -f image2 -r 1/5 -i image%05d.png -vcodec mpeg4 -y movie.mp4