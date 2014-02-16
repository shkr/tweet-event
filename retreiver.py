# -*- coding: utf-8 -*-

import subprocess
import sys
import os
import time
import json
import MySQLdb
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

		self.tokenize = Tokenizer(lowercase=False,normalize=1,ignorestopwords=True).tokenize

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

def GetGeocode(place):
	"""Makes call to GoogleMaps API and returns 
	   longitude,latitude for place name"""

	place = '%20'.join(place.replace(',',' ').split())
	try:	
		out = json.loads(subprocess.check_output('curl "http://maps.googleapis.com/maps/api/geocode/json?address=%s&sensor=false"'%place,stderr=open(os.devnull, 'w'),shell=True))['results'][0]['geometry']['location']
	except IndexError:
		print place
		print json.loads(subprocess.check_output('curl "http://maps.googleapis.com/maps/api/geocode/json?address=%s&sensor=false"'%place,stderr=open(os.devnull, 'w'),shell=True))['results']
		raise IndexError	
	return [out['lng'],out['lat']]

def GetPlaceName(lat,lon):
	"""Makes call to GoogleMaps API and returns 
	   human readable address for lat,lon"""
	out = json.loads(subprocess.check_output('curl "https://maps.googleapis.com/maps/api/geocode/json?latlng=%s,%s&sensor=false"'%(lat,lon),stderr=open(os.devnull, 'w'),shell=True))['formatted_address']
	return out

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

def ClusterGeoTag(place,UsersUnique=True,timeWindow=-1,useall=False):
	"""
	   place        : Name of location which will be covered using heatmap
	   UsersUnique : Plot a tweet from a user only once
	   timeWindow   : The timecovered by one heatmap
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
	
	#Time Vars
	time_start = time.gmtime()

	#TweetList and User List
	tw 			   = []	
	UniqueUids     = []
	TimeWindow     = []

	for item in ListOfTweets:

		#Tweet Dictionary
		item = dict(zip(collect_items,item))
		
		#Data points
		if item['lon']==0:
			continue

		UID   = item['user_id']
		LOC   = (float(item['lat']),float(item['lon'])) 
		TIME  = time.strptime(item['created_at'],"%a %b %d %H:%M:%S +0000 %Y")

		#TimeWindow update
		shiftWindow = ((TIME<time_start) or (time.mktime(TIME)-time.mktime(time_start)>timeWindow)) and timeWindow!=-1
		
		if shiftWindow:
			#Write (long,lat) of Tweets collected to file
			localstart =  time.localtime(time.mktime(time_start)+time.mktime(time.localtime())-time.mktime(time.gmtime()))
			localend   =  time.localtime(time.mktime(TIME)+time.mktime(time.localtime())-time.mktime(time.gmtime()))
			TimeWindow.append([localstart,localend])

			#Generate Clusters Here with tw which contains list of LOCs
			ms 		  = cluster.MeanShift()
			ms.fit(np.vstack(tuple(tw)))
			labels    = ms.labels_
			cluster_centers = ms.cluster_centers_
			labels_unique 	= np.unique(labels)
			n_clusters 		= len(labels_unique)
			geo_centers		= [ GetPlaceName(lat,lon) for lat,lon in cluster_centers]
			
			print("\n timeframe start: %s and end: %s"%(localstart,localend))
			print("number of estimated clusters : %d" % n_clusters_)
			print("the geotag centers : %s"%'\n'.join(geo_centers))

			#Welcome new timeHashed
			time_start 		= TIME
			UniqueUids 		= [] 
			tw              = []


		if UID not in UniqueUids:
			UniqueUids += [UID]
			UserUnique = True

		#Write Co-ordinates to tweet dictionary
		if (UsersUnique and UserUnique) or (not UsersUnique):
			tw.append(LOC)

	#Print last batch of clusters found
	localstart =  time.localtime(time.mktime(time_start)+time.mktime(time.localtime())-time.mktime(time.gmtime()))
	localend   =  time.localtime(time.mktime(TIME)+time.mktime(time.localtime())-time.mktime(time.gmtime()))
	TimeWindow.append([localstart,localend])

	#Generate Clusters Here with tw which contains list of LOCs
	ms 		  = cluster.MeanShift()
	ms.fit(np.vstack(tuple(tw)))
	labels    = ms.labels_
	cluster_centers = ms.cluster_centers_
	labels_unique 	= np.unique(labels)
	n_clusters 		= len(labels_unique)
	geo_centers		= [ GetPlaceName(lat,lon) for lat,lon in cluster_centers]
	
	print("\n timeframe start: %s and end: %s"%(localstart,localend))
	print("number of estimated clusters : %d" % n_clusters_)
	print("the geotag centers : %s"%'\n'.join(geo_centers))

	return True


def CreateHeatMap(place,UsersUnique=False,timeWindow=24*60*60,useall=False):
	"""
	   place        : Name of location which will be covered using heatmap
	   UsersUnique : Plot a tweet from a user only once
	   timeWindow   : The timecovered by one heatmap
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
	scale = 0.1 

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
			filename = "HeatMapTweetsFROM%sTO%s"%(time.strftime('%d%b%HHR%MMN',localstart),time.strftime('%d%b%HHR%MMN',localend)) 
			with open('retreiverData/%s.coords'%filename,'wb') as f:
				for tweet in tw[timeHashed]:
					f.write('%f,%f \n'%tweet)

			#Check if picture file exists
			#try:
				#g = open('%s.png'%filename)
				
			#except IOError:
			subprocess.call(['python','heatmap.py','--csv=retreiverData/%s.coords'%filename,'-s %s'%(scale),'-W','3000','-H','3000','--extent=%f,%f,%f,%f'%(Grid[1],Grid[0],Grid[3],Grid[2]),'-R 25','--osm', '-o %s.png'%filename])
			
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
	
	subprocess.call(['python','heatmap.py','--csv=%s'%('retreiverData/HeatMapTweetsfrom%sto%s.coords'%(time.strftime('%d%b%HHR%MMN',time_start),time.strftime('%d%b%HHR%MMN',time.localtime()))),'-s %s'%(scale),'-H 3000','-W 3000','--extent=%f,%f,%f,%f'%(Grid[1],Grid[0],Grid[3],Grid[2]),'-R 25','--osm', '-o %s.png'%("HeatMapTweetsfrom%sto%s.png"%(time.strftime('%d%b%HHR%MMN',time_start),time.strftime('%d%b%HHR%MMN',time.localtime()))) ])

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
