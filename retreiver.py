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
from utils import locationbox, GetPlaceName, GetGeocode, gmt_to_local
from tweetokenize import Tokenizer
from sklearn import cluster

class TweetIterator:

	def __init__(self,db,place='Boston',collect_items=[],useall=False):
		self.place = place
		self.useall = useall
		#Database : db
		#1. streamer
		#2. streamer2
		#3. placename
		self.db 	= db
		conn = MySQLdb.connect(**mysql_auth)
		self.curr = conn.cursor()#(MySQLdb.cursors.SSCursor)
		Grid          = locationbox[self.place]
		#collect_items = ['user_id','place','lat','lon','created_at','text']
		self.collect_items = ['text'] if collect_items==[] else collect_items
		if self.useall:
			Streamingq  = "SELECT %s FROM streamerallTable WHERE ((lon >= %d AND lon <= %d) AND (lat>=%d AND lat<=%d))"%(','.join(self.collect_items),Grid[0],Grid[2],Grid[1],Grid[3])
		else:
			Streamingq  = "SELECT %s FROM %sTable"%(','.join(self.collect_items),self.db+self.place)
		self.curr.execute(Streamingq)


	def __iter__(self):
		"""dict iterator of items from table"""
		for t in self.curr:
			yield dict(zip(self.collect_items,t))


	def next(self):
		"""Returns next tweet"""
		return dict(zip(self.collect_items,self.curr.fetchone()))

class TweetSnap:

	def __init__(self,db,place="Boston",timeWindow=60*30,UsersUnique=True,Placename2Geocode=True):

		"""
	    place        : Name of location which will be covered using heatmap
	    UsersUnique  : Plot a tweet from a user only once
	    timeWindow   : The timecovered by one parzen window
		"""
		self.timeWindow = timeWindow
		self.UsersUnique  = UsersUnique
		self.Placename2Geocode = Placename2Geocode
		self.collect_items = ['user_id','place','screen_name','lat','lon','created_at','text']
		self.place   = place
		self.Grid = locationbox[self.place]
		self.ObjIter = TweetIterator(db,place,collect_items=self.collect_items)
		self.end     = 0

		#TimeStart - Stores current timePointer for the TweetSnap iterator
		self.time_start = time.gmtime()

		#DataVariables to store with each SnapShot
		self.tw 			   = {'LOC':[],'TEXT':[],'PLACE':[],'CREATED_AT':[],'SCREEN_NAME':[]}
		#UIDS which tweeted in current SnapShot
		self.UniqueUids     = []

		#Catch and throw first snap to start chain of snap
		temp = self.next()
		del temp

	def move_on(self,timeWindow):
		"""Method to skip time when iterating from the aws rdbms"""

		#Initialize variables with class's current status
		time_start = self.time_start

		#Conditional variable needs initialization
		shiftWindow    = 0

		while 1:

			item  = self.ObjIter.next()
			TIME  = time.strptime(item['created_at'],"%a %b %d %H:%M:%S +0000 %Y")
			del item

			#TimeWindow update
			shiftWindow = ((TIME<self.time_start) or (time.mktime(TIME)-time.mktime(self.time_start)>timeWindow)) and timeWindow!=-1

			if shiftWindow:
					#Change time_start and return
					self.time_start = TIME
					return None

	def next(self):

		#Initialize variables with class's current status
		time_start = self.time_start
		tw = self.tw
		UniqueUids = self.UniqueUids

		#Conditional variable needs initialization
		UserUnique     = False
		shiftWindow    = 0

		while 1:

			try:

				item  = self.ObjIter.next()

				#Tweets with no GPS are assigned place_name geocodes
				if item['lon']==0 and self.Placename2Geocode:
						item['lon'],item['lat'] = GetGeocode(item['place'])


				#Additional filter blocks all tweets outside place grid
				if not (item['lon']>=self.Grid[0] and item['lon']<=self.Grid[2] and item['lat']>=self.Grid[1] and item['lat']<=self.Grid[3]):
					continue

				#Unfold tweet into its item variables and destroy tweet
				TEXT   = item['text']
				PLACE = item['place']
				UID   = item['user_id']
				LOC   = (float(item['lat']),float(item['lon']))
				TIME  = time.strptime(item['created_at'],"%a %b %d %H:%M:%S +0000 %Y")
				CREATED_AT = gmt_to_local(TIME,make_string=True)
				SCREEN_NAME = item['screen_name']
				del item

				#TimeWindow update
				shiftWindow = (TIME<self.time_start) or (((time.mktime(TIME)-time.mktime(self.time_start)>self.timeWindow)) and self.timeWindow!=-1)

			except TypeError:
				self.end = 1
				shiftWindow=True

			if shiftWindow:
				#Capture a new timeWindow
				if len(self.tw['LOC'])!=0:
					#Create timestamps for start(stop) if timeWindow captured
					localstart =  time.strftime('%d%b%HHR%MMN',time.localtime(time.mktime(self.time_start)+time.mktime(time.localtime())-time.mktime(time.gmtime())))
					localend   =  time.strftime('%d%b%HHR%MMN',time.localtime(time.mktime(TIME)+time.mktime(time.localtime())-time.mktime(time.gmtime())))

					LOCs = self.tw['LOC']
					TEXTs = self.tw['TEXT']
					PLACEs = self.tw['PLACE']
					CREATED_ATs = self.tw['CREATED_AT']
					SCREEN_NAMEs = self.tw['SCREEN_NAME']

					#Welcome new timeHashed
					self.time_start 		= TIME
					self.UniqueUids 		= []
					self.tw['LOC']		    = []
					self.tw['TEXT']			= []
					self.tw['PLACE']    	= []
					self.tw['CREATED_AT'] 	= []
					self.tw['SCREEN_NAME']  = []

					#Yield
					return {'LOC':LOCs,'TEXT':TEXTs,'TimeWindow':[localstart,localend],'PLACE':PLACEs,'CREATED_AT':CREATED_ATs,'SCREEN_NAME':SCREEN_NAMEs}


			#Check if UID has tweeted in this timeWindow before
			if UID not in self.UniqueUids and self.UsersUnique:
				self.UniqueUids += [UID]
				UserUnique = True

			#Write LOC and TEXT values to tweet dictionary
			if (self.UsersUnique and UserUnique) or (not self.UsersUnique):
				self.tw['LOC'].append(LOC)
				self.tw['TEXT'].append(TEXT)
				self.tw['PLACE'].append(PLACE)
				self.tw['CREATED_AT'].append(CREATED_AT)
				self.tw['SCREEN_NAME'].append(SCREEN_NAME)
				UserUnique = False

	def __iter__(self):

		#Initialize variables with class's current status
		time_start = self.time_start
		tw = self.tw
		UniqueUids = self.UniqueUids

		#Conditional variable needs initialization
		UserUnique     = False

		for item in self.ObjIter:

			#Tweets with no GPS are assigned place_name geocodes
			if item['lon']==0:
					item['lon'],item['lat'] = GetGeocode(item['place'])

			#Block all tweets outside place grid
			if item['lon']==0 or not (item['lon']>=self.Grid[0] and item['lon']<=self.Grid[2] and item['lat']>=self.Grid[1] and item['lat']<=self.Grid[3]):
				continue

			#Unfold tweet into its item variables and destroy tweet
			TEXT   = item['text']
			UID   = item['user_id']
			PLACE = item['place']
			LOC   = (float(item['lat']),float(item['lon']))
			TIME  = time.strptime(item['created_at'],"%a %b %d %H:%M:%S +0000 %Y")
			CREATED_AT = gmt_to_local(TIME,make_string=True)
			SCREEN_NAME = item['screen_name']
			del item

			#TimeWindow update
			shiftWindow = ((TIME<self.time_start) or (time.mktime(TIME)-time.mktime(self.time_start)>self.timeWindow)) and self.timeWindow!=-1

			if shiftWindow:
				#Capture a new timeWindow
				if len(self.tw['LOC'])!=0:
					#Create timestamps for start(stop) if timeWindow captured
					localstart =  time.strftime('%d%b%HHR%MMN',time.localtime(time.mktime(self.time_start)+time.mktime(time.localtime())-time.mktime(time.gmtime())))
					localend   =  time.strftime('%d%b%HHR%MMN',time.localtime(time.mktime(TIME)+time.mktime(time.localtime())-time.mktime(time.gmtime())))
					#Yield
					yield {'LOC':self.tw['LOC'],'TEXT':self.tw['TEXT'],'TimeWindow':[localstart,localend],'PLACE':self.tw['PLACE'],'CREATED_AT':self.tw['CREATED_AT'],'SCREEN_NAME':self.tw['SCREEN_NAME']}

				#Welcome new timeHashed
				self.time_start 		= TIME
				self.UniqueUids 		= []
				self.tw['LOC']			= []
				self.tw['TEXT']			= []
				self.tw['PLACE']    	= []
				self.tw['CREATED_AT'] 	= []
				self.tw['SCREEN_NAME']  = []


			#Check if UID has tweeted in this timeWindow before
			if UID not in self.UniqueUids and self.UsersUnique:
				self.UniqueUids += [UID]
				UserUnique = True

			#Write LOC and TEXT values to tweet dictionary
			if (self.UsersUnique and UserUnique) or (not self.UsersUnique):
				self.tw['LOC'].append(LOC)
				self.tw['TEXT'].append(TEXT)
				self.tw['PLACE'].append(PLACE)
				self.tw['CREATED_AT'].append(CREATED_AT)
				self.tw['SCREEN_NAME'].append(SCREEN_NAME)
				UserUnique = False

		if len(self.tw['LOC'])>0:
			#Create timestamps for start(stop) if timeWindow captured
			localstart =  time.strftime('%d%b%HHR%MMN',time.localtime(time.mktime(self.time_start)+time.mktime(time.localtime())-time.mktime(time.gmtime())))
			localend   =  time.strftime('%d%b%HHR%MMN',time.localtime(time.mktime(TIME)+time.mktime(time.localtime())-time.mktime(time.gmtime())))

			yield {'LOC':self.tw['LOC'],'TEXT':self.tw['TEXT'],'TimeWindow':[localstart,localend],'PLACE':self.tw['PLACE'],'CREATED_AT':self.tw['CREATED_AT'],'SCREEN_NAME':self.tw['SCREEN_NAME']}


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

if __name__=='__main__':
	if sys.argv[1]=='lda':
		DataRetreiveForTopicDiscovery(sys.argv[2])
	elif sys.argv[1]=='heatmap':
		CreateHeatMap(sys.argv[2])
	elif sys.argv[1]=='cluster':
		ClusterGeoTag(sys.argv[2])
#ffmpeg -f image2 -r 1/5 -i image%05d.png -vcodec mpeg4 -y movie.mp4
