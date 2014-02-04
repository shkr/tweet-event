import subprocess
import sys
import os
import time
import json
import MySQLdb
from t_auth import *
from geocode import locationbox
class Cluster:
	def __init__(self,created_at,EventWord,SpatialSigature):
		self.created_at = created_at
		self.lifetime   = 1
		self.Y 			= [EventWord]
		self.SC         = SpatialSigature


def Tokenize(output,tags,regex):
	"""tags  : POS Tagged words to return
	   regex : Matching regex expressions to return
	"""
	WORDS  = output[0].split()
	TAGS   = output[1].split()
	KeyWords  = []

	for i,tag in enumerate(TAGS):
		KeyWords+=[WORDS[i]] if tag in tags else []

	return KeyWords

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
	   list of [longitude,latitude]"""
	print place
	place = place.replace(' ','%20')
	out = json.loads(subprocess.check_output('curl "http://maps.googleapis.com/maps/api/geocode/json?address=%s&sensor=false"'%place,stderr=open(os.devnull, 'w'),shell=True))['results'][0]['geometry']['location']
	return [out['lng'],out['lat']]

def TweetRetreive(time_start=None,timeWindow=60,time_end=None):
	"""
	   time_start : TimeStruct
	   timeWindow : Units are Seconds
	   time_end   : TimeStruct

	   From time_start to time_end send list of tweets at time steps timeWindow,
	   in the form of a tuple (tweetid(tid),KeyWords(W),user_id(uid),[latitude,longitude](l),Time(t)).
	"""

	conn = MySQLdb.connect(**mysql_auth)
	
	curr = conn.cursor()
	collect_items =['tid','lat','lon','created_at','text','user_id','user_location']
	Grid          =[-69,43,-73,41]
	StreamingBoston = "SELECT %s FROM streamerBostonTable WHERE (user_location LIKE ', MA') OR ((lon < %d AND lon > %d) AND (lat<%d AND lat>%d))"%(','.join(collect_items),Grid[0],Grid[2],Grid[1],Grid[3])
	curr.execute(StreamingBoston)
	ListOfTweets = curr.fetchall()
    
	KeyTags    = ['Z','$','G','M','^','#','@']  #Removed V
	KeyRegex   = ['A.M.','AM','digit am','P.M.','PM','digit pm']
	time_start = time.gmtime()
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
		except subprocess.CalledProcessError as e:
			print (e)
			continue
		UID   = item['user_id']
		LOC   = [float(item['lon']),float(item['lat'])] if (item['lon']!=0) else GetGeocode(item['user_location'])
		TIME  = time.strptime(item['created_at'],"%a %b %d %H:%M:%S +0000 %Y")
		
		#TimeWindow update
		shiftWindow = (Time<time_start) or (time.mktime(TIME)-time.mktime(time_start)>timeWindow)
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

    #Time Vars
	time_start = time.gmtime()
	timeHashed = time.strftime("%x %X",time_start)

	#TweetList and User List
	tw 			   = {}		
	tw[timeHashed] = []
	UniqueUids     = []

	for item in ListOfTweets:

		#Tweet Dictionary
		item = dict(zip(collect_items,item))

		#Pass if lat,lon information does not exist
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
			with open("retreiverData/HeatMapTweetsfrom%s.coords"%time.strftime('%a%d%b%Y',time_start),'wb') as f:
				for tweet in tw[timeHashed]:
					f.write('%f,%f \n'%tweet)

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
	with open("retreiverData/HeatMapTweetsfrom%s.coords"%time.strftime('%a%d%b%Y',time_start),'wb') as f:
				for tweet in tw[timeHashed]:
					f.write('%f,%f \n'%tweet)

	#Summarize retreiver action
	with open("retreiverData/HeatMapTweetRetreive.log",'wb') as f:
		f.write("Simple report for tweets from %s\n"%place+"-"*10+"\n")
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
	#TweetTrain()
	CreateHeatMap(sys.argv[1])
