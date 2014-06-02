# -*- encoding: utf-8 -*-

from retreiver import TweetSnap
from Clustering import GMM_clustering, Placename_clustering
from tweetokenize import Tokenizer
from collections import Counter
from SearchUtils import T_Tokenizer, multiple_replace, get_vocabulary
from utils import GetPlaceName, location
from visualization import GeographicalEntropy as Locality
import math
import pandas as pd
import cPickle
import folium
import matplotlib.pyplot as plt


def get_vocabulary(tweet_text,tokenize=None,counter=True):
	tokenize  = T_Tokenizer(lowercase=False,normalize=2,ignorestopwords=True).tokenize if tokenize==None else tokenize
	#Build vocab
	vocab = []
	for text in tweet_text: vocab  += list(set(tokenize(text)));
	for item in vocab:
		if item.lower()=='boston' or item.lower()=='cambridge':
			vocab.remove(item)
	return vocab if counter==False else Counter(vocab)

class Base_Buzz:

	def __init__(self,clustering_algo=None,place='Boston',rate_threshold=100):

		#Sample use case:
		#0. TI = TweetSnap(timeWindow=60*10,UsersUnique=False)
		#1. GMM_clustering(SnapIter=TI,components=range(6,16),visualize=False)
		#2. Placename_clustering(SnapIter=TI,visualize=False)

		self.Clustering = clustering_algo if clustering_algo!=None else Placename_clustering(TweetSnap(db='streamer2'),visualize=False)
		self.Clustering.next()
		self.place = place
		self.tokenize = T_Tokenizer(lowercase=False,normalize=2,ignorestopwords=True).tokenize
		self.rate_threshold = rate_threshold
		self.ResultDict = pd.DataFrame(columns=['word','event_time','location','discovered_time','summary'])


	def evolving_list(self):

		#Skip Snaps with less than threshold tweets
		while len(self.Clustering.Snap['LOC'])<self.rate_threshold:
			print 'Found only %d tweets skipping  %s'%(len(self.Clustering.Snap['LOC']),self.Clustering.Snap['TimeWindow'])
			self.Clustering.next()


		#go = raw_input('Look at next time snap %s ?'%self.Clustering.Snap['TimeWindow'])

		#if go in ['yes','y',1,'go']:
		while not self.Clustering.SnapIter.end:
			#Build clusters from tweetSnap
			labels   = {}
			self.Clustering.build_clusters()

			#Collect indices of different clusters in dict
			for k,l in enumerate(self.Clustering.labels): labels.setdefault(l,[]).append(k);

			#Make vocabulary of text from tokenized tweet
			vocabulary = get_vocabulary([ text for text in self.Clustering.Snap['TEXT'] ],self.tokenize)

			#Search for events in tweetSnap
			for event in self.buzz(labels,vocabulary):
				self.ResultDict = self.ResultDict.append(event,ignore_index=True)
				print event

			#self.Clustering.next()

	def folium_map(self):
		"""Generates a leaflet map with eventful tweets on the map"""

		#Skip Snaps with less than threshold tweets
		while len(self.Clustering.Snap['LOC'])<self.rate_threshold:
			print 'Found only %d tweet(s) skipping for timeWindow %s'%(len(self.Clustering.Snap['LOC']),self.Clustering.Snap['TimeWindow'])
			self.Clustering.next()

		#lat,lon =  location[self.place]['latitude'],location[self.place]['longitude']
		lat, lon = (42.3606249, -71.0591155)

		go = raw_input('Look at next time snap %s ?'%self.Clustering.Snap['TimeWindow'])

		if go in ['yes','y',1,'go']:

			#Build clusters from tweetSnap
			labels   = {}
			self.Clustering.build_clusters()

			#Collect indices of different clusters in dict
			for k,l in enumerate(self.Clustering.labels): labels.setdefault(l,[]).append(k);

			#Make vocabulary of text from tokenized tweet
			vocabulary = get_vocabulary(self.Clustering.Snap['TEXT'],self.tokenize)

			map_1 = folium.Map(location=[lat,lon], zoom_start=8,
												tiles='Stamen Terrain')

			#Search for events in tweetSnap
			for event in self.buzz(labels,vocabulary):
				popup = event.summary().encode('ascii','ignore')
				print 'Event :'+popup
				map_1.simple_marker(location=[event.location[0],event.location[1]], popup=popup)

			map_1.create_map(path='folium_map_%s.html'%self.Clustering.Snap['TimeWindow'])
			del map_1
			#self.Clustering.next()

	def buzz(self,labels,tweets,vocabulary):
		"""Template method described in sub-class"""
		pass

	def SetStart(self,TIME_START):
		if isinstance(TIME_START,str):
			TIME_START  = time.gmtime(time.mktime(time.strptime(TIME_START,"%d %b %H:%M %Z %Y")))
		TIME_DIFF   = time.mktime(TIME_START)  - time.mktime(self.Clustering.SnapIter.time_start)
		if TIME_DIFF>0:
			self.Clustering.SnapIter.move_on(TIME_DIFF)

class HotWords(Base_Buzz):

	def buzz(self,labels,vocabulary,at_least=2):

		Top20 = vocabulary.most_common(20)
		#print 'Top 20 is \n:'
		#print Top20

		for label in labels.keys():
			event  = self.words_in_tweet(labels[label],Top20,at_least)
			if event['summary']!=None:
				yield event

	def words_in_tweet(self,ids,Words,threshold=2,tokenize=None):

		if len(Words[0])==1 and type(Words[0])==unicode:
			pass
		elif len(Words[0])==2 and type(Words[0][0])==unicode:
			Words = [ W[0] for W in Words ]
		else:
			raise TypeError("Words list is not a tuple or list of words")

		Filtered = []

		#Return variables
		event     = {'word':None,'event_time':None,'discovered_time':None,'location':None,'summary':None}

		for k in ids:
			#Type1
			HotWordSize = len(set(filter(lambda x: x in Words,self.tokenize(self.Clustering.Snap['TEXT'][k]))))
			if HotWordSize>=threshold:
				if None in event.values() or len(self.Clustering.Snap['TEXT'][k])/float(HotWordSize)>len(self.tokenize(event['summary']))/float(len(event['word'])):
					event['summary'] = self.Clustering.Snap['TEXT'][k]
					event['event_time'] = self.Clustering.Snap['CREATED_AT'][k]
					event['location'] = GetPlaceName(self.Clustering.Snap['LOC'][k][0],self.Clustering.Snap['LOC'][k][1])
					event['word'] = list(set(filter(lambda x: x in Words,self.tokenize(self.Clustering.Snap['TEXT'][k]))))
					event['discovered_time'] = self.Clustering.Snap['TimeWindow']

		return event

			#Type2
			#Filtered+=filter(lambda x: x in Words,self.tokenize(tw['text']))
			#if len(set(Filtered))>=threshold:
			#	example = tw if (len(tw['text'])<len(example) or event==False) else example
			#	event = True

		return event

class TweetEvent:

	def __init__(self,Time=None,text=None,location=None,property=None,teller=None):
		self.Time = Time
		self.text = text
		self.location = location
		self.property = property
		self.teller   = teller

	def set_time(self,time):
		self.Time  = time

	def set_location(self,location):
		self.location = location

	def set_text(self,text):
		self.text = text

	def set_teller(self,teller):
		self.teller = teller

	def set_property(self,properties):
		self.property = properties

	def summary(self,df=None):

		print 'Hotwords (%s) found at %s and confirmed at %s'%(','.join(self.property) ,self.Time,self.Time)
		print 'Summary :'
		print '%s reported at time %s: %s \n\n'%(self.teller,self.Time,self.text)

class TF_IDF:

	def __init__(self,SearchWords,Documents):

		self.SearchWords  = set(SearchWords)
		self.IDF          = TF_IDF.InverseDocumentFrequencyVector(SearchWords,Documents)
		self.SearchVector = TF_IDF.MultiplyDict(TF_IDF.TermFrequencyVector(SearchWords,SearchWords),self.IDF)

	def score(self,Doc):

		DocVector = TF_IDF.MultiplyDict(TF_IDF.TermFrequencyVector(self.SearchWords,Doc),self.IDF)

		if not any([v for v in DocVector.values()]):
			return 0.0
		else:
			return TF_IDF.CosineSimilarity(self.SearchVector,DocVector)


	@staticmethod
	def MultiplyDict(Dict1,Dict2):

		if set(Dict1.keys())!=set(Dict2.keys()):
			raise ValueError('Incompatible dimensions of Dict1 and Dict2')

		return {key:Dict1[key]*Dict2[key] for key in Dict1.keys()}


	@staticmethod
	def CosineSimilarity(SearchVector,DocVector):

		if set(SearchVector.keys())!=set(DocVector.keys()):
			raise ValueError('Incompatible dimensions of SearchVector and DocVector')

		SVdotDV = 0.0

		ABS_SV  = math.sqrt(sum([v**2 for v in SearchVector.values()]))
		ABS_DV  = math.sqrt(sum([v**2 for v in DocVector.values()]))

		if ABS_DV==0.0:
			return 0.0

		for key in SearchVector.keys():
			SVdotDV += (SearchVector[key]*DocVector[key])

		return SVdotDV/(ABS_SV*ABS_DV)



	@staticmethod
	def InverseDocumentFrequencyVector(SearchWords,Documents):

		DocumentFrequency = {key: 0.0 for key in SearchWords}
		NoOfDocuments     = len(Documents)

		for doc in Documents:
			for key in filter(lambda x:x in doc,SearchWords):
					DocumentFrequency[key]+=1

		return { key:1+math.log(NoOfDocuments/DocumentFrequency[key]) if DocumentFrequency[key]!=0 else 1 for key in SearchWords }

	@staticmethod
	def TermFrequencyVector(SearchWords,Doc):

		TermFrequency = {key: 0.0 for key in SearchWords}
		AddThis = Counter(filter(lambda x:x in SearchWords,Doc))
		for key in AddThis.keys(): TermFrequency[key]+=AddThis[key]
		return TermFrequency

class TF_IDF_Search(Base_Buzz):

	def buzz(self,labels,vocabulary,SearchLength=10):

		SearchWords = [ w[0] for w in vocabulary.most_common(SearchLength) ]
		print 'Top words are :'
		print SearchWords

		for ids in labels.values():

			documents = [ self.tokenize(self.Clustering.Snap['TEXT'][k]) for k in ids ]
			TF_IDF_OBJ     = TF_IDF(SearchWords,documents)
			max_score = 0
			doc_id    = None

			for d,doc in enumerate(documents):

				if TF_IDF_OBJ.score(doc)>max_score:
					doc_id = d
					max_score = TF_IDF_OBJ.score(doc)

			event = TweetEvent()
			event.text = self.Clustering.Snap['TEXT'][ids[doc_id]]
			event.time = self.Clustering.Snap['CREATED_AT'][ids[doc_id]]
			event.location = self.Clustering.Snap['LOC'][ids[doc_id]]
			event.property = set(filter(lambda x: x in SearchWords,self.tokenize(self.Clustering.Snap['TEXT'][ids[doc_id]])))
			event.teller = self.Clustering.Snap['SCREEN_NAME'][ids[doc_id]]

			yield event

class KModesKMeans(Base_Buzz):
	pass

#http://jmlr.org/papers/volume5/lewis04a/a11-smart-stop-list/english.stop
class word2vector(Base_Buzz):
	pass

from visualization import Count
from visualization import GeographicalEntropy as Locality
from scipy.stats import expon
import time
import numpy as np

class NewsWorthyWords:

	def __init__(self,db,timeWindow=60*10,**kwargs):

		print "COLLECTING TWEETS...."
		self.TS = TweetSnap(db=db,timeWindow = timeWindow,Placename2Geocode=False)
		print "COLLECTION OVER...."

		#Variables
		self.SnapStack = []
		self.Candidates= {}
		self.Volume		= []

		#Constants
		self.delta    				 = 1.5
		self.enoughSamples     = 15.0
		self.SnapLim           = 6
		self.StopNewsWords     = ['Boston', 'day', 'time', 'love', 'today', 'Boston-MA']
		#Set TIME_FRAME
		self.SetStart(kwargs.get("TIME_START",time.gmtime(0)))

		#Storage variables for analysis
		self.Storage = []
		self.StorageDict = pd.DataFrame(columns=['word','Poisson','LocalEntropy','GlobalEntropy','start_time','event'])
		self.ResultDict = pd.DataFrame(columns=['word','event_time','location','discovered_time','summary'])

		#Classifier
		self.matrix_w, self.scaler, self.clf = cPickle.load(open('SVClassifier.Store'))

		#Verbosity - 1. Print all messages 2. Print less messages 3. .....
		self.VerboseLevel = kwargs.get('VerboseLevel',1)

	def verbose(self,text,level=1):
		if level<self.VerboseLevel:
			return
		else:
			print text

	def SetStart(self,TIME_START):
		if isinstance(TIME_START,str):
			TIME_START  = time.gmtime(time.mktime(time.strptime(TIME_START,"%d %b %H:%M %Z %Y")))
		TIME_DIFF   = time.mktime(TIME_START)  - time.mktime(self.TS.time_start)
		if TIME_DIFF>0:
			self.TS.move_on(TIME_DIFF)

	def run(self):

		while not self.TS.end:

			#Update SnapStack
			if len(self.SnapStack)==self.SnapLim:
				self.SnapStack = self.SnapStack[1:]
				self.Volume    = self.Volume[1:]
			self.SnapStack.append(self.TS.next())
			self.Volume.append(Count(self.SnapStack[-1]))

			#Update Candidates origin snap as timeWindow has shifted right
			for key,val in self.Candidates.items():
				if val==-self.SnapLim:
					self.Candidates.pop(key)
					self.verbose('This %s word has been removed because it never received enough samples'%key)
				else:
					self.Candidates[key]=val-1


			print('Latest timeWindow %s'%self.SnapStack[-1]['TimeWindow'],2)
			#Algorithm
			self.verbose('Print looking for new events which happened in this timeWindow',2)
			self.FindNewEvent()

			self.verbose('Print confirming old/new candidate events which have not been published')
			self.ConfirmEvent()


			if self.Candidates.keys() !=[]: self.verbose('EventCandidates: %s'%self.Candidates.keys(),2);



	def TotalVolume(self,word,Volume):

		total = 0.0
		k		 = 0

		while k < len(Volume):
			if word in Volume[k].keys():
				total += Volume[k][word]
			k+=1

		return total if total!=0 else 1


	def FindNewEvent(self):

		for word,count in self.Volume[-1].items():

				#Is word count gaussian noise or signal ?
				wordHistory = [float(vol[word]) for vol in self.Volume[:-1] if word in vol.keys() ]
				mean        =  np.mean(wordHistory) if len(wordHistory)>0 else 1
				var				 =  np.std(wordHistory) if len(wordHistory)>=5 else 1

				std_score = (count - mean)/(2*var)

				if std_score>=self.delta and (word not in self.StopNewsWords):

					self.verbose('This %s is not gaussian noise with standard_score = %f '%(word,std_score))
					if word not in self.Candidates.keys() or (self.Volume[self.Candidates[word]][word]<count):
						self.Candidates[word] = -1

	def ConfirmEvent(self):

		for word,no in self.Candidates.items():

			wordHistory = [float(vol.get(word,0.0)) for vol in self.Volume[no:]]
			self.verbose('Confirming candidate Newsword : %s at time = %s with samples=%d and Snapno=%d'%(word,self.SnapStack[no]['TimeWindow'][0],sum(wordHistory),no),2)
			if sum(wordHistory)>=self.enoughSamples:
				self.verbose('This %s word has enough samples from tweets to calculate scores (Poisson,LocalEntropy,StandardDeviation)'%(word),2)
				#Poisson
				Poisson = self.FitPoissonDistribution(word,no)
				#Global and Local Entropy
				GlobalEntropy,LocalEntropy = self.FitSpatialEntropy(word,no)

				#Classifier
				#Define feature vector
				X    = np.array([Poisson,LocalEntropy,GlobalEntropy],dtype=np.float64)
				#Apply Scaler
				X_sc = self.scaler.transform(X)
				#Apply Orthogonality
				X_tr = X_sc.dot(self.matrix_w)
				#Classify new transformed feature vector
				Flag = self.clf.predict(X_tr)[0]

				if Flag==1:
					start_time  = self.SnapStack[no]['TimeWindow'][0]
					confirmed_time = self.SnapStack[-1]['TimeWindow'][0]
					SampleSet   = self.ReportEventQueue(word,no)
					print       "Newsword (%s) at %s confirmed at %s\n"%(word,start_time,confirmed_time)
					print       "Summary : "
					summary     = []
					for user,created_at,tweet,loc in SampleSet:
						print "%s reported at time %s near %s: %s"%(user,created_at,GetPlaceName(loc[0],loc[1]),tweet)
						#summary.append("%s reported at time %s near %s: %s"%(user,created_at,tweet,GetPlaceName(loc[0],loc[1]))
						summary.append([user,created_at,tweet,loc])

					event =  {'word':word,'event_time':start_time,'location':GetPlaceName(np.mean([item[3][0] for item in summary]),np.mean([item[3][1] for item in summary])),'discovered_time':confirmed_time,'summary':'\n'.join([ "%s reported at time %s near %s: %s"%(item[0],item[1],GetPlaceName(item[3][0],item[3][1]),item[2]) for item in summary])}
					print event
					self.ResultDict = self.ResultDict.append(event,ignore_index=True)
					self.Candidates.pop(word)

				else:
					continue



				#Store Data for post-classification
				self.StorageDict = self.StorageDict.append({'word':word,'Poisson':Poisson,'LocalEntropy':LocalEntropy,'GlobalEntropy':GlobalEntropy,'start_time':start_time,'event':event},ignore_index=True)


				#Manual Classifier
				# if flag in ['1','y','yes']:
				# 		print 'This %s word count resembles poisson distribution with lambda=%f'%(word,Lambda)
				# 		self.ReportEventQueue(word,no)
				# 		self.Candidates.pop(word)
				# else:
				# 		print 'This %s word count does not resembles poisson distribution with lambda=%s'%(word,Lambda)

	def FitSpatialEntropy(self,word,no):

		k = no
		tokenize  = T_Tokenizer().tokenize
		#Store locations
		ALLLOC = []
		WORDLOC = []

		while k<0:

			ALLLOC += self.SnapStack[k]['LOC']
			for order,text in enumerate(self.SnapStack[k]['TEXT']):
				if word in tokenize(text):
					WORDLOC.append(self.SnapStack[k]['LOC'][order])

			k+=1

		#Choose Cluster of max ALLLOC, C*
		MakeCluster 	 	= GMM_clustering()
		MakeCluster.Snap = {'LOC':ALLLOC}
		MakeCluster.build_clusters()
		WORDLABELS       = Counter([MakeCluster.labels[ALLLOC.index(LOC)] for LOC in WORDLOC])

		#Global entropy
		GLOBAL_COUNTER = Counter(MakeCluster.labels)
		G_D_pq		   = 0.0
		for cl,number in WORDLABELS.items():
				G_D_pq	+= -1*(number/float(GLOBAL_COUNTER[cl]))*np.log2(number/float(GLOBAL_COUNTER[cl]))
				#G_D_pq	+= -1*((number/sum(WORDLABELS))/float(GLOBAL_COUNTER[cl]/sum(GLOBAL_COUNTER)))*np.log2(number/float(GLOBAL_COUNTER[cl]))


		C_Star					 = WORDLABELS.most_common(1)[0][0]
		C_Star_LOC       = [ ALLLOC[No] for No,label in filter(lambda (enum,x): x==C_Star,enumerate(MakeCluster.labels)) ]
		C_Star_WORD_LOC  = [LOC for LOC in filter(lambda x:x in C_Star_LOC,WORDLOC)]

		#Find D(p||q) of word inside C*
		del MakeCluster
		MakeLocalCluster 	 	= GMM_clustering(components=range(2,8))
		MakeLocalCluster.Snap = {'LOC':C_Star_LOC}
		MakeLocalCluster.build_clusters()

		WORD_LOCAL_COUNTER    = Counter([MakeLocalCluster.labels[C_Star_LOC.index(LOC)] for LOC in C_Star_WORD_LOC])
		LOCAL_ALL_COUNTER		 = Counter( MakeLocalCluster.labels )
		L_D_pq		   = 0.0
		for cl,number in WORD_LOCAL_COUNTER.items():
			  L_D_pq	+= -1*(number/float(LOCAL_ALL_COUNTER[cl]))*np.log2(number/float(LOCAL_ALL_COUNTER[cl]))
				#L_D_pq	+= -1*((number/sum(WORD_LOCAL_COUNTER.values()))/float(LOCAL_ALL_COUNTER[cl]/sum(LOCAL_ALL_COUNTER.values())))*np.log2(number/float(LOCAL_ALL_COUNTER[cl]))

		return [G_D_pq,L_D_pq]

	def FitStdDev(self,word,no):

		k = no
		tokenize  = T_Tokenizer().tokenize
		#Store locations
		WORDLOC= []

		while k<0:
			for order,text in enumerate(self.SnapStack[k]['TEXT']):
				if word in tokenize(text):
					WORDLOC.append(self.SnapStack[k]['LOC'][order])
			k+=1

		return np.std(WORDLOC)

	def FitPoissonDistribution(self,word,no):

		tokenize  = T_Tokenizer().tokenize

		k = no
		Times = []

		ApproxTimes = []

		wordHistory = [vol.get(word,0) for vol in self.Volume[no:]]

		#Store all tweet_times with word in current snap and known history
		while k<0:

			approx = time.mktime(time.strptime(self.SnapStack[k]['TimeWindow'][0]+'2014EDT',"%d%b%HHR%MMN%Y%Z"))
			count  = self.Volume[k].get(word,0)
			ApproxTimes+=[approx]*count

			for order,text in enumerate(self.SnapStack[k]['TEXT']):
				if word in tokenize(text):
					Times.append(\
									time.mktime(time.strptime(self.SnapStack[k]['CREATED_AT'][order],"%d %b %H:%M:%S %Y")))
			k+=1

	  #Calculate time-intervals
		TimeIntervals = [Time-min(Times) for Time in Times]
		ApproxTimeIntervals = sorted([ approx-min(ApproxTimes) for approx in ApproxTimes])
		TimeIntervals.sort()
		self.verbose('Have a look at TimeIntervals(1) and ApproxTimeIntervals(2) and LogLikelihood(3)')
		self.verbose('(1) %s'%TimeIntervals)
		self.verbose('(2) %s'%ApproxTimeIntervals)

		ApproxTimeIntervals = Counter(ApproxTimeIntervals)

		#Calculate ML_Lmbda
		_lmbda      = float(len(TimeIntervals))/sum(TimeIntervals)
		# if sum(ApproxTimeIntervals)!=0:
		# 	_lmbda      = float(len(ApproxTimeIntervals))/sum(ApproxTimeIntervals)
		# else:
		# 	_lmbda      = float(len(TimeIntervals))/sum(TimeIntervals)

		#Calculate Variance for given samples
		# _R2         = 1/_lmbda**2

		#Likelihood calculation and plotting (optional)

		# MaxLogLikelihood
		# _LgLd 			= -1*sum([np.log(_lmbda*np.exp(-_lmbda*x)) for x in TimeIntervals])
		# print '(3)',_LgLd
		#
		# #Simulate a expon_RV with fitted _lmbda
		# _rv         = expon(scale=1/_lmbda)
		#
		# #Plot pdf of counts from _rv and known
		# fig = plt.figure()
		# ax  = fig.add_subplot(111)
		# ax.plot(sorted(ApproxTimeIntervals.keys()),[_rv.cdf(x+600)-_rv.cdf(x) for x in sorted(ApproxTimeIntervals.keys())],'r-',label='fitted')
		# ax.plot(sorted(ApproxTimeIntervals.keys()),[float(ApproxTimeIntervals[key])/sum(wordHistory) for key in sorted(ApproxTimeIntervals.keys()) ],'b-'\
		# 				,label='empirical estimate')
		#
		# plt.legend()
		#
		# #save figure
		# fig.savefig('%s.png'%word)
		#
		# gmm  = GMM_clustering(components=range(4,15))
		# gmm.Snap = self.SnapStack[no]
		# gmm.build_clusters()
		#
		# #flag = raw_input("Fitted curve for %s stored should flag=1 or not with lambda=%f and locality=%f"%(word,_lmbda,Locality(self.SnapStack[no],gmm.labels,word)))
		# plt.close(fig)

		return _lmbda

	def ReportEventQueue(self,word,no,SampleLim=3):

		#Find clusters at start point of event
		gmm  = GMM_clustering(components=range(4,15))
		gmm.Snap = self.SnapStack[no]
		gmm.build_clusters()
		Labels = []
		tokenize  = T_Tokenizer().tokenize
		for k,text in enumerate(gmm.Snap['TEXT']):
			if word in tokenize(text):
				Labels.append(gmm.labels[k])
		Labels = Counter(Labels)
		#Find cluster where word was most common
		StarLabel = Labels.most_common(1)[0][0]

		SampleSet = []
		#Print a tweet from that cluster
		for k,text in enumerate(gmm.Snap['TEXT']):
			if gmm.labels[k] == StarLabel and word in tokenize(text):
				SampleSet.append((gmm.Snap['SCREEN_NAME'][k],gmm.Snap['CREATED_AT'][k],text,gmm.Snap['LOC'][k]))
			if len(SampleSet)>=SampleLim:
				break

		return SampleSet
