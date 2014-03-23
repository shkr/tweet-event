# -*- encoding: utf-8 -*-

from retreiver import TweetSnap
from Clustering import GMM_clustering
from tweetokenize import Tokenizer
from collections import Counter
from SearchUtils import T_Tokenizer, multiple_replace
from utils import GetPlaceName, location
import math
import folium


def get_vocabulary(tweet_text,tokenize=None,count_unique_items=False):
	tokenize  = T_Tokenizer(lowercase=False,normalize=2,ignorestopwords=True).tokenize if tokenize==None else tokenize
	#Build vocab
	vocab = []
	for text in tweet_text: vocab  += tokenize(text) if count_unique_items==False else list(set(tokenize(text)));
	return Counter(vocab)

class Base_Buzz:

	def __init__(self,clustering_algo,place='Boston',rate_threshold=100):

		#Sample use case:
		#0. TI = TweetSnap(timeWindow=60*10,UsersUnique=False)
		#1. GMM_clustering(SnapIter=TI,components=range(6,16),visualize=False)
		#2. Placename_clustering(SnapIter=TI,visualize=False)

		self.Clustering = clustering_algo
		self.Clustering.next()
		self.place = place
		self.tokenize = T_Tokenizer(lowercase=False,normalize=2,ignorestopwords=True).tokenize
		self.rate_threshold = rate_threshold

	def evolving_list(self):

		#Skip Snaps with less than threshold tweets
		while len(self.Clustering.Snap['LOC'])<self.rate_threshold:
			print 'Found only %d tweets skipping  %s'%(len(self.Clustering.Snap['LOC']),self.Clustering.Snap['TimeWindow'])
			self.Clustering.next()


		go = raw_input('Look at next time snap %s ?'%self.Clustering.Snap['TimeWindow'])

		if go in ['yes','y',1,'go']:

			#Build clusters from tweetSnap
			labels   = {}
			self.Clustering.build_clusters()

			#Collect indices of different clusters in dict
			for k,l in enumerate(self.Clustering.labels): labels.setdefault(l,[]).append(k);

			#Make vocabulary of text from tokenized tweet
			vocabulary = get_vocabulary([ text for text in self.Clustering.Snap['TEXT'] ],self.tokenize)

			#Search for events in tweetSnap
			for event in self.buzz(labels,vocabulary):
				print 'Event :'+event.summary().encode('ascii','ignore')

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

			map_1.create_map(path='folium_map_%s.html'%self.Clustering.Snap['TimeWindow'][0])
			del map_1
			#self.Clustering.next()

	def buzz(self,labels,tweets,vocabulary):
		"""Template method described in sub-class"""
		pass

class HotWords(Base_Buzz):

	def buzz(self,labels,vocabulary,at_least=2):

		Top20 = vocabulary.most_common(20)
		print 'Top 20 is \n:'
		print Top20

		for label in labels.keys():
			event  = self.words_in_tweet(labels[label],Top20,at_least)
			if event.text!=None:
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
		event    = TweetEvent()

		for k in ids:

			#Type1
			if len(set(filter(lambda x: x in Words,self.tokenize(self.Clustering.Snap['TEXT'][k]))))>=threshold:
				if event.text==None or (len(self.Clustering.Snap['TEXT'][k])<len(event.text) and len(set(filter(lambda x: x in Words,self.tokenize(self.Clustering.Snap['TEXT'][k]))))>len(event.property)):
					event.text = self.Clustering.Snap['TEXT'][k]
					event.time = self.Clustering.Snap['CREATED_AT'][k]
					event.location = self.Clustering.Snap['LOC'][k]
					event.property = set(filter(lambda x: x in Words,self.tokenize(self.Clustering.Snap['TEXT'][k])))
					event.teller = self.Clustering.Snap['SCREEN_NAME'][k]

			#Type2
			#Filtered+=filter(lambda x: x in Words,self.tokenize(tw['text']))
			#if len(set(Filtered))>=threshold:
			#	example = tw if (len(tw['text'])<len(example) or event==False) else example
			#	event = True

		return event

class TweetEvent:

	def __init__(self,time=None,text=None,location=None,property=None,teller=None):
		self.time = time
		self.text = text
		self.location = location
		self.property = property
		self.teller   = teller

	def set_time(self,time):
		self.time  = time

	def set_location(self,location):
		self.location = location

	def set_text(self,text):
		self.text = text

	def set_teller(self,teller):
		self.teller = teller

	def set_property(self,properties):
		self.property = properties

	def summary(self):
		return unicode(self.teller + ' tweeted : ' + multiple_replace({'\"':'\\"','\'':"\\'"},self.text) + ' at time '+ self.time + ' near '+ GetPlaceName(self.location[0],self.location[1],zoom=8)[:-15] + ' because '+'/'.join(self.property))


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

class semantichashing(Base_Buzz):
	pass
