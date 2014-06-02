import time
from tweetokenize import Tokenizer
from utils import gmt_to_local, local_to_gmt
import re
import cPickle
from collections import Counter

def multiple_replace(dict, text):
  # Create a regular expression  from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

  # For each match, look-up corresponding value in dictionary
  return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)

def is_sublist(a, b):
  if a.size == 0: return True
  if b.size == 0: return False
  return (b[:a.size] == a).all() or is_sublist(a, b[1:])

def get_vocabulary(tweet_text,tokenize=None,counter=True):
  tokenize  = T_Tokenizer(lowercase=False,normalize=2,ignorestopwords=True).tokenize if tokenize==None else tokenize
  #Build vocab
  vocab = []
  for text in tweet_text: vocab  += list(set(tokenize(text)));
  for item in vocab:
    if item.lower()=='boston' or item.lower()=='cambridge':
      vocab.remove(item)
  return vocab if counter==False else Counter(vocab)

class T_Tokenizer:

  def __init__(self,lowercase=False,normalize=2,ignorestopwords=True,phrases='Phrases.cPickle'):

    self.method = Tokenizer(lowercase=lowercase,normalize=normalize,ignorestopwords=ignorestopwords).tokenize
    self.Phrases= cPickle.load(open(phrases))

  def tokenize(self,text):
    ans = filter(lambda x: x is not None,[ i if (len(i)>1 and i[0]!='@' and i not in ['USERNAME','URL','PHONENUMBER','TIME','NUMBER'])\
                   else None for i in self.method(re.sub('#','',text))])
    ans = self.make_phrases(' '.join(ans))

    return ans

  def make_phrases(self,text,threshold=0):

    for phrase,value in self.Phrases.items():
      if phrase in text and value>threshold and ('-'+phrase not in text):
        text = text.replace(phrase,'-'.join(phrase.split()))

    return text.split()


class SearchTopic:

	# Template for TweetIter
	# TweetIter(collect_items=['screen_name','lat','lon','created_at','text'])

	def __init__(self,TweetIter,keywords,timerange=None,bbox=None,search_name=''):

		self.TweetIter     = TweetIter
		self.timerange     = timerange
		self.bbox          = bbox

		if self.timerange!=None or self.bbox!=None:
			self.TweetIter = SearchTweets(self.TweetIter,timerange,bbox,search_name).retreive()

		self.keywords      = keywords
		self.tweets        = []
		self.search_name   = search_name
		self.tokenize      = Tokenizer(lowercase=False,normalize=2,ignorestopwords=True).tokenize

	def retreive(self):

		for tw in self.TweetIter:

			if self.timerange!=None or self.bbox!=None:
				item                = {}
				item['screen_name'] = tw.split('\t _tweeted_ \t')[0]
				item['text']        = tw.split('\t _tweeted_ \t')[1].split('\t at_time \t')[0]
				item['created_at']  = tw.split('\t _tweeted_ \t')[1].split('\t at_time \t')[1]
				tw                  = item
			else:
				tw['created_at']  = gmt_to_local(time.strptime(tw['created_at'],"%a %b %d %H:%M:%S +0000 %Y"),make_string=True)

			words_in_tweet = filter(lambda x: x.isalnum() and x not in ['USERNAME','URL','PHONENUMBER','TIME','NUMBER'],self.tokenize(tw['text']))
			if any(w in self.keywords for w in words_in_tweet):
				TEXT   = '@'+tw['screen_name']+'\t _tweeted_ \t'+tw['text'] + '\t at_time \t' + tw['created_at']
				self.tweets.append(TEXT)
			else:
				pass

		return self.tweets

"""
Experiment with SearchTweets

1. tr   = ('Feb 03 19:52:00 EST 2014','Feb 04 19:52:00 EST 2014')
   bbox = [(42.366286,-71.063456),(42.366096,-71.060516)]
   search_name = TD_Garden
2. tr   = ('Feb 04 19:52:00 EST 2014','Feb 05 19:52:00 EST 2014')
   bbox = [(42.340273,-71.108392),(42.337703,-71.103157)]
   search_name = Winsor school
3. tr   = ('Feb 07 19:52:00 EST 2014','Feb 08 19:52:00 EST 2014')
   bbox = [(42.373701,-71.023388),(42.356801, -71.015406)]
   search_name = "BOS Terminal C Feb 07"
4. tr   = ('Feb 09 19:52:00 EST 2014','Feb 10 19:52:00 EST 2014')
   bbox = [(42.243672,-71.118956),(42.238556,-71.109471)]
   search_name = Curry College ground
5. tr   = ('Feb 14 19:52:00 EST 2014','Feb 15 19:52:00 EST 2014')
   bbox = [(42.376892,-71.120958),(42.371566,-71.113279)]
   search_name = 'Harvard Square and nearby park 14th Feb'


   Additional

   1. ((42.375835, -71.031903),
 (42.348818, -71.005725),
 (42.372157, -71.007184),
 (42.364486, -71.035422)) - BLA
   03Feb to 06Feb
   2. ((42.375835, -71.031903),
 (42.348818, -71.005725),
 (42.372157, -71.007184),
 (42.36702, -71.042031))  - MilkSt and StateSt
   03Feb to 06Feb


"""

class SearchTweets:

	def __init__(self,TweetIter,timerange,bbox,search_name=''):

		# Template for TweetIter
		# TweetIter(collect_items=['screen_name','lat','lon','created_at','text'])
		self.TweetIter     = TweetIter(collect_items=['text','created_at','screen_name','lat','lon'])
		if timerange==None and bbox==None:
			raise ValueError('No search criteria (time or bbox) provided')
		self.timerange     = (local_to_gmt(timerange[0]),local_to_gmt(timerange[1]))


		#if bbox==None or len(bbox)==2:
		self.bbox = bbox if bbox!=None else [(90.0,-180.0),(-90.0,+180.0)]
		#if len(bbox)==4:
		#	self.bbox = ( [max(bbox[0][0],bbox[2][0]),min(bbox[0][1],bbox[3][1])],[min(bbox[1][0],bbox[3][0]),max(bbox[1][1],bbox[2][1])])
		#else:
		#	raise ValueError('bbox is not of length 2 or 4')
		self.tweets        = []
		self.search_name   = search_name

	def retreive(self):

		min_time = None
		max_time = None

		for tw in self.TweetIter:

			TIME  = time.strptime(tw['created_at'],"%a %b %d %H:%M:%S +0000 %Y")

			if (tw['lat']<=self.bbox[0][0] and tw['lat']>=self.bbox[1][0]) and (tw['lon']>=self.bbox[0][1] and tw['lon']<=self.bbox[1][1] and TIME>=self.timerange[0]):
				TEXT   = '@'+tw['screen_name']+'\t _tweeted_ \t'+tw['text'] + '\t at_time \t' + gmt_to_local(TIME,make_string=True)
				self.tweets.append(TEXT)

				if min_time==None:
					min_time = TIME
				elif min_time>TIME:
					min_time = TIME

				if max_time==None:
					max_time = TIME
				elif max_time<TIME:
					max_time = TIME

			else:
				pass

			if TIME>self.timerange[1]:
				break
			else:
				pass

		#self.min_time = gmt_to_local(min_time,make_string=True)
		#self.max_time = gmt_to_local(max_time,make_string=True)

		return self.tweets
