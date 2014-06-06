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
    self.QueueStack     = []
    self.Candidates    = {}
    self.Vocabulary		= []

    #Constants
    self.delta    				 = 3  #GaussianDistortion
    self.MinWordSamples     = 15.0 #Has to be greater than 8 See SetFeatureTable method for this restriction

    self.QueueLim           = 6  #MaximumQueueLimit

    self.StopNewsWords     = ['Boston', 'day', 'time', 'love', 'today', 'Boston-MA']  #Default StopWordList

    #Set TIME_FRAME
    self.SetStart(kwargs.get("TIME_START",time.gmtime(0)))

    #Storage variables for analysis
    self.FeatureDict = pd.DataFrame(columns=['word','Poisson','LocalEntropy','GlobalEntropy','start_time','event'])
    self.ResultDict = pd.DataFrame(columns=['word','event_time','location','discovered_time','summary'])

    #Classifier
    self.matrix_w, self.scaler, self.clf = cPickle.load(open('SVClassifier.Store'))

    #Verbosity - 1. Print all messages 2. Print less messages 3. .....
    self.OnlyMessage = kwargs.get('OnlyMessage',0)

  def message(self,text):
    if self.OnlyMessage:
      print text
    else:
      pass

  def SetStart(self,TIME_START):
    if isinstance(TIME_START,str):
      TIME_START  = time.gmtime(time.mktime(time.strptime(TIME_START,"%d %b %H:%M %Z %Y")))
    TIME_DIFF   = time.mktime(TIME_START)  - time.mktime(self.TS.time_start)
    if TIME_DIFF>0:
      self.TS.move_on(TIME_DIFF)

  def run(self):

    while not self.TS.end:

      #Update QueueStack
      if len(self.QueueStack)==self.QueueLim:
        self.QueueStack = self.QueueStack[1:]
        self.Vocabulary    = self.Vocabulary[1:]
      self.QueueStack.append(self.TS.next())
      self.Vocabulary.append(Count(self.QueueStack[-1]))

      #Update Candidates origin snap as timeWindow has shifted right
      for key,val in self.Candidates.items():
        if val==-self.QueueLim:
          self.Candidates.pop(key)
          self.message('This %s word has been removed because it never received enough samples'%key)
        else:
          self.Candidates[key]=val-1


      print('Latest timeWindow %s'%self.QueueStack[-1]['TimeWindow'])
      #Algorithm
      #1. Add to candidates list
      self.FilterWords()
      #1.1
      if self.TableON==1 and len(self.Candidates.keys())!=0:
        self.SetFeatureTable()
      #2. Find news-word in candidate list
      self.ConfirmEvent()
      #Status of candidate list
      if self.Candidates.keys() !=[]: self.message('EventCandidates: %s'%self.Candidates.keys());

  def FilterWords(self):

    for word,count in self.Vocabulary[-1].items():

        #Is word count gaussian noise or signal ?
        wordHistory = [float(vol[word]) for vol in self.Vocabulary[:-1] if word in vol.keys() ]
        mean        =  np.mean(wordHistory) if len(wordHistory)>0 else 1
        variance 	 =  np.std(wordHistory) if len(wordHistory)>=5 else 1

        Z_score = (count - mean)/variance

        if Z_score>=self.delta and (word not in self.StopNewsWords):

          self.message('This %s is not gaussian noise with standard_score = %f '%(word,Z_score))
          if word not in self.Candidates.keys() or (self.Vocabulary[self.Candidates[word]][word]<count):
            self.Candidates[word] = -1

  def ConfirmEvent(self):

    for word,no in self.Candidates.items():

      wordHistory = [float(vol.get(word,0.0)) for vol in self.Vocabulary[no:]]

      self.message('Confirming candidate Newsword : %s at time = %s with samples=%d and Queueno=%d'%(word,self.QueueStack[no]['TimeWindow'][0],sum(wordHistory),no))

      if sum(wordHistory)>=self.MinWordSamples:

        self.message('This %s word has enough samples from tweets to calculate scores (Poisson,LocalEntropy,StandardDeviation)'%(word))

        #Poisson
        Poisson = self.FitPoissonDistribution(word,no)
        #Global and Local Entropy
        GlobalEntropy,LocalEntropy = self.FitSpatialEntropy(word,no)

        #Poisson, GlobalEntropy, LocalEntropy = self.GetFeatures(word,no)

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
          start_time  = self.QueueStack[no]['TimeWindow'][0]
          confirmed_time = self.QueueStack[-1]['TimeWindow'][0]
          SampleSet   = self.ReportEventQueue(word,no)
          print       "Newsword (%s) at %s confirmed at %s\n"%(word,start_time,confirmed_time)
          print       "Summary : "
          summary     = []
          for user,created_at,tweet,loc in SampleSet:
            print "%s reported at time %s near %s: %s"%(user,created_at,loc,tweet)
            #summary.append("%s reported at time %s near %s: %s"%(user,created_at,tweet,GetPlaceName(loc[0],loc[1]))
            summary.append([user,created_at,tweet,loc])

          event =  {'word':word,'event_time':start_time,'location':GetPlaceName(np.mean([item[3][0] for item in summary]),np.mean([item[3][1] for item in summary])),'discovered_time':confirmed_time,'summary':'\n'.join([ "%s reported at time %s near %s: %s"%(item[0],item[1],item[3],item[2]) for item in summary])}
          print event
          self.ResultDict = self.ResultDict.append(event,ignore_index=True)
          self.Candidates.pop(word)

        else:
          continue



        #Store Data for post-classification
        self.FeatureDict = self.FeatureDict.append({'word':word,'Poisson':Poisson,'LocalEntropy':LocalEntropy,'GlobalEntropy':GlobalEntropy,'start_time':start_time,'event':event},ignore_index=True)


        #Manual Classifier
        # if flag in ['1','y','yes']:
        # 		print 'This %s word count resembles poisson distribution with lambda=%f'%(word,Lambda)
        # 		self.ReportEventQueue(word,no)
        # 		self.Candidates.pop(word)
        # else:
        # 		print 'This %s word count does not resembles poisson distribution with lambda=%s'%(word,Lambda)

  def SetFeatureTable(self):

    tokenize  = T_Tokenizer().tokenize
    self.Feature = {}
    k = -len(self.QueueStack)

    #Store locations
    ALL_LOC  = []
    WORD_LOC = {}
    C_Star_LOC = {}
    C_Star_Labels = {}

    #Get List of locations of all tweets Collected : ALL_LOC
    #Get List of locations where "word" appears in tweets posted after it was declared as an event
    #    : WORD_LOC[word]
    while k<0:
       ALL_LOC += self.QueueStack[k]['LOC']
       for order,text in enumerate(self.QueueStack[k]['TEXT']):
         for word,no in self.Candidates.items():
           if word in tokenize(text) and order>=no:
             WORD_LOC.setdefault(word,[]).append(self.QueueStack[k]['LOC'][order])

       k+=1

    #Global Clustering
    MakeCluster 	 	= GMM_clustering(components=range(3,8))
    MakeCluster.Snap = {'LOC':ALL_LOC}
    MakeCluster.build_clusters()
    #Input : ALL_LOC & Output : Global labels for locations of tweets
    GLOBAL_LABELS    = Counter(MakeCluster.labels)

    #Local Clustering for each cluster in lists
    for C_Star in GLOBAL_LABELS.keys():

      #Input : C_Star_LOC ; All tweet locations withing C_Star cluster
      C_Star_LOC[C_Star]    = [ ALL_LOC[No] for No,label in filter(lambda (enum,x): x==C_Star,enumerate(MakeCluster.labels)) ]
      if len(C_Star_LOC[C_Star])>=(self.MinWordSamples/3.0):
        MakeLocalCluster 	 	= GMM_clustering(components=range(2,min(8,int(self.MinWordSamples/3))))
        MakeLocalCluster.Snap = {'LOC':C_Star_LOC[C_Star]}
        MakeLocalCluster.build_clusters()

        #Output : C_Star_Labels ; Labels for All tweet locations withing C_Star cluster
        C_Star_Labels[C_Star] = MakeLocalCluster.labels

    #Set GlobalEntropy and LocalEntropy for each Candidate word
    for word,no in self.Candidates.items():

      #Global entropy
      #1. Initialize to 0
      G_D_pq 		   = 0.0
      #2. List of all non-zero counts for global clusters where 'word' appears in tweet
      WORD_LABELS   = Counter([MakeCluster.labels[ALL_LOC.index(LOC)] for LOC in WORD_LOC[word]])
      #3. Calculate entropy by summing up over all clusters
      for cl,number in WORD_LABELS.items():
          G_D_pq	+= -1*(number/float(GLOBAL_LABELS[cl]))*np.log2(number/float(GLOBAL_LABELS[cl]))
          #G_D_pq	+= -1*((number/sum(WORDLABELS))/float(GLOBAL_COUNTER[cl]/sum(GLOBAL_COUNTER)))*np.log2(number/float(GLOBAL_COUNTER[cl]))

      #Local entropy
      #1. Most populated cluster with 'word'
      C_Star					 = WORD_LABELS.most_common(1)[0][0]
      #2. List of all non-zero counts for global clusters where 'word' appears in tweet
      WORD_LOCAL_LABELS     = Counter([C_Star_Labels[C_Star][C_Star_LOC[C_Star].index(LOC)] for LOC in WORD_LOC[word] if LOC in C_Star_LOC[C_Star]])
      LOCAL_LABELS 		     = Counter( C_Star_Labels[C_Star] )
      #3. Calculate entropy by summing up over all local clusters
      L_D_pq		   = 0.0
      for cl,number in WORD_LOCAL_LABELS.items():
          L_D_pq	+= -1*(number/float(LOCAL_LABELS[cl]))*np.log2(number/float(LOCAL_LABELS[cl]))
          #L_D_pq	+= -1*((number/sum(WORD_LOCAL_COUNTER.values()))/float(LOCAL_ALL_COUNTER[cl]/sum(LOCAL_ALL_COUNTER.values())))*np.log2(number/float(LOCAL_ALL_COUNTER[cl]))

      self.Feature[word] = [G_D_pq,L_D_pq,self.GetPoissonRate(word,no)]

  def FitSpatialEntropy(self,word,no):

    if self.TableON:
      return [self.Feature[word][0],self.Feature[word][1]]

    k = no
    tokenize  = T_Tokenizer().tokenize
    #Store locations
    ALLLOC = []
    WORDLOC = []

    while k<0:

      ALLLOC += self.QueueStack[k]['LOC']
      for order,text in enumerate(self.QueueStack[k]['TEXT']):
        if word in tokenize(text):
          WORDLOC.append(self.QueueStack[k]['LOC'][order])

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


  def GetPoissonRate(self,word,no):

    tokenize  = T_Tokenizer().tokenize

    k = no
    Times = []
    ApproxTimes = []

    #Store all tweet_times with word in current snap and known history
    while k<0:

      approx = time.mktime(time.strptime(self.QueueStack[k]['TimeWindow'][0]+'2014EDT',"%d%b%HHR%MMN%Y%Z"))
      count  = self.Vocabulary[k].get(word,0)
      ApproxTimes+=[approx]*count

      for order,text in enumerate(self.QueueStack[k]['TEXT']):
        if word in tokenize(text):
          Times.append(\
                  time.mktime(time.strptime(self.QueueStack[k]['CREATED_AT'][order],"%d %b %H:%M:%S %Y")))
      k+=1

    #Calculate time-intervals
    TimeIntervals       = sorted([Time-min(Times) for Time in Times])
    ApproxTimeIntervals = sorted([ approx-min(ApproxTimes) for approx in ApproxTimes])

    #Calculate ML_Lmbda
    if sum(ApproxTimeIntervals)!=0:
      _lmbda      = float(len(ApproxTimeIntervals))/sum(ApproxTimeIntervals)
    else:
      _lmbda      = float(len(TimeIntervals))/sum(TimeIntervals)

    return _lmbda



  def ReportEventQueue(self,word,no,SampleLim=3):

    #Find clusters at start point of event
    gmm  = GMM_clustering(components=range(4,15))
    gmm.Snap = self.QueueStack[no]
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
