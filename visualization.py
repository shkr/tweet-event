from retreiver import TweetSnap, TweetIterator
from Clustering import Placename_clustering
from Clustering import GMM_clustering
from sklearn import preprocessing
from SearchUtils import T_Tokenizer, get_vocabulary
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import prettyplotlib as ppl
import pandas as pd
from matplotlib.table import Table
import numpy as np
import smopy
from utils import gmt_to_local, locationbox, GetGeocode
import time
import operator
from collections import Counter

# Set up some better defaults for matplotlib
import brewer2mpl
from matplotlib import rcParams

#colorbrewer2 Dark2 qualitative color table
dark2_colors = brewer2mpl.get_map('Dark2', 'Qualitative', 7).mpl_colors

rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 150
rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.facecolor'] = 'white'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'white'
rcParams['patch.facecolor'] = dark2_colors[0]
rcParams['font.family'] = 'StixGeneral'

def clustering_distribution(ClusterType=Placename_clustering):

  #Table
  #Col           Row
  #ClusterNames  No.OfTweets

  #BarChart
  #X-Axis        Y-Axis
  #ClusterNames  No.OfTweets

  #OSM-Map
  #ClusterName -> Color -> Print tweets as a colored point
  #Draw boundary of the cluster

  # print "COLLECTING TWEETS...."
  # if ClusterType==Placename_clustering:
  #   TS = TweetSnap(timeWindow=-1,Placename2Geocode=False)
  # else:
  #   TS = TweetSnap(timeWindow = -1)
  # print "COLLECTION OVER...."


  # Clustering = ClusterType(TS,visualize=False)
  # Clustering.next()
  # Clustering.build_clusters()

  TS = TweetIterator(collect_items=['place'])
  PLACES = TS.curr.fetchall()
  ClusterLabels = [str(place[0]) for place in PLACES]

  LabelCounter  = Counter(ClusterLabels)
  LabelSet      = [ item[0] for item in LabelCounter.most_common() ]

  #Print Results
  # gs      = gridspec.GridSpec(1,2,width_ratios=[1,2])
  #
  # fig1    = plt.figure(figsize=(24,12),dpi=200)
  #
  #
  # ax0     = fig1.add_subplot(gs[0,0])
  # ax1     = fig1.add_subplot(gs[0,1])
  #
  # rowLabels = [ '%d. %s'%(no+1,label) for no,label in enumerate(LabelSet)]
  # cellText  = [ LabelCounter[label] for label in LabelSet ]
  # rowLabels.reverse()
  # cellText.reverse()
  # for y, label, text in zip(range(len(cellText)),rowLabels,cellText):
  #   ax0.text(0.0001,(float(y+1)/80.0),s='%s : %s'%(label,text),size=12)
  #
  # ax0.xaxis.set_visible(False)
  # ax0.yaxis.set_visible(False)
  #
  # ppl.bar(ax1,range(len(LabelSet)),[LabelCounter[label] for label in LabelSet],grid='y')
  # ax1.set_xticks(np.arange(len(LabelSet)))
  # ax1.set_xticklabels(LabelSet,rotation='vertical')
  #
  # fig1.savefig('clustering_distribution.png',dpi=200,bbox_inches="tight")
  # plt.close(fig1)

  print 'here'

  #Plot numbers on map
  #Presentation-related lists
  visual_patterns = ['xb','xg','xr','xc','xm','xy','xk','xw']

  Grid = locationbox['Boston']

  #Plot results
  fig2 = plt.figure(dpi=200)
  mpl  = fig2.add_subplot(111)
  map_ = smopy.Map([Grid[1],Grid[0],Grid[3],Grid[2]],z=13)
  mpl.set_xticks([])
  mpl.set_yticks([])
  mpl.grid(False)
  mpl.set_xlim(0, map_.w)
  mpl.set_ylim(map_.h, 0)
  mpl.axis('off')

  mpl  = map_.show_mpl(mpl)

  for k,label in enumerate(LabelSet):
    try:
      lon,lat = GetGeocode(label)
      if lon<Grid[0] or lon>Grid[2] or lat<Grid[1] or lat>Grid[3] or LabelCounter[label]<10:
        raise ValueError('OutofBounds')
    except:
      print label
      continue
    x,y = map_.to_pixels(lat,lon)
    mpl.plot(x,y,'%s'%visual_patterns[k%len(visual_patterns)])
    mpl.text(x+1, y, '%d at %s'%(LabelCounter[label],label), fontsize=8,color=visual_patterns[k%len(visual_patterns)][1])

  fig2.savefig('clustering_distribution_map.png',dpi=200,bbox_inches="tight")


def print_vocabulary_report(db,scale=60*20,**kwargs):

  print "COLLECTING TWEETS...."
  TS = TweetSnap(db=db,timeWindow = scale,Placename2Geocode=False)
  print "COLLECTION OVER...."

  TIME_START = kwargs.get("TIME_START",time.gmtime(0))
  TIME_END   = kwargs.get("TIME_END",time.gmtime(time.time()))
  HotWordSize = kwargs.get("HotWordSize",8)

  if isinstance(TIME_START,str):
    TIME_START  = time.gmtime(time.mktime(time.strptime(TIME_START,"%d %b %H:%M %Z %Y")))
  if isinstance(TIME_END,str):
    TIME_END    = time.gmtime(time.mktime(time.strptime(TIME_END,"%d %b %H:%M %Z %Y")))

  TIME_DIFF   = time.mktime(TIME_START)  - time.mktime(TS.time_start)

  if TIME_DIFF>0:
    TS.move_on(TIME_DIFF-scale)

  volume = []
  HotWordsList = []
  ColorGradient = {}
  TweetCountDict    = {}
  TimeList      = []

  while (TS.time_start<TIME_END and not TS.end):

    #Capture nextSnap and initialize time_start of next snap
    snap = TS.next()
    timeWindow = gmt_to_local(TS.time_start,make_string=True,format='%a %H:%M')
    #Volume of tweets
    volume.append(len(snap['LOC']))

    #HotWords List
    Vocab_dict = dict(get_vocabulary(snap['TEXT']).most_common(HotWordSize))
    TimeList.append(timeWindow)

    ColorGradient[timeWindow] = {}

    for word in Vocab_dict.keys():
      ColorGradient[timeWindow][word] = Vocab_dict[word]/float(sum(Vocab_dict.values()))
      if word in TweetCountDict.keys():
        TweetCountDict[word] += Vocab_dict[word]
      else:
        TweetCountDict[word] = Vocab_dict[word]
    print "LOOPING2"



  SortedTweetCount = sorted(TweetCountDict.iteritems(),key=operator.itemgetter(1))
  WordList         = [item[0] for item in SortedTweetCount]
  TweetCountArray = np.array([item[1] for item in SortedTweetCount],dtype=int)
  del SortedTweetCount


  ColorMap = np.empty([len(WordList),len(TimeList)],dtype=float)

  for rw,word in enumerate(WordList):
    for cl,timeWindow in enumerate(TimeList):
      if word in ColorGradient[timeWindow].keys():
        ColorMap[rw][cl] = ColorGradient[timeWindow][word]
      else:
        ColorMap[rw][cl] = 0

  ###PRINT RESULTS
  gs      = gridspec.GridSpec(2,2,width_ratios=[1,2],height_ratios=[1,4])
  gs.update(left=0.05,right=0.48,wspace=0.00000000000000000000000000000000000000005,hspace=0.00000000000000000000000000000000000000005)

  fig1    = plt.figure(figsize=(36,50),dpi=200)


  ax0     = fig1.add_subplot(gs[0,1])
  ax1     = fig1.add_subplot(gs[1,1])
  ax2     = fig1.add_subplot(gs[1,0])
  ax3     = fig1.add_subplot(gs[0,0])

  #TweetVolume
  ax0.grid(True, 'major', color='w', linestyle='-', linewidth=0.7)
  ax0.grid(True, 'minor', color='0.92', linestyle='-', linewidth=0.35)
  ax0.set_axis_bgcolor('0.95')

  ASCII_WordList = [ word.encode('ascii','ignore') for word in WordList ]
  ax0.plot(np.arange(len(TimeList)),volume,label='NumberOfTweets',linewidth=0.75)
  ax0.legend(loc='upper left',ncol=4)
  ax0.set_xlim(0,len(TimeList)-1)
  ax0.xaxis.tick_top()
  ax0.yaxis.tick_right()
  ax0.set_xticks(np.arange(0,len(TimeList),3))
  ax0.set_xticklabels(TimeList,rotation='vertical')

  #HotWordColorMap
  ax1.imshow(ColorMap,cmap=plt.cm.binary,vmin=ColorMap.min(),vmax=ColorMap.max(),aspect='auto',origin='lower')
  ax1.yaxis.tick_right()
  ax1.set_yticks(np.arange(len(WordList)))
  ax1.set_yticklabels(WordList)
  ax1.set_xticks(np.arange(len(TimeList)))
  ax1.set_xticklabels(TimeList,rotation='vertical')

  ax1.grid(True, 'major', color='w', linestyle='-', linewidth=0.7)
  ax1.grid(True, 'minor', color='0.92', linestyle='-', linewidth=0.35)

  #TweetVolumeDistributionOverHotWords
  ax2.grid(True, 'major', color='w', linestyle='-', linewidth=0.7)
  ax2.grid(True, 'minor', color='0.92', linestyle='-', linewidth=0.35)
  ax2.set_axis_bgcolor('0.95')

  ax2.invert_xaxis()
  ax2.barh(np.arange(len(WordList)),TweetCountArray,align='center')

  #add the numbers to the side of each bar
  PreviousValue = None
  for p, ch in zip(np.arange(len(WordList)), TweetCountArray):
      if ch!=PreviousValue:
        ax2.annotate(str(ch), xy=(ch + 2.5, p - 0.25), va='center')
        PreviousValue = ch
      else:
        continue


  ax2.set_yticks(np.arange(len(WordList)))
  ax2.set_yticklabels(WordList)#,rotation='horizontal')
  ax2.set_ylim(0,len(WordList)-1+0.25)

  #Plot table with assisting information
  #1. Date : Day, Date Year and TIME_START to TIME_END
  #2. TIME_START
  #3. TIME_END
  #4. TIME_WINDOW
  #5. No. of HotWords per TimeWindow
  #6. Total No. of unique HotWords Found
  #7. Max #of Tweets for HotWord & HotWord
  #8. Min #of Tweets for HotWord & HotWord
  #9. Max #of Tweets in a timeWindow & timeWindow
  #10.Mix #of Tweets in a timeWindow & timeWindow

  rowLabels = ['1. Date','2. Start time','3. End time','4. Time Window (seconds)','5. No.Of HotWords per TimeWindow','6. No. of unique hotwords','7. Max #of tweets for HotWord','8. Min #of tweets for HotWord','9. Max #of tweets in a time window','10. Min #of tweets in a time window']
  DateStart = gmt_to_local(TIME_START,make_string=True,format='%a %d %b %Y')
  DateEnd   = gmt_to_local(TIME_END,make_string=True,format='%a %d %b %Y')
  Date      = DateStart if DateStart==DateEnd else DateStart+' to '+DateEnd
  start_time= gmt_to_local(TIME_START,make_string=True,format='%d %b %H:%M')
  end_time  = gmt_to_local(TIME_END,make_string=True,format='%d %b %H:%M')
  cellText  = [Date,start_time,end_time,scale,HotWordSize,len(set(WordList)),TweetCountArray.max(),TweetCountArray.min(),str(max(volume)),str(min(volume))]
  rowLabels.reverse()
  cellText.reverse()
  colLabels = ['Value']
  for y, label, text in zip(range(len(cellText)),rowLabels,cellText):
    ax3.text(0.05,(float(y)/20)+0.05,s='%s : %s'%(label,text),size=20)
  ax3.xaxis.set_visible(False)
  ax3.yaxis.set_visible(False)

  fig1.savefig('%s_to_%spng'%(start_time,end_time),dpi=200,bbox_inches="tight")
  plt.close(fig1)

  # Seperate figures
  # TweetVolume
  # fig0     = plt.figure(figsize=(12,8),dpi=200)
  # ax0      = fig0.add_subplot(111)
  # ax0.grid(True, 'major', color='w', linestyle='-', linewidth=0.7)
  # ax0.grid(True, 'minor', color='0.92', linestyle='-', linewidth=0.35)
  # ax0.set_axis_bgcolor('0.95')
  #
  # ASCII_WordList = [ word.encode('ascii','ignore') for word in WordList ]
  # ax0.plot(np.arange(len(TimeList)),volume,label='VolumeOfTweets',linewidth=0.75)
  # ax0.legend(loc='upper left',ncol=4)
  # ax0.set_xlim(0,len(TimeList)-1)
  # ax0.set_xticks(np.arange(len(TimeList)))
  # ax0.set_xticklabels(TimeList,rotation='vertical')
  #
  #
  # fig0.savefig('TV.png',dpi=200,bbox_inches="tight")
  # plt.close(fig0)
  #
  # #HotWordColorMap
  # fig1     = plt.figure(figsize=(12,24),dpi=200)
  # ax1     = fig1.add_subplot(111)
  # ax1.imshow(ColorMap,cmap=plt.cm.binary,vmin=ColorMap.min(),vmax=ColorMap.max(),aspect='equal',origin='lower')
  # ax1.yaxis.tick_right()
  # ax1.set_yticks(np.arange(len(WordList)))
  # ax1.set_yticklabels(WordList)
  # ax1.set_xticks(np.arange(len(TimeList)))
  # ax1.set_xticklabels(TimeList,rotation='vertical')
  #
  # ax1.grid(True, 'major', color='w', linestyle='-', linewidth=0.7)
  # ax1.grid(True, 'minor', color='0.92', linestyle='-', linewidth=0.35)
  #
  # fig1.savefig('HWCM2.png',dpi=200,bbox_inches="tight")
  # plt.close(fig1)
  #
  #
  # #TweetVolumeDistributionOverHotWords
  # fig2     = plt.figure(figsize=(12,24),dpi=200)
  # ax2      = fig2.add_subplot(111)
  # ax2.grid(True, 'major', color='w', linestyle='-', linewidth=0.7)
  # ax2.grid(True, 'minor', color='0.92', linestyle='-', linewidth=0.35)
  # ax2.set_axis_bgcolor('0.95')
  #
  # ax2.invert_xaxis()
  # ax2.barh(np.arange(len(WordList)),TweetCountArray,align='center')
  #
  # #add the numbers to the side of each bar
  # PreviousValue = None
  # for p, ch in zip(np.arange(len(WordList)), TweetCountArray):
  #     if ch!=PreviousValue:
  #       print PreviousValue,ch
  #       plt.annotate(str(ch), xy=(ch + 2.5, p - 0.25), va='center')
  #       PreviousValue = ch
  #     else:
  #       print PreviousValue,ch
  #       continue
  #
  #
  # ax2.set_yticks(np.arange(len(WordList)))
  # ax2.set_yticklabels(WordList)#,rotation='horizontal')
  # ax2.set_ylim(0,len(WordList))
  #
  # fig2.savefig('TWHW2.png',dpi=200,bbox_inches="tight")
  # plt.close(fig2)

threshold = 1
#1 Virality
def PoissonRate(Snap,given_word=None):

  tokenize  = T_Tokenizer().tokenize

  if given_word==None:
    PoissonRate= {}
    TweetTime  = {}


    for k,text in enumerate(Snap['TEXT']):
      for word in set(tokenize(text)):
        TweetTime.setdefault(word,[]).append(time.mktime(time.strptime(Snap['CREATED_AT'][k],"%d %b %H:%M:%S %Y")))

    for word,times in TweetTime.items():

      times = sorted(list(times))
      if len(times)>threshold:
        timeintervals = [times[-k]-times[-k-1] for k in range(1,len(times))]
        PoissonRate[word] = 1.0/np.average(timeintervals)

    return Counter(PoissonRate)

  else:

    PoissonRate= 0
    TweetTime  = []

    for k,text in enumerate(Snap['TEXT']):
      if given_word.lower() in set([word.lower() for word in tokenize(text)]):
        TweetTime.append(time.mktime(time.strptime(Snap['CREATED_AT'][k],"%d %b %H:%M:%S %Y")))


    times = sorted(list(TweetTime))
    if len(times)>threshold:
        timeintervals = [times[-k]-times[-k-1] for k in range(1,len(times))]
        PoissonRate = 1.0/np.average(timeintervals)

    return PoissonRate

#2 Delta(Volume)
def DeltaVolume(Snap1,Snap2):

  tokenize  = T_Tokenizer().tokenize

  TweetCount1={}
  TweetCount2={}
  KLDivergence={}

  for text in Snap1['TEXT']:
    for word in set(tokenize(text)):
      TweetCount1.setdefault(word,0)
      TweetCount1[word]+=1

  for text in Snap2['TEXT']:
    for word in set(tokenize(text)):
      TweetCount2.setdefault(word,0)
      TweetCount2[word]+=1

  for word,count in TweetCount2.items():
    p2 = float(count)/sum(TweetCount2.values())
    if word in TweetCount1.keys():
      p1 = float(TweetCount1[word])/sum(TweetCount1.values())
      KLDivergence[word] = -p2*np.log(p2/p1)

  return Counter(KLDivergence)


#5 Geographical Entropy
def GeographicalEntropy(Snap,labels,given_word=None):

  tokenize  = T_Tokenizer().tokenize

  if given_word==None:
    Entropy= {}
    TextLocation = {}
    for k,text in enumerate(Snap['TEXT']):
      for word in set(tokenize(text)):
        TextLocation.setdefault(word,[]).append(labels[k])

    for word,locs in TextLocation.items():
      if len(locs)>threshold:
        Distribution = Counter(locs)
        for count in Distribution.values():
            Entropy.setdefault(word,np.log2(len(set(labels))))
            Entropy[word]+= (float(count)/len(locs))*np.log2((float(count)/len(locs)))

    return Counter(Entropy)
  else:
    Entropy= 0
    locs = []
    for k,text in enumerate(Snap['TEXT']):
      if given_word.lower() in set([word.lower() for word in tokenize(text)]):
        locs.append(labels[k])

    if len(locs)>threshold:
      Distribution = Counter(locs)
      for count in Distribution.values():
          Entropy=np.log2(len(set(labels)))
          Entropy+= (float(count)/len(locs))*np.log2((float(count)/len(locs)))

    return Entropy



#6 Relevance to area
def Ttest(Snap):

  Ttest= {}
  tokenize  = T_Tokenizer().tokenize
  TweetLoc= {}

  for k,text in enumerate(Snap['TEXT']):
    for word in set(tokenize(text)):
      TweetLoc.setdefault(word,[]).append(Snap['LOC'][k])

  Mean = [np.average([loc[0] for loc in Snap['LOC']]),np.average([loc[1] for loc in Snap['LOC']])]

  for word,locs in TweetLoc.items():

      if len(locs)>threshold:
        Lats = [loc[0] for loc in locs]
        Lons = [loc[1] for loc in locs]
        Std  = np.sqrt(np.std(Lats)**2 + np.std(Lons)**2)
        Ttest[word] = np.sqrt((np.average(Lats)-Mean[0])**2+(np.average([loc[1] for loc in locs])-Mean[1])**2)/(Std/np.sqrt(len(locs)))

  return Counter(Ttest)

def find_phrases():

  print "COLLECTING TWEETS"
  TI = TweetIterator(db='streamer',collect_items=['text'])
  print "COLLECTION OVER"
  PhraseCount   = {}
  tokenize  = T_Tokenizer().tokenize
  TweetText  = {}

  cut_off_vocab = 100


  for k,tweet in enumerate(TI):
    text = tweet['text']
    for pos,word in enumerate(tokenize(text)):
      TweetText.setdefault(word,[]).append([pos,k])

  words = TweetText.keys()
  for k in range(0,len(words)-1):
    word1 = words[k];
    tweets1 = TweetText[word1]
    if len(tweets1)>cut_off_vocab:

      lefttweets1 = {'/'.join([str(pos-1),str(tweet)]) for pos,tweet in tweets1}
      righttweets1 = {'/'.join([str(pos+1),str(tweet)]) for pos,tweet in tweets1}

      l = k + 1
      while l<len(words):
        word2 = words[l]; tweets2 = {'/'.join([str(pos),str(tweet)]) for pos,tweet in TweetText[word2]}
        if len(tweets2)>cut_off_vocab:
          if len(tweets2.intersection(lefttweets1))>cut_off_vocab:
            PhraseCount[' '.join([word1,word2])] = -len(tweets2.intersection(lefttweets1))*np.log2(float(len(tweets2.intersection(lefttweets1)))/(len(tweets1)*len(tweets2)))
          if len(tweets2.intersection(righttweets1))>cut_off_vocab:
            PhraseCount[' '.join([word2,word1])] = -len(tweets2.intersection(righttweets1))*np.log2(float(len(tweets2.intersection(righttweets1)))/(len(tweets1)*len(tweets2)))
        l+=1
  return Counter(PhraseCount)

def Count(Snap,given_word=None):


  tokenize  = T_Tokenizer().tokenize

  if given_word==None:
    TextCount   = {}
    for k,text in enumerate(Snap['TEXT']):
      for word in set(tokenize(text)):
        TextCount.setdefault(word,0)
        TextCount[word]+=1
    return Counter(TextCount)
  else:
    TextCount = 0
    for k,text in enumerate(Snap['TEXT']):
      if given_word.lower() in set([word.lower() for word in tokenize(text)]):
        TextCount+=1
    return TextCount


def set_SnapIter(db,timeWindow,**kwargs):

  print "COLLECTING TWEETS...."
  TS = TweetSnap(db=db,timeWindow = timeWindow,Placename2Geocode=False)
  print "COLLECTION OVER...."

  TIME_START = kwargs.get("TIME_START",time.gmtime(0))
  TIME_END   = kwargs.get("TIME_END",time.gmtime(time.time()))

  if isinstance(TIME_START,str):
    TIME_START  = time.gmtime(time.mktime(time.strptime(TIME_START,"%d %b %H:%M %Z %Y")))
  if isinstance(TIME_END,str):
    TIME_END    = time.gmtime(time.mktime(time.strptime(TIME_END,"%d %b %H:%M %Z %Y")))

  TIME_DIFF   = time.mktime(TIME_START)  - time.mktime(TS.time_start)

  if TIME_DIFF>0:
    TS.move_on(TIME_DIFF-timeWindow)

  return TS


def newsworthy_words(db,timeWindow,**kwargs):

  print "COLLECTING TWEETS...."
  TS = TweetSnap(db=db,timeWindow = timeWindow,Placename2Geocode=False)
  print "COLLECTION OVER...."

  TIME_START = kwargs.get("TIME_START",time.gmtime(0))
  TIME_END   = kwargs.get("TIME_END",time.gmtime(time.time()))
  HotWordSize = kwargs.get("HotWordSize",25)


  if isinstance(TIME_START,str):
    TIME_START  = time.gmtime(time.mktime(time.strptime(TIME_START,"%d %b %H:%M %Z %Y")))
  if isinstance(TIME_END,str):
    TIME_END    = time.gmtime(time.mktime(time.strptime(TIME_END,"%d %b %H:%M %Z %Y")))

  TIME_DIFF   = time.mktime(TIME_START)  - time.mktime(TS.time_start)

  if TIME_DIFF>0:
    TS.move_on(TIME_DIFF-timeWindow)


  Day = {}

  while (TS.time_start<TIME_END and not TS.end):

    #Capture nextSnap and initialize time_start of next snap
    snap = TS.next()
    if len(snap['TEXT'])<100:
      continue
    gmm  = GMM_clustering()
    #1. Virality
    Virality  = PoissonRate(snap)
    #2. DeltaVolume
    #Broadcast = DeltaVolume(snap0,snap)
    #3. Locality
    gmm.Snap = snap
    gmm.build_clusters()
    Locality  = GeographicalEntropy(snap,gmm.labels)
    #4. Prevalence
    #Prevalence= Ttest(snap)
    #5. Count
    Volume     = Count(snap)

    #Prepare Dataframe
    #Union
    #HotWords= list(set(dict(Virality.most_common(HotWordSize)).keys()+dict(Broadcast.most_common(HotWordSize)).keys()+dict(Locality.most_common(HotWordSize)).keys()+dict(Prevalence.most_common(HotWordSize)).keys()+dict(Volume.most_common(HotWordSize)).keys()))
    #Intersection
    #print "Simmering words"
    #print 'Virality',set(dict(Virality.most_common(HotWordSize)).keys())
    #print 'Broadcast',set(dict(Broadcast.most_common(HotWordSize)).keys())
    #print 'Locality',set(dict(Locality.most_common(HotWordSize)).keys())
    #print set(dict(Prevalence.most_common(HotWordSize)).keys())
    #print 'Volume',set(dict(Volume.most_common(HotWordSize)).keys())
    #print "*"*5

    #HotWords= set(dict(Virality.most_common(HotWordSize)).keys())&set(dict(Broadcast.most_common(HotWordSize)).keys())&set(dict(Locality.most_common(HotWordSize)).keys())&set(dict(Prevalence.most_common(HotWordSize)).keys())&set(dict(Volume.most_common(HotWordSize)).keys())
    HotWords= list(set(dict(Virality.most_common(HotWordSize)).keys())&set(dict(Locality.most_common(HotWordSize)).keys())&set(dict(Volume.most_common(HotWordSize)).keys()))
    if not len(HotWords)>0:
      continue

    Virality= [Virality[key] if key in Virality.keys() else 0 for key in HotWords]
    # Broadcast=[Broadcast[key] if key in Broadcast.keys() else 0 for key in HotWords]
    Locality= [Locality[key] if key in Locality.keys() else 0 for key in HotWords]
    # Prevalence=[Prevalence[key] if key in Prevalence.keys() else 0 for key in HotWords]
    Volume=[Volume[key] if key in Volume.keys() else 0 for key in HotWords]

    #scaler           = preprocessing.MinMaxScaler([0,100]).fit_transform
    #scaledVirality   = list(scaler(np.array([Virality]).T).flatten())
    # scaledBroadcast  = scaler(Broadcast)
    #scaledLocality   = list(scaler(np.array([Locality]).T).flatten())
    # scaledPrevalence = scaler(Prevalence)
    #scaledVolume     = list(scaler(np.array([Volume],dtype=np.float16).T).flatten())
    Score            = [vi+lo+vo for vi,lo,vo in zip(Virality,Locality,Volume)]

    df             = pd.DataFrame({'Words':HotWords,'Virality':Virality,'Locality':Locality,'Volume':Volume,'Score':Score})
    #df_scaled      = pd.DataFrame({'Words':HotWords,'Virality':scaledVirality,'Locality':scaledLocality,'Volume':scaledVolume,'Score':Score})

    Day['to'.join(snap['TimeWindow'])]=df

  return Day

def visualize_word(db,timeWindow,given_word,**kwargs):

  print "COLLECTING TWEETS...."
  TS = TweetSnap(db=db,timeWindow = timeWindow,Placename2Geocode=False)
  print "COLLECTION OVER...."

  TIME_START = kwargs.get("TIME_START",time.gmtime(0))
  TIME_END   = kwargs.get("TIME_END",time.gmtime(time.time()))

  if isinstance(TIME_START,str):
    TIME_START  = time.gmtime(time.mktime(time.strptime(TIME_START,"%d %b %H:%M %Z %Y")))
  if isinstance(TIME_END,str):
    TIME_END    = time.gmtime(time.mktime(time.strptime(TIME_END,"%d %b %H:%M %Z %Y")))

  TIME_DIFF   = time.mktime(TIME_START)  - time.mktime(TS.time_start)

  if TIME_DIFF>0:
    TS.move_on(TIME_DIFF-timeWindow)

  Virality = []
  Volume   = []
  Locality = []
  TimeWindow=[]
  Word     = []

  while (TS.time_start<TIME_END and not TS.end):

    #Capture nextSnap and initialize time_start of next snap
    snap = TS.next()
    if len(snap['TEXT'])<100:
      continue
    #gmm  = GMM_clustering()
    for item in given_word:
      #1. Virality
      #Virality.append(PoissonRate(snap,given_word=item))
      #2. Locality
      #gmm.Snap = snap
      #gmm.build_clusters()
      #Locality.append(GeographicalEntropy(snap,gmm.labels,given_word=item))
      #3. Volume
      Volume.append(Count(snap,given_word=item))
      #4. TimeWindow
      TimeWindow.append(snap['TimeWindow'][0])
      #5. Word
      Word.append(item)

  #Prepare Dataframe
  df             = pd.DataFrame({'Virality':Virality,'Locality':Locality,'Volume':Volume,'TimeWindow':TimeWindow,'Word':Word})

  return df

def visualize_timeframe(db,timeWindow,**kwargs):

  print "COLLECTING TWEETS...."
  TS = TweetSnap(db=db,timeWindow = timeWindow,Placename2Geocode=False)
  print "COLLECTION OVER...."

  TIME_START = kwargs.get("TIME_START",time.gmtime(0))
  TIME_END   = kwargs.get("TIME_END",time.gmtime(time.time()))
  VocabSize   = kwargs.get("VocabSize",500)

  if isinstance(TIME_START,str):
    TIME_START  = time.gmtime(time.mktime(time.strptime(TIME_START,"%d %b %H:%M %Z %Y")))
  if isinstance(TIME_END,str):
    TIME_END    = time.gmtime(time.mktime(time.strptime(TIME_END,"%d %b %H:%M %Z %Y")))

  TIME_DIFF   = time.mktime(TIME_START)  - time.mktime(TS.time_start)

  if TIME_DIFF>0:
    TS.move_on(TIME_DIFF-timeWindow)

  #Create Dataframe
  df             = pd.DataFrame(columns=['Virality','Locality','Volume','Words','TimeWindow'])

  while (TS.time_start<TIME_END and not TS.end):

    #InitializeColumns
    Virality = {}
    Volume   = {}
    Locality = {}
    TimeWindow=[]

    #Capture nextSnap and initialize time_start of next snap
    snap = TS.next()
    if len(snap['TEXT'])<100:
      continue
    gmm  = GMM_clustering()
    #1. Virality
    Virality  = te(snap)
    #2. Locality
    gmm.Snap = snap
    gmm.build_clusters()
    Locality = GeographicalEntropy(snap,gmm.labels)
    #3. Volume
    Volume   = Count(snap)

    #HotWords= set(dict(Virality.most_common(HotWordSize)).keys())&set(dict(Broadcast.most_common(HotWordSize)).keys())&set(dict(Locality.most_common(HotWordSize)).keys())&set(dict(Prevalence.most_common(HotWordSize)).keys())&set(dict(Volume.most_common(HotWordSize)).keys())
    Words= list(set(dict(Virality.most_common(VocabSize)).keys())&set(dict(Locality.most_common(VocabSize)).keys())&set(dict(Volume.most_common(VocabSize)).keys()))
    if not len(Words)>0:
      continue

    Virality= [Virality[key] if key in Virality.keys() else 0 for key in Words]
    # Broadcast=[Broadcast[key] if key in Broadcast.keys() else 0 for key in HotWords]
    Locality= [Locality[key] if key in Locality.keys() else 0 for key in Words]
    # Prevalence=[Prevalence[key] if key in Prevalence.keys() else 0 for key in HotWords]
    Volume=[Volume[key] if key in Volume.keys() else 0 for key in Words]

    #4. TimeWindow
    TimeWindow= [snap['TimeWindow'][0]]*len(Words)
    #5. Words
    Words= Words

    #Append to Dataframe
    df = df.append(pd.DataFrame({'Virality':Virality,'Locality':Locality,'Volume':Volume,'Words':Words,'TimeWindow':TimeWindow}),ignore_index=True)

  return df


if __name__=='__main__':
  print_tweet_report(TIME_START="13 Mar 00:00 EDT 2014",TIME_END='14 Mar 00:00 EDT 2014')
