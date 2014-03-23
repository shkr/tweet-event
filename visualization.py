from retreiver import TweetSnap, TweetIterator
from Clustering import Placename_clustering
from Clustering import GMM_clustering
from semantic import get_vocabulary
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import prettyplotlib as ppl
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


def print_vocabulary_report(scale=60*20,**kwargs):

  print "COLLECTING TWEETS...."
  TS = TweetSnap(timeWindow = scale,Placename2Geocode=False)
  print "COLLECTION OVER...."

  TIME_START = kwargs.get("TIME_START",time.gmtime(0))
  TIME_END   = kwargs.get("TIME_END",time.gmtime(time.time()))
  HotWordSize = kwargs.get("HotWordSize",10)

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

  while TS.time_start<TIME_END:

    #Capture nextSnap and initialize time_start of next snap
    snap = TS.next()
    timeWindow = gmt_to_local(TS.time_start,make_string=True,format='%a %H:%M')
    #Volume of tweets
    volume.append(len(snap['LOC']))

    #HotWords List
    Vocab_dict = dict(get_vocabulary(snap['TEXT'],count_unique_items=True).most_common(HotWordSize))
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

  fig1    = plt.figure(figsize=(36,40),dpi=200)


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
  ax0.set_xticks(np.arange(len(TimeList)))
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

  print 'HERE'
  print TIME_START
  print TIME_END

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
        plt.annotate(str(ch), xy=(ch + 2.5, p - 0.25), va='center')
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



if __name__=='__main__':
  print_tweet_report(TIME_START="13 Mar 00:00 EDT 2014",TIME_END='14 Mar 00:00 EDT 2014')
