from t_auth import auth
import datetime
from TwitterSearch.TwitterSearch import *
import MySQLdb

location = {'Delhi':dict(latitude=28.360,longitude= 77.120,radius=25)}
try:
     tso = TwitterSearchOrder()
     tso.setCount(7)
     tso.setIncludeEntities(False)
     tso.setResultType('mixed')
     #tso.setUntil(datetime.datetime.strptime('31-12-13','%d-%m-%y').date())
     tso.setGeocode(**location['Delhi'])
     ts = TwitterSearch(**auth)
     
     conn = MySQLdb(host='',port=0,user='',passwd='',db='')
     curr = conn.cursor()
     with curr:
          curr.execute("CREATE TABLE IF NOT EXISTS TestTable (Id INT PRIMARY KEY AUTO_INCREMENT,tid INT,text VARCHAR(160))")
          ct=1
          for tweet in ts.searchTweetsIterable(tso):
               ct+=1
               #print( '@%s tweeted: %s' % ( tweet['user']['screen_name'], tweet['text'] ) )
               curr.execute("""INSERT INTO TESTTABLE (tid,text)""",%(tweet['id'],tweet['text'])
               if ct==50:
                    break
         
except TwitterSearchException as e:
     print (e)
         
         
         
         
