#READ ME : TweetEvent v0.01 

## Contents
### 1. TweetCollector

1. What Twitter apis is it connected to ?

  -RESTAPIV1.1

  -StreamingAPIV1.1

2. How to stream from RESTAPIV1.1 :

  $> python collector.py boston

3. How to stream from StreamingAPI v1.1:

  $> python Streaming.py boston

4. Which locations can it collect tweets from ?

  Locations and their geocodes are listed in geocode.py ; Additional locations can be inserted as dictionary key-value    pairs

  Source : http://maps.googleapis.com/maps/api/geocode/json?address=%s&sensor=false"%place_name

