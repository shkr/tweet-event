#READ ME : TweetEvent v0.01

## TweetCollector

*Twitter apis connected to :
-Tweet collection from RESTAPIV1.1
-Tweet collection from StreamingAPIV1.1

*How to stream from RESTAPIV1.1 :
$> python collectory.py boston

*How to stream from StreamingAPI v1.1:
$> python streaming.py boston

*Allowed locations 
Location geocodes are listed in geocode.py ; Additional locations can be inserted as dictionary key-value pairs
Source : http://maps.googleapis.com/maps/api/geocode/json?address=%s&sensor=false"%place_name
