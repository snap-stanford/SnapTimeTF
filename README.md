# SnapTimeTF

*reader.py* contains functions for reading from tfrecords.
This requires data_folder to point to the correct location ("/dfs/scratch0/mvc/test/snap_tf/bool_norm").
Calling *read()* will return a dense batch (raw tfrecord values), as well as framed values, which contains windows over the raw data.
