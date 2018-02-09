# SnapTimeTF

*reader.py* contains functions for reading from tfrecords.
This requires data_folder to point to the correct location ("/dfs/scratch0/mvc/test/snap_tf/bool_norm").

Calling *read()* will return a dense batch (raw tfrecord values), as well as framed values, which contains windows over the raw data.

Example

```python
reader = Reader()
shape = reader.get_shapes()
dense_values, frame_values, num_sensors, num_timesteps = self.reader.read(batch_size, shape=shape)
predictions = tf.layers.dense(dense_values)

with tf.Session() as sess:
    sess.run(predictions)
```
