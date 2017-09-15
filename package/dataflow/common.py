import tensorflow as tf
from .base import DataFlow
import numpy as np

def assert_type(v, tp):
    assert isinstance(v, tp), "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

def FeedInput(inputs, placeholders):
    """
    expects inputs be from package.dataflow.base.DataFlow
    """
    assert_type(inputs, DataFlow) 
    cur_batch = inputs.next_batch()

    assert len(cur_batch) == len(placeholders), "[FeedInput] size different"
    feed = dict(zip(placeholders, cur_batch))
    # print(feed)
    return feed
    # return tf.train.SessionRunArgs(fetches=[], feed_dict=feed)

