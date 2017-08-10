import tensorflow as tf
import numpy as np

def init():
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run() #required by string_input_producer since we did  not specify epoch
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    return coord, threads

def fini(coord, threads, filename_queue):
    if filename_queue:
        filename_queue.close(cancel_pending_enqueues=True)
    coord.request_stop()
    coord.join(threads)

def build_shuffle_batch_test():
    input = tf.random_normal([])
    BATCH_SIZE = 2
    min_after_dequeue = BATCH_SIZE * 2
    capacity = min_after_dequeue + 5 * BATCH_SIZE
    input_dump = tf.Print(input, [input])
    shuffled_batch = tf.train.shuffle_batch(
        [input_dump],
        batch_size=BATCH_SIZE,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return shuffled_batch

def build_where_test():
    input = tf.random_uniform([2, 5, 5]) > 0.5
    size = np.prod(input.shape)
    input_dump = tf.Print(input, [input], summarize=size)
    where = tf.where(input_dump)
    return where

def build_map_fn_test():
    input = tf.to_int32(tf.random_uniform([5, 2]) > 0.5)
    size = np.prod(input.shape)
    input_dump = tf.Print(input, [input], summarize=size)
    m = tf.map_fn(lambda l: l * 2, input_dump)
    return m

def log_graph(log_dir):
    writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
    writer.flush()
    writer.close()

with tf.Session() as sess:
    result = build_map_fn_test()
    log_graph('./log')
    try:
        coord, threads = init()
        print sess.run([result])
    except Exception, e:
        coord.request_stop(e)
        print e
    finally:
        fini(coord, threads, None)
