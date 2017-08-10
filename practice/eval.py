import tensorflow as tf

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


with tf.Session() as sess:
    input = tf.random_normal([])
    shuffled_batch = build_shuffle_batch_test()

    try:
        coord, threads = init()
        print sess.run([shuffled_batch])
        print sess.run([shuffled_batch])
        print sess.run([shuffled_batch])
    except Exception, e:
        coord.request_stop(e)
        print e
    finally:
        fini(coord, threads, None)
