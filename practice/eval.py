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

with tf.Session() as sess:
    coord, threads = init()
    try:
        print sess.run([])
    except Exception, e:
        coord.request_stop(e)
        print e
    finally:
        fini(coord, threads, None)
