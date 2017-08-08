import tensorflow as tf
import glob

BATCH_SIZE = 3

def inputs(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized,
        features={
            'label': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string),
        })

    record_image = tf.decode_raw(features['image'], tf.uint8)

    # Changing the image into this shape helps train and visualize the output by converting it to
    # be organized like an image.
    image = tf.reshape(record_image, [250, 151, 1])

    label = tf.cast(features['label'], tf.string)

    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * BATCH_SIZE
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label], batch_size=BATCH_SIZE, capacity=capacity, min_after_dequeue=min_after_dequeue)

    # Converting the images to a float of [0,1) to match the expected input to convolution2d
    float_image_batch = tf.image.convert_image_dtype(image_batch, tf.float32)
    # Find every directory name in the imagenet-dogs directory (n02085620-Chihuahua, ...)
    labels = list(map(lambda c: c.split("/")[-1], glob.glob("../../imagenet-dogs-shrink/*")))

    # Match every label from label_batch and return the index where they exist in the list of classes
    train_labels = tf.map_fn(lambda l: tf.where(tf.equal(labels, l))[0,0:1][0], label_batch, dtype=tf.int64)
    return float_image_batch, train_labels


def build_nn(float_image_batch):
    conv2d_layer_one = tf.contrib.layers.convolution2d(
    float_image_batch,
    num_outputs=32,     # The number of filters to generate
    kernel_size=(5,5),          # It's only the filter height and width.
    activation_fn=tf.nn.relu,
    # weights_initializer=tf.random_normal,
    stride=(2, 2),
    trainable=True)
    pool_layer_one = tf.nn.max_pool(conv2d_layer_one,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME')

    # Note, the first and last dimension of the convolution output hasn't changed but the
    # middle two dimensions have.
    print conv2d_layer_one.get_shape(), pool_layer_one.get_shape()

    conv2d_layer_two = tf.contrib.layers.convolution2d(
        pool_layer_one,
        num_outputs=64,        # More output channels means an increase in the number of filters
        kernel_size=(5,5),
        activation_fn=tf.nn.relu,
        # weights_initializer=tf.random_normal,
        stride=(1, 1),
        trainable=True)

    pool_layer_two = tf.nn.max_pool(conv2d_layer_two,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME')

    print conv2d_layer_two.get_shape(), pool_layer_two.get_shape()

    flattened_layer_two = tf.reshape(
        pool_layer_two,
        [
            BATCH_SIZE,  # Each image in the image_batch
            -1           # Every other dimension of the input
        ])

    print flattened_layer_two.get_shape()

    # The weights_initializer parameter can also accept a callable, a lambda is used here  returning a truncated normal
    # with a stddev specified.
    hidden_layer_three = tf.contrib.layers.fully_connected(
        flattened_layer_two,
        512,
        #    weights_initializer=lambda i, dtype: tf.truncated_normal([38912, 512], stddev=0.1),
        activation_fn=tf.nn.relu
    )

    # Dropout some of the neurons, reducing their importance in the model
    hidden_layer_three = tf.nn.dropout(hidden_layer_three, 0.1)

    # The output of this are all the connections between the previous layers and the 120 different dog breeds
    # available to train on.
    final_fully_connected = tf.contrib.layers.fully_connected(
        hidden_layer_three,
        120,  # Number of dog breeds in the ImageNet Dogs dataset
        #    weights_initializer=lambda i, dtype: tf.truncated_normal([512, 120], stddev=0.1)
    )
    return final_fully_connected

def loss(predict_label_batch, train_label_batch):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=predict_label_batch, labels=train_label_batch))

def train(total_loss):
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.01,
        batch * 3,
        120,
        0.95,
        staircase=True)

    return tf.train.AdamOptimizer(
        learning_rate, 0.9).minimize(
        total_loss, global_step=batch)

def init():
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run() #required by string_input_producer since we did  not specify epoch
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    return coord, threads

def fini(coord, threads, filename_queue):
    filename_queue.close(cancel_pending_enqueues=True)
    coord.request_stop()
    coord.join(threads)

with tf.Session() as sess:
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once("./output/training-images/*.tfrecords"))
    X, Y = inputs(filename_queue)
    Y_ = build_nn(X)
    train_prediction = tf.nn.softmax(Y_)
    total_loss = loss(Y_, Y)
    train_op = train(total_loss)

    print 'ready to run computation graph'
    coord, threads = init()

    # actual training loop
    training_steps = 1000
    for step in range(training_steps):
        sess.run([train_op])
        # for debugging and learning purposes, see how the loss gets decremented thru training steps
        if step % 10 == 0:
            print "loss: ", sess.run([total_loss])

    print sess.run([train_prediction])

    fini(coord, threads, filename_queue)
