from __future__ import print_function, division

import argparse
import functools
import time
import tensorflow as tf, numpy as np, os, random
from utils import get_files, get_img_random_crop
import keras.backend as K
from model import AdaINModel
import threading


parser = argparse.ArgumentParser()
### Directories
parser.add_argument('--checkpoint', type=str,
                    dest='checkpoint', help='Checkpoint save dir', 
                    required=True)
parser.add_argument('--log-path', type=str,
                    dest='log_path', help='Logging dir path')
parser.add_argument('--content-path', type=str,
                    dest='content_path', help='Content images folder')
parser.add_argument('--style-path', type=str,
                    dest='style_path', help='Style images folder')

### Loss weights
parser.add_argument('--content-weight', type=float,
                    dest='content_weight',
                    default=1)
parser.add_argument('--style-weight', type=float,
                    dest='style_weight',
                    default=1e-2)
parser.add_argument('--tv-weight', type=float,
                    dest='tv_weight',
                    default=0)

### Model opts
parser.add_argument('--gram', action='store_true', 
                    help='Use gram matrices for style loss instead of mean/std',
                    default=False)

### Train opts
parser.add_argument('--learning-rate', type=float,
                    dest='learning_rate',
                    help='Learning rate',
                    default=1e-4)
parser.add_argument('--lr-decay', type=float,
                    dest='lr_decay',
                    help='Learning rate decay',
                    default=5e-5)
parser.add_argument('--max-iter', type=int,
                    dest='max_iter', help='Max # of training iterations',
                    default=160000)
parser.add_argument('--batch-size', type=int,
                    dest='batch_size', help='Batch size',
                    default=8)
parser.add_argument('--save-iter', type=int,
                    dest='save_iter', help='Checkpoint save frequency',
                    default=200)
parser.add_argument('--summary-iter', type=int,
                    dest='summary_iter', help='Summary write frequency',
                    default=20)
parser.add_argument('--alpha', type=float,
                    dest='alpha',
                    default=1)
args = parser.parse_args()


def batch_gen(folder, batch_shape):
    '''Resize images to 512, randomly crop a 256 square, and normalize'''
    files = np.asarray(get_files(folder))
    while True:
        X_batch = np.zeros(batch_shape, dtype=np.float32)

        idx = 0

        while idx < batch_shape[0]:  # Build batch sample by sample
            try:
                f = np.random.choice(files)

                X_batch[idx] = get_img_random_crop(f, resize=512, crop=256).astype(np.float32)
                X_batch[idx] /= 255.    # Normalize between [0,1]
                
                assert(not np.isnan(X_batch[idx].min()))
            except Exception as e:
                # Do not increment idx if we failed 
                print(e)
                continue
            idx += 1

        yield X_batch


def train():
    batch_shape = (args.batch_size,256,256,3)

    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        ### Setup data loading queue
        queue_input_content = tf.placeholder(tf.float32, shape=batch_shape)
        queue_input_style = tf.placeholder(tf.float32, shape=batch_shape)
        queue = tf.FIFOQueue(capacity=100, dtypes=[tf.float32, tf.float32], shapes=[[256,256,3], [256,256,3]])
        enqueue_op = queue.enqueue_many([queue_input_content, queue_input_style])
        dequeue_op = queue.dequeue()
        content_batch_op, style_batch_op = tf.train.batch(dequeue_op, batch_size=args.batch_size, capacity=100)

        def enqueue(sess):
            content_images = batch_gen(args.content_path, batch_shape)
            style_images   = batch_gen(args.style_path, batch_shape)
            while True:
                content_batch = next(content_images)
                style_batch   = next(style_images)

                sess.run(enqueue_op, feed_dict={queue_input_content: content_batch,
                                                queue_input_style:   style_batch})

        ### Build the model graph and train/summary ops
        model = AdaINModel(mode='train',
                           batch_size=args.batch_size,
                           content_weight=args.content_weight, 
                           style_weight=args.style_weight,
                           tv_weight=args.tv_weight,
                           learning_rate=args.learning_rate,
                           lr_decay=args.lr_decay,
                           use_gram=args.gram)

        saver = tf.train.Saver(max_to_keep=None)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        with tf.Session(config=config) as sess:
            enqueue_thread = threading.Thread(target=enqueue, args=[sess])
            enqueue_thread.isDaemon()
            enqueue_thread.start()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            log_path = args.log_path if args.log_path is not None else os.path.join(args.checkpoint,'log')
            summary_writer = tf.summary.FileWriter(log_path, sess.graph)

            sess.run(tf.global_variables_initializer())
 
            if os.path.exists(os.path.join(args.checkpoint,'checkpoint')):
                ckpt = tf.train.get_checkpoint_state(args.checkpoint)
                if ckpt and ckpt.model_checkpoint_path:
                    print("Restoring from checkpoint", ckpt.model_checkpoint_path)
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    raise Exception("No checkpoint found...")
                

            model.vgg19.load_weights()

            for iteration in range(args.max_iter):
                start = time.time()
                
                content_batch, style_batch = sess.run([content_batch_op, style_batch_op])

                fetches = {
                    'train':        model.train_op,
                    'global_step':  model.global_step,
                    'summary':      model.summary_op,
                    'lr':           model.learning_rate,
                    'content_loss': model.content_loss,
                    'style_loss':   model.style_loss,
                    'tv_loss':      model.tv_loss
                }

                feed_dict = { model.content_imgs: content_batch,
                              model.style_imgs:   style_batch,
                              model.alpha:        args.alpha }

                results = sess.run(fetches, feed_dict=feed_dict)

                ### Log the summaries
                if iteration % args.summary_iter == 0:
                    summary_writer.add_summary(results['summary'], results['global_step'])

                ### Save checkpoint
                if iteration % args.save_iter == 0:
                    save_path = saver.save(sess, os.path.join(args.checkpoint, 'model.ckpt'), results['global_step'])
                    print("Model saved in file: %s" % save_path)

                ### Debug
                print("Step: {}  LR: {:.7f}  Content: {:.5f}  Style: {:.5f}  TV: {:.5f}  Time: {:.5f}".format(results['global_step'], results['lr'], results['content_loss'], results['style_loss'], results['tv_loss'], time.time() - start))

            # Last save
            save_path = saver.save(sess, os.path.join(args.checkpoint, 'model.ckpt'), results['global_step'])
            print("Model saved in file: %s" % save_path)

            coord.request_stop()


if __name__ == '__main__':
    train()
