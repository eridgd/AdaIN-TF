import os
import numpy as np
from model import AdaINModel
import tensorflow as tf


class AdaINTest(object):
    def __init__(self, checkpoint_dir, device_t='/gpu:0', small_model=False):        
        model = AdaINModel(small_model=small_model, mode='test')

        self.stylized = model.decoded
        self.content_imgs = model.content_imgs
        self.style_imgs = model.style_imgs
        self.alpha_tensor = model.alpha

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        with tf.device(device_t):
            saver = tf.train.Saver()

            if os.path.isdir(checkpoint_dir):
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print("Restoring from checkpoint", ckpt.model_checkpoint_path)
                    saver.restore(self.sess, ckpt.model_checkpoint_path)
                else:
                    raise Exception("No checkpoint found...")

    @staticmethod
    def preprocess(image):
        if len(image.shape) == 3:  # Add batch dimension
            image = np.expand_dims(image, 0)
        return image / 255.        # Range [0,1]

    @staticmethod
    def postprocess(image):
        return (image * 255.).astype(np.uint8)

    def predict(self, content, style, alpha=1):
        content = self.preprocess(content)
        style = self.preprocess(style)

        stylized = self.sess.run(self.stylized, feed_dict={self.content_imgs: content,
                                                      self.style_imgs:   style,
                                                      self.alpha_tensor: alpha})
        return self.postprocess(stylized[0])

### TODO:
# Process single image
# Process video file