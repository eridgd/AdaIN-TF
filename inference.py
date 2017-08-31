import os
import numpy as np
from model import AdaINModel
import tensorflow as tf


class AdaINference(object):
    '''Styilze images with trained AdaIN model'''

    def __init__(self, checkpoint_dir, sess=None, device='/gpu:0'): 
        '''
            Args:
                checkpoint_dir: Path to trained model checkpoint
                device: String for device ID to load model onto
        '''       
        self.model = AdaINModel(mode='test')

        self.stylized = self.model.decoded
        self.content_imgs = self.model.content_imgs
        self.style_imgs = self.model.style_imgs
        self.alpha_tensor = self.model.alpha

        if sess is None:
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
        self.sess = sess

        with tf.device(device):
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
        return np.uint8(np.clip(image, 0, 1) * 255)

    def predict(self, content, style, alpha=1):
        '''Stylize a single content/style pair'''
        content = self.preprocess(content)
        style = self.preprocess(style)

        stylized = self.sess.run(self.stylized, feed_dict={self.content_imgs: content,
                                                           self.style_imgs:   style,
                                                           self.alpha_tensor: alpha})

        return self.postprocess(stylized[0])

    def predict_interpolate(self, content, styles, style_weights, alpha=1):
        '''Stylize a weighted sum of multiple style encodings for a single content'''
        content_stacked = np.stack([content]*len(styles))  # Repeat content for each style
        style_stacked = np.stack(styles)
        content_stacked = self.preprocess(content_stacked)
        style_stacked = self.preprocess(style_stacked)

        encoded = self.sess.run(self.model.adain_encoded, feed_dict={self.content_imgs: content_stacked,
                                                                     self.style_imgs:   style_stacked,
                                                                     self.alpha_tensor: alpha})

        # Weight & combine AdaIN transformed encodings
        style_weights = np.array(style_weights).reshape((-1, 1, 1, 1))
        encoded_weighted = encoded * style_weights
        encoded_interpolated = np.sum(encoded_weighted, axis=0, keepdims=True)

        stylized = self.sess.run(self.stylized, feed_dict={self.model.adain_encoded_pl: encoded_interpolated})

        return self.postprocess(stylized[0])
