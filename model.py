import tensorflow as tf
from vgg_normalised import vgg_from_t7
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, Lambda, UpSampling2D
from keras.initializers import VarianceScaling
from ops import adain, pad_reflect, Conv2DReflect, torch_decay, gram_matrix, mse, sse
import functools


class AdaINModel(object):
    '''Adaptive Instance Normalization model from https://arxiv.org/abs/1703.06868
    '''
    def __init__(self, mode='train', small_model=False, *args, **kwargs):
        self.build_model(small_model=small_model)

        if mode == 'train':
            self.build_train(**kwargs)
            self.build_summary()

    def build_model(self, small_model=False):
        self.content_imgs = tf.placeholder(shape=(None, None, None, 3), name='content_imgs', dtype=tf.float32)
        self.style_imgs = tf.placeholder(shape=(None, None, None, 3), name='style_imgs', dtype=tf.float32)
        
        self.alpha = tf.placeholder_with_default(1., shape=[], name='alpha')

        ### Load shared VGG model up to relu4_1
        with tf.name_scope('encoder'):
            self.vgg_model = vgg_from_t7('vgg_normalised.t7', target_layer='relu4_1')
        print(self.vgg_model.summary())

        ### Build encoders for content layer
        with tf.name_scope('content_layer_encoder'):
            # Build content layer encoding model
            content_layer = self.vgg_model.get_layer('relu4_1').output
            self.content_encoder_model = Model(inputs=self.vgg_model.input, outputs=content_layer)

            # Setup content layer encodings for content/style images
            self.content_encoded = self.content_encoder_model(self.content_imgs)
            self.style_encoded = self.content_encoder_model(self.style_imgs)
            
            # Apply affine Adaptive Instance Norm transform
            self.adain_encoded = adain(self.content_encoded, self.style_encoded, self.alpha)

        ### Build decoder
        with tf.name_scope('decoder'):
            n_channels = self.adain_encoded.get_shape()[-1].value
            self.decoder_model = self.build_decoder(input_shape=(None, None, n_channels), small_model=small_model)
            
            # Stylized/decoded output from AdaIN transformed encoding
            self.decoded = self.decoder_model(Lambda(lambda x: x)(self.adain_encoded)) # Lambda converts TF tensor to Keras
            self.decoded = tf.Print(self.decoded, [tf.reduce_min(self.decoded), tf.reduce_max(self.decoded)])

        # Content layer encoding for stylized out
        self.decoded_encoded = self.content_encoder_model(self.decoded)

    def build_decoder(self, input_shape, small_model=False):
        if small_model:
            arch = [                                                            #  HxW  / InC->OutC
                    Conv2DReflect(128, 3, padding='valid', activation='relu'),  # 32x32 / 512->256
                    UpSampling2D(),                                             # 32x32 -> 64x64
                    Conv2DReflect(64, 3, padding='valid', activation='relu'),   # 64x64 / 256->128
                    UpSampling2D(),                                             # 64x64 -> 128x128
                    Conv2DReflect(32, 3, padding='valid', activation='relu'),   # 128x128 / 128->64
                    UpSampling2D(),                                             # 128x128 -> 256x256
                    Conv2DReflect(32, 3, padding='valid', activation='relu'),   # 256x256 / 64->64
                    Conv2DReflect(3, 3, padding='valid', activation=None)] # 256x256 / 64->3
                    # Conv2DReflect(3, 3, padding='valid', activation='sigmoid')] # 256x256 / 64->3
        else:
            arch = [                                                            
                    Conv2DReflect(256, 3, padding='valid', activation='relu'),  # 32x32 / 512->256
                    UpSampling2D(),                                             # 32x32 -> 64x64
                    Conv2DReflect(256, 3, padding='valid', activation='relu'),  # 64x64 / 256->256
                    Conv2DReflect(256, 3, padding='valid', activation='relu'),  # 64x64 / 256->256
                    Conv2DReflect(256, 3, padding='valid', activation='relu'),  # 64x64 / 256->256
                    Conv2DReflect(128, 3, padding='valid', activation='relu'),  # 64x64 / 256->128
                    UpSampling2D(),                                             # 64x64 -> 128x128
                    Conv2DReflect(128, 3, padding='valid', activation='relu'),  # 128x128 / 128->128
                    Conv2DReflect(64, 3, padding='valid', activation='relu'),   # 128x128 / 128->64
                    UpSampling2D(),                                             # 128x128 -> 256x256
                    Conv2DReflect(64, 3, padding='valid', activation='relu'),   # 256x256 / 64->64
                    Conv2DReflect(3, 3, padding='valid', activation=None)] # 256x256 / 64->3
                    # Conv2DReflect(3, 3, padding='valid', activation='sigmoid')] # 256x256 / 64->3
        
        code = Input(shape=input_shape, name='decoder_input')
        x = code

        with tf.variable_scope('decoder'):
            for layer in arch:
                x = layer(x) 
            
        decoder = Model(code, x, name='decoder_model')
        print(decoder.summary())
        return decoder

    def build_train(self, 
                    batch_size=8,
                    content_weight=1, 
                    style_weight=1e-2, 
                    tv_weight=0,
                    learning_rate=1e-4, 
                    lr_decay=5e-5, 
                    use_gram=False):
        ### Extract style layer feature maps
        with tf.name_scope('style_layers'):
            # Build style model for blockX_conv1 tensors for X:[1,2,3,4]
            relu_layers = [ 'relu1_1',
                            'relu2_1',
                            'relu3_1',
                            'relu4_1' ]
            # relu_layers = [ 'relu1_1',
            #                 'relu1_2',
            #                 'relu2_1',
            #                 'relu2_2',
            #                 'relu3_1',
            #                 'relu3_2',
            #                 'relu3_3',
            #                 'relu3_4',
            #                 'relu4_1' ]
            style_layers = [self.vgg_model.get_layer(l).output for l in relu_layers]
            self.style_layer_model = Model(inputs=self.vgg_model.input, outputs=style_layers)

            self.style_fmaps = self.style_layer_model(self.style_imgs)
            self.decoded_fmaps = self.style_layer_model(self.decoded)

        ### Losses
        with tf.name_scope('losses'):
            # Content loss between stylized encoding and AdaIN encoding
            self.content_loss = content_weight * mse(self.decoded_encoded, self.adain_encoded)

            if not use_gram:    # Collect style losses for means/stds
                mean_std_losses = []
                for s_map, d_map in zip(self.style_fmaps, self.decoded_fmaps):
                    s_mean, s_var = tf.nn.moments(s_map, [1,2])
                    d_mean, d_var = tf.nn.moments(d_map, [1,2])
                    m_loss = sse(d_mean, s_mean) / batch_size  # normalized w.r.t. batch size
                    s_loss = sse(tf.sqrt(d_var), tf.sqrt(s_var)) / batch_size  # normalized w.r.t. batch size

                    mean_std_loss = m_loss + s_loss
                    mean_std_loss = style_weight * mean_std_loss

                    mean_std_losses.append(mean_std_loss)

                self.style_loss = tf.reduce_sum(mean_std_losses)
            else:                # Use gram matrices for style loss instead
                gram_losses = []
                for s_map, d_map in zip(self.style_fmaps, self.decoded_fmaps):
                    s_gram = gram_matrix(s_map)
                    d_gram = gram_matrix(d_map)
                    gram_loss = mse(d_gram, s_gram)
                    gram_losses.append(gram_loss)
                self.style_loss = tf.reduce_sum(gram_losses) / batch_size

            if tv_weight > 0:
                self.tv_loss = tv_weight * tf.reduce_mean(tf.image.total_variation(self.decoded))
            else:
                self.tv_loss = tf.constant(0.)

            # Weight & combine content/style losses
            # self.total_loss = content_weight*self.content_loss + style_weight*self.style_loss + tv_weight*self.tv_loss
            self.total_loss = self.content_loss + self.style_loss + self.tv_loss

        ### Training ops
        with tf.name_scope('train'):
            self.global_step = tf.Variable(0, name='global_step_train', trainable=False)
            # self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 100, 0.96, staircase=False)
            self.learning_rate = torch_decay(learning_rate, self.global_step, lr_decay)
            d_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.9)

            t_vars = tf.trainable_variables()
            # Only train decoder vars, encoder is frozen
            self.d_vars = [var for var in t_vars if 'decoder' in var.name]

            self.train_op = d_optimizer.minimize(self.total_loss, var_list=self.d_vars, global_step=self.global_step)

    def build_summary(self):
        ### Summaries
        with tf.name_scope('summary'):
            content_loss_summary = tf.summary.scalar('content_loss', self.content_loss)
            style_loss_summary = tf.summary.scalar('style_loss', self.style_loss)
            tv_loss_summary = tf.summary.scalar('tv_loss', self.tv_loss)
            total_loss_summary = tf.summary.scalar('total_loss', self.total_loss)

            clip = lambda x: tf.clip_by_value(x, 0, 1)
            content_imgs_summary = tf.summary.image('content_imgs', clip(self.content_imgs))
            style_imgs_summary = tf.summary.image('style_imgs', clip(self.style_imgs))
            decoded_images_summary = tf.summary.image('decoded_images', clip(self.decoded))
            
            # # Visualize first three filters of encoding layers
            # sliced = lambda x: tf.slice(x, [0,0,0,0], [-1,-1,-1,3])
            # content_encoded_summary = tf.summary.image('content_encoded', sliced(self.content_encoded))
            # style_encoded_summary = tf.summary.image('style_encoded', sliced(self.style_encoded))
            # adain_encoded_summary = tf.summary.image('adain_encoded', sliced(self.adain_encoded))
            # decoded_encoded_summary = tf.summary.image('decoded_encoded', sliced(self.decoded_encoded))

            # for var in self.d_vars:
            #     tf.summary.histogram(var.op.name, var)

            self.summary_op = tf.summary.merge_all()
