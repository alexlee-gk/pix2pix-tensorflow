from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import math
import os
import random
import re
import time

import numpy as np
import tensorflow as tf
from docile.ops import conv2d, conv_pool2d, upsample_conv2d, lrelu, instancenorm, sigmoid_kl_with_logits
from docile.utils import preprocess, deprocess
from tensorflow.python.platform import gfile

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")
parser.add_argument("--gpu_mem_frac", type=float, default=None, help="fraction of gpu memory to use")

parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--use_context_frames", type=int, default=1)
parser.add_argument("--context_frames", type=int, default=2)
parser.add_argument("--state_dim", type=int, default=4)
parser.add_argument("--scale_size", type=int, default=256, help="scale images to this size")
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=10.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
parser.add_argument("--gan_loss_type", choices=["GAN", "LSGAN"], default="LSGAN")

a = parser.parse_args()

EPS = 1e-12

Examples = collections.namedtuple("Examples", "paths, context_orig_images, orig_images, context_images, images, gen_images, context_states, states, gen_states, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")


def gan_loss(logits, labels, gan_loss_type):
    # use 1.0 (or 1.0 - discrim_label_smooth) for real data and 0.0 for fake data
    if gan_loss_type == 'GAN':
        # discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
        # gen_loss = tf.reduce_mean(-tf.log(predict_fake + EPS))
        if labels in (0.0, 1.0):
            labels = tf.constant(labels, dtype=logits.dtype, shape=logits.get_shape())
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
        else:
            loss = tf.reduce_mean(sigmoid_kl_with_logits(logits, labels))
    elif gan_loss_type == 'LSGAN':
        # discrim_loss = tf.reduce_mean((tf.square(predict_real - 1) + tf.square(predict_fake)))
        # gen_loss = tf.reduce_mean(tf.square(predict_fake - 1))
        loss = tf.reduce_mean(tf.square(logits - labels))
    else:
        raise ValueError('Unknown GAN loss type %s' % gan_loss_type)
    return loss


def load_examples():
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    filenames = gfile.Glob(os.path.join(a.input_dir, '*'))
    if not filenames:
        raise RuntimeError('No data files found.')

    # extract information from filename to count the number of data points in the dataset
    count = 0
    for filename in filenames:
        match = re.search('(\d+)_to_(\d+)_timestep_(\d+).tfrecords', os.path.basename(filename))
        start_traj_iter = int(match.group(1))
        end_traj_iter = int(match.group(2))
        count += end_traj_iter - start_traj_iter + 1

    filename_queue = tf.train.string_input_producer(filenames, shuffle=a.mode == 'train')
    reader = tf.TFRecordReader()
    paths, serialized_example = reader.read(filename_queue)

    features = {
        'context_frames': tf.FixedLenFeature([], tf.int64),
        'orig_image_height': tf.FixedLenFeature([], tf.int64),
        'orig_image_width': tf.FixedLenFeature([], tf.int64),
        'image_height': tf.FixedLenFeature([], tf.int64),
        'image_width': tf.FixedLenFeature([], tf.int64),
        'state_dim': tf.FixedLenFeature([], tf.int64),
        'context_orig_images': tf.FixedLenFeature([], tf.string),
        'orig_image': tf.FixedLenFeature([], tf.string),
        'context_images': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string),
        'gen_image': tf.FixedLenFeature([], tf.string),
        'context_states': tf.FixedLenFeature([a.context_frames, a.state_dim], tf.float32),
        'state': tf.FixedLenFeature([a.state_dim], tf.float32),
        'gen_state': tf.FixedLenFeature([a.state_dim], tf.float32),
    }
    features = tf.parse_single_example(serialized_example, features=features)

    context_frames = tf.cast(features['context_frames'], tf.int32)
    orig_image_height = tf.cast(features['orig_image_height'], tf.int32)
    orig_image_width = tf.cast(features['orig_image_width'], tf.int32)
    image_height = tf.cast(features['image_height'], tf.int32)
    image_width = tf.cast(features['image_width'], tf.int32)
    state_dim = tf.cast(features['state_dim'], tf.int32)

    with tf.control_dependencies([tf.assert_equal(context_frames, a.context_frames)]):
        context_frames = tf.constant(a.context_frames, tf.int32)
    with tf.control_dependencies([tf.assert_equal(state_dim, a.state_dim)]):
        state_dim = tf.constant(a.state_dim, tf.int32)
    orig_image_shape = [orig_image_height, orig_image_width, 3]
    image_shape = [image_height, image_width, 3]
    state_shape = [state_dim]

    def decode_and_preprocess_image(image_feature, image_shape):
        image = tf.decode_raw(image_feature, tf.uint8)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.reshape(image, image_shape)

        # make sure image is square
        image_height, image_width, _ = image_shape[-3:]
        with tf.control_dependencies([tf.assert_equal(image_height, image_width)]):
            image_size = tf.identity(image_height)

        image = preprocess(image)

        # upsample with bilinear interpolation but downsample with area interpolation
        image = tf.cond(tf.equal(image_size, a.scale_size),
                        lambda: image,
                        lambda: tf.cond(tf.less(image_size, a.scale_size),
                                        lambda: tf.image.resize_images(image, [a.scale_size, a.scale_size],
                                                                       method=tf.image.ResizeMethod.BILINEAR),
                                        lambda: tf.image.resize_images(image, [a.scale_size, a.scale_size],
                                                                       method=tf.image.ResizeMethod.AREA)))
        image = tf.reshape(image, image_shape[:-3] + [a.scale_size, a.scale_size] + image_shape[-1:])
        return image

    orig_image = decode_and_preprocess_image(features['orig_image'], orig_image_shape)
    image = decode_and_preprocess_image(features['image'], image_shape)
    gen_image = decode_and_preprocess_image(features['gen_image'], image_shape)
    state = tf.reshape(features['state'], state_shape)
    gen_state = tf.reshape(features['gen_state'], state_shape)

    if a.use_context_frames:
        context_orig_images = decode_and_preprocess_image(features['context_orig_images'], [context_frames] + orig_image_shape)
        context_images = decode_and_preprocess_image(features['context_images'], [context_frames] + image_shape)
        context_states = tf.reshape(features['context_states'], [context_frames] + state_shape)

    if a.use_context_frames:
        paths_batch, context_orig_images_batch, orig_image_batch, context_images_batch, image_batch, gen_image_batch, context_states_batch, state_batch, gen_state_batch = \
            tf.train.batch([paths, context_orig_images, orig_image, context_images, image, gen_image, context_states, state, gen_state], batch_size=a.batch_size)
        context_orig_images_batch = tf.unstack(context_orig_images_batch, axis=1)
        context_images_batch = tf.unstack(context_images_batch, axis=1)
        context_states_batch = tf.unstack(context_states_batch, axis=1)
    else:
        paths_batch, orig_image_batch, image_batch, gen_image_batch, state_batch, gen_state_batch = \
            tf.train.batch([paths, orig_image, image, gen_image, state, gen_state], batch_size=a.batch_size)
        context_orig_images_batch = []
        context_images_batch = []
        context_states_batch = []
    steps_per_epoch = int(math.ceil(count / a.batch_size))

    return Examples(
        paths=paths_batch,
        context_orig_images=context_orig_images_batch,
        orig_images=orig_image_batch,
        context_images=context_images_batch,
        images=image_batch,
        gen_images=gen_image_batch,
        context_states=context_states_batch,
        states=state_batch,
        gen_states=gen_state_batch,
        count=count,
        steps_per_epoch=steps_per_epoch,
    )


def create_generator(generator_inputs, generator_outputs_channels, context_images):
    layers = []
    if context_images:
        generator_inputs = tf.concat([generator_inputs] + context_images, axis=3)

    if a.scale_size == 256:
        layer_specs = [
            (a.ngf, 2),      # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
            (a.ngf * 2, 2),  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            (a.ngf * 4, 2),  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            (a.ngf * 8, 2),  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            (a.ngf * 8, 2),  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            (a.ngf * 8, 2),  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            (a.ngf * 8, 2),  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            (a.ngf * 8, 2),  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ]
    elif a.scale_size == 128:
        layer_specs = [
            (a.ngf, 2),
            (a.ngf * 2, 2),
            (a.ngf * 4, 2),
            (a.ngf * 8, 2),
            (a.ngf * 8, 2),
            (a.ngf * 8, 2),
            (a.ngf * 8, 2),
        ]
    elif a.scale_size == 64:
        layer_specs = [
            (a.ngf, 2),
            (a.ngf * 2, 2),
            (a.ngf * 4, 2),
            (a.ngf * 8, 2),
            (a.ngf * 8, 2),
            (a.ngf * 8, 2),
        ]
    else:
        raise NotImplementedError

    with tf.variable_scope("encoder_1"):
        out_channels, stride = layer_specs[0]
        if stride == 1:
            output = conv2d(generator_inputs, out_channels, kernel_size=(4, 4))
        else:
            output = conv_pool2d(generator_inputs, out_channels, kernel_size=(4, 4), strides=(stride,) * 2, pool_mode='avg')
        layers.append(output)

    for out_channels, stride in layer_specs[1:]:
        assert stride in (1, 2)
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            if stride == 1:
                convolved = conv2d(rectified, out_channels, kernel_size=(4, 4))
            else:
                convolved = conv_pool2d(rectified, out_channels, kernel_size=(4, 4), strides=(stride,) * 2, pool_mode='avg')
            output = instancenorm(convolved)
            layers.append(output)

    if a.scale_size == 256:
        layer_specs = [
            (a.ngf * 8, 2, 0.5),                   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            (a.ngf * 8, 2, 0.5),                   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (a.ngf * 8, 2, 0.5),                   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (a.ngf * 8, 2, 0.0),                   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (a.ngf * 4, 2, 0.0),                   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (a.ngf * 2, 2, 0.0),                   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (a.ngf, 2, 0.0),                       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
            (generator_outputs_channels, 2, 0.0),  # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
        ]
    elif a.scale_size == 128:
        layer_specs = [
            (a.ngf * 8, 2, 0.5),
            (a.ngf * 8, 2, 0.5),
            (a.ngf * 8, 2, 0.5),
            (a.ngf * 4, 2, 0.0),
            (a.ngf * 2, 2, 0.0),
            (a.ngf, 2, 0.0),
            (generator_outputs_channels, 2, 0.0),
        ]
    elif a.scale_size == 64:
        layer_specs = [
            (a.ngf * 8, 2, 0.5),
            (a.ngf * 8, 2, 0.5),
            (a.ngf * 4, 2, 0.0),
            (a.ngf * 2, 2, 0.0),
            (a.ngf, 2, 0.0),
            (generator_outputs_channels, 2, 0.0),
        ]
    else:
        raise NotImplementedError

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, stride, dropout) in enumerate(layer_specs[:-1]):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            if stride == 1:
                output = conv2d(rectified, out_channels, kernel_size=(4, 4))
            else:
                output = upsample_conv2d(rectified, out_channels, kernel_size=(4, 4), strides=(stride,) * 2)
            output = instancenorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    with tf.variable_scope("decoder_1"):
        out_channels, stride, dropout = layer_specs[-1]
        assert dropout == 0.0  # no dropout at the last layer
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        if stride == 1:
            output = conv2d(rectified, out_channels, kernel_size=(4, 4))
        else:
            output = upsample_conv2d(rectified, out_channels, kernel_size=(4, 4), strides=(stride,) * 2)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


def create_discriminator(discrim_inputs, discrim_targets, context_images):
    layers = []
    if context_images:
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)
    else:
        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets] + context_images, axis=3)

    if a.scale_size == 256:
        layer_specs = [
            (a.ndf, 2),      # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
            (a.ndf * 2, 2),  # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
            (a.ndf * 4, 2),  # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
            (a.ndf * 8, 1),  # layer_4: [batch, 32, 32, ndf * 4] => [batch, 32, 32, ndf * 8]
            (1, 1),          # layer_5: [batch, 32, 32, ndf * 8] => [batch, 32, 32, 1]
        ]
    elif a.scale_size == 128:
        layer_specs = [
            (a.ndf, 2),
            (a.ndf * 2, 2),
            (a.ndf * 4, 1),
            (a.ndf * 8, 1),
            (1, 1),
        ]
    elif a.scale_size == 64:
        layer_specs = [
            (a.ndf, 2),
            (a.ndf * 2, 1),
            (a.ndf * 4, 1),
            (a.ndf * 8, 1),
            (1, 1),
        ]
    else:
        raise NotImplementedError

    with tf.variable_scope("layer_1"):
        out_channels, stride = layer_specs[0]
        convolved = conv_pool2d(input, out_channels, kernel_size=(4, 4), strides=(stride,) * 2, pool_mode='avg')
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    for out_channels, stride in layer_specs[1:-1]:
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            if stride == 1:
                convolved = conv2d(layers[-1], out_channels, kernel_size=(4, 4))
            else:
                convolved = conv_pool2d(layers[-1], out_channels, kernel_size=(4, 4), strides=(stride,) * 2, pool_mode='avg')
            normalized = instancenorm(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        out_channels, stride = layer_specs[-1]
        if stride == 1:
            logits = conv2d(rectified, out_channels, kernel_size=(4, 4))
        else:
            logits = conv_pool2d(rectified, out_channels, kernel_size=(4, 4), strides=(stride,) * 2, pool_mode='avg')
        layers.append(logits)  # don't apply sigmoid to the logits in case we want to use LSGAN

    return layers[-1]


def create_model(inputs, targets, context_images):
    with tf.variable_scope("generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels, context_images)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 32, 32, 1]
            logits_real = create_discriminator(inputs, targets, context_images)
            if a.gan_loss_type == 'LSGAN':
                predict_real = logits_real
            else:
                predict_real = tf.sigmoid(logits_real)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 32, 32, 1]
            logits_fake = create_discriminator(inputs, outputs, context_images)
            if a.gan_loss_type == 'LSGAN':
                predict_fake = logits_fake
            else:
                predict_fake = tf.sigmoid(logits_fake)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        # discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
        discrim_loss_GAN = gan_loss(logits_real, 1.0, a.gan_loss_type) + gan_loss(logits_fake, 0.0, a.gan_loss_type)
        discrim_loss = discrim_loss_GAN * a.gan_weight

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        # gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_GAN = gan_loss(logits_fake, 1.0, a.gan_loss_type)
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )


def main():
    if tf.__version__.split('.')[0] != "1":
        raise Exception("Tensorflow version 1 required")

    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"ngf", "ndf", "use_context_frames", "context_frames", "state_dim", "scale_size"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    config = tf.ConfigProto()
    if a.gpu_mem_frac is not None:
        config.gpu_options.per_process_gpu_memory_fraction = a.gpu_mem_frac

    examples = load_examples()
    print("examples count = %d" % examples.count)

    # inputs and images are [batch_size, height, width, channels]
    model = create_model(examples.gen_images, examples.images, context_images=examples.context_images)

    context_images = [deprocess(context_image) for context_image in examples.context_images]
    gen_images = deprocess(examples.gen_images)
    images = deprocess(examples.images)
    outputs = deprocess(model.outputs)

    # reverse any processing on images so they can be written to disk or displayed to user
    if context_images:
        with tf.name_scope("convert_context_images"):
            converted_context_images = tf.image.convert_image_dtype(tf.concat(context_images, axis=-2), dtype=tf.uint8, saturate=True)

    with tf.name_scope("convert_gen_images"):
        converted_gen_images = tf.image.convert_image_dtype(gen_images, dtype=tf.uint8, saturate=True)

    with tf.name_scope("convert_images"):
        converted_images = tf.image.convert_image_dtype(images, dtype=tf.uint8, saturate=True)

    with tf.name_scope("convert_outputs"):
        converted_outputs = tf.image.convert_image_dtype(outputs, dtype=tf.uint8, saturate=True)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "gen_images": tf.map_fn(tf.image.encode_png, converted_gen_images, dtype=tf.string, name="input_pngs"),
            "images": tf.map_fn(tf.image.encode_png, converted_images, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }
        if context_images:
            display_fetches['context_images'] = tf.map_fn(tf.image.encode_png, converted_context_images, dtype=tf.string, name="context_images_pngs")

    # summaries
    if context_images:
        with tf.name_scope("context_images_summary"):
            tf.summary.image("context_images", converted_context_images)

    with tf.name_scope("gen_images_summary"):
        tf.summary.image("gen_images", converted_gen_images)

    with tf.name_scope("images_summary"):
        tf.summary.image("images", converted_images)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)

    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=5)
    logdir = a.output_dir if (a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session(config=config) as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            # testing
            raise NotImplementedError
        else:
            # training
            start = time.time()

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["gen_loss_L1"] = model.gen_loss_L1

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                results = sess.run(fetches)

                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm (%0.1fh) (%0.1fd)" % (train_epoch, train_step, rate, remaining / 60, remaining / 60 / 60, remaining / 60 / 60 / 24))
                    print("discrim_loss", results["discrim_loss"])
                    print("gen_loss_GAN", results["gen_loss_GAN"])
                    print("gen_loss_L1", results["gen_loss_L1"])

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break


if __name__ == '__main__':
    main()
