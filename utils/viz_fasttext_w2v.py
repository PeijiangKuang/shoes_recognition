# encoding: utf-8

from tqdm import tqdm
from argparse import ArgumentParser, ArgumentTypeError
import numpy as np
import os


def is_file(fname):
    if os.path.isfile(fname):
        return fname
    else:
        return False


def is_path(pname):
    if os.path.isdir(pname):
        return pname
    else:
        return False


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--w2v-fname", type=is_file)
    ap.add_argument("--log-dir", type=is_path, default='projections')
    ap.add_argument("--skip-header", action="store_true")
    ap.add_argument("--value-sep", type=str, default='')

    args = ap.parse_args()

    with open(args.w2v_fname, 'r') as ef:
        if args.skip_header:
            ef.readline()
        vectors = list()
        with open(os.path.join(args.log_dir, "embeddings.meta"), 'w') as mf:
            for line in tqdm(ef):
                if args.value_sep == '':
                    if ',' in line:
                        vals = line.strip().split(',')
                    elif ' ' in line or '\t' in line:
                        vals = line.strip().split()
                else:
                    vals = line.strip().split(args.value_sep)
                label = 'h'+vals[0]
                vector = [np.float32(x) for x in vals[1:]]
                vectors.append(vector)

                if label:
                    mf.write(label + "\n")
                else:
                    mf.write("<Empty Line>\n")
        vectors = np.array(vectors)

        import tensorflow as tf
        from tensorflow.contrib.tensorboard.plugins import projector
        sess = tf.InteractiveSession()
        embeddings_init_value = tf.placeholder(tf.float32, vectors.shape)
        embeddings = tf.Variable(embeddings_init_value, trainable=False, name='embeddings')
        sess.run(tf.global_variables_initializer(), feed_dict={embeddings_init_value: vectors})

        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(args.log_dir, sess.graph)

        # projector
        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = 'embeddings'
        embed.metadata_path = 'embeddings.meta'

        projector.visualize_embeddings(writer, config)
        saver.save(sess, os.path.join(args.log_dir, 'embeddings.ckpt'))
        print('Run \ntensorboard --logdir={0}\n to visualize embeddings in tensorboard'.format(args.log_dir))
