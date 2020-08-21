import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_util
from google.protobuf import text_format
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util
from numpy import prod, sum
import os
from absl import app
from absl import flags

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


flags.DEFINE_string("pb_file_path", None, "model pb file path")
flags.mark_flag_as_required("pb_file_path")

FLAGS = flags.FLAGS

def main(_):
    flops = 0
    GRAPH_PB_PATH = FLAGS.pb_file_path  # path to your .pb file
    with tf.Session() as sess:
        print("load graph")
        with gfile.FastGFile(GRAPH_PB_PATH, "rb") as f:
            graph_def = tf.GraphDef()
            graph = tf.get_default_graph()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
            num_layers = 0
            for op in graph.get_operations():
                if (
                    op.type == "Conv2D"
                    or op.type == "DepthwiseConv2dNative"
                    or op.type == "Conv2DBackpropInput"
                ):
                    '''
                    How to calculate flops for conv layer:

                    k_w: kernel width
                    k_h: kernel height
                    k_in: input kernel size
                    k_out: output kernel size
                    O_w: output width
                    O_h: output height
                    O_c: output channel, == k_out

                    flops = O_w * O_h * O_c * (k_w * k_h * k_in)
                          = O_w * O_h * k_out * (k_w * k_h * k_in)
                          = O_w * O_h * (k_w * k_h * k_in * k_out)
                    '''

                    # op.inputs[0] is the features map
                    # op.inputs[1] is the weights
                    flops += (
                        op.outputs[0].shape[1]
                        * op.outputs[0].shape[2]
                        * prod(op.inputs[1].shape)
                    )

                    print(op.name, op.outputs[0].shape, op.inputs[1].shape)
                    num_layers += 1
                    print("Total layers: ", num_layers)
                    print("FLOPs: ", flops)

if __name__ == "__main__":
    app.run(main)
