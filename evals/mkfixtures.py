"""Build the tiny models anya's tests run against.

Committed to nbs/fixtures so CI needs only the inference runtimes, not tensorflow or onnx.
Run with: python mkfix.py <outdir>
"""
import sys, zipfile, numpy as np
from pathlib import Path

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- onnx
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper as nh

def save(g, path, **kw):
    m = h.make_model(g, opset_imports=[h.make_opsetid('', 13)], **kw)
    m.ir_version = 9
    onnx.checker.check_model(m)
    onnx.save(m, str(out/path))
    print(' ', path, (out/path).stat().st_size, 'bytes')

# classifier: NCHW 1x3x8x8 -> mean over space -> 3x4 matmul -> logits (1,4)
w = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], np.float32)   # channel k -> class k
g = h.make_graph(
    [h.make_node('ReduceMean', ['image'], ['m'], axes=[2, 3], keepdims=0),
     h.make_node('MatMul', ['m', 'w'], ['logits'])],
    'tiny_cls', [h.make_tensor_value_info('image', T.FLOAT, [1, 3, 8, 8])],
    [h.make_tensor_value_info('logits', T.FLOAT, [1, 4])], [nh.from_array(w, 'w')])
save(g, 'tiny_cls.onnx')

# detector: ignores the picture, emits one box for class 1 in a (1,7,20) YOLO-shaped head
det = np.zeros((1, 7, 120), np.float32)
det[0, :4, 0] = [50, 50, 20, 20]      # cx, cy, w, h in input pixels
det[0, 5, 0] = 0.9                    # class 1 of 3
g = h.make_graph(
    [h.make_node('Constant', [], ['out'], value=nh.from_array(det, 'v')),
     h.make_node('Identity', ['out'], ['output0'])],
    'tiny_det', [h.make_tensor_value_info('image', T.FLOAT, [1, 3, 100, 100])],
    [h.make_tensor_value_info('output0', T.FLOAT, [1, 7, 120])])
save(g, 'tiny_det.onnx')

# segmenter: argmax of a 1x3x8x8 logit map, driven by which channel of the input is brightest
g = h.make_graph(
    [h.make_node('Identity', ['image'], ['masks'])],
    'tiny_seg', [h.make_tensor_value_info('image', T.FLOAT, [1, 3, 8, 8])],
    [h.make_tensor_value_info('masks', T.FLOAT, [1, 3, 8, 8])])
save(g, 'tiny_seg.onnx')

# embedder: dynamic batch, so the batching path is exercised on something real
we = np.eye(3, 128, dtype=np.float32)
g = h.make_graph(
    [h.make_node('ReduceMean', ['image'], ['m'], axes=[2, 3], keepdims=0),
     h.make_node('MatMul', ['m', 'we'], ['embedding'])],
    'tiny_emb', [h.make_tensor_value_info('image', T.FLOAT, ['batch', 3, 8, 8])],
    [h.make_tensor_value_info('embedding', T.FLOAT, ['batch', 128])], [nh.from_array(we, 'we')])
save(g, 'tiny_emb.onnx')

# ---------------------------------------------------------------- tflite
import tensorflow as tf

def keras_cls():
    i = tf.keras.Input((8, 8, 3), batch_size=1)
    x = tf.keras.layers.GlobalAveragePooling2D()(i)
    o = tf.keras.layers.Dense(4, use_bias=False,
                              kernel_initializer=tf.constant_initializer(
                                  np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], np.float32)))(x)
    return tf.keras.Model(i, o)

def rep():
    for _ in range(8): yield [np.random.rand(1, 8, 8, 3).astype(np.float32)]

m = keras_cls()
c = tf.lite.TFLiteConverter.from_keras_model(m)
(out/'tiny_cls.tflite').write_bytes(c.convert())
print('  tiny_cls.tflite', (out/'tiny_cls.tflite').stat().st_size, 'bytes')

c = tf.lite.TFLiteConverter.from_keras_model(m)
c.optimizations = [tf.lite.Optimize.DEFAULT]
c.representative_dataset = rep
c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
c.inference_input_type = tf.uint8
c.inference_output_type = tf.uint8
(out/'tiny_cls_quant.tflite').write_bytes(c.convert())
print('  tiny_cls_quant.tflite', (out/'tiny_cls_quant.tflite').stat().st_size, 'bytes')

# TFLite metadata rides along as a zip appended to the flatbuffer: that is how a real
# litert-community classifier ships its labels, so one fixture carries them the same way.
import shutil
shutil.copy(out/'tiny_cls.tflite', out/'tiny_cls_labels.tflite')
with zipfile.ZipFile(out/'tiny_cls_labels.tflite', 'a') as z:
    z.writestr('labels.txt', 'red\ngreen\nblue\nnone\n')
print('  tiny_cls_labels.tflite', (out/'tiny_cls_labels.tflite').stat().st_size, 'bytes')
