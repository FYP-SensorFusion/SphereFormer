import tensorflow as tf

# Assume you have a `model` object that is your trained model
# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# This ensures that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# This ensures that if any ops can't be quantized, the converter throws an error
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8

# Provide a representative dataset to ensure we correctly generate the quantization parameters
def representative_dataset():
    for _ in range(100):
        yield [np.random.normal(0, 1, (1, 224, 224, 3)).astype(np.float32)]
converter.representative_dataset = representative_dataset

tflite_quant_model = converter.convert()