import numpy as np
import tensorflow as tf


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="../Model/cnn_model_quantized.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
print(input_details[0])
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
print(input_data)
interpreter.set_tensor(input_details[0]['index'], input_data)


interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
gest_id = {0:'wave_mode', 1:'fist_pump_mode', 2:'random_motion_mode', 3:'speed_mode'}
print(gest_id[np.argmax(output_data, axis=1)[0]])
