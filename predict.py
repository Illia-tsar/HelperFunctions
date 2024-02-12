import tensorflow as tf
from preprocess import preprocess_image


def predict(img_path, model_path):
    img = preprocess_image(img_path, 128, 128, 114)

    model = tf.lite.Interpreter(model_path=model_path)
    input_metadata = model.get_input_details()
    output_metadata = model.get_output_details()
    model.allocate_tensors()

    model.set_tensor(input_metadata[0]['index'], img)
    model.invoke()

    output = model.get_tensor(output_metadata[0]["index"])
    return output
