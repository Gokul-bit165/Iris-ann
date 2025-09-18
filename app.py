import gradio as gr
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("artifacts/iris_ann.h5")

class_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    predictions = model.predict(input_data)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    return f"🌸 Predicted: {class_names[predicted_class]} (Confidence: {confidence:.2f})"

with gr.Blocks() as demo:
    gr.Markdown("# 🌸 Iris Flower Classifier (ANN)")
    gr.Markdown("Enter the flower measurements and get the predicted species.")

    with gr.Row():
        sepal_length = gr.Number(label="Sepal Length (cm)", value=5.1)
        sepal_width = gr.Number(label="Sepal Width (cm)", value=3.5)
        petal_length = gr.Number(label="Petal Length (cm)", value=1.4)
        petal_width = gr.Number(label="Petal Width (cm)", value=0.2)

    predict_btn = gr.Button("🔍 Predict")
    output = gr.Textbox(label="Prediction")

    predict_btn.click(
        predict_iris,
        inputs=[sepal_length, sepal_width, petal_length, petal_width],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()
