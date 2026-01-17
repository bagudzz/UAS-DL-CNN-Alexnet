import gradio as gr
from inference import load_model, predict

WEIGHTS_PATH = "weights/alexnet_catsdogs.pth"
model, idx_to_class, device = load_model(WEIGHTS_PATH)

def infer(image):
    return predict(image, model, idx_to_class, device)

demo = gr.Interface(
    fn=infer,
    inputs=gr.Image(type="pil", label="Upload gambar"),
    outputs=gr.Label(num_top_classes=2, label="Prediksi"),
    title="Cats vs Dogs - AlexNet",
    description="Klasifikasi kucing vs anjing menggunakan AlexNet (transfer learning)."
)

if __name__ == "__main__":
    demo.launch()
