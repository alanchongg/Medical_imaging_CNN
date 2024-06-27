import tkinter as tk
from tkinter import filedialog, Label
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
import torch
from Cnn import *
from torchvision import transforms
import os
import time

path=(os.getcwd()+'/Cnn_model/model.pt')
model=Cnn4A1LR1()
checkpoint=torch.load(path)
model.load_state_dict(checkpoint['Cnn4A1LR1'])

transform=transforms.Compose([transforms.Resize((299, 299)),
                            transforms.ToTensor(),])
classes=['Lung_Opacity', 'Normal', 'Viral Pneumonia']

def predict_image(img_path):
    start_time=time.time()
    img=Image.open(img_path)
    img=transform(img).unsqueeze(0)
    with torch.no_grad():
        out=model(img)
        _, pred=torch.max(out, 1)
        print('time taken:', time.time()-start_time)
        return classes[pred]

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Lung Diagonsis Assistant")
        self.geometry("800x600")
        default_image_path = "placeholder.jpg"

        self.img = Image.open(default_image_path)
        self.img = self.img.resize((299, 299), Image.LANCZOS)
        self.img = ImageTk.PhotoImage(self.img)

        self.image_label = Label(self, image=self.img)
        self.image_label.pack(pady=20)

        self.result_label = Label(self, text="", font=("Arial", 16))
        self.result_label.pack(pady=20)

        self.button = tk.Button(self, text="Select image", command=self.open_file_dialog)
        self.button.pack(pady=20)

    def open_file_dialog(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg")])
        if file_path:
            self.display_image(file_path)

    def display_image(self, file_path):
        img = Image.open(file_path)
        img = img.resize((299, 299), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)

        self.image_label.config(image=img)
        self.image_label.image = img

        result = predict_image(file_path)
        self.result_label.config(text=f"Prediction: {result}")

if __name__ == "__main__":
    app = App()
    app.mainloop()