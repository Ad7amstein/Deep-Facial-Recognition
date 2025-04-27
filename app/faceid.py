"""Main App File"""

# Import kivy dependencies
from kivy.app import App

# Import kivy UX components
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies
import os
import cv2
import PIL
import numpy as np
import torch
from torchvision import transforms
from model import SiameseNN

APP_DATA_PATH = r"E:\Data Science\projects\Deep-Facial-Recognition\app\application_data"
MODEL_PATH = (
    r"E:\Data Science\projects\Deep-Facial-Recognition\app\siamese_network_v2.pth"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CampApp(App):
    """Camera app"""

    def __init__(self):
        """Init method"""
        super(CampApp, self).__init__()
        # Main Layout Components
        self.web_cam = Image(size_hint=(1, 0.8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1, 0.1))
        self.verification_label = Label(
            text="Verification Uninitiated", size_hint=(1, 0.1)
        )
        self.capture = cv2.VideoCapture(0)
        # Create a new instance
        self.model = SiameseNN()

        # Load in the saved state_dict()
        self.model.load_state_dict(torch.load(f=MODEL_PATH))

        # Send the model to the target device
        self.model.to(DEVICE)

    def build(self):
        """Build the app"""
        # Add items to layout
        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        Clock.schedule_interval(self.update, 1.0 / 33.0)

        return layout

    def update(self, *args):
        """Update the widget"""

        # Read frame from cv
        _, frame = self.capture.read()
        x, y = 30, 300
        frame = frame[x : x + 250, y : y + 250, :]

        # Flip horizontall and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt="bgr"
        )
        img_texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.web_cam.texture = img_texture

    def preprocess(self, file_path):
        """Preprocess the image"""
        transform = transforms.Compose(
            [transforms.Resize((105, 105)), transforms.ToTensor()]
        )

        img = PIL.Image.open(file_path).convert("RGB")
        img = transform(img)
        return img

    def verify(self, *args):
        """Verification function"""
        # Specify thresholds
        detection_threshold = 0.9
        verification_threshold = 0.8

        # Capture input image from webcam
        SAVE_PATH = os.path.join(APP_DATA_PATH, "input_image", "input_img.jpg")
        _, frame = self.capture.read()
        x, y = 30, 300
        frame = frame[x : x + 250, y : y + 250]
        cv2.imwrite(SAVE_PATH, frame)

        results = []
        self.model.to(DEVICE)
        self.model.eval()
        for image in os.listdir(os.path.join(APP_DATA_PATH, "verification_images")):
            in_img_path = os.path.join(APP_DATA_PATH, "input_image", "input_img.jpg")
            vr_img_path = os.path.join(APP_DATA_PATH, "verification_images", image)
            input_img = torch.unsqueeze(self.preprocess(in_img_path), dim=0).to(DEVICE)
            validation_img = torch.unsqueeze(self.preprocess(vr_img_path), dim=0).to(
                DEVICE
            )
            with torch.inference_mode():
                # Forward pass (model outputs)
                pred_logits = self.model(input_img, validation_img)

                # GGet prediction probabilities (logits -> probabilities)
                pred_prob = torch.sigmoid(pred_logits.squeeze())

                # Get pred_prob off the GPU
                results.append(pred_prob.cpu())

        # Stack the pre_probs to turn  list into a tensor
        results = torch.stack(results)
        detection = torch.sum(results > detection_threshold)
        verification = detection / len(results)
        verified = verification > verification_threshold

        # Set verification text
        self.verification_label.text = (
            "Hi 3llam 🫡" if verified else "Not Recognized ❌"
        )

        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)
        # Logger.info(torch.sum(results > 0.2).item())
        # Logger.info(torch.sum(results > 0.4).item())
        # Logger.info(torch.sum(results > 0.5).item())
        # Logger.info(torch.sum(results > 0.8).item())

        return results, verified


if __name__ == "__main__":
    CampApp().run()
