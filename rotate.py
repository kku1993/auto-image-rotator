import click
import cv2
import dlib
import numpy as np
import os

from pathlib import Path
from PIL import Image, ImageFile


class Rotator:
    def __init__(self, directory: str, overwrite_files: bool=False):
        self.detector = dlib.get_frontal_face_detector()
        self.directory = directory
        self.overwrite_files =overwrite_files

    def analyze_images(self):
        # Recursively loop through all files and subdirectories.
        # os.walk() is a recursive generator.
        # The variable "root" is dynamically updated as walk() recursively traverses directories.
        images = []
        for root_dir, sub_dir, files in os.walk(self.directory):
            for file_name in files:
                if file_name.lower().endswith((".jpeg", ".jpg", ".png")):
                    file_path = str(os.path.join(root_dir, file_name))
                    images.append(file_path)

        # Analyze each image file path - rotating when needed.
        rotations = {}
        with click.progressbar(images, label=f"Analyzing {len(images)} Images...") as filepaths:
            for filepath in filepaths:
                image = self.open_image(filepath)
                rotation = self.analyze_image(image, filepath)

                if rotation:
                    rotations[filepath] = rotation

        with click.progressbar(rotations.items(), label=f"Rotating {len(rotations)} Images...") as items:
          for filepath, rotation in items:
                print(f" - {filepath} (Rotated {rotation} Degrees)")

    def analyze_image(self, image: ImageFile, filepath: str) -> int:
        """Cycles through 4 image rotations of 90 degrees.
           Saves the image at the current rotation if faces are detected.
        """
        for cycle in range(0, 4):
            if cycle > 0:
                # Rotate the image an additional 90 degrees for each non-zero cycle.
                image = image.rotate(90, expand=True)

            image_copy = np.asarray(image)
            image_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

            faces = self.detector(image_gray, 0)
            if len(faces) == 0:
                continue

            if cycle > 0:
                return cycle * 90

        return 0

    def open_image(self, filepath: str) -> ImageFile:
        """Intentionally opens an image file using Pillow.
           If opened with OpenCV, the saved image is a much larger file size than the original
           (regardless of whether saved via OpenCV or Pillow).
        """
        return Image.open(filepath)

    def save_image(self, image: ImageFile, filepath: str) -> bool:
        """Saves the rotated image using Pillow."""

        if not self.overwrite_files:
            filepath = filepath.replace(".", "-rotated.", 1)

        try:
            image.save(filepath)
            return True
        except:
            return False


@click.command()
@click.argument("directory", type=click.STRING, required=True)
@click.option("--overwrite", type=click.BOOL, default=False)
def cli(directory: str, overwrite: bool=False):
    rotator = Rotator(directory, overwrite)
    rotator.analyze_images()


if __name__ == "__main__":
    cli()
