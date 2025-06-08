import os
from pathlib import Path

import click
import cv2
import dlib
from jpegtran import JPEGImage


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
                if file_name.lower().endswith((".jpeg", ".jpg")):
                    file_path = str(os.path.join(root_dir, file_name))
                    images.append(file_path)

        # Analyze each image file path to identify the amount of rotation.
        rotations = {}
        with click.progressbar(images, label=f"Analyzing {len(images)} Images...") as filepaths:
            for filepath in filepaths:
                rotation = self.analyze_image(filepath)
                if rotation:
                    rotations[filepath] = rotation

        # For images that need to be rotated, rotate and save them.
        with click.progressbar(rotations.items(), label=f"Rotating {len(rotations)} Images...") as items:
          for filepath, rotation in items:
                print(f" - {filepath} (Rotated {rotation} Degrees)")
                img = JPEGImage(filepath)
                img = img.rotate(rotation)
                self.save_image(img, filepath)


    def analyze_image(self, filepath: str) -> int:
        """Cycles through 4 image rotations of 90 degrees.
           Saves the image at the current rotation if faces are detected.
        """
        image = cv2.imread(filepath)
        for cycle in range(0, 4):
            if cycle > 0:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector(image_gray, 0)
            if len(faces) == 0:
                continue

            if cycle > 0:
                return cycle * 90

        return 0

    def save_image(self, image: JPEGImage, filepath: str) -> bool:
        filename, ext = os.path.splitext(filepath)
        if not self.overwrite_files:
            filepath = filename + '-rotated' + ext
        image.save(filepath)


@click.command()
@click.argument("directory", type=click.STRING, required=True)
@click.option("--overwrite", type=click.BOOL, default=False)
def cli(directory: str, overwrite: bool=False):
    rotator = Rotator(directory, overwrite)
    rotator.analyze_images()


if __name__ == "__main__":
    cli()
