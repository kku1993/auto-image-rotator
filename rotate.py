import logging
import os
from concurrent.futures import ProcessPoolExecutor

import click
import cv2
import dlib
from jpegtran import JPEGImage


class Rotator:
    def __init__(self, overwrite_files: bool=False):
        self.detector = dlib.get_frontal_face_detector()
        self.overwrite_files = overwrite_files

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


def init_worker(overwrite: bool=False):
    global rotator
    print("Initialize worker...")
    rotator = Rotator(overwrite)


def worker(filepath: str):
    global rotator
    rotation = rotator.analyze_image(filepath)
    if rotation:
        print(f" - {filepath} (Rotated {rotation} Degrees)")
        img = JPEGImage(filepath)
        img = img.rotate(rotation)
        rotator.save_image(img, filepath)


@click.command()
@click.argument("directory", type=click.STRING, required=True)
@click.option("--overwrite", type=click.BOOL, default=False, help="If true, overwrites original image file with rotated version. Default=False")
@click.option("--max_workers", type=click.INT, default=None, help="The number of parallel processes to run. Default=number of CPU cores.")
def cli(directory: str, overwrite: bool=False, max_workers: int=None):
    # Recursively loop through all files and subdirectories.
    # os.walk() is a recursive generator.
    # The variable "root" is dynamically updated as walk() recursively traverses directories.
    images = []
    for root_dir, sub_dir, files in os.walk(directory):
        for file_name in files:
            if file_name.lower().endswith((".jpeg", ".jpg")):
                file_path = str(os.path.join(root_dir, file_name))
                images.append(file_path)

    print(f"Processing {len(images)} images...")

    # Analyze each image file path to identify the amount of rotation.
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(overwrite,)) as executor:
        list(executor.map(worker, images))


if __name__ == "__main__":
    cli()
