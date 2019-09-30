from pepper.framework import *
from pepper import config

from PIL import Image


class ImageObjectsApp(AbstractApplication, ObjectDetectionComponent):
    def on_object(self, objects):
        if objects:
            obj = objects[0]

            img = obj.image.get_image(obj.image_bounds)

            Image.fromarray(img).show()

            exit()


if __name__ == '__main__':
    ImageObjectsApp(config.get_backend()).run()
