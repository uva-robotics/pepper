# -*- coding: future_fstrings -*-

from pepper.framework import *
from pepper import config

from PIL import Image

import os


class ImageObjectsApp(AbstractApplication, ObjectDetectionComponent):
    def on_object(self, objects):
        if objects:
            obj = objects[0]

            img = obj.image.get_image(obj.image_bounds)

            img_name = f'{obj.name}_{obj.id}'

            if not os.path.exists('./images'):
                os.mkdir('./images')

            Image.fromarray(img).save(f'./images/{img_name}.jpg')



if __name__ == '__main__':
    ImageObjectsApp(config.get_backend()).run()
