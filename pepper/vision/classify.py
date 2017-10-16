from io import BytesIO
import socket
import yaml
import numpy as np


class ClassifyClient:
    def __init__(self, address):
        """
        Classify Images using Inception Model

        Parameters
        ----------
        address: (str, int)
            Address of Inception Model Host
        """
        self.address = address

    def classify(self, image):
        """
        Parameters
        ----------
        image: PIL.Image.Image

        Returns
        -------
        classification: list of (float, list)
            List of confidence-object pairs, where object is a list of object synonyms
        """

        jpeg = self._convert_to_jpeg(image)
        jpeg_size = np.array([len(jpeg)], np.uint32)

        s = socket.socket()
        s.connect(self.address)
        s.sendall(jpeg_size)
        s.sendall(jpeg)
        response = yaml.load(s.recv(4096).decode())
        return response

    def _convert_to_jpeg(self, image):
        """
        Parameters
        ----------
        image: PIL.Image.Image

        Returns
        -------
        encoded_image: bytes
        """

        with BytesIO() as jpeg_buffer:
            image.save(jpeg_buffer, format='JPEG')
            return jpeg_buffer.getvalue()

