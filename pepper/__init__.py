import logging

# Logging Module Default Settings, for debugging, switch to level=logging.DEBUG
logging.basicConfig(format='%(asctime)-15s %(levelname)s %(message)s', level=logging.INFO)

# Pepper Address, reference point for all apps (because it will change a lot!)
ADDRESS = '192.168.1.103', 9559


# Import to Package Level
from pepper.input.microphone import WaveMicrophone, SystemMicrophone, PepperMicrophone
from pepper.input.camera import PepperCamera, SystemCamera, CameraTarget, CameraColorSpace, CameraResolution

from .speech.asr import *
from .speech.utterance import Utterance

from .vision.face import OpenFace, FaceBounds, GenderClassifyClient
from .vision.object import ObjectClassifyClient

from .people.clustering import PeopleCluster, load_people, load_data_set

from .knowledge.wolfram import Wolfram

from .app import App, FlowApp

