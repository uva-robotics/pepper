"""Example Application that displays what it sees in the browser"""

from pepper.framework import *
from pepper import config
from pepper.knowledge import Wikipedia      # Class to Query Wikipedia using Natural Language


class DisplayApp(AbstractApplication,           # Each Application inherits from AbstractApplication
                 StatisticsComponent,           # Show Performance Statistics in Terminal
                 DisplayComponent,              # Display what Robot (or Computer) sees in browser
                 SceneComponent,                # Scene (dependency of DisplayComponent)
                 ContextComponent,              # Context (dependency of DisplayComponent)
                 ObjectDetectionComponent,      # Object Detection (dependency of DisplayComponent)
                 FaceRecognitionComponent,      # Face Recognition (dependency of DisplayComponent)
                 SpeechRecognitionComponent,    # Speech Recognition Component (dependency)
                 TextToSpeechComponent):        # Text to Speech (dependency)
    
    pass  # Application does not need to react to events :)





if __name__ == '__main__':

    # Run DisplayApp with Backend specified in Global Config File
    DisplayApp(config.get_backend()).run()
