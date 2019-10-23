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

    #     pass  # Application does not need to react to events :)

    GREET_TIMEOUT = 15  # Only Greet people once every X seconds

    def __init__(self, backend):
        """Greets New and Known People"""
        super(DisplayApp, self).__init__(backend)

        self.name_time = {}  # Dictionary of <name, time> pairs, to keep track of who is greeted when

    def on_transcript(self, hypotheses, audio):
        question = hypotheses[0].transcript
        print(question)
        result = Wikipedia.query(question)

        if result:

            # Obtain answer and Thumbnail Image URL from Wikipedia
            answer, url = result
            
            # Limit Answer to a single sentence
            answer = answer.split('.')[0]

            # Tell Answer to Human
            self.say(answer)

        else:

            # Tell Human you don't know
            self.say("I don't know!")





if __name__ == '__main__':

    # Run DisplayApp with Backend specified in Global Config File
    DisplayApp(config.get_backend()).run()
