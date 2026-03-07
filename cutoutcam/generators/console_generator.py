from ..analyzers.face_analyzer import Analizer
from ..state import FState


class ConsoleGenerator:
    def __init__(self, analyzer: Analizer):
        self.analyzer = analyzer

    def generate(self):
        if self.analyzer._nose == FState.CENTER:
            print("_______")
            print("(     )")
            print("(  .  )")
            print("(     )")
            print("-------")
        elif self.analyzer._nose == FState.LEFT:
            print("_______")
            print("(     )")
            print("( .   )")
            print("(     )")
            print("-------")
        elif self.analyzer._nose == FState.RIGHT:
            print("_______")
            print("(     )")
            print("(   . )")
            print("(     )")
            print("-------")
        elif self.analyzer._nose == FState.UP:
            print("_______")
            print("(  .  )")
            print("(     )")
            print("(     )")
            print("-------")
        elif self.analyzer._nose == FState.DOWN:
            print("_______")
            print("(     )")
            print("(     )")
            print("(  .  )")
            print("-------")
        elif self.analyzer._nose == FState.UP_LEFT:
            print("_______")
            print("( .   )")
            print("(     )")
            print("(     )")
            print("-------")
        elif self.analyzer._nose == FState.UP_RIGHT:
            print("_______")
            print("(   . )")
            print("(     )")
            print("(     )")
            print("-------")
        elif self.analyzer._nose == FState.DOWN_LEFT:
            print("_______")
            print("(     )")
            print("(     )")
            print("( .   )")
            print("-------")
        elif self.analyzer._nose == FState.DOWN_RIGHT:
            print("_______")
            print("(     )")
            print("(     )")
            print("(   . )")
            print("-------")
