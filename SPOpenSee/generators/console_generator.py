from ..analyzers.face_analyzer import Analizer


class ConsoleGenerator:
    def __init__(self, analyzer: Analizer):
        self.analyzer = analyzer

    def generate(self):
        if self.analyzer._nose == Analizer.NoseState.CENTER:
            print("_______")
            print("(     )")
            print("(  .  )")
            print("(     )")
            print("-------")
        elif self.analyzer._nose == Analizer.NoseState.LEFT:
            print("_______")
            print("(     )")
            print("( .   )")
            print("(     )")
            print("-------")
        elif self.analyzer._nose == Analizer.NoseState.RIGHT:
            print("_______")
            print("(     )")
            print("(   . )")
            print("(     )")
            print("-------")
        elif self.analyzer._nose == Analizer.NoseState.UP:
            print("_______")
            print("(  .  )")
            print("(     )")
            print("(     )")
            print("-------")
        elif self.analyzer._nose == Analizer.NoseState.DOWN:
            print("_______")
            print("(     )")
            print("(     )")
            print("(  .  )")
            print("-------")
        elif self.analyzer._nose == Analizer.NoseState.UP_LEFT:
            print("_______")
            print("( .   )")
            print("(     )")
            print("(     )")
            print("-------")
        elif self.analyzer._nose == Analizer.NoseState.UP_RIGHT:
            print("_______")
            print("(   . )")
            print("(     )")
            print("(     )")
            print("-------")
        elif self.analyzer._nose == Analizer.NoseState.DOWN_LEFT:
            print("_______")
            print("(     )")
            print("(     )")
            print("( .   )")
            print("-------")
        elif self.analyzer._nose == Analizer.NoseState.DOWN_RIGHT:
            print("_______")
            print("(     )")
            print("(     )")
            print("(   . )")
            print("-------")
