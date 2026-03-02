from .analizer import Analizer


class CharacterGenerator:
    def __init__(self, char_id: str = 'default'):
        self.char_id = char_id


class ConsoleGenerator:
    def __init__(self, analyzer: Analizer):
        self.analyzer = analyzer

    def generate(self):
        if self.analyzer._nose == Analizer.NoseState.CENTER:
            print('(  .  )')
        elif self.analyzer._nose == Analizer.NoseState.LEFT:
            print('( .   )')
        elif self.analyzer._nose == Analizer.NoseState.RIGHT:
            print('(   . )')
