from .analizer import Analizer


class SquareGenerator:
    ...


class CharacterGenerator:
    def __init__(self, char_id: str = 'default'):
        self.char_id = char_id


class ConsoleGenerator:
    def __init__(self, analyzer: Analizer):
        self.analyzer = analyzer

    def generate(self):
        if self.analyzer._nose == Analizer.NoseState.CENTER:
            print('_______')
            print('(     )')
            print('(  .  )')
            print('(     )')
            print('-------')
        elif self.analyzer._nose == Analizer.NoseState.LEFT:
            print('_______')
            print('(     )')
            print('( .   )')
            print('(     )')
            print('-------')
        elif self.analyzer._nose == Analizer.NoseState.RIGHT:
            print('_______')
            print('(     )')
            print('(   . )')
            print('(     )')
            print('-------')
        elif self.analyzer._nose == Analizer.NoseState.UP:
            print('_______')
            print('(  .  )')
            print('(     )')
            print('(     )')
            print('-------')
        elif self.analyzer._nose == Analizer.NoseState.DOWN:
            print('_______')
            print('(     )')
            print('(     )')
            print('(  .  )')
            print('-------')
        elif self.analyzer._nose == Analizer.NoseState.UP_LEFT:
            print('_______')
            print('( .   )')
            print('(     )')
            print('(     )')
            print('-------')
        elif self.analyzer._nose == Analizer.NoseState.UP_RIGHT:
            print('_______')
            print('(   . )')
            print('(     )')
            print('(     )')
            print('-------')
        elif self.analyzer._nose == Analizer.NoseState.DOWN_LEFT:
            print('_______')
            print('(     )')
            print('(     )')
            print('( .   )')
            print('-------')
        elif self.analyzer._nose == Analizer.NoseState.DOWN_RIGHT:
            print('_______')
            print('(     )')
            print('(     )')
            print('(   . )')
            print('-------')
