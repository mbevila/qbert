import enum


def _unchanged(string: str) -> str:
    return string


def _longest(string: str) -> str:
    return max(string.replace("_", " ").split(" "), key=len)


def _first(string: str) -> str:
    return string.replace("_", " ").split(" ")[0]


def _last(string: str) -> str:
    return string.replace("_", " ").split(" ")[-1]


def make_offset(synset):
    return "wn:" + str(synset.offset()).zfill(8) + synset.pos()


class MWEStrategy(enum.Enum):

    UNCHANGED = _unchanged
    LONGEST = _longest
    FIRST = _first
    LAST = _last