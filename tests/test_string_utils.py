from nlp_architect.utils.string_utils import StringUtils


def test_is_determiner():
    assert StringUtils.is_determiner('the')
    assert StringUtils.is_determiner('on') is False


def test_is_preposition():
    assert StringUtils.is_preposition('the') is False
    assert StringUtils.is_preposition('on')


def test_is_pronoun():
    assert StringUtils.is_pronoun('anybody')
    assert StringUtils.is_pronoun('the') is False


def test_is_stopword():
    assert StringUtils.is_stop('always')
    assert StringUtils.is_stop('sunday') is False
