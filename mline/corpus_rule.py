"""User rule (2026-10-24 ~10:10, program note 5.71): no model is trained on the CNN corpus again; datasets are chosen by purpose (natural
language -> a natural-language dataset). The training entry points (mline.longrun, mline.rltm_train, mline.rl0) call check_trainable on
the corpus they would train on and stop with a clear message if it is the CNN one. Evaluating models that were trained on it is not
training and is not blocked. The corpus directory comes from MLINE_DATA (mline.train.DATA), so a purpose-chosen corpus needs no code change.
"""
import json
from pathlib import Path

CLOSED_SOURCES = ('cnn_dailymail',)


class ClosedCorpus(SystemExit):
    pass


def source_of(directory):
    path = Path(directory) / 'corpus.json'
    return json.loads(path.read_bytes().decode('utf8')).get('source') if path.exists() else None


def check_trainable(directory):
    """Raise ClosedCorpus if `directory` is a corpus closed for training; return its recorded source otherwise."""
    source = source_of(directory)
    if source in CLOSED_SOURCES or 'cnn' in Path(directory).name.lower():
        raise ClosedCorpus(f'training on {directory} (source {source}) is closed by the user rule of 2026-10-24 (note 5.71): '
                           'set MLINE_DATA to a purpose-chosen corpus, e.g. __RLT_ROOT__/mline-data/fineweb-edu-bytes')
    return source
