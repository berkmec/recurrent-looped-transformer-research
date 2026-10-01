import importlib
import numpy as np
import pytest
import torch


def module():
    try:return importlib.import_module('integration.run_scale')
    except ModuleNotFoundError:pytest.fail('larger-scale evidence harness not implemented')


def test_checkpoint_selection_uses_validation_prefix_loss_not_test_or_accuracy():
    m=module()
    rows=[{'step':100,'validation_loss':.3,'validation_accuracy':1.},
          {'step':200,'validation_loss':.2,'validation_accuracy':.9},
          {'step':300,'validation_loss':.2,'validation_accuracy':1.}]
    assert m.select_checkpoint(rows)==200


def test_weight_hash_ignores_mapping_order_but_detects_values():
    m=module();a={'z':torch.tensor([1.,2.]),'a':torch.tensor([3.])}
    b={'a':a['a'].clone(),'z':a['z'].clone()}
    assert m.weight_hash(a)==m.weight_hash(b)
    b['z'][1]=5.
    assert m.weight_hash(a)!=m.weight_hash(b)


def test_scale_batches_match_core_training_stream_exactly():
    m=module();x,y=m.data(17,32,32,100000+1009)
    assert np.array_equal(x.numpy(),np.random.default_rng(104258).choice(3,(32,32),p=[.4375,.4375,.125]))
    assert np.array_equal(y[:,-1].numpy(),((x.numpy()==1).sum(1)%2))


def test_protocol_write_never_overwrites_prior_evidence(tmp_path):
    m=module();path=tmp_path/'protocol.json'
    m.exclusive_json(path,{'v':1})
    with pytest.raises(FileExistsError):m.exclusive_json(path,{'v':2})
    assert path.read_text().find('1')>=0
