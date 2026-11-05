from pathlib import Path

from training.config_utils import load_detector_config

ROOT = Path(__file__).resolve().parents[1] / 'training' / 'config' / 'detector'


def test_main_config_inherits_dfd_hr():
    cfg = load_detector_config(ROOT / 'dfd_hr_pmil.yaml')
    assert cfg['model_name'] == 'dfd_hr_pmil'
    assert cfg['backbone_config']['capacity'] == 0.75          # from dfd_hr.yaml
    assert cfg['resolution'] == 448
    assert cfg['part_mil']['freeze_base'] is True
    assert '_base_' not in cfg


def test_ablation_overrides_only_listed_keys():
    cfg = load_detector_config(ROOT / 'pmil' / 'abl_no_boundary.yaml')
    assert cfg['part_mil']['parts'] == ['eyes_brows', 'nose', 'mouth', 'skin']
    assert cfg['part_mil']['tau'] == 0.5
    assert cfg['part_mil']['init_from'] == './checkpoints/ckpt_best.pth'
    e2e = load_detector_config(ROOT / 'pmil' / 'e2e_scratch.yaml')
    assert e2e['part_mil']['freeze_base'] is False and e2e['part_mil']['init_from'] is None
    assert e2e['nEpochs'] == 20


def test_all_configs_load():
    for path in [ROOT / 'dfd_hr.yaml', *sorted((ROOT / 'pmil').glob('*.yaml'))]:
        assert load_detector_config(path)['model_name'] in {'dfd_hr', 'dfd_hr_pmil'}
