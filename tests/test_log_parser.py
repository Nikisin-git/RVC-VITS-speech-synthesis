from app.workers.log_parser import parse_rvc_train_line, parse_tts_train_line
from app.workers.progress_tracker import ProgressState


def test_parse_rvc_epoch_line():
    ev = parse_rvc_train_line("Epoch 10/200, loss_g: 3.521 elapsed 0:05")
    assert ev is not None and ev.kind == "epoch"
    assert ev.epoch == 10 and ev.total_epochs == 200
    assert ev.loss is not None and abs(ev.loss - 3.521) < 1e-6


def test_parse_rvc_save_line():
    ev = parse_rvc_train_line("Saving ckpt at epoch_25")
    assert ev is not None and ev.kind == "checkpoint" and ev.epoch == 25


def test_parse_rvc_stage_line():
    ev = parse_rvc_train_line("=== stage: preprocess ===")
    assert ev is not None and ev.kind == "stage" and ev.message == "preprocess"


def test_parse_tts_epoch():
    ev = parse_tts_train_line(" > EPOCH: 5/100")
    assert ev is not None and ev.kind == "epoch" and ev.epoch == 5 and ev.total_epochs == 100


def test_progress_tracker():
    s = ProgressState()
    s.apply(parse_rvc_train_line("Epoch 50/200, loss: 1.0"))
    assert s.current_epoch == 50 and s.total_epochs == 200
    assert s.percent == 25
    s.apply(parse_rvc_train_line("Saving ckpt at epoch_50"))
    assert s.checkpoints_saved == 1
