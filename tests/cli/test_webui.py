"""The web UI's model-free parts: the device list, the size parser, the pipeline cache
key, the train config writer and the page's wiring (no Gradio server, no weights)."""

import ast
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu
ROOT = Path(__file__).resolve().parents[2]
HAVE_GRADIO = importlib.util.find_spec("gradio") is not None
needs_gradio = pytest.mark.skipif(not HAVE_GRADIO, reason="gradio is not installed")

# the writer's form, as the page passes it to `TRAIN_FIELDS`
TRAIN = {
    "weights": "models/MOSS-Image-Base",
    "data": "/data/images",
    "source": "folder",
    "name": "my run",
    "out": "/tmp/webui-test-out",
    "algo": "lora",
    "preset": "attn-mlp",
    "dim": 16,
    "alpha": 16.0,
    "lr": 1e-4,
    "seed": 0,
    "optimizer": "adamw",
    "precision": "auto",
    "parallel": "none",
    "device_index": 1,
    "memory_profile": "balanced",
    "steps": 10,
    "warmup": 1,
    "batch": 1,
    "grad_acc": 1,
    "resolution_mode": "resize",
    "min_side": 512,
    "max_side": 2048,
    "upscale": "never",
    "crop": "center",
    "batching": "tokens",
    "max_tokens": 12288,
    "overlong": "drop",
    "save_every": 5,
    "sample_every": 0,
    "sample_prompts": "",
    "latent_cache": "online",
    "text_cache": "online",
    "resume": False,
}


def _webui():
    """The module (imported lazily: gradio is an extra)."""
    return pytest.importorskip("kohakumoss.webui")


def _knobs(path: Path) -> dict:
    """The config's module-level literals, as Python parses them back."""
    tree = ast.parse(path.read_text(encoding="ascii"))
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
    }


@needs_gradio
def test_devices_are_named_with_an_index():
    """`device_choices` lists every device with its index, then the host."""
    w = _webui()
    choices = w.device_choices()
    assert choices[-1] == "cpu", choices
    for name in choices[:-1]:
        assert name != "cpu" and ":" in name, name
        assert name.split(":")[1].isdigit(), name


def test_device_indices_skips_the_host():
    """`device_indices`: (first accelerator index, every accelerator index); the host
    is never a training device."""
    w = _webui()
    first, all_of = w.device_indices(["npu:0", "npu:1", "cpu"])
    assert (first, all_of) == (0, [0, 1])
    assert w.device_indices(["cpu"]) == (0, [])


def test_parse_size_is_height_by_width():
    w = _webui()
    assert w.parse_size("1024x1024") == (1024, 1024)
    assert w.parse_size("832x1216") == (832, 1216), "height first, as SIZES is (H, W)"
    assert w.parse_size("512X768") == (512, 768), "the X is case-insensitive"


def test_parse_prompts_is_one_per_line():
    """`SAMPLE_PROMPTS`: one prompt per non-blank line; a list passes through."""
    w = _webui()
    assert w.parse_prompts("a cat\n\n  a dog  \n") == ["a cat", "a dog"]
    assert w.parse_prompts("") == []
    assert w.parse_prompts(None) == []
    assert w.parse_prompts(["a cat", " "]) == ["a cat"]


@needs_gradio
def test_generation_key_covers_what_loads_the_model():
    """The cache key holds every load-time setting and nothing of the sampler's."""
    w = _webui()
    base = dict(
        weights="w",
        variant="base",
        transformer="",
        adapter="",
        adapter_mode="merge",
        precision="auto",
        device="npu:0",
        memory_profile="",
    )
    key = w.generation_key(**base)
    other_sampler = dict(steps=10, cfg=5.0, seed=1)
    assert w.generation_key(**base) == key, "the sampler settings are not in the key"
    assert other_sampler, "the sampler settings are passed to generate, not the key"
    for field, value in (
        ("weights", "w2"),
        ("precision", "bf16"),
        ("device", "cpu"),
        ("adapter", "a.safetensors"),
        ("adapter_mode", "bypass"),
        ("memory_profile", "low"),
        ("transformer", "x.safetensors"),
        ("variant", "sft"),
    ):
        changed = w.generation_key(**{**base, field: value})
        assert changed != key, f"{field} does not change the key"


def test_train_config_is_ascii_and_a_valid_python_config(tmp_path):
    """The writer emits one module-level literal per knob, ASCII, that Python parses
    back to the same values (a kogine config is a module)."""
    w = _webui()
    path = w.train_config({**TRAIN, "out": str(tmp_path / "out")})
    got = _knobs(path)
    assert got["ALGO"] == "lora" and got["DIM"] == 16
    assert got["DEVICES"] == (1,), "PARALLEL='none' trains on the form's index"
    assert got["DATASETS"] == [{"source": "folder", "path": "/data/images"}]
    assert got["PARALLEL"] == "none"
    assert path.name == "my_run.py", "the run name is sanitised for a file name"
    assert path.read_text(encoding="ascii").lstrip().startswith('"""'), (
        "the docstring names the script, so `kohakumoss-config check` reads the file"
    )


def test_train_config_takes_every_accelerator_for_a_parallel_run(tmp_path):
    """A multi-device PARALLEL gets every accelerator of the host: one index in the form
    is the *first* device, not the only one (a tp run with DEVICES=(0,) never starts)."""
    w = _webui()
    mono = w.device_choices()
    if len(mono) < 3:
        pytest.skip("this host has fewer than two accelerators")
    first, all_of = w.device_indices(mono)
    path = w.train_config(
        {**TRAIN, "out": str(tmp_path / "o"), "parallel": "ddp", "optimizer": "muon"}
    )
    got = _knobs(path)
    assert got["DEVICES"] == tuple(all_of), got["DEVICES"]
    assert got["OPTIMIZER"] == "muon", "ddp carries Muon (only tp / fsdp refuse it)"


def test_train_config_forces_a_resident_plan_for_a_sharding_layout(tmp_path):
    """`tp` / `fsdp` / `pipeline` / `threaded` need a resident plan: the form's
    MEMORY_PROFILE is overridden rather than left to fail the knob check."""
    w = _webui()
    path = w.train_config(
        {**TRAIN, "out": str(tmp_path / "o"), "parallel": "tp", "memory_profile": "low"}
    )
    assert _knobs(path)["MEMORY_PROFILE"] == "ckpt-resident"
    kept = w.train_config(
        {**TRAIN, "out": str(tmp_path / "o"), "parallel": "ddp", "memory_profile": "low"}
    )
    assert _knobs(kept)["MEMORY_PROFILE"] == "low", "ddp runs every plan"


def test_train_config_writes_the_sample_prompts_it_needs(tmp_path):
    """SAMPLE_EVERY > 0 needs SAMPLE_PROMPTS: the form's lines are written as the list,
    and an empty field is an error rather than a config that cannot start."""
    w = _webui()
    path = w.train_config(
        {
            **TRAIN,
            "out": str(tmp_path / "o"),
            "sample_every": 500,
            "sample_prompts": "a cat\n\na dog on a mat\n",
        }
    )
    assert _knobs(path)["SAMPLE_PROMPTS"] == ["a cat", "a dog on a mat"]
    off = w.train_config({**TRAIN, "out": str(tmp_path / "o")})
    assert "SAMPLE_PROMPTS" not in _knobs(off), "no sampling, no prompt list"
    import gradio as gr

    with pytest.raises(gr.Error, match="SAMPLE_PROMPTS"):
        w.train_config(
            {**TRAIN, "out": str(tmp_path / "o"), "sample_every": 500,
             "sample_prompts": ""}
        )


def test_check_config_names_the_run_s_own_problems(tmp_path):
    """`check_config` runs the project's knob checks on the written file, so a bad
    combination is reported before a training process starts. A path the user has not
    created is theirs, not a knob problem."""
    w = _webui()
    bad = w.train_config(
        {**TRAIN, "out": str(tmp_path / "o"), "parallel": "tp", "optimizer": "muon"}
    )
    problems = w.check_config(bad)
    assert problems, "tp + muon must be refused"
    assert any("muon" in p and "tp" in p for p in problems), problems
    assert not any("does not exist" in p for p in problems), problems
    good = w.train_config(
        {**TRAIN, "out": str(tmp_path / "o"), "parallel": "ddp", "optimizer": "muon"}
    )
    assert not [p for p in w.check_config(good) if "DEVICES" in p], "ddp takes two"


def test_train_config_refuses_an_empty_path(tmp_path):
    """An empty WEIGHTS / DATASETS path is an error, not a config with "" in it."""
    w = _webui()
    values = {**TRAIN, "weights": "", "out": str(tmp_path)}
    import gradio as gr

    with pytest.raises(gr.Error, match="WEIGHTS"):
        w.train_config(values)
    with pytest.raises(gr.Error, match="DATASETS"):
        w.train_config({**values, "weights": "w", "data": ""})


def test_train_fields_match_the_form_order():
    """`TRAIN_FIELDS` is the order the page passes the form's values in (Gradio binds
    positional inputs), so `start_training` can rebuild the dict."""
    w = _webui()
    assert w.TRAIN_FIELDS[0] == "weights" and w.TRAIN_FIELDS[-1] == "resume"
    assert len(w.TRAIN_FIELDS) == len(set(w.TRAIN_FIELDS)) == len(TRAIN)
    assert tuple(TRAIN) == w.TRAIN_FIELDS, "the test form and the page disagree"
    got = w._train_values(tuple(range(len(w.TRAIN_FIELDS))))
    assert got["weights"] == 0 and got["resume"] == len(w.TRAIN_FIELDS) - 1


@needs_gradio
def test_the_page_has_both_tabs():
    """The Generate tab and the (optional) Train tab are wired."""
    w = _webui()
    page = w.build_ui()
    import gradio as gr

    assert isinstance(page, gr.Blocks)
    kinds = [b.__class__.__name__ for b in page.blocks.values()]
    assert "TabItem" in kinds or "Tab" in kinds, kinds
    # the tab labels are the text its Markdown renders
    texts = " ".join(
        str(getattr(b, "value", "") or "") for b in page.blocks.values()
    )
    assert "Training" in texts or "Finetune" in texts, texts[:400]
    assert "Generate" in texts, texts[:400]


def test_release_closes_the_engine():
    """`Engine.close` drops the cached pipeline (so the next load rebuilds it)."""
    w = _webui()
    engine = w.Engine()
    engine.key, engine.pipe = ("k",), object()
    engine.close()
    assert engine.pipe is None and engine.key is None
    built = []
    first = object()
    got = engine.get(("a",), lambda: built.append(1) or first)
    again = engine.get(("a",), lambda: built.append(2) or object())
    assert got is first and again is first, "the second get rebuilt the pipeline"
    assert built == [1], built
    other = engine.get(("b",), lambda: built.append(3) or object())
    assert other is not first and built == [1, 3], "a new key must rebuild"


# --------------------------------------------------------------- job state / recovery

FAKE_TRAINER = """
import sys, time
for i in range(400):
    print(f"tick {i}", flush=True)
    time.sleep(0.05)
"""


def _fake_kogine(tmp_path, body=FAKE_TRAINER):
    """A stand-in `kogine` command that runs a small script (no training, no weights)."""
    import sys

    script = tmp_path / "fake_train.py"
    script.write_text(body)
    runner = tmp_path / "fake_kogine.sh"
    runner.write_text(f'#!/bin/sh\n"{sys.executable}" "{script}"\n')
    runner.chmod(0o755)
    return runner


def test_a_job_is_found_again_by_a_new_trainer(tmp_path, monkeypatch):
    """The page holds no job state of its own: `state.json` + the log file are the
    hand-off, so a reloaded (or restarted) page finds the running child by pid."""
    import time

    w = _webui()
    monkeypatch.setattr(w, "kogine", lambda: [str(_fake_kogine(tmp_path))])
    first = w.Trainer(state_dir=tmp_path / "state")
    config = tmp_path / "state" / "run.py"
    first.write_config(config, {"WEIGHTS": "x"})
    first.start(config, output=tmp_path, name="fake-run")
    try:
        assert first.running()
        time.sleep(0.6)
        # a fresh Trainer stands for the page process after a reload
        second = w.Trainer(state_dir=tmp_path / "state")
        assert second.job is not None, "no job recovered from state.json"
        assert second.job.name == "fake-run"
        assert second.job.status == "running"
        assert second.job.alive(), "the child must outlive the page"
        assert "tick" in second.log(), "the log is on disk, not in a pipe"
        assert second.job.pid == first.job.pid
    finally:
        first.stop()


@pytest.mark.slow
def test_the_child_outlives_the_page_process(tmp_path):
    """A killed page process does not kill the job: its stdout is a file, not a pipe
    (a pipe would raise BrokenPipeError in the child). Two real processes: slow."""
    import subprocess
    import sys
    import time

    w = _webui()
    state = tmp_path / "state"
    state.mkdir()
    fake = _fake_kogine(tmp_path)
    page = tmp_path / "page.py"
    page.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
        "import kohakumoss.webui as w\n"
        "from pathlib import Path\n"
        f"w.kogine = lambda: [{str(fake)!r}]\n"
        f"t = w.Trainer(state_dir=Path({str(state)!r}))\n"
        f"cfg = Path({str(state / 'run.py')!r})\n"
        't.write_config(cfg, {"WEIGHTS": "x"})\n'
        f"t.start(cfg, output=Path({str(tmp_path)!r}), name='survivor')\n"
        "print(t.job.pid, flush=True)\n"
    )
    out = subprocess.run(
        [sys.executable, str(page)], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    pid = int(out.stdout.strip())
    try:
        time.sleep(1.0)
        assert Path(f"/proc/{pid}").exists(), "the child died with the page"
        before = len((state / w.LOG_FILE).read_text().splitlines())
        time.sleep(1.0)
        after = len((state / w.LOG_FILE).read_text().splitlines())
        assert after > before, "the child stopped writing its log"
        recovered = w.Trainer(state_dir=state)
        assert recovered.job.status == "running" and recovered.job.alive()
    finally:
        import os
        import signal

        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass


def test_a_dead_process_is_not_reported_running(tmp_path):
    """A job whose pid is gone reads as finished, not as a running job the page lost."""
    import json

    w = _webui()
    state = tmp_path / "state"
    state.mkdir()
    (state / w.STATE_FILE).write_text(
        json.dumps(
            {
                "config": str(tmp_path / "run.py"),
                "output": str(tmp_path),
                "name": "dead",
                "pid": 999999,
                "status": "running",
            }
        )
    )
    trainer = w.Trainer(state_dir=state)
    assert trainer.job is not None
    assert trainer.job.status == "finished", trainer.job.status
    assert not trainer.running()
    saved = json.loads((state / w.STATE_FILE).read_text())
    assert saved["status"] == "finished", "the correction is written back"


def test_the_form_is_restored_from_the_last_config(tmp_path, monkeypatch):
    """`restore_form` reads the last job's config back into the form's fields, so a
    reloaded page shows the running run's settings instead of the defaults."""
    w = _webui()
    state = tmp_path / "state"
    config = w.train_config({**TRAIN, "out": str(tmp_path / "out"), "parallel": "ddp"})
    state.mkdir()
    (state / w.STATE_FILE).write_text(
        json.dumps(
            {
                "config": str(config),
                "output": str(config.parent),
                "name": "my run",
                "pid": 0,
                "status": "finished",
            }
        )
    )
    monkeypatch.setattr(w, "TRAINER", w.Trainer(state_dir=state))
    updates = w.restore_form()
    assert len(updates) == len(w.TRAIN_FIELDS)
    got = {name: upd.get("value") for name, upd in zip(w.TRAIN_FIELDS, updates)}
    assert got["weights"] == TRAIN["weights"]
    assert got["parallel"] == "ddp"
    assert got["out"] == str(tmp_path / "out")
    assert got["sample_prompts"] == TRAIN["sample_prompts"]


# --------------------------------------------------------------------------- metrics


def _write_metrics(root: Path, rows):
    run = root / "run"
    run.mkdir(parents=True, exist_ok=True)
    (run / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def test_metrics_are_read_from_the_run_s_own_file(tmp_path):
    """`metrics.jsonl` is the run's record: one row every LOG_EVERY steps."""
    w = _webui()
    _write_metrics(
        tmp_path,
        [
            {"step": 10, "loss": 0.5, "grad_norm": 0.06, "lr": 2e-6, "step_time": 4.8},
            {"step": 20, "loss": 0.4, "grad_norm": 0.05, "lr": 4e-6, "step_time": 4.7},
        ],
    )
    frame = w.read_metrics(tmp_path)
    assert list(frame["step"]) == [10, 20]
    assert frame["loss"].iloc[-1] == pytest.approx(0.4)
    long = w.metric_series(frame)
    assert set(long["metric"]) == {"loss", "grad_norm", "lr", "s_per_step"}
    assert len(long) == 8, "four metrics over two rows"
    text = w.metric_summary(frame, w.Job(config="c", output=str(tmp_path), name="run"))
    assert "step **20**" in text and "loss **0.4000**" in text
    assert "4.70 s/step" in text


def test_metrics_of_a_missing_run_are_an_empty_frame(tmp_path):
    """No job yet (or a run that has not reached LOG_EVERY): an empty frame, not an
    error, so the page can render the empty chart."""
    w = _webui()
    empty = w.read_metrics(tmp_path / "nope")
    assert empty.empty
    assert list(w.metric_series(empty)) == ["step", "metric", "value"]
    assert w.metric_table(empty).empty
    assert "No training job" in w.metric_summary(empty, None)


def test_metric_summary_reports_a_running_job(tmp_path):
    """The headline names the job, its state and where its output is."""
    w = _webui()
    _write_metrics(tmp_path, [{"step": 5, "loss": 1.5}])
    job = w.Job(config="c", output=str(tmp_path), name="moss-run", status="running")
    text = w.metric_summary(w.read_metrics(tmp_path), job)
    assert "moss-run" in text and "training" in text and str(tmp_path) in text
    assert "step **5**" in text


# ------------------------------------------------------------------------------- npu

NPU_SMI = """
+------------------------------------------------------------------------------------------------------------------+
| NPU   Name                | Health        | Power(W)             Temp(C)                 Hugepages-Usage(page)   |
| Chip  Phy-ID              | Bus-Id        | AICore(%)            Memory-Usage(MB)        HBM-Usage(MB)           |
+===========================+===============+======================================================================+
| 1     Ascend910           | OK            | 623.2                68                      0    / 0                |
| 0     2                   | 0000:99:00.0  | 100                  0    / 0                39584/ 65536            |
+------------------------------------------------------------------------------------------------------------------+
| 1     Ascend910           | OK            | -                    69                      0    / 0                |
| 1     3                   | 0000:9B:00.0  | 100                  0    / 0                37592/ 65536            |
+===========================+===============+======================================================================+
"""


def test_npu_smi_parses_power_temp_and_hbm():
    """`npu-smi info`: two lines per chip (card, then chip); the wide last cell holds
    several padded fields, so they are pulled out by shape (`A / B` is a fraction)."""
    w = _webui()
    chips = w.parse_npu_smi(NPU_SMI)
    assert len(chips) == 2, chips
    first, second = chips
    assert (first.card, first.chip) == (1, 0)
    assert first.name == "Ascend910" and first.health == "OK"
    assert first.power_w == pytest.approx(623.2)
    assert first.temp_c == pytest.approx(68)
    assert first.aicore_pct == pytest.approx(100)
    assert (first.hbm_used_mb, first.hbm_total_mb) == (39584, 65536)
    assert first.hbm_pct == pytest.approx(60.4, abs=0.1)
    assert second.power_w is None, "a '-' cell is 'no reading', not 0 W"
    assert (second.chip, second.temp_c) == (1, 69)


def test_npu_smi_stops_at_the_process_table():
    """The process table after the chips also starts with two digits; its cells are
    names, not readings, and must not be parsed as a chip (or crash the parser)."""
    w = _webui()
    text = NPU_SMI + """
+---------------------------+---------------+------------------------------------+
| NPU     Chip              | Process id    | Process name                       |
+===========================+===============+====================================+
| 1       0                 | 1158043       | python3.12                         |
+===========================+===============+====================================+
"""
    chips = w.parse_npu_smi(text)
    assert len(chips) == 2, chips
    assert chips[0].hbm_used_mb == 39584


def test_npu_panel_is_markdown_and_a_table():
    """The panel renders a per-chip line and a table row with the live fields."""
    w = _webui()
    chips = w.parse_npu_smi(NPU_SMI)
    text = w.npu_summary(chips)
    assert "`1:0`" in text and "623 W" in text and "68 C" in text
    assert "HBM 39584 / 65536 MB (60%)" in text
    frame = w.npu_frame(chips)
    assert list(frame["card"]) == [1, 1]
    assert {"power W", "temp C", "AICore %", "HBM"} <= set(frame.columns)


def test_npu_panel_without_a_device():
    """A host without `npu-smi` says so instead of rendering an empty chart."""
    w = _webui()
    assert w.npu_summary([]).startswith("No `npu-smi`")
    assert w.npu_frame([]).empty


def test_npu_stats_uses_the_real_command(monkeypatch):
    """`npu_stats` runs `npu-smi info` once and parses its table."""
    w = _webui()

    class Done:
        stdout = NPU_SMI

    monkeypatch.setattr(w.shutil, "which", lambda name: "/usr/bin/npu-smi")
    monkeypatch.setattr(w.subprocess, "run", lambda *a, **k: Done())
    chips = w.npu_stats()
    assert len(chips) == 2 and chips[0].hbm_used_mb == 39584


def test_a_dead_job_s_exit_code_is_read_from_its_log(tmp_path):
    """A page that was not running when the job ended cannot know the code; the log
    does (the launcher's last line), so `load` reports finished / failed correctly."""
    w = _webui()
    state = tmp_path / "state"
    state.mkdir()
    (state / w.STATE_FILE).write_text(
        json.dumps(
            {
                "config": str(tmp_path / "run.py"),
                "output": str(tmp_path),
                "name": "done",
                "pid": 999999,
                "status": "running",
            }
        )
    )
    (state / w.LOG_FILE).write_text(
        "step 1 ...\n[INFO] exported run-comfy.safetensors\n"
        "Script executed successfully\n"
    )
    trainer = w.Trainer(state_dir=state)
    assert trainer.job.status == "finished"
    assert trainer.job.exit_code == 0

    (state / w.LOG_FILE).write_text("Traceback (most recent call last)\n")
    (state / w.STATE_FILE).write_text(
        json.dumps(
            {
                "config": str(tmp_path / "run.py"),
                "output": str(tmp_path),
                "name": "failed",
                "pid": 999998,
                "status": "running",
            }
        )
    )
    trainer = w.Trainer(state_dir=state)
    assert trainer.job.status == "finished", "no 'success' line and no code: finished"
    assert trainer.job.exit_code is None

    (state / w.LOG_FILE).write_text("\n--- training exited with code 3 ---\n")
    (state / w.STATE_FILE).write_text(
        json.dumps(
            {
                "config": str(tmp_path / "run.py"),
                "output": str(tmp_path),
                "name": "crashed",
                "pid": 999997,
                "status": "running",
            }
        )
    )
    trainer = w.Trainer(state_dir=state)
    assert (trainer.job.status, trainer.job.exit_code) == ("failed", 3)


def test_read_tail_keeps_the_last_lines_of_a_big_log(tmp_path):
    """The log is on disk and can be large; the page reads only its tail."""
    w = _webui()
    path = tmp_path / "big.log"
    path.write_text("\n".join(f"line {i}" for i in range(5000)) + "\n")
    tail = w.read_tail(path, 3)
    assert tail.splitlines()[-1] == "line 4999"
    assert len(tail.splitlines()) == 3
    assert w.read_tail(tmp_path / "missing.log", 5) == ""


# ------------------------------------------------------------------ resolution / buckets


def test_sides_and_planner_knobs_round_trip():
    """The form's (min side, max side) pair maps onto the planner's TARGET_AREA and
    ASPECT_RANGE: the area is their product, the bounds their ratio."""
    w = _webui()
    assert w.area_and_range(512, 2048) == (1048576, (0.25, 4.0))
    assert w.area_and_range(1024, 1024) == (1048576, (1.0, 1.0))
    assert w.area_and_range(2048, 512) == w.area_and_range(512, 2048), "order free"
    assert w.sides_of(1048576, (0.25, 4.0)) == (512, 2048)
    assert w.sides_of(1048576, (1.0, 1.0)) == (1024, 1024)
    import gradio as gr

    with pytest.raises(gr.Error, match="positive"):
        w.area_and_range(0, 1024)


def test_bucket_mode_uses_the_sides_as_its_bounds():
    """For `bucket` the two sides are the extremes of the set, so both the area and the
    aspect window come from them."""
    w = _webui()
    assert w.resolution_knobs("bucket", 512, 2048) == (1048576, (0.25, 4.0))
    assert w.resolution_knobs("bucket", 768, 1344)[1] == (0.5714, 1.75)


def test_the_other_modes_keep_an_open_aspect_window():
    """`resize` / `square_crop` / `shifted_square` do not size to buckets, so a square
    pair must set the area only: taking the ratio too would crop every image square."""
    w = _webui()
    for mode in ("resize", "square_crop", "shifted_square"):
        area, bounds = w.resolution_knobs(mode, 1024, 1024)
        assert area == 1048576
        assert bounds == (0.25, 4.0), f"{mode} would crop every image to a square"
    from kohakumoss_data.resolution.geometry import Geometry

    area, bounds = w.resolution_knobs("resize", 1024, 1024)
    geo = Geometry(area=area, aspect_range=bounds)
    h, w = geo.fit(1088, 1536)
    assert w != h and abs(w / h - 1088 / 1536) < 0.05, "the aspect must survive"


def test_bucket_preview_lists_the_planner_s_buckets():
    """`RESOLUTION_MODE='bucket'` sizes every image to one of these buckets; the preview
    is the planner's own list (not a reimplementation)."""
    w = _webui()
    text, frame, _ = w.bucket_preview(512, 2048, "never")
    assert "97 buckets" in text
    assert len(frame) == 97
    assert set(frame.columns) == {"width", "height", "aspect", "image tokens"}
    assert frame["aspect"].iloc[0] == pytest.approx(0.25, abs=0.01)
    assert frame["aspect"].iloc[-1] == pytest.approx(4.0, abs=0.01)
    assert frame["width"].min() == 512 and frame["width"].max() == 2048
    # every bucket is ~the target area, so the token counts are close
    assert frame["image tokens"].min() > 3900
    assert frame["image tokens"].max() < 4100


def test_bucket_preview_follows_the_two_sides():
    """A narrower side pair gives fewer buckets, from the small side to the large."""
    w = _webui()
    wide, wide_frame, _ = w.bucket_preview(512, 2048, "never")
    narrow, narrow_frame, _ = w.bucket_preview(768, 1344, "never")
    assert len(narrow_frame) < len(wide_frame)
    assert narrow_frame["width"].min() == 768
    assert narrow_frame["width"].max() == 1344
    assert narrow_frame["aspect"].min() == pytest.approx(0.57, abs=0.02)
    assert narrow_frame["aspect"].max() == pytest.approx(1.75, abs=0.02)


def test_a_square_pair_in_bucket_mode_is_one_bucket():
    """`1024 x 1024` in bucket mode is a single bucket, as the sides say."""
    w = _webui()
    text, frame, _ = w.bucket_preview(1024, 1024, "never")
    assert len(frame) == 1 and "1 bucket" in text
    assert tuple(frame.iloc[0][["width", "height"]]) == (1024, 1024)


def test_bucket_preview_reports_bad_sides():
    """Bad sides are a message, not an exception in the UI."""
    w = _webui()
    text, frame, _ = w.bucket_preview(-1, 1024, "never")
    assert "sides" in text
    assert frame.empty


def test_resolution_change_warns_about_the_latent_cache():
    """An offline latent cache keys on the resolution knobs: changing them misses."""
    w = _webui()
    note = w.resolution_note("bucket", "offline")
    assert "latent cache" in note and "RESOLUTION_MODE" in note
    assert "online" in note, "the note must name the way out"
    assert w.resolution_note("bucket", "online") == ""
    assert w.resolution_note("bucket", "off") == ""


def test_the_config_writes_every_resolution_knob(tmp_path):
    """The form's (min side, max side) pair reaches the config as the script's knobs."""
    w = _webui()
    path = w.train_config(
        {
            **TRAIN,
            "out": str(tmp_path / "out"),
            "resolution_mode": "bucket",
            "min_side": 768,
            "max_side": 1344,
            "upscale": "to_area",
            "crop": "random",
            "batching": "tokens_ffd",
            "overlong": "truncate_text",
        }
    )
    got = _knobs(path)
    assert got["RESOLUTION_MODE"] == "bucket"
    assert got["TARGET_AREA"] == 768 * 1344
    assert got["ASPECT_RANGE"] == (0.5714, 1.75), "written as the planner's tuple"
    assert got["UPSCALE"] == "to_area"
    assert got["CROP"] == "random"
    assert got["BATCHING"] == "tokens_ffd"
    assert got["OVERLONG"] == "truncate_text"
    back = w.read_config(path)
    assert (back["min_side"], back["max_side"]) == (768, 1344), "read back as the form's"


def test_an_offline_cache_with_random_crop_is_refused_before_launch(tmp_path):
    """`LATENT_CACHE='offline'` needs a centre crop; the pre-launch check catches it
    (the run would stop at build otherwise)."""
    w = _webui()
    path = w.train_config(
        {**TRAIN, "out": str(tmp_path / "o"), "crop": "random", "latent_cache": "offline"}
    )
    problems = w.check_config(path)
    assert any("CROP" in p or "crop" in p for p in problems), problems


# ------------------------------------------------- late-run growth (the frozen tab)

def _write_long_run(root: Path, rows: int = 1500) -> Path:
    """A run shaped like a 15,000-step one: a row every 10 steps."""
    run = root / "run"
    run.mkdir(parents=True, exist_ok=True)
    (run / "metrics.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "step": step,
                    "loss": 0.5 - step / 100000,
                    "grad_norm": 0.05,
                    "lr": 2e-5,
                    "step_time": 4.7,
                }
            )
            for step in range(10, rows * 10 + 1, 10)
        )
        + "\n"
    )
    return run / "metrics.jsonl"


def test_read_metrics_reads_only_the_tail(tmp_path):
    """A long run's file is read from its tail, so one tick cannot load a whole run."""
    w = _webui()
    _write_long_run(tmp_path, rows=3000)
    frame = w.read_metrics(tmp_path, limit=500)
    assert len(frame) == 500
    assert int(frame["step"].iloc[-1]) == 30000, "the newest row is kept"
    assert frame["step"].is_monotonic_increasing


def test_the_chart_is_thinned_to_a_fixed_size(tmp_path):
    """The plot keeps a bounded number of points whatever the run length: a chart
    rebuilt with every row is what froze the tab late in a run."""
    w = _webui()
    _write_long_run(tmp_path, rows=1500)
    frame = w.read_metrics(tmp_path)
    long = w.metric_series(frame)
    assert len(long) <= w.CHART_POINTS * 4, len(long)
    assert len(long) < len(frame) * 4, "a 1500-row run must be thinned"
    assert long["step"].max() == 15000, "the newest step survives the thinning"
    assert set(long["metric"]) == {"loss", "grad_norm", "lr", "s_per_step"}


def test_thin_keeps_the_extremes_and_a_stable_order():
    """Thinning keeps the first and last row and a monotonic step axis."""
    w = _webui()
    frame = w.pd.DataFrame({"step": range(1000), "loss": range(1000)})
    out = w.thin(frame, 100)
    assert len(out) == 100
    assert out["step"].iloc[0] == 0 and out["step"].iloc[-1] == 999
    assert out["step"].is_monotonic_increasing
    assert len(w.thin(frame.head(50), 100)) == 50, "a short frame is untouched"


def test_refresh_skips_panels_that_did_not_change(tmp_path):
    """The timer's handler returns an empty update for an unchanged panel, so the
    browser is not asked to rebuild the chart / log / NPU table every tick."""
    import json as _json

    w = _webui()
    state = tmp_path / "state"
    state.mkdir()
    out = tmp_path / "out"
    _write_long_run(out, rows=40)
    (state / w.LOG_FILE).write_text("line one\nline two\n")
    w.clear_caches()
    w.TRAINER = w.Trainer(state_dir=state)
    w.TRAINER.job = w.Job(
        config="c", output=str(out), name="run", status="running", pid=1
    )
    page = w.build_ui()
    handler = next(
        fn.fn
        for fn in page.fns.values()
        if len(getattr(fn, "inputs", []) or []) == 1
        and len(getattr(fn, "outputs", []) or []) == 8
    )

    seen, first = handler({})
    assert not _skipped(first), "the first tick must fill every panel"
    seen, second = handler(seen)
    assert _skipped(second) == 7, f"unchanged panels: {_skipped(second)}/7"

    with (state / w.LOG_FILE).open("a") as handle:
        handle.write("line three\n")
    seen, third = handler(seen)
    assert _resent(third) == [1], f"only the log should resend, got {_resent(third)}"

    with (out / "run" / "metrics.jsonl").open("a") as handle:
        handle.write(_json.dumps({"step": 999999, "loss": 0.1, "grad_norm": 0.1}) + "\n")
    seen, fourth = handler(seen)
    assert _resent(fourth) == [2, 3, 4], _resent(fourth)


def _skipped(updates) -> int:
    return sum(
        1
        for value in updates
        if isinstance(value, dict) and value.get("__type__") == "update"
    )


def _resent(updates) -> list[int]:
    return [
        index
        for index, value in enumerate(updates)
        if not (isinstance(value, dict) and value.get("__type__") == "update")
    ]


def test_one_read_serves_every_panel(tmp_path, monkeypatch):
    """A tick reads `metrics.jsonl` and forks `npu-smi` once (not once per panel)."""
    w = _webui()
    state = tmp_path / "state"
    state.mkdir()
    out = tmp_path / "out"
    _write_long_run(out, rows=40)
    w.clear_caches()
    w.TRAINER = w.Trainer(state_dir=state)
    w.TRAINER.job = w.Job(
        config="c", output=str(out), name="run", status="running", pid=1
    )
    page = w.build_ui()
    handler = next(
        fn.fn
        for fn in page.fns.values()
        if len(getattr(fn, "inputs", []) or []) == 1
        and len(getattr(fn, "outputs", []) or []) == 8
    )
    calls = {"metrics": 0, "smi": 0}
    real_metrics, real_smi = w.read_metrics, w.npu_stats
    monkeypatch.setattr(
        w, "read_metrics", lambda *a, **k: (calls.__setitem__("metrics", calls["metrics"] + 1), real_metrics(*a, **k))[1]
    )
    monkeypatch.setattr(
        w, "npu_stats", lambda *a, **k: (calls.__setitem__("smi", calls["smi"] + 1), real_smi(*a, **k))[1]
    )
    w.clear_caches()
    handler({})
    assert calls["metrics"] <= 1, calls
    assert calls["smi"] <= 1, calls


def test_an_unchanged_metrics_file_is_parsed_once(tmp_path):
    """The tick re-reads a file only when it grew: a still run costs no parsing."""
    w = _webui()
    _write_long_run(tmp_path, rows=200)
    w.clear_caches()
    first = w.read_metrics(tmp_path)
    second = w.read_metrics(tmp_path)
    assert second is first, "the same file must not be parsed twice"
    with (tmp_path / "run" / "metrics.jsonl").open("a") as handle:
        handle.write(json.dumps({"step": 99999, "loss": 0.1}) + "\n")
    third = w.read_metrics(tmp_path)
    assert third is not first, "a grown file is re-read"
    assert int(third["step"].iloc[-1]) == 99999
