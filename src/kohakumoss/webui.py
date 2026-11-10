"""The KohakuMoss web UI: a Gradio page over the pipeline's inference and (optionally)
a training launcher.

    kohakumoss-webui                       # http://127.0.0.1:7860
    kohakumoss-webui --host 0.0.0.0 --port 7861
    python -m kohakumoss.webui [same flags]

Two tabs:

- **Generate** loads one `MossPipeline` on first use (the weights, adapters and
  placement come from the form and are cached per settings tuple), then runs
  `MossPipeline.generate` per click with a live step progress bar.
- **Train** writes a kogine config from the form and runs `scripts/train/moss.py` in a
  child process, streaming its log into the page; a running job can be stopped.

The page holds one model in memory: switching the weights / adapters / precision
releases the old pipeline first. See docs/guides/webui.md.
"""

import argparse
import ast
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import gradio as gr
import pandas as pd

from kohakumoss.inference import MossPipeline, knobs
from kohakumoss.inference.placement_report import forward_tokens
from kohakumoss.logging.log import get_logger, setup
from kohakumoss.offload.accel import configure_allocator, default_type
from kohakumoss_kernels import compile_setup

log = get_logger(__name__)

TRAIN_SCRIPT = "scripts/train/moss.py"


def project_root() -> Path:
    """The clone the page runs from: the package's own parent (`kohakumoss.webui` ->
    `src/kohakumoss` -> `src` -> the clone) when it holds the train script, else the
    working directory (a checkout started from its root)."""
    packaged = Path(__file__).resolve().parents[2]
    if (packaged / TRAIN_SCRIPT).is_file():
        return packaged
    return Path.cwd()


ROOT = project_root()
SIZES = ("1024x1024", "832x1216", "1216x832", "1024x768", "768x1024", "512x512")
PRECISIONS = ("auto", "bf16", "fp16", "fp32")
MEMORY_PROFILES = ("balanced", "speed", "low", "minimal", "ckpt-resident", "auto")
# the inference side leaves it unset (the pipeline's own placement)
SAMPLE_MEMORY_PROFILES = ("", *MEMORY_PROFILES)
ADAPTER_MODES = ("merge", "bypass")
CFG_MODES = ("packed", "two_pass")
ALGOS = ("lora", "locon", "loha", "lokr", "dora", "oft", "boft", "glora", "ia3", "full")
OPTIMIZERS = ("adamw", "adamw16", "muon", "prodigy", "schedule_free")
PARALLELS = ("none", "ddp", "fsdp", "pipeline", "threaded", "tp")
SOURCES = ("kohya", "folder", "jsonl", "parquet", "webdataset")
CACHES = ("online", "offline")
# RESOLUTION_MODE: how each image is sized before the VAE (docs/concepts/data.md)
RESOLUTION_MODES = ("resize", "bucket", "square_crop", "shifted_square")
# the batching rule over a batch's samples (docs/concepts/data.md#batching)
BATCHINGS = ("tokens", "tokens_ffd", "count")
# a sample longer than MAX_TOKENS: drop it, cut its text, or drop random image tokens
OVERLONG_MODES = ("drop", "truncate_text", "drop_tokens")
UPSCALES = ("never", "to_area")
CROPS = ("center", "random")
ASPECT_RANGES = ("0.25:4", "0.33:3", "0.5:2", "0.67:1.5", "1:1")
LOG_LINES = 400
# bytes read from the log's tail for `LOG_LINES` lines; a step line is ~110 bytes, so
# this leaves room for wrapped tracebacks without reading a whole multi-MB log
LOG_BYTES = 96 * 1024
STEP_POLL_S = 1.5
# the metrics chart keeps at most this many points per curve (a 15k-step run writes
# 1500 rows; plotting all of them every tick is what froze the browser late in a run)
CHART_POINTS = 240
# `npu-smi info` costs ~35 ms (a process launch): read it once per this many seconds
NPU_POLL_S = 4.0
# the live table under the chart
TABLE_ROWS = 20
# a run longer than this many rows is read from its tail only
METRICS_TAIL_ROWS = 4000


def device_of(kind: str, index: int = 0) -> str:
    """`"npu"` -> `"npu:0"`; the host is `"cpu"`."""
    return "cpu" if kind == "cpu" else f"{kind}:{index}"


def device_choices() -> list[str]:
    """Every accelerator device this host has, then the host."""
    kind = default_type()
    out = []
    if kind != "cpu":
        mod = getattr(__import__("torch"), kind)
        out = [device_of(kind, i) for i in range(mod.device_count())]
    return out + ["cpu"]


def kogine() -> list[str]:
    """The command prefix that runs kogine: this interpreter's environment's `kogine`
    console script, else one on PATH, else `python -m kohakuengine.cli`."""
    candidate = Path(sys.executable).with_name("kogine")
    if candidate.is_file():
        return [str(candidate)]
    found = shutil.which("kogine")
    if found:
        return [found]
    return [sys.executable, "-m", "kohakuengine.cli"]


def device_indices(devices: list[str]) -> tuple[int, list[int]]:
    """(the host's default index, every accelerator index) of a `device_choices` list;
    the host is never a training device."""
    accelerators = [
        int(name.split(":")[1]) for name in devices if ":" in name and name != "cpu"
    ]
    return (accelerators[0] if accelerators else 0), accelerators


# --------------------------------------------------------------------------- generate


class Engine:
    """One loaded pipeline, keyed by the settings that built it.

    A settings change drops the old pipeline (freeing its VRAM) before the new one is
    built; a failed build leaves nothing behind, so the next click retries. The lock
    serialises loads and generations (one model, one device)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.key = None
        self.pipe = None

    def get(self, key: tuple, build) -> MossPipeline:
        """The cached pipeline of `key`, built by `build()` when it changed."""
        with self.lock:
            if self.key == key and self.pipe is not None:
                return self.pipe
            self.close()
            pipe = build()
            self.key, self.pipe = key, pipe
            return pipe

    def close(self) -> None:
        """Release the current pipeline and its device memory."""
        if self.pipe is None:
            return
        try:
            self.pipe.release()
        except Exception:  # noqa: BLE001 - a broken teardown must not block a rebuild
            log.warning("releasing the previous pipeline failed", exc_info=True)
        self.pipe, self.key = None, None


ENGINE = Engine()


def generation_key(
    weights, variant, transformer, adapter, adapter_mode, precision, device, memory_profile
) -> tuple:
    """Everything that decides what the pipeline loads (not the sampler settings)."""
    return (
        str(weights),
        str(variant),
        str(transformer),
        str(adapter),
        str(adapter_mode),
        str(precision),
        str(device),
        str(memory_profile),
    )


def build_pipeline(
    weights,
    variant,
    transformer,
    adapter,
    adapter_mode,
    precision,
    device,
    memory_profile,
    text_cpu,
    size,
    steps,
    cfg,
    cfg_mode,
    batch,
) -> MossPipeline:
    """`MossPipeline.from_pretrained` from the form's settings."""
    if not str(weights).strip():
        raise gr.Error("WEIGHTS is empty: point it at a ComfyUI models folder, a weights root or a pipeline dir")
    configure_allocator(device)
    compile_setup.apply()
    adapters = [knobs.parse_adapter(a) for a in [adapter] if str(a).strip()]
    targets = [(parse_size(size)[0], parse_size(size)[1])]
    memory = str(memory_profile or "") or None
    return MossPipeline.from_pretrained(
        str(weights).strip(),
        variant=str(variant) or "base",
        transformer=str(transformer).strip() or None,
        dtype=knobs.compute_dtype(str(precision), device),
        device=device,
        adapters=adapters,
        adapter_mode=str(adapter_mode),
        text_device="cpu" if text_cpu else None,
        memory=memory,
        tune="off",
        forward_tokens=forward_tokens(
            targets, min(int(batch) or 1, len(targets)), float(cfg), str(cfg_mode)
        ),
        vae_dtype=knobs.resolve_vae_dtype("auto", device),
        text_plan=knobs.text_spec({"TEXT_ENCODE": {}, "TEXT_DEVICE": "cpu" if text_cpu else ""}),
        vae_plan=knobs.vae_spec({}),
    )


def parse_size(text: str) -> tuple[int, int]:
    """`"1024x1024"` -> `(1024, 1024)` (height, width)."""
    h, w = str(text).lower().split("x")
    return int(h), int(w)


class Progress:
    """A `step_callback` that reports the sampler's progress to Gradio."""

    def __init__(self, steps: int, progress: gr.Progress):
        self.steps = max(int(steps), 1)
        self.progress = progress
        self.seen = 0

    def __call__(self, step: int, sample) -> None:
        self.seen = step
        self.progress(min(step / self.steps, 0.98), desc=f"sampling {step}/{self.steps}")


def generate(
    weights,
    variant,
    transformer,
    adapter,
    adapter_mode,
    precision,
    device,
    memory_profile,
    text_cpu,
    prompt,
    negative,
    size,
    steps,
    cfg,
    shift,
    seed,
    batch,
    cfg_mode,
    skip_layers,
    progress=gr.Progress(),
):
    """One Generate click: build (or reuse) the pipeline, sample, return the images."""
    key = generation_key(
        weights, variant, transformer, adapter, adapter_mode, precision, device, memory_profile
    ) + (bool(text_cpu),)
    progress(0.0, desc="loading the model")
    pipe = ENGINE.get(
        key,
        lambda: build_pipeline(
            weights,
            variant,
            transformer,
            adapter,
            adapter_mode,
            precision,
            device,
            memory_profile,
            text_cpu,
            size,
            steps,
            cfg,
            cfg_mode,
            batch,
        ),
    )
    prompt = str(prompt or "").strip()
    if not prompt:
        raise gr.Error("the prompt is empty")
    h, w = parse_size(size)
    progress(0.02, desc="encoding the prompt")
    started = time.perf_counter()
    result = pipe.generate(
        [prompt],
        negative_prompts=[str(negative or "")],
        sizes=[(h, w)],
        steps=int(steps),
        cfg=float(cfg),
        shift=float(shift),
        skip_layers=knobs.skip_layers(str(skip_layers)),
        seeds=[int(seed)],
        batch=min(int(batch) or 1, 1),
        cfg_mode=str(cfg_mode),
        progress=False,
        step_callback=Progress(int(steps), progress),
    )
    took = time.perf_counter() - started
    progress(1.0, desc=f"done in {took:.1f}s")
    meta = {
        "prompt": prompt,
        "negative": str(negative or ""),
        "size": f"{h}x{w}",
        "steps": int(steps),
        "cfg": float(cfg),
        "shift": float(shift),
        "seed": int(seed),
        "seconds": round(took, 2),
    }
    return result.images[0], json.dumps(meta, indent=1)


# ------------------------------------------------------------------------------ train

STATE_ROOT = Path("out") / "webui"
STATE_FILE = "state.json"
LOG_FILE = "train.log"
METRICS = "metrics.jsonl"


@dataclass
class Job:
    """The training job the page knows about, as `state.json` stores it.

    A page reload rebuilds this from disk: the child writes its own log file (not a
    pipe), so it keeps training while the page is gone, and `pid` is how the new page
    finds it again."""

    config: str
    output: str
    name: str
    pid: int = 0
    status: str = "running"  # running | finished | stopped | failed | unknown
    exit_code: int | None = None
    started: float = 0.0
    updated: float = 0.0

    @property
    def run_dir(self) -> Path:
        """The run's directory: OUTPUT_DIR (where kogine writes run/)."""
        return Path(self.output)

    def alive(self) -> bool:
        """`pid` is a live process of this host (not a zombie)."""
        if self.pid <= 0:
            return False
        try:
            os.kill(self.pid, 0)
        except (ProcessLookupError, PermissionError):
            return False
        try:
            stat = Path(f"/proc/{self.pid}/stat").read_text()
        except OSError:
            return True
        state = stat.rsplit(")", 1)[-1].split()[0] if ")" in stat else ""
        return state not in ("Z", "X")


def read_tail(path: Path, lines: int, bytes_: int = LOG_BYTES) -> str:
    """The last `lines` lines of `path` as one string; "" when it does not exist.

    Reads at most `bytes_` from the end (a 15 MB log is the norm late in a run), so the
    cost of one tick does not grow with the run's length."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            block = min(size, max(bytes_, 1))
            handle.seek(size - block)
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def tail_stamp(path: Path) -> tuple[int, int] | None:
    """(file size, mtime in ns) of `path`: changes whenever a line is appended. Used to
    skip the log panel when nothing was written since the last tick."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_size, stat.st_mtime_ns


class Trainer:
    """One training job: a `scripts/train/moss.py` child process.

    The child writes `<state dir>/train.log` itself (stdout/stderr redirect, no pipe),
    so it outlives the page: a reload (or a restarted page) reads the log and the run's
    metrics back from disk and finds the process again by its pid. `state.json` in the
    state dir is the hand-off."""

    def __init__(self, state_dir: Path | None = None):
        self.lock = threading.Lock()
        self.job: Job | None = None
        self.state_dir = state_dir or Path(
            os.environ.get("KOHAKUMOSS_WEBUI_STATE") or (ROOT / STATE_ROOT)
        )
        self.state_path = self.state_dir / STATE_FILE
        self.load()

    # -- state ---------------------------------------------------------------
    def load(self) -> Job | None:
        """The job `state.json` names.

        A job marked `running` whose process is gone is resolved from the evidence the
        run left on disk: its log's last line (the launcher's own "exited with code N",
        or "Script executed successfully") and its metrics file. A page that was not
        running when the job ended cannot know the exit code, so the state is
        `finished` unless the log says otherwise."""
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.job = Job(**raw)
        except (OSError, ValueError, TypeError):
            self.job = None
            return None
        if self.job.status == "running" and not self.job.alive():
            self.job.status, self.job.exit_code = self._outcome()
            self.save()
        return self.job

    def _outcome(self) -> tuple[str, int | None]:
        """(status, exit code) of a job whose process is gone, from its log."""
        tail = read_tail(self.log_path(), 40)
        for line in reversed(tail.splitlines()):
            found = re.search(r"training exited with code (-?\d+)", line)
            if found:
                code = int(found.group(1))
                return ("finished" if code == 0 else "failed"), code
        if "Script executed successfully" in tail:
            return "finished", 0
        return "finished", None

    def save(self) -> None:
        """Write `state.json` atomically (`tmp` + replace). A directory that vanished
        under a running child (a test's tmp, a cleaned output tree) is not an error the
        watcher thread should die on."""
        if self.job is None:
            return
        self.job.updated = time.time()
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(self.job), indent=1), encoding="utf-8")
            tmp.replace(self.state_path)
        except OSError:
            log.warning(f"could not write {self.state_path}", exc_info=True)

    # -- process -------------------------------------------------------------
    def running(self) -> bool:
        return self.job is not None and self.job.status == "running"

    def log_path(self) -> Path:
        """The child's log: one file per run, under the state dir."""
        return self.state_dir / LOG_FILE

    def log(self) -> str:
        return read_tail(self.log_path(), LOG_LINES)

    def write_config(self, path: Path, knobs_dict: dict) -> None:
        """Write a kogine config: a docstring naming the train script, then one
        module-level literal per knob, ASCII."""
        path.parent.mkdir(parents=True, exist_ok=True)
        body = [
            '"""Training config written by the KohakuMoss web UI, for scripts/train/',
            'moss.py. Every setting is a module-level knob: edit it, then',
            '`kogine run scripts/train/moss.py --config this_file.py`."""',
            "",
        ]
        for name, value in knobs_dict.items():
            body.append(f"{name} = {value!r}")
        path.write_text("\n".join(body) + "\n", encoding="ascii")

    def start(self, config: Path, output: Path, name: str) -> None:
        """Run the train script with `config`; the child writes its own log file."""
        if self.running():
            raise gr.Error("a training job is already running; stop it first")
        clear_caches()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_path()
        log_path.write_text("", encoding="utf-8")
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "KOHAKUMOSS_WEBUI": "1"}
        command = [*kogine(), "run", TRAIN_SCRIPT, "--config", str(config)]
        with log_path.open("ab") as sink:
            proc = subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdin=subprocess.DEVNULL,
                stdout=sink,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        self.job = Job(
            config=str(config),
            output=str(output),
            name=name,
            pid=proc.pid,
            status="running",
            started=time.time(),
        )
        self.save()
        threading.Thread(target=self._watch, args=(proc.pid,), daemon=True).start()

    def _watch(self, pid: int) -> None:
        """Wait for the child; the log is already on disk, only the status changes."""
        try:
            _, status = os.waitpid(pid, 0)
            code = os.waitstatus_to_exitcode(status)
        except (ChildProcessError, OSError):
            code = None
        with self.lock:
            if self.job is None or self.job.pid != pid:
                return
            self.job.exit_code = code
            self.job.status = "finished" if code == 0 else "failed"
            self.save()
        with self.log_path().open("a", encoding="utf-8") as handle:
            handle.write(f"\n--- training exited with code {code} ---\n")

    def stop(self) -> None:
        """SIGTERM the job's process group (the launcher tears its group down)."""
        if self.job is None or not self.running():
            return
        try:
            os.killpg(os.getpgid(self.job.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(self.job.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self.job.status = "stopped"
        self.save()

    def save_log(self, name: str) -> Path:
        """Write the log tail to `<state dir>/saved/<name>.log`."""
        root = self.state_dir / "saved"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', name or 'train')}.log"
        path.write_text(self.log(), encoding="utf-8")
        return path


TRAINER = Trainer()


# --------------------------------------------------------------------------- metrics


_METRICS_CACHE: dict = {"stamp": None, "frame": None}
_NPU_CACHE: dict = {"at": 0.0, "chips": []}


def clear_caches() -> None:
    """Drop the file-backed caches (`read_metrics`, `npu_stats`). Tests and a job
    switch call this so a different run's rows are never served from the old one."""
    _METRICS_CACHE["stamp"], _METRICS_CACHE["frame"] = None, None
    _NPU_CACHE["at"], _NPU_CACHE["chips"] = 0.0, []


def read_metrics(output: str | Path, limit: int = METRICS_TAIL_ROWS) -> pd.DataFrame:
    """`<OUTPUT_DIR>/run/metrics.jsonl` as a (step, loss, grad_norm, lr, speed) frame.

    kogine writes one JSON row every `LOG_EVERY` steps, so this is the run's own record
    (not a scrape of the console). Only the file's tail is read (`limit` rows): a
    15,000-step run's file is a few MB, and the page re-reads it on every tick. A
    missing file is an empty frame, not an error. An unchanged file is parsed once."""
    path = Path(output) / "run" / METRICS
    stamp = tail_stamp(path)
    if stamp is not None and _METRICS_CACHE["stamp"] == (path, stamp):
        return _METRICS_CACHE["frame"]
    lines: list[str] = []
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            block = min(size, max(limit, 1) * 400)
            handle.seek(size - block)
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return pd.DataFrame(columns=["step", "loss", "grad_norm", "lr", "s_per_step"])
    rows = []
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if "step" not in row:
            continue
        rows.append(
            {
                "step": int(row["step"]),
                "loss": row.get("loss"),
                "grad_norm": row.get("grad_norm"),
                "lr": row.get("lr"),
                "s_per_step": row.get("step_time"),
            }
        )
    frame = pd.DataFrame(rows[-limit:])
    if stamp is not None:
        _METRICS_CACHE["stamp"], _METRICS_CACHE["frame"] = (path, stamp), frame
    return frame


def thin(frame: pd.DataFrame, points: int = CHART_POINTS) -> pd.DataFrame:
    """`frame` thinned to at most `points` rows, keeping the last one.

    A chart redrawn every tick with every row of a long run is what makes the tab
    unresponsive: the browser rebuilds the SVG (and the JSON behind it) on each poll.
    Keeping an even stride plus the newest row preserves the shape of the curve."""
    if len(frame) <= points:
        return frame
    stride = len(frame) / points
    keep = [frame.index[int(i * stride)] for i in range(points)]
    keep[-1] = frame.index[-1]
    return frame.loc[keep]


def metric_series(frame: pd.DataFrame, points: int = CHART_POINTS) -> pd.DataFrame:
    """The long form `gr.LinePlot` reads: one row per (step, metric), thinned."""
    if frame.empty:
        return pd.DataFrame(columns=["step", "metric", "value"])
    keep = [c for c in ("loss", "grad_norm", "lr", "s_per_step") if c in frame]
    long = frame.melt(
        id_vars=["step"], value_vars=keep, var_name="metric", value_name="value"
    )
    long = long.dropna(subset=["value"])
    if len(frame) <= points:
        return long
    # thin the source frame once, so all four curves keep the same steps
    steps = set(thin(frame, points)["step"])
    return long[long["step"].isin(steps)]


def metric_summary(frame: pd.DataFrame, job: Job | None) -> str:
    """The headline numbers above the charts (Markdown)."""
    if job is None:
        return "**No training job yet.** Fill the form and press *Start training*."
    state = {
        "running": "training",
        "finished": "finished",
        "stopped": "stopped",
        "failed": "failed",
        "unknown": "unknown",
    }.get(job.status, job.status)
    parts = [f"**{job.name}**: {state}", f"`{job.output}`"]
    if job.started:
        parts.append(f"started {time.strftime('%H:%M:%S', time.localtime(job.started))}")
    if frame.empty:
        parts.append("no metrics yet (the first row lands at `LOG_EVERY` steps)")
    else:
        last = frame.iloc[-1]
        first = frame.iloc[0]
        parts.append(f"step **{int(last['step'])}**")
        if pd.notna(last.get("loss")):
            parts.append(f"loss **{float(last['loss']):.4f}**")
        if pd.notna(last.get("grad_norm")):
            parts.append(f"grad norm **{float(last['grad_norm']):.3f}**")
        if pd.notna(last.get("lr")):
            parts.append(f"lr **{float(last['lr']):.2e}**")
        if pd.notna(last.get("s_per_step")):
            parts.append(f"**{float(last['s_per_step']):.2f} s/step**")
        if len(frame) > 1:
            parts.append(f"({len(frame)} rows, from step {int(first['step'])})")
    return "  \n".join(parts)


def metric_table(frame: pd.DataFrame, rows: int = TABLE_ROWS) -> pd.DataFrame:
    """The last rows, for the table under the charts."""
    if frame.empty:
        return pd.DataFrame(columns=["step", "loss", "grad_norm", "lr", "s_per_step"])
    out = frame.tail(rows).copy()
    out["step"] = out["step"].astype(int)
    return out.iloc[::-1]


# ------------------------------------------------------------------------------- npu


@dataclass
class ChipStat:
    """One Ascend chip's live state, as `npu-smi info` reports it."""

    card: int
    chip: int
    name: str = ""
    health: str = ""
    power_w: float | None = None
    temp_c: float | None = None
    aicore_pct: float | None = None
    hbm_used_mb: int | None = None
    hbm_total_mb: int | None = None

    @property
    def hbm_pct(self) -> float | None:
        if not self.hbm_total_mb:
            return None
        return 100.0 * (self.hbm_used_mb or 0) / self.hbm_total_mb


NUM = re.compile(r"-?\d+(?:\.\d+)?")
FRACTION = re.compile(r"(\d+)\s*/\s*(\d+)")


FIELD_SEP = re.compile(r"\s{2,}")


def _fields(cell: str) -> list[float | tuple[int, int] | None]:
    """One `npu-smi` table cell as its fields, in column order.

    The cell packs several columns with padding rather than separators
    (`"-      69      0    / 0"`), so a fraction (`0 / 0`) is first masked to one token,
    then the runs of two or more spaces are the column breaks. `None` is a `-` cell
    (no reading), which is why the fields cannot be read by value order alone."""
    fractions: list[tuple[int, int]] = []

    def mask(match: re.Match) -> str:
        fractions.append((int(match.group(1)), int(match.group(2))))
        return f"\x00{len(fractions) - 1}\x00"

    masked = FRACTION.sub(mask, cell)
    out: list[float | tuple[int, int] | None] = []
    for token in FIELD_SEP.split(masked.strip()):
        token = token.strip()
        if not token:
            continue
        if token.startswith("\x00") and token.endswith("\x00"):
            out.append(fractions[int(token.strip("\x00"))])
        elif token == "-":
            out.append(None)
        else:
            try:
                out.append(float(token))
            except ValueError:
                return None
    return out


def parse_npu_smi(text: str) -> list[ChipStat]:
    """The `npu-smi info` table: one row per chip.

    Two lines per chip. The first is the card line (`<NPU> <Name> | Health | power,
    temperature, hugepages`), the second the chip line (`<Chip> <Phy-ID> | Bus-Id |
    AICore%, memory, HBM`); the two are told apart by the second token (a name vs a
    numeric PHY id), and a header row (`NPU Name`, `Chip Phy-ID`) has no leading
    digit. The process table that follows (`Process id | Process name | ...`) ends the
    chip section: its rows also start with two digits, but hold a process name."""
    chips: dict[int, ChipStat] = {}
    card = 0
    pending: dict | None = None
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        if "Process id" in cells[0] or "Process id" in cells[1]:
            break
        head = cells[0].split()
        if len(head) != 2 or not head[0].isdigit():
            continue
        fields = _fields(cells[2])
        if not fields:
            continue
        if head[1].isdigit():
            # chip line: <chip> <phy-id> | bus | AICore, memory, HBM
            chip = int(head[0])
            hbm = next((f for f in reversed(fields) if isinstance(f, tuple)), None)
            base = pending or {}
            chips[chip] = ChipStat(
                card=int(base.get("card", card)),
                chip=chip,
                name=str(base.get("name", "")),
                health=str(base.get("health", "")),
                power_w=base.get("power_w"),
                temp_c=base.get("temp_c"),
                aicore_pct=_first_number(fields),
                hbm_used_mb=None if hbm is None else hbm[0],
                hbm_total_mb=None if hbm is None else hbm[1],
            )
            continue
        # card line: <NPU> <Name> | health | power, temperature, hugepages
        card = int(head[0])
        pending = {
            "card": card,
            "name": head[1],
            "health": cells[1],
            "power_w": _at(fields, 0),
            "temp_c": _at(fields, 1),
        }
    return [chips[k] for k in sorted(chips)]


def _at(fields: list, index: int) -> float | None:
    """`fields[index]` when it is a number (a `-` or a fraction is None)."""
    if index >= len(fields) or not isinstance(fields[index], (int, float)):
        return None
    return float(fields[index])


def _first_number(fields: list) -> float | None:
    """The first number in `fields` (a chip line's AICore%)."""
    for field in fields:
        if isinstance(field, (int, float)):
            return float(field)
    return None


def _num(text: str) -> float | None:
    """A cell as a float; "-", "" and anything non-numeric is None."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _fraction(text: str) -> tuple[int | None, int | None]:
    """`"3119 / 65536"` -> (3119, 65536)."""
    parts = [p.strip() for p in str(text).split("/")]
    if len(parts) != 2:
        return None, None
    used, total = _num(parts[0]), _num(parts[1])
    return (
        None if used is None else int(used),
        None if total is None else int(total),
    )



def npu_stats(max_age: float = 0.0) -> list[ChipStat]:
    """This host's Ascend chips (empty on a host without `npu-smi`).

    One `npu-smi info` costs ~35 ms; a page that polls every tick would fork it
    continuously, so a read younger than `max_age` seconds is reused (the timer asks
    for a fresh one at its own interval)."""
    now = time.monotonic()
    if max_age > 0 and _NPU_CACHE["at"] and now - _NPU_CACHE["at"] < max_age:
        return _NPU_CACHE["chips"]
    if shutil.which("npu-smi") is None:
        return []
    try:
        text = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    chips = parse_npu_smi(text)
    _NPU_CACHE["at"], _NPU_CACHE["chips"] = now, chips
    return chips


def npu_frame(chips: list[ChipStat]) -> pd.DataFrame:
    """The chips as the status table's rows."""
    if not chips:
        return pd.DataFrame(
            columns=["card", "chip", "name", "health", "power W", "temp C", "AICore %", "HBM"]
        )
    return pd.DataFrame(
        [
            {
                "card": c.card,
                "chip": c.chip,
                "name": c.name,
                "health": c.health,
                "power W": c.power_w,
                "temp C": c.temp_c,
                "AICore %": c.aicore_pct,
                "HBM": (
                    ""
                    if c.hbm_total_mb is None
                    else f"{c.hbm_used_mb} / {c.hbm_total_mb} MB ({c.hbm_pct:.0f}%)"
                ),
            }
            for c in chips
        ]
    )


def npu_summary(chips: list[ChipStat]) -> str:
    """A one-line-per-chip Markdown summary (the table's caption)."""
    if not chips:
        return "No `npu-smi` on this host: the NPU panel is empty."
    lines = []
    for c in chips:
        bits = [f"`{c.card}:{c.chip}` {c.name} {c.health}"]
        if c.power_w is not None:
            bits.append(f"{c.power_w:.0f} W")
        if c.temp_c is not None:
            bits.append(f"{c.temp_c:.0f} C")
        if c.aicore_pct is not None:
            bits.append(f"AICore {c.aicore_pct:.0f}%")
        if c.hbm_pct is not None:
            bits.append(f"HBM {c.hbm_used_mb} / {c.hbm_total_mb} MB ({c.hbm_pct:.0f}%)")
        lines.append("- " + ", ".join(bits))
    return "\n".join(lines)


def train_config(values: dict) -> Path:
    """Write the form's train config and return its path.

    The parallel fields are resolved here rather than in the form: a multi-device
    `PARALLEL` takes every accelerator of the host (the form's index is the first one),
    and the layout decides the memory profile: `tp` / `fsdp` / `pipeline` / `threaded`
    need a resident plan, `ddp` runs whatever the form names."""
    weights = str(values.get("weights") or "").strip()
    data = str(values.get("data") or "").strip()
    if not weights:
        raise gr.Error("WEIGHTS is empty")
    if not data:
        raise gr.Error("DATASETS' path is empty")
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(values.get("name") or "webui-run"))
    parallel = str(values["parallel"])
    first, accelerators = device_indices(device_choices())
    devices = tuple(accelerators) if parallel != "none" and accelerators else (
        int(values["device_index"]),
    )
    memory = str(values.get("memory_profile") or "balanced")
    if parallel in ("tp", "fsdp", "pipeline", "threaded"):
        memory = "ckpt-resident"
    sample_every = int(values["sample_every"])
    prompts = parse_prompts(values.get("sample_prompts"))
    if sample_every > 0 and not prompts:
        raise gr.Error(
            "SAMPLE_EVERY > 0 needs at least one sample prompt: fill SAMPLE_PROMPTS "
            "or set SAMPLE_EVERY=0"
        )
    area, aspect_range = resolution_knobs(
        str(values["resolution_mode"]), values["min_side"], values["max_side"]
    )
    k = {
        "WEIGHTS": weights,
        "DATASETS": [{"source": str(values.get("source") or "kohya"), "path": data}],
        "OUTPUT_DIR": str(Path(str(values.get("out") or (ROOT / "out" / name)))),
        "ALGO": str(values["algo"]),
        "PRESET": str(values["preset"]),
        "DIM": int(values["dim"]),
        "ALPHA": float(values["alpha"]),
        "LR": float(values["lr"]),
        "OPTIMIZER": str(values["optimizer"]),
        "PRECISION": str(values["precision"]),
        "STEPS": int(values["steps"]),
        "SCHEDULE": {
            "mode": "cosine",
            "warmup": int(values["warmup"]),
            "end": -1,
            "min_value": 0.1,
        },
        "BATCH_SIZE": int(values["batch"]),
        "GRAD_ACC": int(values["grad_acc"]),
        "RESOLUTION_MODE": str(values["resolution_mode"]),
        "TARGET_AREA": area,
        "ASPECT_RANGE": aspect_range,
        "UPSCALE": str(values["upscale"]),
        "CROP": str(values["crop"]),
        "BATCHING": str(values["batching"]),
        "MAX_TOKENS": int(values["max_tokens"]),
        "OVERLONG": str(values["overlong"]),
        "SAVE_EVERY": int(values["save_every"]),
        "SAMPLE_EVERY": sample_every,
        "SAMPLE_PROMPTS": prompts,
        "MEMORY_PROFILE": memory,
        "PARALLEL": parallel,
        "DEVICES": devices,
        "LATENT_CACHE": str(values["latent_cache"]),
        "TEXT_CACHE": str(values["text_cache"]),
        "SEED": int(values["seed"]),
        "TRANSFORMER": "",
        "VARIANT": "base",
    }
    if not prompts:
        k.pop("SAMPLE_PROMPTS")
    if values.get("resume"):
        k["RESUME"] = "latest"
    root = Path(k["OUTPUT_DIR"])
    path = root / f"{name}.py"
    TRAINER.write_config(path, k)
    return path


def parse_prompts(text) -> list[str]:
    """`SAMPLE_PROMPTS`: one prompt per non-blank line (a bare string is one prompt)."""
    if isinstance(text, (list, tuple)):
        return [str(p).strip() for p in text if str(p).strip()]
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def sides_of(area, aspect_range) -> tuple[int, int]:
    """(min side, max side) of an `area` at `aspect_range`.

    The bucket planner's widest and tallest bucket are `sqrt(area * hi)` wide and
    `sqrt(area * lo)` tall (and the transpose), so the two sides a user thinks in are
    exactly the extremes of the set: `min_side * max_side == area`."""
    low, high = float(aspect_range[0]), float(aspect_range[1])
    return round(math.sqrt(float(area) * low)), round(math.sqrt(float(area) * high))


def area_and_range(min_side, max_side) -> tuple[int, tuple[float, float]]:
    """(TARGET_AREA, ASPECT_RANGE) of a (min side, max side) pair, the planner's own
    parameters. `min_side * max_side` is the area; the aspect bounds are the two
    ratios, `min / max` and `max / min`."""
    low, high = float(min_side), float(max_side)
    if low <= 0 or high <= 0:
        raise gr.Error(f"the sides must be positive, got {min_side!r} x {max_side!r}")
    if low > high:
        low, high = high, low
    return round(low * high), (round(low / high, 4), round(high / low, 4))


# Resize keeps each image's own aspect, so its bounds must not follow the sides: a
# square pair (`1024 x 1024`) would otherwise crop every image to a square. The bucket
# planner is the one whose extremes *are* the sides.
SIDE_BOUNDS = {"bucket": True, "resize": False, "square_crop": False,
               "shifted_square": False}
# the aspect window the non-bucket modes keep when the sides say nothing about it
OPEN_ASPECT = (0.25, 4.0)


def resolution_knobs(mode, min_side, max_side) -> tuple[int, tuple[float, float]]:
    """(TARGET_AREA, ASPECT_RANGE) of the form's mode and (min side, max side).

    `bucket` sizes every image to a bucket of the set, whose extremes are exactly the
    two sides: the area is their product and the aspect bounds their ratio. The other
    modes take the area from the sides but keep an open aspect window (they resize /
    crop to their own shape, so a square pair must not force a square target)."""
    area, bounds = area_and_range(min_side, max_side)
    return area, (bounds if SIDE_BOUNDS.get(mode, False) else OPEN_ASPECT)


def buckets_note(sizes, min_side=None, max_side=None) -> str:
    """The one-line summary over a bucket list; the sides are the requested extremes
    when the set really spans them (bucket mode), else omitted."""
    if not sizes:
        return "no buckets for these sides"
    widths = [w for h, w in sizes]
    heights = [h for h, w in sizes]
    count = "1 bucket" if len(sizes) == 1 else f"{len(sizes)} buckets"
    span = ""
    if min_side is not None and max_side is not None:
        if min(widths) == min_side and max(widths) == max_side:
            span = f" between `{min_side}` and `{max_side}` px"
    return (
        f"**{count}**{span}: widths {min(widths)}..{max(widths)}, "
        f"heights {min(heights)}..{max(heights)}. A batch never mixes sizes."
    )


def bucket_preview(
    min_side, max_side, upscale, mode: str = "bucket"
) -> tuple[str, pd.DataFrame, str]:
    """The bucket planner's sizes for a (min side, max side) pair.

    With `RESOLUTION_MODE='bucket'` every image goes to the nearest log-aspect bucket
    of ~`TARGET_AREA`; the table is what the run will actually ask the VAE to encode
    (`geometry.buckets`). The other modes do not use buckets; their preview keeps the
    open aspect window `resolution_knobs` gives them."""
    from kohakumoss_data.resolution.geometry import Geometry

    try:
        area, aspect = resolution_knobs(mode, min_side, max_side)
        geo = Geometry(area=area, aspect_range=aspect, upscale=str(upscale))
    except (ValueError, TypeError) as err:
        return f"the sides: {err}", pd.DataFrame(), "fix the sides"
    sizes = geo.buckets()
    rows = [
        {
            "width": w,
            "height": h,
            "aspect": round(w / h, 3),
            "image tokens": geo.tokens((h, w)),
        }
        for h, w in sizes
    ]
    frame = pd.DataFrame(rows)
    sides = (
        (int(min_side), int(max_side)) if SIDE_BOUNDS.get(mode, False) else (None, None)
    )
    return buckets_note(sizes, *sides), frame, ""


def resolution_note(mode: str, latent_cache: str) -> str:
    """The warning a resolution change carries: the latent cache keys on it."""
    if latent_cache != "offline":
        return ""
    return (
        "**This changes the latent cache key** (`RESOLUTION_MODE`, `TARGET_AREA`, "
        "`ASPECT_RANGE`, `UPSCALE`, `CROP`): an `offline` cache built for other "
        "values misses and the run stops at the first encode. Rebuild it, or set "
        "`LATENT_CACHE='online'` to encode and fill it during training."
    )


def start_training(*values) -> tuple[str, str, str]:
    """Train tab's Start: write the config, validate it, launch, return
    (status Markdown, log, metrics summary).

    The config is checked with the project's own knob checks before a child process is
    spent on it, so a bad setting comes back as a one-line Gradio error instead of a
    launcher traceback in the log."""
    values = _train_values(values)
    config = train_config(values)
    problems = check_config(config)
    if problems:
        raise gr.Error("invalid configuration:\n  - " + "\n  - ".join(problems))
    output = config.parent
    TRAINER.start(config, output, str(values.get("name") or config.stem))
    return job_status(), TRAINER.log(), job_summary()


def job_status() -> str:
    """The status line under the buttons (the page's `load` fills it too)."""
    job = TRAINER.job
    if job is None:
        return "no job yet"
    state = {
        "running": "training",
        "finished": "finished",
        "stopped": "stopped",
        "failed": "failed",
        "unknown": "unknown",
    }.get(job.status, job.status)
    bits = [f"**{job.name}**: {state}"]
    if job.pid:
        bits.append(f"pid {job.pid}")
    if job.exit_code is not None:
        bits.append(f"exit {job.exit_code}")
    bits.append(f"`{job.config}`")
    return "  \n".join(bits)


def job_summary(frame: pd.DataFrame | None = None, job: Job | None = None) -> str:
    """The metrics headline of the current job (an already-read `frame` is reused)."""
    job = job if job is not None else TRAINER.job
    if job is None:
        return metric_summary(pd.DataFrame(), None)
    if frame is None:
        frame = read_metrics(job.output)
    return metric_summary(frame, job)


def stop_training() -> tuple[str, str]:
    """Train tab's Stop."""
    TRAINER.stop()
    return job_status(), TRAINER.log()


def tail_log() -> str:
    """The log tail (the page polls it)."""
    job = TRAINER.job
    if job is None:
        return ""
    if job.status == "running" and not job.alive():
        # the process ended while no page was open: say so, do not claim it runs
        job.status = "finished"
        TRAINER.save()
    state = "running" if job.status == "running" else job.status
    return f"[{state}] {TRAINER.log()}"


def save_log(name: str) -> str:
    """Write the log to a file and report its path."""
    return f"saved to {TRAINER.save_log(name)}"


def restore_form() -> list:
    """The Train tab's fields as the last job's config set them (the page's `load`).

    A reloaded page shows the running job's own settings, not the form's defaults: the
    config is a plain module of literals, so it is read back as one."""
    job = TRAINER.load()
    if job is None or not Path(job.config).is_file():
        return [gr.update() for _ in TRAIN_FIELDS]
    try:
        values = read_config(Path(job.config))
    except (OSError, SyntaxError, ValueError):
        return [gr.update() for _ in TRAIN_FIELDS]
    return [gr.update(value=values.get(field)) for field in TRAIN_FIELDS]


def read_config(path: Path) -> dict:
    """The Train-tab fields a written config sets (the inverse of `train_config`)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    knobs = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    datasets = knobs.get("DATASETS") or [{}]
    devices = knobs.get("DEVICES") or (0,)
    schedule = knobs.get("SCHEDULE") or {}
    prompts = knobs.get("SAMPLE_PROMPTS") or []
    return {
        "weights": knobs.get("WEIGHTS"),
        "data": datasets[0].get("path"),
        "source": datasets[0].get("source", "folder"),
        "name": path.stem,
        "out": knobs.get("OUTPUT_DIR"),
        "algo": knobs.get("ALGO"),
        "preset": knobs.get("PRESET"),
        "dim": knobs.get("DIM"),
        "alpha": knobs.get("ALPHA"),
        "lr": knobs.get("LR"),
        "seed": knobs.get("SEED"),
        "optimizer": knobs.get("OPTIMIZER"),
        "precision": knobs.get("PRECISION"),
        "parallel": knobs.get("PARALLEL"),
        "device_index": devices[0] if devices else 0,
        "memory_profile": knobs.get("MEMORY_PROFILE"),
        "steps": knobs.get("STEPS"),
        "warmup": schedule.get("warmup"),
        "batch": knobs.get("BATCH_SIZE"),
        "grad_acc": knobs.get("GRAD_ACC"),
        "resolution_mode": knobs.get("RESOLUTION_MODE", "resize"),
        "min_side": sides_of(
            knobs.get("TARGET_AREA", 1024 * 1024),
            knobs.get("ASPECT_RANGE", (0.25, 4.0)),
        )[0],
        "max_side": sides_of(
            knobs.get("TARGET_AREA", 1024 * 1024),
            knobs.get("ASPECT_RANGE", (0.25, 4.0)),
        )[1],
        "upscale": knobs.get("UPSCALE"),
        "crop": knobs.get("CROP"),
        "batching": knobs.get("BATCHING"),
        "max_tokens": knobs.get("MAX_TOKENS"),
        "overlong": knobs.get("OVERLONG"),
        "save_every": knobs.get("SAVE_EVERY"),
        "sample_every": knobs.get("SAMPLE_EVERY"),
        "sample_prompts": "\n".join(str(p) for p in prompts),
        "latent_cache": knobs.get("LATENT_CACHE"),
        "text_cache": knobs.get("TEXT_CACHE"),
        "resume": "RESUME" in knobs,
    }


def check_config(config: Path) -> list[str]:
    """The knob problems of a written config: every one the run itself would report,
    except a path the user has not created yet (it is their file to point somewhere
    real). Imported here because the checker loads the train script."""
    from kohakumoss.cli.config.validate import MISSING, config_problems

    return [p for p in config_problems(config) if MISSING not in p]


TRAIN_FIELDS = (
    "weights",
    "data",
    "source",
    "name",
    "out",
    "algo",
    "preset",
    "dim",
    "alpha",
    "lr",
    "seed",
    "optimizer",
    "precision",
    "parallel",
    "device_index",
    "memory_profile",
    "steps",
    "warmup",
    "batch",
    "grad_acc",
    "resolution_mode",
    "min_side",
    "max_side",
    "upscale",
    "crop",
    "batching",
    "max_tokens",
    "overlong",
    "save_every",
    "sample_every",
    "sample_prompts",
    "latent_cache",
    "text_cache",
    "resume",
)


def _train_values(values) -> dict:
    """The form's positional fields as the dict `train_config` reads."""
    return dict(zip(TRAIN_FIELDS, values))


# --------------------------------------------------------------------------------- UI


def build_ui() -> gr.Blocks:
    """The page: the Generate tab, then the optional Train tab."""
    devices = device_choices()
    with gr.Blocks(title="KohakuMoss") as page:
        gr.Markdown(
            "# KohakuMoss\n"
            "Text-to-image with MOSS-Image, and (optionally) finetuning it. "
            f"Devices on this host: `{', '.join(devices)}`."
        )
        with gr.Tabs():
            with gr.Tab("Generate"):
                _generate_tab(devices)
            with gr.Tab("Train"):
                restore, restore_outputs = _train_tab(devices)
        # inside the Blocks context: `load` needs it. A reload of the page runs this,
        # so a job started before the reload is shown again (form, log, metrics, NPU).
        page.load(restore, outputs=restore_outputs)
    return page


def _generate_tab(devices: list[str]) -> None:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Model")
            weights = gr.Textbox(
                label="WEIGHTS",
                placeholder="ComfyUI/models  or  models/MOSS-Image-Base  or  a pipeline dir",
                value=os.environ.get("KOHAKUMOSS_WEIGHTS", ""),
            )
            with gr.Row():
                variant = gr.Dropdown(["base", "sft"], value="base", label="VARIANT")
                precision = gr.Dropdown(PRECISIONS, value="auto", label="PRECISION")
            transformer = gr.Textbox(
                label="TRANSFORMER (optional)",
                placeholder="a DiT file / dir override; empty = WEIGHTS",
            )
            with gr.Row():
                adapter = gr.Textbox(label="ADAPTER", placeholder="my_lora.safetensors:0.8")
                adapter_mode = gr.Dropdown(ADAPTER_MODES, value="merge", label="ADAPTER_MODE")
            with gr.Row():
                device = gr.Dropdown(devices, value=devices[0], label="DEVICE")
                memory_profile = gr.Dropdown(
                    SAMPLE_MEMORY_PROFILES, value="", label="MEMORY_PROFILE"
                )
            text_cpu = gr.Checkbox(
                value=False, label="text encoder on the host (frees VRAM, slower)"
            )
            gr.Markdown("### Prompt")
            prompt = gr.Textbox(
                label="PROMPT", lines=3, placeholder="a lighthouse on a rocky coast at sunset"
            )
            negative = gr.Textbox(label="NEGATIVE", lines=2)
            with gr.Row():
                size = gr.Dropdown(SIZES, value="1024x1024", label="SIZE (HxW)")
                seed = gr.Number(value=0, precision=0, label="SEED")
            with gr.Row():
                steps = gr.Slider(4, 100, value=50, step=1, label="STEPS")
                cfg = gr.Slider(1.0, 10.0, value=3.0, step=0.1, label="CFG")
            with gr.Accordion("Sampler details", open=False):
                with gr.Row():
                    shift = gr.Number(value=5.0, label="SHIFT")
                    batch = gr.Number(value=1, precision=0, label="BATCH")
                with gr.Row():
                    cfg_mode = gr.Dropdown(CFG_MODES, value="packed", label="CFG_MODE")
                    skip_layers = gr.Textbox(value="default", label="SKIP_LAYERS")
            run = gr.Button("Generate", variant="primary")
            release = gr.Button("Release the model (free VRAM)")
        with gr.Column(scale=1):
            image = gr.Image(label="Result", type="pil", height=640)
            meta = gr.Code(label="Settings", language="json", lines=14)

    run.click(
        generate,
        inputs=[
            weights,
            variant,
            transformer,
            adapter,
            adapter_mode,
            precision,
            device,
            memory_profile,
            text_cpu,
            prompt,
            negative,
            size,
            steps,
            cfg,
            shift,
            seed,
            batch,
            cfg_mode,
            skip_layers,
        ],
        outputs=[image, meta],
        show_progress="full",
    )
    release.click(lambda: (ENGINE.close(), "released")[1], outputs=[meta])


def _train_tab(devices: list[str]) -> None:
    first_device, _ = device_indices(devices)
    gr.Markdown(
        "### Finetune\n"
        "The form writes a kogine config and runs `scripts/train/moss.py` in a child "
        "process. The child writes its own log file, so it keeps training when the page "
        "is reloaded or restarted: the page finds the job again and shows its metrics "
        "and the live NPU state."
    )
    with gr.Row():
        with gr.Column(scale=1):
            weights = gr.Textbox(label="WEIGHTS", placeholder="models/MOSS-Image-Base")
            with gr.Row():
                data = gr.Textbox(label="DATASETS path", placeholder="/data/my-images")
                source = gr.Dropdown(SOURCES, value="folder", label="source")
            with gr.Row():
                name = gr.Textbox(value="webui-run", label="run name")
                out = gr.Textbox(label="OUTPUT_DIR", placeholder="out/webui-run")
            with gr.Row():
                algo = gr.Dropdown(ALGOS, value="lora", label="ALGO")
                preset = gr.Textbox(value="attn-mlp", label="PRESET")
            with gr.Row():
                dim = gr.Number(value=32, precision=0, label="DIM (rank)")
                alpha = gr.Number(value=32.0, label="ALPHA")
            with gr.Row():
                lr = gr.Number(value=1e-4, label="LR")
                seed = gr.Number(value=0, precision=0, label="SEED")
        with gr.Column(scale=1):
            with gr.Row():
                optimizer = gr.Dropdown(OPTIMIZERS, value="adamw", label="OPTIMIZER")
                precision = gr.Dropdown(PRECISIONS, value="auto", label="PRECISION")
                parallel = gr.Dropdown(PARALLELS, value="none", label="PARALLEL")
                device_index = gr.Number(
                    value=first_device, precision=0, label="DEVICE index (first)"
                )
            with gr.Row():
                memory_profile = gr.Dropdown(
                    MEMORY_PROFILES, value="balanced", label="MEMORY_PROFILE"
                )
                latent_cache = gr.Dropdown(CACHES, value="online", label="LATENT_CACHE")
                text_cache = gr.Dropdown(CACHES, value="online", label="TEXT_CACHE")
            with gr.Row():
                steps = gr.Number(value=1000, precision=0, label="STEPS")
                warmup = gr.Number(
                    value=100, precision=0, label="SCHEDULE warmup (steps)"
                )
                batch = gr.Number(value=1, precision=0, label="BATCH_SIZE")
                grad_acc = gr.Number(value=1, precision=0, label="GRAD_ACC")
            with gr.Accordion("Resolution / batching", open=False):
                with gr.Row():
                    resolution_mode = gr.Dropdown(
                        RESOLUTION_MODES,
                        value="resize",
                        label="RESOLUTION_MODE",
                        info="resize: native aspect | bucket: one size per aspect bucket",
                    )
                    min_side = gr.Number(
                        value=512,
                        precision=0,
                        label="smallest side (px)",
                        info="the short side of the most extreme bucket",
                    )
                    max_side = gr.Number(
                        value=1536,
                        precision=0,
                        label="largest side (px)",
                        info="the long side of the most extreme bucket",
                    )
                with gr.Row():
                    upscale = gr.Dropdown(UPSCALES, value="never", label="UPSCALE")
                    crop = gr.Dropdown(CROPS, value="center", label="CROP")
                    batching = gr.Dropdown(
                        BATCHINGS,
                        value="tokens",
                        label="BATCHING",
                        info="tokens | tokens_ffd | count (BATCH_SIZE)",
                    )
                    max_tokens = gr.Number(
                        value=12288, precision=0, label="MAX_TOKENS (per batch)"
                    )
                    overlong = gr.Dropdown(
                        OVERLONG_MODES, value="drop", label="OVERLONG"
                    )
                preview = gr.Button("Preview the buckets")
                bucket_text = gr.Markdown(
                    "", label="bucket preview", visible=True
                )
                bucket_table = gr.Dataframe(
                    pd.DataFrame(),
                    label="bucket sizes (bucket mode)",
                    interactive=False,
                    visible=True,
                )
            with gr.Row():
                save_every = gr.Number(value=500, precision=0, label="SAVE_EVERY")
                sample_every = gr.Number(
                    value=0, precision=0, label="SAMPLE_EVERY (0 = no sampling)"
                )
            sample_prompts = gr.Textbox(
                label="SAMPLE_PROMPTS (one per line)",
                lines=3,
                placeholder="a lighthouse on a rocky coast at sunset",
            )
            resume = gr.Checkbox(value=False, label="RESUME from the newest checkpoint")
            with gr.Row():
                start = gr.Button("Start training", variant="primary")
                stop = gr.Button("Stop")
                save = gr.Button("Save log")
    gr.Markdown(
        "*A multi-device `PARALLEL` trains on every accelerator of this host "
        f"({', '.join(devices[:-1]) or 'none'}); `DEVICE index` is only used by "
        "`none`. `tp` / `fsdp` / `pipeline` / `threaded` need a resident plan "
        "(`ckpt-resident`), and `tp` refuses `muon`.*"
    )

    # ---------------------------------------------------------------- job panel
    with gr.Row():
        status = gr.Markdown(job_status(), label="Status")
        refresh = gr.Button("Refresh now", scale=0, min_width=120)
    with gr.Row():
        with gr.Column(scale=3):
            metrics_text = gr.Markdown()
            plot = gr.LinePlot(
                metric_series(read_metrics(TRAINER.job.output if TRAINER.job else "")),
                x="step",
                y="value",
                color="metric",
                title="Training metrics",
                x_title="step",
                y_title="value",
                height=320,
                tooltip="all",
                buttons=["fullscreen", "export"],
            )
            recent = gr.Dataframe(
                metric_table(read_metrics(TRAINER.job.output if TRAINER.job else "")),
                label="last rows (metrics.jsonl)",
                interactive=False,
                wrap=True,
            )
        with gr.Column(scale=2):
            npu_text = gr.Markdown(npu_summary(npu_stats()), label="NPU")
            npu_table = gr.Dataframe(
                npu_frame(npu_stats()), label="Ascend devices", interactive=False
            )
            logbox = gr.Textbox(
                label="Log",
                lines=16,
                max_lines=30,
                autoscroll=True,
                interactive=False,
                value=TRAINER.log(),
            )

    # ------------------------------------------------------------------ wiring
    values = {
        "weights": weights,
        "data": data,
        "source": source,
        "name": name,
        "out": out,
        "algo": algo,
        "preset": preset,
        "dim": dim,
        "alpha": alpha,
        "lr": lr,
        "seed": seed,
        "optimizer": optimizer,
        "precision": precision,
        "parallel": parallel,
        "device_index": device_index,
        "memory_profile": memory_profile,
        "steps": steps,
        "warmup": warmup,
        "batch": batch,
        "grad_acc": grad_acc,
        "resolution_mode": resolution_mode,
        "min_side": min_side,
        "max_side": max_side,
        "upscale": upscale,
        "crop": crop,
        "batching": batching,
        "max_tokens": max_tokens,
        "overlong": overlong,
        "save_every": save_every,
        "sample_every": sample_every,
        "sample_prompts": sample_prompts,
        "latent_cache": latent_cache,
        "text_cache": text_cache,
        "resume": resume,
    }
    assert tuple(values) == TRAIN_FIELDS, tuple(values)
    fields = list(values.values())
    start.click(
        start_training, inputs=fields, outputs=[status, logbox, metrics_text]
    )
    stop.click(stop_training, outputs=[status, logbox])
    save.click(save_log, inputs=[name], outputs=[status])

    def preview_buckets(mode, small, large, up, cache):
        """The bucket sizes and the latent-cache warning for the form's settings."""
        text, frame, _ = bucket_preview(small, large, up, mode)
        area, aspect = resolution_knobs(mode, small, large)
        side = round(math.sqrt(area))
        if mode == "bucket":
            usage = (
                f"`bucket` puts each image in the nearest of these sizes "
                f"(target area {area:,} px, aspect {aspect[0]}..{aspect[1]})."
            )
        elif mode == "resize":
            span = sides_of(area, aspect)
            usage = (
                f"`resize` does not use buckets: every image keeps its own aspect at "
                f"~{area:,} px (the sides set the area only, the aspect window stays "
                f"{aspect[0]}..{aspect[1]}). The table below is the window a `bucket` "
                f"run of the same area ({span[0]}x{span[1]}) would use."
            )
        else:
            shape = "a centre crop" if mode == "square_crop" else "a seeded crop"
            usage = (
                f"`{mode}` does not use buckets: every image is {shape} of "
                f"{side}x{side} (the sides set the area only). The table below is the "
                "window a `bucket` run of the same area would use."
            )
        text = f"{usage}\n\n{text}"
        if up == "never":
            text += (
                "\n\n`UPSCALE='never'`: an image smaller than the smallest bucket keeps "
                "its own size, so the run also produces buckets below the smallest side "
                "(visible in the table). Set `UPSCALE='to_area'` to scale those up."
            )
        note = resolution_note(mode, cache)
        if note:
            text = f"{text}\n\n{note}"
        return gr.update(value=text, visible=True), gr.update(
            value=frame, visible=not frame.empty
        )

    preview.click(
        preview_buckets,
        inputs=[resolution_mode, min_side, max_side, upscale, latent_cache],
        outputs=[bucket_text, bucket_table],
    )
    for probe in (
        resolution_mode,
        min_side,
        max_side,
        upscale,
        latent_cache,
    ):
        probe.change(
            preview_buckets,
            inputs=[resolution_mode, min_side, max_side, upscale, latent_cache],
            outputs=[bucket_text, bucket_table],
        )

    def snapshot():
        """One read of the live state: status, log, metrics, NPU.

        A single read serves every panel (the old version read `metrics.jsonl` twice and
        forked `npu-smi` twice per tick), and it reports the fingerprints the refresh
        uses to skip panels that did not change."""
        job = TRAINER.job
        output = job.output if job else ""
        frame = read_metrics(output) if output else pd.DataFrame()
        chips = npu_stats(max_age=NPU_POLL_S)
        state = job_status()
        log = tail_log()
        summary = job_summary(frame, job)
        series = metric_series(frame)
        table = metric_table(frame)
        return {
            "status": state,
            "log": log,
            "log_stamp": tail_stamp(TRAINER.log_path()),
            "summary": summary,
            "series": series,
            "series_stamp": (len(series), int(series["step"].max()) if len(series) else 0),
            "table": table,
            "npu_text": npu_summary(chips),
            "npu_frame": npu_frame(chips),
            "npu_stamp": [
                (c.card, c.chip, c.power_w, c.temp_c, c.aicore_pct, c.hbm_used_mb)
                for c in chips
            ],
        }

    PANELS = ("status", "log", "summary", "series", "table", "npu_text", "npu_frame")

    def refresh_all(seen: dict | None):
        """Push only the panels whose content changed since the last tick.

        Late in a long run the metrics file and the log are big and grow every step: an
        unconditional re-send of the chart, the table and the log is what made the tab
        unresponsive. `seen` (one `gr.State` per browser session) holds the last pushed
        fingerprints, so an unchanged panel costs nothing."""
        shot = snapshot()
        seen = dict(seen) if isinstance(seen, dict) else {}
        stamps = {
            "status": shot["status"],
            "log": shot["log_stamp"],
            "summary": shot["summary"],
            "series": shot["series_stamp"],
            "table": shot["series_stamp"],
            "npu_text": shot["npu_stamp"],
            "npu_frame": shot["npu_stamp"],
        }
        out = []
        for key in PANELS:
            if seen.get(key) == stamps[key]:
                out.append(gr.update())
            else:
                seen[key] = stamps[key]
                out.append(shot[key])
        return seen, out

    outputs = [status, logbox, metrics_text, plot, recent, npu_text, npu_table]
    refresh_state = gr.State({})
    refresh.click(
        refresh_all, inputs=[refresh_state], outputs=[refresh_state, *outputs]
    )
    timer = gr.Timer(STEP_POLL_S)
    timer.tick(refresh_all, inputs=[refresh_state], outputs=[refresh_state, *outputs])

    def restore():
        """The page's `load`: the last job's form, status, log, metrics and NPU."""
        TRAINER.load()
        _, panels = refresh_all({})
        return restore_form() + list(panels)

    return restore, fields + outputs



def main(argv: list[str] | None = None) -> None:
    """`kohakumoss-webui`: parse the flags and serve the page."""
    ap = argparse.ArgumentParser(
        prog="kohakumoss-webui", description="The KohakuMoss web UI (Gradio)"
    )
    ap.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0: all)")
    ap.add_argument("--port", type=int, default=7860, help="port (default 7860)")
    ap.add_argument("--share", action="store_true", help="a public Gradio link")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser")
    args = ap.parse_args(argv)
    setup()
    log.info(f"serving the web UI on http://{args.host}:{args.port}")
    build_ui().queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        show_error=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
