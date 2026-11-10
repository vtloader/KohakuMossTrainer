# Web UI

A Gradio page over the pipeline's inference, and optionally over training.

```bash
pip install -e ".[webui]"          # gradio, once
kohakumoss-webui                   # http://127.0.0.1:7860
kohakumoss-webui --host 0.0.0.0 --port 7861 --share
```

| flag | effect |
| --- | --- |
| `--host` | bind address; `0.0.0.0` reaches the page from another machine (default `127.0.0.1`) |
| `--port` | port (default `7860`) |
| `--share` | a public `gradio.live` link |
| `--no-browser` | do not open a browser on start |

The page detects the accelerator itself: the `DEVICE` dropdown lists every device this
host has (`npu:0`, `npu:1`, `cuda:0`, `xpu:0`, then `cpu`).

## Generate

The form's settings are the sample script's knobs
([inference.md](inference.md)), with the same names:

| field | meaning |
| --- | --- |
| `WEIGHTS` | a ComfyUI models folder, a weights root (`base/`, `sft/`) or one pipeline dir |
| `VARIANT`, `PRECISION`, `TRANSFORMER` | the arch preset, the compute dtype, an optional DiT override |
| `ADAPTER`, `ADAPTER_MODE` | `my_lora.safetensors:0.8` and `merge` / `bypass` |
| `DEVICE`, `MEMORY_PROFILE` | where the DiT runs, and its memory profile (`auto` picks the fastest that fits) |
| text encoder on the host | `TEXT_DEVICE` = `cpu`: frees VRAM, slower per prompt |
| `PROMPT`, `NEGATIVE`, `SIZE`, `SEED` | one image per Generate click |
| `STEPS`, `CFG`, `SHIFT`, `CFG_MODE`, `SKIP_LAYERS`, `BATCH` | the sampler |

The model loads on the first click and stays resident; changing any load-time field
(weights, adapter, precision, device, ...) releases the old pipeline and builds a new
one, so VRAM never holds two. **Release the model** frees it without stopping the page.
The step slider in the progress bar follows the sampler's own `step_callback`.

## Train (optional)

The second tab writes a [kogine config](configs.md) from the form and runs
`scripts/train/moss.py` in a child process.

- **Start training** writes `<OUTPUT_DIR>/<run name>.py`, checks it with the project's
  own knob checks, and launches it (with the `kogine` of the page's own venv);
- **Stop** sends `SIGTERM` to the job's process group (the launcher tears its group
  down, so the dataloader workers and encoder processes go with it);
- **Save log** writes the log tail to `out/webui/saved/<run name>.log`.

### The job outlives the page

The child writes its own log file (`out/webui/train.log`), not a pipe, and the page
keeps `out/webui/state.json` (config path, output dir, pid, status). So:

| you do | what happens |
| --- | --- |
| reload the browser tab | the page's `load` finds the job, restores the form from its config, and shows the log, metrics and NPU panel again |
| restart `kohakumoss-webui` | same: the new page reads `state.json`, sees the pid alive, and picks the job back up |
| close the page during training | **the job keeps training** (it is its own process group); the next page finds it |
| stop the page and let the job finish | the next page reports it `finished`, with the exit code the log's last line records |

A job whose pid is gone but whose `state.json` still says `running` (the page was not
there when it ended) is resolved from its log: `Script executed successfully` /
`training exited with code N` decide `finished` / `failed` and the code.

### Live panels

| panel | source | refresh |
| --- | --- | --- |
| status | `out/webui/state.json` | every `1.5 s` |
| log | `out/webui/train.log` (last 400 lines, ≤ 96 KiB read) | every `1.5 s` |
| **metrics** | `<OUTPUT_DIR>/run/metrics.jsonl`: the run's own rows (`loss`, `grad_norm`, `lr`, `step_time`); one chart line per metric, a table of the last 20 rows | every `1.5 s` |
| **NPU** | `npu-smi info`: power (W), temperature (C), AICore %, HBM used / total per chip | every `1.5 s` (the reading itself is reused for 4 s: one call costs ~35 ms) |

The metrics come from the run, not from scraping the console: the trainer appends one
JSON row every `LOG_EVERY` steps, so a reloaded page plots the whole history (it reads
the file, not a live buffer). **Refresh now** forces a redraw without waiting for the
timer.

#### A long run does not slow the page down

Late in a 15,000-step run the metrics file has thousands of rows and the log is tens of
MB. Four things keep the tab responsive:

| what | how |
| --- | --- |
| the chart | thinned to at most `CHART_POINTS` (240) points per curve, keeping the newest step; a 1500-row run plots 960 points instead of 6000 |
| the file reads | the metrics file is read from its **tail only** (4000 rows) and the log from its last 96 KiB, so one tick costs the same at step 10 and step 15,000 |
| the requests | a panel whose content did not change is skipped (`gr.State`, one per browser session, holds the last pushed fingerprints): an idle tick sends **nothing** |
| the `npu-smi` forks | one call per tick (it used to fork twice per tick for the two panels), and the reading is reused for 4 s |

Measured on a simulated 15,000-step run (1500 metric rows, a 15 MB log):

| tick | before | after |
| --- | --- | --- |
| nothing changed | ~165 KiB every 0.5 s | **0 KiB** |
| a new log line | ~165 KiB | **41 KiB** |
| a new metric row | ~165 KiB | **22 KiB** |
| one tick's server time | ~40 ms | **~4 ms** |

The form's `source` picks the dataset reader (`kohya`, `folder`, `jsonl`, `parquet`,
`webdataset`; [datasets.md](datasets.md)). `folder` reads a flat tree of images (with
`.txt` / `.caption` sidecars), `kohya` the `<repeats>_<name>/` subfolders; a flat
dataset under `kohya` is the "no images in the subfolders" error.

### Resolution and batching

The **Resolution / batching** section holds the data layout knobs
([data.md](../concepts/data.md#resolution-batching-over-long-samples)). The two
resolution fields are **sides**, not a bare aspect ratio: enter the smallest and the
largest side you want, and the page derives the planner's `TARGET_AREA` /
`ASPECT_RANGE` from them.

| field | meaning |
| --- | --- |
| `RESOLUTION_MODE` | `resize` (native aspect), `bucket` (one size per aspect bucket, the batch never mixes sizes), `square_crop`, `shifted_square` |
| **smallest side** | the short side of the most extreme bucket (e.g. `512`) |
| **largest side** | the long side of the most extreme bucket (e.g. `1536`) |
| `UPSCALE` | `never` keeps an image smaller than its bucket at its native size (its own bucket); `to_area` scales it up |
| `CROP` | `center`, or `random` (a seeded crop position); an `offline` latent cache needs `center` |
| `BATCHING` | `tokens` (greedy up to `MAX_TOKENS`), `tokens_ffd` (denser, order shuffled), `count` (`BATCH_SIZE`) |
| `MAX_TOKENS` | joint image + text tokens per batch |
| `OVERLONG` | a sample over `MAX_TOKENS`: `drop`, `truncate_text`, `drop_tokens` |

How the sides map onto the planner:

```
TARGET_AREA  = min_side * max_side
ASPECT_RANGE = (min_side / max_side, max_side / min_side)     # bucket mode
```

so `512 x 1536` is `TARGET_AREA=786432`, `ASPECT_RANGE=(0.3333, 3.0)` — exactly the
`512x1536` … `1536x512` bucket set (65 buckets). `resize` / `square_crop` /
`shifted_square` take the area from the sides but keep the open `(0.25, 4)` aspect
window: they size each image to its own shape, so a square pair (`1024 x 1024`) must
not turn every target into a square.

**Preview the buckets** (and any change of the sizing fields) lists the exact sizes the
`bucket` planner will use — the planner's own `geometry.buckets()`, not a
reimplementation:

```
37 buckets between 768 and 1344 px: widths 768..1344, heights 768..1344.
A batch never mixes sizes.
```

with a table of width / height / aspect / image tokens per bucket. The other modes do
not use buckets, so their preview explains what they do instead and still shows the
window for reference.

With `UPSCALE='never'` an image smaller than the smallest bucket keeps its own size, so
the run also produces sizes below the smallest side (the table shows them). Use
`UPSCALE='to_area'` to scale them up instead.

> **Changing the resolution invalidates an offline latent cache.** The latents cache
> keys on `RESOLUTION_MODE`, `TARGET_AREA`, `ASPECT_RANGE`, `UPSCALE`, `CROP`,
> `FLIP`, `CACHE_DTYPE`, `BACKGROUND`, `RESAMPLE` and the encoder identity. With
> `LATENT_CACHE='offline'` a change makes every entry miss and the run stops at the
> first encode; rebuild the cache or set `LATENT_CACHE='online'` (the first pass
> encodes and fills it). The page shows this warning as the cache dropdown changes.

### Fields the page resolves itself

| field | what the page does |
| --- | --- |
| `DEVICES` | a multi-device `PARALLEL` (`ddp`, `fsdp`, `pipeline`, `threaded`, `tp`) trains on **every accelerator of the host**; the form's `DEVICE index` is the first one, and is only used by `none` |
| `MEMORY_PROFILE` | `tp` / `fsdp` / `pipeline` / `threaded` force `ckpt-resident` (they cannot stream or offload the base); `ddp` and `none` keep the form's profile |
| `LATENT_CACHE` / `TEXT_CACHE` | `online` (default: the first pass encodes and caches) or `offline` (a prebuilt cache; see [datasets.md](datasets.md#3-cache-offline-mode)) |
| `RESOLUTION_MODE`, smallest / largest side, `UPSCALE`, `CROP`, `BATCHING`, `OVERLONG` | the sides are converted to the planner's `TARGET_AREA` / `ASPECT_RANGE`; the resolution set keys the latent cache, so **Preview the buckets** also shows the cache warning |
| `SCHEDULE` | the form's `warmup` steps with the base's cosine shape; it must be below `STEPS` |
| `SAMPLE_PROMPTS` | one prompt per line; `SAMPLE_EVERY > 0` refuses an empty list |

Start is refused (a one-line error, no child process) for a combination the run itself
would reject, e.g. `PARALLEL='tp'` with `OPTIMIZER='muon'`: Muon orthogonalises a whole
matrix, and a TP rank only holds a slice of it. Use `PARALLEL='ddp'` with Muon, or
`tp` with `adamw`.

The written config is an ordinary config: every knob is a module-level literal, so the
same file runs without the page:

```bash
kogine run scripts/train/moss.py --config <OUTPUT_DIR>/<run name>.py
kohakumoss-config check <OUTPUT_DIR>/<run name>.py    # the same knob checks
```

Training runs in a separate process, so the page stays responsive; inference and
training on the same device at the same time will run out of memory, so keep the
`DEVICE` of the Generate tab elsewhere (or do not press Generate) while a job runs.

The state lives under `out/webui/` (`state.json`, `train.log`, `saved/`), so a page
started from another working directory still finds the job: set
`KOHAKUMOSS_WEBUI_STATE` to move it (see `Trainer(state_dir=...)`).

## Notes

- One model per page process. Two browser tabs share it (the same pipeline), which is
  what you want; two page *processes* would each load their own copy.
- The page never writes a config it does not show: the Train tab's settings are exactly
  the file it saves.
- `gradio` is an optional extra (`[webui]`): every other entry point works without it.
