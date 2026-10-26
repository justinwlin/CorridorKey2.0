# Running CorridorKey on remote GPUs

`remote/run.py` runs CorridorKey on GPU machines other than the one you are
sitting at. Use it when your computer has no NVIDIA GPU, or to process a
folder of clips on several GPUs at once. It runs on macOS, Linux, and Windows
and needs only Python 3.10+ and the `ssh` client. No local GPU, PyTorch, or
model weights are needed.

The machines come from a **provider**. Some providers use machines you
already have, and others rent machines for the length of the run.

## Providers

| Provider | Machines | Guide |
|---|---|---|
| `ssh` | Your own Linux machines with an NVIDIA GPU, reached over SSH | [providers/ssh](providers/ssh/README.md) |
| `runpod` | Runpod GPU pods, started for the run and shut down afterwards | [providers/runpod](providers/runpod/README.md) |

Choose one with `--provider`. The default is `runpod`. Any cloud whose
machines you can SSH into also works with `--provider ssh`. To add a provider
that starts and stops machines for you, see
[ADDING_A_PROVIDER.md](ADDING_A_PROVIDER.md).

## How a run works

For each machine, `run.py`:

1. Connects to the machine, starting it first if the provider rents it.
2. Uploads this repository's code and your clips over SSH.
3. Runs [setup_machine.sh](setup_machine.sh), which installs `ffmpeg` if it is
   missing, installs the Python requirements without changing the machine's
   PyTorch, and downloads the weights (~5 GB) unless they are already there.
4. Runs `infer.py` on each clip it takes from the queue and downloads the
   outputs. Each clip's files are deleted from the machine afterwards.
5. Shuts down the machine if the provider started it, including when the run
   fails or you press Ctrl-C. Machines you own are left running.

## Running

```
python remote/run.py --provider <provider> [provider flags] \
  --input shot.mp4 --output_dir out
```

Every flag that `run.py` does not define is passed to `infer.py` unchanged, so
the examples in the main README work as they are:

```
python remote/run.py --provider ssh --ssh_host gpu-box \
  --input shot.mp4 \
  --output_dir out_frame1 \
  --hint_first_frame matte_frame_0001.png \
  --hint_quality 0.95 \
  --outputs alpha fg cutout
```

`--hint_first_frame` and `--hint_video` take local paths and are uploaded with
the clip.

| Flag | Default | Purpose |
|---|---|---|
| `--provider` | `runpod` | Where the machines come from. |
| `--input` | required | Video file, frame directory, or folder of clips. |
| `--output_dir` | required | Local directory for the outputs. |
| `--machines` | 1, or every `--ssh_host` for `ssh` | Machines to use in parallel for a folder of clips. |
| `--ready_timeout` | 900 | Seconds to wait for a machine to accept SSH. |
| `--keep_machines` | off | Leave started machines running after the run and print the SSH command to connect. They keep billing until you delete them. |

## Several clips on several GPUs

Point `--input` at a folder of clips:

```
python remote/run.py --provider ssh --ssh_host box1 --ssh_host box2 \
  --input clips/ --output_dir out
python remote/run.py --provider runpod --machines 3 \
  --input clips/ --output_dir out
```

The folder is read the same way as with the drag-and-drop launcher: video
files and `Input`/`AlphaHint` frame folders are clips, companion hint files
(`<clip>FrameHint.png`, `<clip>_AlphaHint.mp4`) are applied automatically, and
each clip's outputs go to `out/out_<clip>`.

Each machine takes the next unprocessed clip until none are left. A single
clip always runs on one machine, because splitting a clip would break the
temporal context between its frames. If a machine drops out mid-clip, that
clip goes back in the queue for the others.

## Tests

```
python remote/tests/test_run.py
```

The tests replace SSH and the provider APIs with fakes, so they need no cloud
account, SSH server, or GPU.
