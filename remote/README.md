# Running CorridorKey on remote GPUs

`remote/run.py` runs CorridorKey on GPU machines other than the one you are
sitting at. Those can be:

- **Your own machines**, reached over SSH: a workstation in the studio, a
  render node, or a server you rent anywhere.
- **Machines rented on demand from a cloud provider**, which `run.py` starts
  for the run and shuts down afterwards. Runpod is supported today, and other
  providers can be added.

Use it when your computer has no NVIDIA GPU, or to process a folder of clips on
several GPUs at once. It runs on macOS, Linux, and Windows and needs only
Python 3.10+ and the `ssh` client. No local GPU, PyTorch, or model weights
are needed.

For each run it:

1. Connects to each machine, starting it first if it comes from a provider.
2. Uploads this repository's code and your clips over SSH.
3. Installs the requirements and downloads the weights on the machine.
4. Runs `infer.py` on each clip and downloads the outputs.
5. Shuts down the machines it started, including when the run fails or you
   press Ctrl-C. Your own machines are left running.

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

## Your own machines over SSH

The machine needs Linux, an NVIDIA GPU, `ffmpeg`, and a Python environment
with CUDA PyTorch. The setup steps in the main README produce this. You must
be able to `ssh` into it without typing a password, using your SSH agent or a
key file.

```
python remote/run.py --provider ssh \
  --ssh_host artist@gpu-box \
  --ssh_python /home/artist/CorridorKey2.0/venv/bin/python \
  --input shot.mp4 --output_dir out
```

| Flag | Default | Purpose |
|---|---|---|
| `--ssh_host` | required | `[user@]host[:port]`, or a `Host` alias from `~/.ssh/config`. Repeat for several machines. |
| `--ssh_python` | `python3` | Python with CUDA PyTorch. The requirements are installed into it, so use a virtual environment rather than the system Python. |
| `--ssh_key` | your SSH agent and config | Private key file to use. |
| `--ssh_workdir` | `corridorkey-remote` | Directory for code, weights, and job files. Relative paths are under the SSH user's home. |

The first run on a machine downloads the ~5 GB of weights into
`--ssh_workdir`. Later runs reuse them. Host keys are checked against your
`known_hosts`, and a host you have never connected to is added on first use.
Each clip's files are deleted from the machine once its outputs have been
downloaded.

## Runpod

Create an API key at
[console.runpod.io/user/settings](https://console.runpod.io/user/settings),
then:

```
export RUNPOD_API_KEY=your_key
python remote/run.py --input shot.mp4 --output_dir out
```

Runpod is the default `--provider`. Pods bill per second to your Runpod
account while they run, and the script prints each pod's hourly price when it
starts it. Each run uses a new SSH key, so nothing needs to be set up on your
account.

| Flag | Default | Purpose |
|---|---|---|
| `--runpod_gpu` | 48 GB cards (RTX A6000, A40, L40, RTX 6000 Ada, L40S), then RTX 4090, RTX 5090, RTX PRO 6000, and A100 80GB | GPU types to try, in order. Repeat the flag to give your own list. |
| `--runpod_network_volume` | none | Network volume to keep the weights on between runs. |
| `--runpod_volume_gb` | 60 | Pod disk for weights, inputs, and outputs when there is no network volume. |
| `--runpod_cloud_type` | `SECURE` | `COMMUNITY` is cheaper, with less availability. |
| `--runpod_image` | `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` | Container image. |

48 GB cards let CorridorKey use its largest temporal window up to 1440p (see
[Long Clips And Low VRAM](../README.md#long-clips-and-low-vram)). The list
goes on to other cards so runs still start when those are sold out.

Without a network volume, each pod downloads the ~5 GB of weights at the start
of the run. With `--runpod_network_volume`, they are downloaded once and reused
by later runs and by every pod in a run. Create the volume in the Runpod
console. Pods then start in that volume's data center.

## Several clips on several GPUs

Point `--input` at a folder of clips:

```
python remote/run.py --input clips/ --output_dir out --machines 3
python remote/run.py --provider ssh --ssh_host box1 --ssh_host box2 \
  --input clips/ --output_dir out
```

The folder is read the same way as with the drag-and-drop launcher: video
files and `Input`/`AlphaHint` frame folders are clips, companion hint files
(`<clip>FrameHint.png`, `<clip>_AlphaHint.mp4`) are applied automatically, and
each clip's outputs go to `out/out_<clip>`.

Each machine takes the next unprocessed clip until none are left, then is
shut down if a provider started it. `--machines` sets how many to start, and
defaults to 1 for cloud providers. With `--provider ssh`, every `--ssh_host`
is used unless `--machines` is lower. A single clip always runs on one
machine, because splitting a clip would break the temporal context between its
frames.

## Other options

- `--keep_machines` leaves started machines running after the run and prints
  the SSH command to connect. They keep billing until you delete them.
- `--ready_timeout` is how long to wait for a machine to accept SSH (default
  900 seconds).

## Adding a provider

Any cloud that rents a CUDA Linux machine with SSH access can be added in one
file of about 100 lines. See [ADDING_A_PROVIDER.md](ADDING_A_PROVIDER.md) for
the interface, the machine requirements, and a testing checklist.

## Tests

```
python remote/tests/test_run.py
```

The tests use a fake provider and a fake SSH connection, so they need no cloud
account, SSH server, or GPU.
