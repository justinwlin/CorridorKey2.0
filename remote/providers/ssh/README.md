# Provider: ssh

Runs CorridorKey on Linux machines you already have, such as a studio
workstation, a render node, or a server rented anywhere. `run.py` connects to
them over SSH and never starts or shuts them down.

See [the remote README](../../README.md) for how a run works and the flags
that apply to every provider.

## Requirements

On each machine:

- Linux with an NVIDIA GPU (RTX 30-series or newer).
- `ffmpeg`. It is installed automatically only when you connect as root.
- A Python environment with CUDA PyTorch. Following the setup in the main
  README on the machine gives you one. The CorridorKey requirements are
  installed into this environment, and its PyTorch is left as it is.
- About 10 GB free for the weights, plus room for clips and outputs.

On your computer:

- `ssh` that connects to the machine without a password prompt, using your
  SSH agent, a key in `~/.ssh/config`, or `--ssh_key`.

Check with `ssh <host> nvidia-smi` before your first run.

## Usage

```
python remote/run.py --provider ssh \
  --ssh_host artist@gpu-box \
  --ssh_python /home/artist/CorridorKey2.0/venv/bin/python \
  --input shot.mp4 --output_dir out
```

Several machines share a folder of clips:

```
python remote/run.py --provider ssh \
  --ssh_host gpu-box-1 --ssh_host gpu-box-2 \
  --input clips/ --output_dir out
```

## Flags

| Flag | Default | Purpose |
|---|---|---|
| `--ssh_host` | required | `[user@]host[:port]`, or a `Host` alias from `~/.ssh/config`. Repeat for several machines. |
| `--ssh_python` | `python3` | Python with CUDA PyTorch. The requirements are installed into it, so use a virtual environment rather than the system Python. |
| `--ssh_key` | your SSH agent and config | Private key file to use. |
| `--ssh_workdir` | `corridorkey-remote` | Directory for code, weights, and job files. Relative paths are under the SSH user's home. |

Every `--ssh_host` is used unless `--machines` is lower.

## Notes

- The first run on a machine downloads the ~5 GB of weights into
  `--ssh_workdir`. Later runs reuse them.
- Host keys are checked against your `known_hosts`. A host you have never
  connected to is added on first use.
