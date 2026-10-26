# Provider: runpod

Rents GPU pods on [Runpod](https://www.runpod.io) for the length of a run,
through the Runpod REST API v2, and terminates them when the run ends. Pods
bill per second to your own Runpod account.

See [the remote README](../../README.md) for how a run works and the flags
that apply to every provider.

## Setup

Create an API key at
[console.runpod.io/user/settings](https://console.runpod.io/user/settings)
and put it in the environment:

```
export RUNPOD_API_KEY=your_key
```

Nothing else needs to be set up on the account. Each run creates a new SSH
key and gives it to its pods, so you don't need to register SSH keys.

## Usage

```
python remote/run.py --provider runpod --input shot.mp4 --output_dir out
python remote/run.py --provider runpod --machines 3 --input clips/ --output_dir out
```

`runpod` is the default provider, so `--provider runpod` can be left out. The
script prints each pod's GPU and hourly price when it starts it.

## Flags

| Flag | Default | Purpose |
|---|---|---|
| `--runpod_gpu` | 48 GB cards (RTX A6000, A40, L40, RTX 6000 Ada, L40S), then RTX 4090, RTX 5090, RTX PRO 6000, and A100 80GB | GPU types to try, in order. Repeat the flag to give your own list. |
| `--runpod_network_volume` | none | Network volume to keep the weights on between runs. |
| `--runpod_volume_gb` | 60 | Pod disk for weights, inputs, and outputs when there is no network volume. |
| `--runpod_cloud_type` | `SECURE` | `COMMUNITY` is cheaper, with less availability. |
| `--runpod_image` | `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` | Container image. It needs CUDA PyTorch and an SSH server that reads `PUBLIC_KEY`, as Runpod's official images do. |

48 GB cards let CorridorKey use its largest temporal window up to 1440p (see
[Long Clips And Low VRAM](../../../README.md#long-clips-and-low-vram)). The
list goes on to other cards so runs still start when those are sold out. Each
type is tried in turn until one is available, and only hosts with CUDA 12.8 or
newer drivers are used.

## Keeping the weights between runs

Without a network volume, each pod downloads the ~5 GB of weights at the start
of the run, which takes a minute or two. With `--runpod_network_volume`, they
are downloaded once and reused by later runs and by every pod in a run. Create
the volume in the Runpod console. Pods then start in that volume's data
center, so the GPU types you ask for must be available there.
