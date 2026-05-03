# Running the cross-domain NER experiments on ITU HPC

Files in this folder:

| File              | Purpose                                                            |
|-------------------|--------------------------------------------------------------------|
| `install_env.job` | One-time SLURM job: installs miniconda + creates the `ner` env.    |
| `smoke_test.job`  | 30-min sanity check on `scavenge` (`run_experiment.py --debug`).   |
| `train_iter.job`  | The real training job. Submit once per config.                     |
| `logs/`           | Auto-created on first job. SLURM `.out`/`.err` go here.            |

Everything below assumes your ITU username is **`olgy`**, project will live at
`/home/olgy/Cross_domain_NER_Project_2026`, and `/home/olgy/tmp` already exists.

---

## 0. From your laptop — push these scripts first

```bash
cd ~/Documents/Github/Cross_domain_NER_Project_2026
git add hpc/
git commit -m "HPC SLURM job scripts"
git push
```

## 1. Log in to the HPC

```bash
ssh olgy@hpc.itu.dk
```

## 2. Clone the repo (login node — this is fine, it's just git)

```bash
cd /home/olgy
git clone https://github.com/Oliverdron/Cross_domain_NER_Project_2026.git
cd Cross_domain_NER_Project_2026
mkdir -p hpc/logs
```

> If the repo is private, generate a GitHub personal access token and use it as
> the password when git asks. Or set up an SSH key on the HPC and add it to
> GitHub.

## 3. Install the conda env (as a job — never on the login node)

```bash
sbatch hpc/install_env.job
squeue -u olgy        # watch ST go from PD → R → (gone)
```

When it disappears from the queue, check the log:

```bash
ls hpc/logs/
cat hpc/logs/install_*.out
```

You should see something like `cuda available : True` (or `False` if the
install job didn't get a GPU — that's fine, the env is still installed
correctly; the real check happens in the training job).

## 4. Smoke test (recommended — ~10 min on a V100)

```bash
sbatch hpc/smoke_test.job
squeue -u olgy
tail -f hpc/logs/ner_smoke_*.out   # Ctrl-C to stop tailing
```

If this finishes without an error, the full job will run.

## 5. Launch the real training jobs

Submit one job per config — they'll run independently and can land on
different nodes.

```bash
sbatch --job-name=iter_conll \
       --export=ALL,CFG=experiments/config_conll.yaml \
       hpc/train_iter.job

sbatch --job-name=iter_astro \
       --export=ALL,CFG=experiments/config_astro.yaml \
       hpc/train_iter.job
```

Each will print its job ID, e.g. `Submitted batch job 12345`.

## 6. Monitor

```bash
squeue -u olgy                          # is it running?
tail -f hpc/logs/iter_conll_<JOBID>.out # live training log
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS,ReqMem
nvidia-smi                              # only useful if you ssh'd into the compute node
```

To cancel a job: `scancel <JOBID>`.

## 7. Get results back to your laptop

The runs land in `/home/olgy/Cross_domain_NER_Project_2026/runs/`. From your
laptop:

```bash
scp -rp olgy@hpc.itu.dk:/home/olgy/Cross_domain_NER_Project_2026/runs ./runs_hpc
```

Or selectively, just the summary CSVs:

```bash
scp olgy@hpc.itu.dk:'/home/olgy/Cross_domain_NER_Project_2026/runs/iter_*/summary.csv' .
```

---

## Why these specific SLURM settings?

| Setting                  | Value                  | Reason                                                                                            |
|--------------------------|------------------------|---------------------------------------------------------------------------------------------------|
| `--partition=acltr`      | acltr                  | GPU queue. 3-day max for students. `scavenge` only gives 24h and we need margin for `iter_astro`. |
| `--gres=gpu:v100:1`      | 1× V100 32GB           | bert-base @ batch 32, seq 512 fits comfortably; no point asking for more.                         |
| `--cpus-per-task=8`      | 8 CPU cores            | Tokenisation + dataloader workers. More than this is wasted for a single-GPU run.                 |
| `--mem=48G`              | 48 GB RAM              | Headroom for tokenised datasets + HF cache. Astro paragraphs (seq 512) are the heavy ones.        |
| `--time=2-00:00:00`      | 2 days                 | 27 fine-tunes × ~5–15 epochs each (early stop, patience 8). Safe even for the slow `iter_astro`.  |
| `module load CUDA/12.1.1`| CUDA 12.1              | Matches the `cu121` torch wheels installed in step 3.                                             |

If a job hits the 2-day wall and you want more margin, bump `--time=3-00:00:00`
(the student max on `acltr`). If `acltr` is busy and the job sits in `PD`
forever, swap to `--partition=scavenge --time=24:00:00` and re-submit — but
then split each config into smaller chunks (e.g. one seed at a time) so 24h is
enough.

## Common gotchas

- **"Permission denied" on git push from HPC:** clone with HTTPS + a personal
  access token, not SSH, unless you've added a HPC SSH key to GitHub.
- **`module: command not found` inside a job:** you likely shelled into the
  compute node from a different login session — the SLURM jobs source modules
  themselves, so this only matters for interactive use.
- **`No devices were found` from `nvidia-smi`:** you forgot `--gres=gpu:...`.
  CPU partitions never give you a GPU even if the node physically has one.
- **HF download stalls:** the first run downloads `bert-base-cased` (~440 MB).
  `HF_HOME=/home/olgy/.cache/huggingface` keeps it there for re-runs.
- **`numpy<2` conflict:** keep the pin in `requirements.txt`; `seqeval` and
  some older transformers paths still misbehave on numpy 2.
