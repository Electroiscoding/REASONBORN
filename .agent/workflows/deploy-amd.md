---
description: Provision AMD MI300X Droplet, connect IDE, and begin pre-training
---

# ReasonBorn AMD MI300X Deployment Workflow

## Step 1: Provision the AMD GPU Droplet

1. **Log in** to DigitalOcean Cloud Panel.
2. **Create** → **Droplets**.
3. **Region:** NYC or SFO (AMD GPU availability).
4. **Image:** Marketplace → search **"AMD ROCm"** or **"Ubuntu 22.04 AI/ML"** (pre-installed ROCm drivers).
5. **Size:** GPU Dedicated → MI300X (1x 192GB VRAM or 8x 1.5TB).
6. **Auth:** SSH Key only. If you need to generate one:

```powershell
ssh-keygen -t ed25519
type %userprofile%\.ssh\id_ed25519.pub
```

Paste the output into DigitalOcean → New SSH Key.

7. Click **Create Droplet** and **copy the IPv4 address**.

## Step 2: Configure Local SSH

Open `C:\Users\botma\.ssh\config` and add:

```
Host ReasonBorn-AMD
    HostName <PASTE_YOUR_DROPLET_IP_HERE>
    User root
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking no
```

## Step 3: Connect Antigravity IDE

1. Open Antigravity IDE.
2. Install **Remote - SSH** extension (`Ctrl+Shift+X`).
3. `Ctrl+Shift+P` → **Remote-SSH: Connect to Host...** → select **ReasonBorn-AMD**.
4. Wait ~30s for remote server install. Green badge: `>< SSH: ReasonBorn-AMD`.

## Step 4: Verify Hardware & Clone

// turbo
```bash
rocm-smi
```

Then clone the repo:
```bash
git clone <your_github_repo_url>
cd REASONBORN
```

Or drag-and-drop from local explorer into IDE.

## Step 5: Build & Boot ROCm Container

// turbo
```bash
bash scripts/deploy/boot_rocm.sh
```

This runs the Docker build + container launch with AMD GPU passthrough.

## Step 6: Begin Pre-Training

Inside the Docker container:

// turbo
```bash
python scripts/data/prepare_pretraining_data.py --output_dir data/processed/
```

Then launch training:

// turbo
```bash
bash scripts/training/train_reasonborn.sh configs/training/pretraining.yaml data/processed/
```
