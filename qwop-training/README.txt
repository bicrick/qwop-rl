QWOP Training Setup for Lambda Labs
====================================

IMPORTANT: For Ubuntu ARM64 Instances
--------------------------------------
This setup is optimized for Ubuntu ARM64 (aarch64) instances.
It uses distro-provided Chromium + ChromeDriver for compatibility.

Quick Start:
-----------
1. Connect to Lambda instance via Cursor Remote SSH

2. Clone this repo:
   git clone <your-repo-url> qwop-wr
   cd qwop-wr/qwop-training

3. Run automated setup:
   ./setup_lambda.sh


4. Activate environment and train:
   source ~/.bashrc
   conda activate qwop
   ./lambda_train.sh train_ppo

What Gets Installed:
-------------------
- qwop-gym is installed via pip (no need to copy qwop-gym directory)
- Chromium and ChromeDriver installed together (ensures version match)
- All dependencies in isolated conda environment
- Virtual display for headless training

Training Commands:
-----------------
./lambda_train.sh train_ppo     # PPO (recommended)
./lambda_train.sh train_dqn     # DQN
./lambda_train.sh train_qrdqn   # QRDQN
./lambda_train.sh spectate      # Watch trained model

TensorBoard (Real-time Monitoring):
-----------------------------------
On remote instance:
  tensorboard --logdir data/ --host 0.0.0.0 --port 6006

Cursor will detect the port and offer to forward it automatically.
Or manually: Command Palette -> "Forward a Port" -> 6006
Then open http://localhost:6006 in your browser

Configuration:
-------------
- Edit config/*.yml files to customize training hyperparameters
- config/env.yml contains browser/driver paths (auto-generated)

Architecture Notes (ARM64/aarch64):
----------------------------------
- Google does NOT provide official ChromeDriver for linux-arm64
- We use Ubuntu's chromium-browser + chromium-chromedriver packages
- These are compiled for ARM64 and version-matched
- System ChromeDriver is copied to project directory for consistency

Troubleshooting:
---------------
1. Version mismatch errors:
   sudo apt-get install --reinstall chromium-browser chromium-chromedriver

2. Display/rendering errors:
   sudo apt-get install xvfb
   
3. Conda not found after setup:
   source ~/.bashrc
   OR
   eval "$(~/miniconda3/bin/conda shell.bash hook)"

4. Check installation:
   chromium --version
   chromedriver --version
   (Major versions should match!)

5. Test qwop-gym:
   conda activate qwop
   qwop-gym play
