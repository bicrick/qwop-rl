QWOP Training Setup for Lambda Labs
====================================

Quick Start:
-----------
1. SSH into Lambda instance
2. Clone this repo: git clone <repo-url>
3. cd qwop-training
4. ./setup_lambda.sh
5. conda activate qwop
6. ./lambda_train.sh train_ppo

What's included:
---------------
- setup_lambda.sh    : Automated setup (installs everything)
- lambda_train.sh    : Run training with virtual display
- verify_setup.sh    : Verify installation

The setup script will:
- Install Miniconda
- Create Python 3.10 environment
- Install Chrome/Chromium
- Download ChromeDriver
- Install qwop-gym from PyPI
- Patch QWOP source
- Create config files

Training:
--------
./lambda_train.sh train_ppo     # PPO (recommended)
./lambda_train.sh train_dqn     # DQN
./lambda_train.sh train_qrdqn   # QRDQN
./lambda_train.sh spectate      # Watch trained model

Monitor with TensorBoard:
------------------------
tensorboard --logdir data/ --host 0.0.0.0 --port 6006

# From local machine:
ssh -L 6006:localhost:6006 ubuntu@<lambda-ip>
# Open http://localhost:6006

Config files:
------------
Edit config/*.yml to customize training parameters

Troubleshooting:
---------------
- Display errors: sudo apt-get install xvfb
- ChromeDriver mismatch: Download from https://googlechromelabs.github.io/chrome-for-testing/
- Check setup: ./verify_setup.sh
