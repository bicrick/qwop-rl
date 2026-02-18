# =============================================================================
# Copyright 2023 Simeon Manolov <s.manolloff@gmail.com>.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import safe_mean
from gymnasium.wrappers import TimeLimit
import os
import math
import stable_baselines3
import sb3_contrib


from . import common


class LogCallback(BaseCallback):
    """Logs user-defined `info` values into tensorboard"""

    def _on_step(self) -> bool:
        for k in common.INFO_KEYS:
            v = safe_mean([ep_info[k] for ep_info in self.model.ep_info_buffer])
            self.model.logger.record(f"user/{k}", v)
        return True

    on_step = _on_step  # Fixes a bug with stable-baselines3 in version 2.2.1


def init_model(
    venv,
    seed,
    model_load_file,
    learner_cls,
    learner_kwargs,
    learning_rate,
    log_tensorboard,
    out_dir,
):
    alg = None
    # Make a copy to avoid modifying the original (for metadata saving)
    model_kwargs = dict(learner_kwargs)
    
    # Inject device selection (cuda > mps > cpu)
    device = common.get_device(model_kwargs.pop("device", "auto"))
    model_kwargs["device"] = device
    print(f"Using device: {device}")

    match learner_cls:
        case "A2C":
            alg = stable_baselines3.A2C
        case "PPO":
            alg = stable_baselines3.PPO
        case "DQN":
            alg = stable_baselines3.DQN
        case "QRDQN":
            alg = sb3_contrib.QRDQN
        case "EQRDQN":
            from qwop_gym.algorithms.enhanced_qrdqn import EnhancedQRDQN
            from qwop_gym.buffers.prioritized_replay import PrioritizedReplayBuffer
            alg = EnhancedQRDQN
            
            # Extract PER parameters from model_kwargs (don't modify original learner_kwargs)
            # These will be passed to PrioritizedReplayBuffer
            per_alpha = model_kwargs.pop("per_alpha", 0.6)
            per_beta_start = model_kwargs.pop("per_beta_start", 0.4)
            per_beta_frames = model_kwargs.pop("per_beta_frames", 100000)
            per_eps = model_kwargs.pop("per_eps", 1e-6)
            
            # Set replay buffer class and kwargs
            model_kwargs["replay_buffer_class"] = PrioritizedReplayBuffer
            model_kwargs["replay_buffer_kwargs"] = {
                "alpha": per_alpha,
                "beta_start": per_beta_start,
                "beta_frames": per_beta_frames,
                "eps": per_eps,
            }
        case "SAC":
            alg = stable_baselines3.SAC
        case "DSAC":
            from qwop_gym.algorithms.discrete_sac import DiscreteSAC
            from qwop_gym.buffers.prioritized_replay import PrioritizedReplayBuffer
            alg = DiscreteSAC
            
            # Extract PER and SAC parameters from model_kwargs
            use_per = model_kwargs.pop("use_per", False)
            
            if use_per:
                per_alpha = model_kwargs.pop("per_alpha", 0.6)
                per_beta_start = model_kwargs.pop("per_beta_start", 0.4)
                per_beta_frames = model_kwargs.pop("per_beta_frames", 100000)
                per_eps = model_kwargs.pop("per_eps", 1e-6)
                
                # Set replay buffer class and kwargs
                model_kwargs["replay_buffer_class"] = PrioritizedReplayBuffer
                model_kwargs["replay_buffer_kwargs"] = {
                    "alpha": per_alpha,
                    "beta_start": per_beta_start,
                    "beta_frames": per_beta_frames,
                    "eps": per_eps,
                }
        case "RPPO":
            alg = sb3_contrib.RecurrentPPO
        case _:
            raise Exception("Unexpected learner_cls: %s" % learner_cls)

    if model_load_file:
        print("Loading %s model from %s" % (alg.__name__, model_load_file))
        try:
            # Try to load the full model (including optimizer state)
            model = alg.load(model_load_file, env=venv)
            print("Successfully loaded model with optimizer state")
        except (ValueError, RuntimeError) as e:
            # If optimizer state doesn't match (due to changed hyperparameters),
            # load just the policy and create a fresh model with new hyperparameters
            if "optimizer" in str(e).lower() or "parameter group" in str(e).lower():
                print(f"Note: Optimizer state mismatch (expected when changing hyperparameters)")
                print(f"Loading policy weights only, reinitializing optimizer with new config...")
                
                # Create a fresh model with new hyperparameters
                kwargs = dict(model_kwargs, learning_rate=learning_rate, seed=seed)
                model = alg(env=venv, **kwargs)
                
                # Load only the policy weights from the checkpoint, skip optimizer
                # Use SB3's load_from_zip_file to properly extract weights
                import torch
                from stable_baselines3.common.save_util import load_from_zip_file
                
                data, params, pytorch_variables = load_from_zip_file(
                    model_load_file,
                    device='cpu',  # Load to CPU first for compatibility, model will use correct device
                    print_system_info=False
                )
                
                # Load the policy network weights only (skip optimizer)
                model.policy.load_state_dict(params['policy'], strict=False)
                print("Policy weights loaded successfully")
                
                # For QRDQN, also load q_net and q_net_target if they exist separately
                if hasattr(model, 'q_net') and 'q_net' in params:
                    model.q_net.load_state_dict(params['q_net'])
                    print("Q-network weights loaded successfully")
                if hasattr(model, 'q_net_target') and 'q_net_target' in params:
                    model.q_net_target.load_state_dict(params['q_net_target'])
                    print("Target Q-network weights loaded successfully")
                
                print("Optimizer and replay buffer reinitialized with new hyperparameters")
            else:
                # Re-raise if it's a different error
                raise
    else:
        kwargs = dict(model_kwargs, learning_rate=learning_rate, seed=seed)
        model = alg(env=venv, **kwargs)

    if log_tensorboard:
        os.makedirs(out_dir, exist_ok=True)
        log = logger.configure(folder=out_dir, format_strings=["tensorboard"])
        model.set_logger(log)

    return model


#
# A note about tensorboard logging of user-defined values in `info`:
#
# On each step, if env is done, Monitor wrapper will read `info_keywords`
# from `info` and copy them into `info["episode"]`:
# https://github.com/DLR-RM/stable-baselines3/blob/v1.8.0/stable_baselines3/common/monitor.py#L103
#
# Then, on each step, SB3 algos (PPO/DQN/...) put all `info["episode"]`
# dicts from the vec_env's step into `ep_info_buffer`:
# https://github.com/DLR-RM/stable-baselines3/blob/v1.8.0/stable_baselines3/common/base_class.py#L441
#
# This buffer can then be accessed in SB3 callbacks, which also have
# access to the SB3 log - and that's how user-defined values in `info`
# (set by QwopEnv) can be logged into tensorboard.
#
def create_vec_env(seed, max_episode_steps, n_envs=1, vec_env_type="dummy"):
    """
    Create a vectorized environment.
    
    Args:
        seed: Random seed for environment
        max_episode_steps: Maximum steps per episode
        n_envs: Number of parallel environments
        vec_env_type: Type of vectorization - "dummy" (sequential) or "subproc" (parallel)
    
    Returns:
        Vectorized environment
    """
    vec_env_cls = SubprocVecEnv if vec_env_type == "subproc" else None
    
    if vec_env_type == "subproc":
        print(f"Using SubprocVecEnv with {n_envs} parallel environments")
        print("Note: Each environment will spawn its own browser instance")
    
    venv = make_vec_env(
        "local/QWOP-v1",
        n_envs=n_envs,
        env_kwargs={"seed": seed},
        monitor_kwargs={"info_keywords": common.INFO_KEYS},
        wrapper_class=TimeLimit,
        wrapper_kwargs={"max_episode_steps": max_episode_steps},
        vec_env_cls=vec_env_cls,
    )

    return venv


def train_sb3(
    learner_cls,
    seed,
    run_id,
    model_load_file,
    learner_kwargs,
    learner_lr_schedule,
    total_timesteps,
    max_episode_steps,
    n_checkpoints,
    out_dir_template,
    log_tensorboard,
    n_envs=1,
    vec_env_type="dummy",
    demo_file=None,
    demo_injection_ratio=0.5,
    demo_prefill_count=0,
):
    venv = create_vec_env(seed, max_episode_steps, n_envs, vec_env_type)

    try:
        out_dir = common.out_dir_from_template(out_dir_template, seed, run_id)
        learning_rate = common.lr_from_schedule(learner_lr_schedule)

        model = init_model(
            venv=venv,
            seed=seed,
            model_load_file=model_load_file,
            learner_cls=learner_cls,
            learner_kwargs=learner_kwargs,
            learning_rate=learning_rate,
            log_tensorboard=log_tensorboard,
            out_dir=out_dir,
        )

        # Build callback list
        callbacks = [
            LogCallback(),
            CheckpointCallback(
                save_freq=math.ceil(total_timesteps / n_checkpoints),
                save_path=out_dir,
                name_prefix="model",
            ),
        ]

        # Add DQNfD callback if demo_file is provided
        if demo_file is not None:
            from qwop_gym.callbacks import DQNfDCallback
            callbacks.append(
                DQNfDCallback(
                    demo_file=demo_file,
                    injection_ratio=demo_injection_ratio,
                    prefill_count=demo_prefill_count,
                    verbose=1,
                )
            )
            print(f"DQNfD enabled: demo_file={demo_file}, "
                  f"injection_ratio={demo_injection_ratio}, "
                  f"prefill_count={demo_prefill_count}")

        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False,
            progress_bar=True,
            callback=callbacks,
        )

        # The CheckpointCallback kinda makes this redundant...
        common.save_model(out_dir, model)

        return {"out_dir": out_dir}
    finally:
        venv.close()
