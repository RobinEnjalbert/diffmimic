# Copyright 2022 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Brax training acting functions."""

import time
from typing import Callable, Sequence, Tuple, Any, Dict

import jax
import numpy as np
from brax.training.types import Metrics
from brax.training.types import PRNGKey
from brax.training.types import Policy
from brax.training.types import PolicyParams
from brax.training.types import Transition
from brax.v1 import envs
from brax.v1.physics.base import QP


def actor_step(env: envs.Env,
               env_state: envs.State,
               policy: Policy,
               key: PRNGKey,
               extra_fields: Sequence[str] = ()) -> Tuple[envs.State, QP]:
    """Collect data."""

    actions, policy_extras = policy(env_state.obs, key)
    n_state = env.step(env_state, actions)
    state_extras = {x: n_state.info[x] for x in extra_fields}
    return n_state, env_state.qp


def generate_unroll(env: envs.Env,
                    env_state: envs.State,
                    policy: Policy,
                    key: PRNGKey,
                    unroll_length: int,
                    extra_fields: Sequence[str] = ()) -> Tuple[envs.State, Transition]:
    """Collect trajectories of given unroll_length."""

    @jax.jit
    def f(carry, unused_t):
        state, current_key = carry
        current_key, next_key = jax.random.split(current_key)
        n_state, qp = actor_step(env, state, policy, current_key, extra_fields=extra_fields)
        return (n_state, next_key), qp

    (final_state, _), qp_list = jax.lax.scan(f, (env_state, key), (), length=unroll_length)
    return final_state, qp_list


# TODO: Consider moving this to its own file.
class Evaluator:

    def __init__(self,
                 eval_env: envs.Env,
                 eval_policy_fn: Callable[[PolicyParams], Policy],
                 num_eval_envs: int,
                 episode_length: int,
                 action_repeat: int,
                 key: PRNGKey):
        """
        Class to run evaluations.

        :param eval_env: Batched environment to run evals on.
        :param eval_policy_fn: Function returning the policy from the policy parameters.
        :param num_eval_envs: Each env will run 1 episode in parallel for each eval.
        :param episode_length: Maximum length of an episode.
        :param action_repeat: Number of physics steps per env step.
        :param key: RNG key.
        """

        self._key = key
        self._eval_wall_time = 0.

        eval_env = envs.wrappers.EvalWrapper(eval_env)

        def generate_eval_unroll(policy_params: PolicyParams, _key: PRNGKey) -> (envs.State, QP):
            reset_keys = jax.random.split(_key, num_eval_envs)
            eval_first_state = eval_env.reset(reset_keys)
            return generate_unroll(eval_env,
                                   eval_first_state,
                                   eval_policy_fn(policy_params),
                                   _key,
                                   unroll_length=episode_length // action_repeat)

        self._generate_eval_unroll = jax.jit(generate_eval_unroll)
        self._steps_per_unroll = episode_length * num_eval_envs

    def run_evaluation(self,
                       policy_params: PolicyParams,
                       training_metrics: Metrics,
                       aggregate_episodes: bool = True) -> Tuple[Dict[str, float], Any]:
        """
        Run one epoch of evaluation.
        """

        self._key, unroll_key = jax.random.split(self._key)

        t = time.time()
        eval_state, qp_list = self._generate_eval_unroll(policy_params, unroll_key)
        eval_metrics = eval_state.info['eval_metrics']
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {f'eval/episode_{name}': np.mean(value) if aggregate_episodes else value
                   for name, value in eval_metrics.episode_metrics.items()}
        metrics = {name: value / np.mean(eval_metrics.episode_steps)
                   for name, value in metrics.items()}
        metrics['eval/avg_episode_length'] = np.mean(eval_metrics.episode_steps)
        metrics['eval/epoch_eval_time'] = epoch_eval_time
        metrics['eval/sps'] = self._steps_per_unroll / epoch_eval_time
        self._eval_wall_time = self._eval_wall_time + epoch_eval_time
        metrics = {'eval/wall_time': self._eval_wall_time,
                   **training_metrics,
                   **metrics}

        return metrics, qp_list
