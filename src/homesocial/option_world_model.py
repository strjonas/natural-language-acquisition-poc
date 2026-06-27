from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .agents import TeacherFollowingAgent
from .attribution import _needs_array
from .env import Action, HomeostaticSocialGrid
from .imitation import load_checkpoint, save_checkpoint
from .observations import observation_vector
from .option_counterfactual_language import OPTION_NAMES, _option_action
from .recurrent_ac import RecurrentActorCritic, RecurrentConfig, action_mask
from .self_state_language import _trend
from .teachers import build_teacher, masks_language, normalize_teacher_mode


@dataclass(frozen=True)
class OptionBranchSample:
    observations: mx.array
    actions: mx.array
    next_observations: mx.array
    next_needs: mx.array
    rewards: mx.array
    current_needs: mx.array
    option_label: int
    trend_label: int


@dataclass(frozen=True)
class OptionBranchDataset:
    samples: tuple[OptionBranchSample, ...]
    config: RecurrentConfig
    horizon: int


@dataclass(frozen=True)
class OptionBranchBatch:
    observations: mx.array
    step_masks: mx.array
    observation_lengths: mx.array
    actions: mx.array
    action_masks: mx.array
    action_lengths: mx.array
    next_observations: mx.array
    next_needs: mx.array
    rewards: mx.array
    current_needs: mx.array
    option_labels: mx.array
    trend_labels: mx.array


@dataclass(frozen=True)
class OptionWorldModelResult:
    samples: int
    average_horizon: float
    step_need_mse: float
    final_need_mse: float
    final_observation_mse: float
    reward_mse: float
    trend_accuracy: float
    lowest_need_accuracy: float


@dataclass(frozen=True)
class OptionWorldTrainResult:
    samples: int
    epochs: int
    final_loss: float
    checkpoint_path: str | None


def collect_option_branch_dataset(
    config: RecurrentConfig,
    *,
    episodes: int,
    seed: int,
    teacher_mode: str = "grounded",
    horizon: int = 6,
    max_samples: int | None = None,
) -> OptionBranchDataset:
    normalized_teacher = normalize_teacher_mode(teacher_mode)
    mask_language = (
        masks_language(normalized_teacher) or not config.include_language_channel
    )
    env = HomeostaticSocialGrid(
        width=config.width,
        height=config.height,
        seed=seed,
        max_steps=config.max_steps,
        teacher=build_teacher(normalized_teacher, seed=seed),
        randomize_world=config.randomize_world,
        diagnostic_mode=config.diagnostic_mode,
        body_dynamics_mode=config.body_dynamics_mode,
    )
    samples: list[OptionBranchSample] = []

    for episode in range(episodes):
        observation = env.reset(seed=seed + episode)
        agent = TeacherFollowingAgent()
        history: list[np.ndarray] = []
        terminated = False
        truncated = False
        while not terminated and not truncated:
            history.append(
                observation_vector(
                    observation,
                    width=env.width,
                    height=env.height,
                    include_language=config.include_language_channel,
                    mask_language=mask_language,
                    include_object_kinds=config.include_object_kinds,
                    interoception_mode=config.interoception_mode,
                    body_dynamics_mode=config.body_dynamics_mode,
                )
            )
            before_needs = _needs_array(observation.needs)
            for option_index, option_name in enumerate(OPTION_NAMES):
                sample = _option_branch_sample(
                    env,
                    observation,
                    history,
                    option_name=option_name,
                    option_index=option_index,
                    current_needs=before_needs,
                    horizon=max(1, horizon),
                    config=config,
                    mask_language=mask_language,
                )
                samples.append(sample)
                if max_samples is not None and len(samples) >= max_samples:
                    return OptionBranchDataset(
                        samples=tuple(samples[:max_samples]),
                        config=config,
                        horizon=max(1, horizon),
                    )

            policy_observation = observation
            if mask_language:
                policy_observation = observation.__class__(
                    step_count=observation.step_count,
                    position=observation.position,
                    direction=observation.direction,
                    needs=observation.needs,
                    visible=observation.visible,
                    object_ahead=observation.object_ahead,
                    teacher_utterance=None,
                    last_event=observation.last_event,
                )
            action = agent.act(policy_observation)
            action_index = tuple(Action).index(action)
            if action_mask(observation)[action_index] <= 0.0:
                action = Action.MOVE_FORWARD
            observation, _reward, terminated, truncated, _info = env.step(action)

    return OptionBranchDataset(
        samples=tuple(samples),
        config=config,
        horizon=max(1, horizon),
    )


def train_option_world_model(
    model: RecurrentActorCritic,
    config: RecurrentConfig,
    dataset: OptionBranchDataset,
    *,
    checkpoint_path: str | None = None,
    epochs: int = 6,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    seed: int | None = None,
    step_needs_weight: float = 2.0,
    final_needs_weight: float = 8.0,
    observation_prediction_weight: float = 0.03,
    reward_prediction_weight: float = 0.4,
) -> OptionWorldTrainResult:
    if not dataset.samples:
        raise ValueError("Cannot train option world model on an empty dataset.")

    train_seed = config.seed if seed is None else seed
    rng = np.random.default_rng(train_seed)
    mx.random.seed(train_seed)
    optimizer = optim.Adam(learning_rate=learning_rate)
    samples = list(dataset.samples)
    final_loss = 0.0

    def loss_fn(
        observations: mx.array,
        step_masks: mx.array,
        observation_lengths: mx.array,
        actions: mx.array,
        action_masks: mx.array,
        action_lengths: mx.array,
        next_observations: mx.array,
        next_needs: mx.array,
        rewards: mx.array,
    ) -> mx.array:
        return _option_world_loss(
            model,
            observations,
            step_masks,
            observation_lengths,
            actions,
            action_masks,
            action_lengths,
            next_observations,
            next_needs,
            rewards,
            step_needs_weight,
            final_needs_weight,
            observation_prediction_weight,
            reward_prediction_weight,
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for _epoch in range(max(1, epochs)):
        rng.shuffle(samples)
        for start in range(0, len(samples), max(1, batch_size)):
            batch = pad_option_branch_samples(samples[start : start + max(1, batch_size)])
            loss, grads = loss_and_grad(
                batch.observations,
                batch.step_masks,
                batch.observation_lengths,
                batch.actions,
                batch.action_masks,
                batch.action_lengths,
                batch.next_observations,
                batch.next_needs,
                batch.rewards,
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            final_loss = float(loss)

    if checkpoint_path is not None:
        save_checkpoint(model, config, checkpoint_path)

    return OptionWorldTrainResult(
        samples=len(dataset.samples),
        epochs=max(1, epochs),
        final_loss=final_loss,
        checkpoint_path=checkpoint_path,
    )


def evaluate_option_world_model(
    model: RecurrentActorCritic,
    dataset: OptionBranchDataset,
    *,
    batch_size: int = 128,
) -> OptionWorldModelResult:
    if not dataset.samples:
        raise ValueError("Cannot evaluate option world model on an empty dataset.")

    step_need_errors: list[np.ndarray] = []
    final_need_errors: list[np.ndarray] = []
    final_observation_errors: list[np.ndarray] = []
    reward_errors: list[np.ndarray] = []
    predicted_trends: list[int] = []
    target_trends: list[int] = []
    predicted_lowest: list[int] = []
    target_lowest: list[int] = []
    action_lengths: list[int] = []

    samples = list(dataset.samples)
    for start in range(0, len(samples), max(1, batch_size)):
        batch = pad_option_branch_samples(samples[start : start + max(1, batch_size)])
        predicted_observations, predicted_needs, predicted_rewards = (
            _option_rollout_predictions(
                model,
                batch.observations,
                batch.step_masks,
                batch.observation_lengths,
                batch.actions,
                batch.action_masks,
            )
        )
        final_predicted_needs = _last_action_value(
            predicted_needs,
            batch.action_lengths,
        )
        final_target_needs = _last_action_value(batch.next_needs, batch.action_lengths)
        final_predicted_observations = _last_action_value(
            predicted_observations,
            batch.action_lengths,
        )
        final_target_observations = _last_action_value(
            batch.next_observations,
            batch.action_lengths,
        )

        need_error = np.asarray(
            mx.mean((predicted_needs - batch.next_needs) ** 2, axis=-1)
        )
        masks = np.asarray(batch.action_masks)
        step_need_errors.append(need_error[masks > 0.0])
        final_need_errors.append(
            np.asarray(mx.mean((final_predicted_needs - final_target_needs) ** 2, axis=-1))
        )
        final_observation_errors.append(
            np.asarray(
                mx.mean(
                    (final_predicted_observations - final_target_observations) ** 2,
                    axis=-1,
                )
            )
        )
        reward_error = np.asarray((predicted_rewards - batch.rewards) ** 2)
        reward_errors.append(reward_error[masks > 0.0])

        predicted_final = np.asarray(final_predicted_needs)
        target_final = np.asarray(final_target_needs)
        current = np.asarray(batch.current_needs)
        predicted_trends.extend(
            _trend(current[index], predicted_final[index])
            for index in range(predicted_final.shape[0])
        )
        target_trends.extend(
            int(value) for value in np.asarray(batch.trend_labels).tolist()
        )
        predicted_lowest.extend(np.argmin(predicted_final, axis=1).tolist())
        target_lowest.extend(np.argmin(target_final, axis=1).tolist())
        action_lengths.extend(np.asarray(batch.action_lengths).tolist())

    return OptionWorldModelResult(
        samples=len(dataset.samples),
        average_horizon=float(np.mean(action_lengths)),
        step_need_mse=float(np.mean(np.concatenate(step_need_errors))),
        final_need_mse=float(np.mean(np.concatenate(final_need_errors))),
        final_observation_mse=float(np.mean(np.concatenate(final_observation_errors))),
        reward_mse=float(np.mean(np.concatenate(reward_errors))),
        trend_accuracy=float(
            np.mean(np.asarray(predicted_trends) == np.asarray(target_trends))
        ),
        lowest_need_accuracy=float(
            np.mean(np.asarray(predicted_lowest) == np.asarray(target_lowest))
        ),
    )


def pad_option_branch_samples(samples: list[OptionBranchSample]) -> OptionBranchBatch:
    if not samples:
        raise ValueError("Cannot pad an empty option branch batch.")

    batch_size = len(samples)
    max_observation_length = max(sample.observations.shape[0] for sample in samples)
    max_action_length = max(sample.actions.shape[0] for sample in samples)
    input_size = samples[0].observations.shape[-1]

    observations = np.zeros(
        (batch_size, max_observation_length, input_size),
        dtype=np.float32,
    )
    step_masks = np.zeros((batch_size, max_observation_length), dtype=np.float32)
    observation_lengths = np.zeros(batch_size, dtype=np.int32)
    actions = np.zeros((batch_size, max_action_length), dtype=np.int32)
    action_masks = np.zeros((batch_size, max_action_length), dtype=np.float32)
    action_lengths = np.zeros(batch_size, dtype=np.int32)
    next_observations = np.zeros(
        (batch_size, max_action_length, input_size),
        dtype=np.float32,
    )
    next_needs = np.zeros((batch_size, max_action_length, 4), dtype=np.float32)
    rewards = np.zeros((batch_size, max_action_length), dtype=np.float32)
    current_needs = np.zeros((batch_size, 4), dtype=np.float32)
    option_labels = np.zeros(batch_size, dtype=np.int32)
    trend_labels = np.zeros(batch_size, dtype=np.int32)

    for index, sample in enumerate(samples):
        observation_length = sample.observations.shape[0]
        action_length = sample.actions.shape[0]
        observations[index, :observation_length] = np.asarray(sample.observations)
        step_masks[index, :observation_length] = 1.0
        observation_lengths[index] = observation_length
        actions[index, :action_length] = np.asarray(sample.actions)
        action_masks[index, :action_length] = 1.0
        action_lengths[index] = action_length
        next_observations[index, :action_length] = np.asarray(sample.next_observations)
        next_needs[index, :action_length] = np.asarray(sample.next_needs)
        rewards[index, :action_length] = np.asarray(sample.rewards)
        current_needs[index] = np.asarray(sample.current_needs)
        option_labels[index] = sample.option_label
        trend_labels[index] = sample.trend_label

    return OptionBranchBatch(
        observations=mx.array(observations, dtype=mx.float32),
        step_masks=mx.array(step_masks, dtype=mx.float32),
        observation_lengths=mx.array(observation_lengths, dtype=mx.int32),
        actions=mx.array(actions, dtype=mx.int32),
        action_masks=mx.array(action_masks, dtype=mx.float32),
        action_lengths=mx.array(action_lengths, dtype=mx.int32),
        next_observations=mx.array(next_observations, dtype=mx.float32),
        next_needs=mx.array(next_needs, dtype=mx.float32),
        rewards=mx.array(rewards, dtype=mx.float32),
        current_needs=mx.array(current_needs, dtype=mx.float32),
        option_labels=mx.array(option_labels, dtype=mx.int32),
        trend_labels=mx.array(trend_labels, dtype=mx.int32),
    )


def _option_branch_sample(
    env: HomeostaticSocialGrid,
    observation,
    history: list[np.ndarray],
    *,
    option_name: str,
    option_index: int,
    current_needs: np.ndarray,
    horizon: int,
    config: RecurrentConfig,
    mask_language: bool,
) -> OptionBranchSample:
    branch = deepcopy(env)
    branch_observation = observation
    actions: list[int] = []
    next_observations: list[np.ndarray] = []
    next_needs: list[np.ndarray] = []
    rewards: list[float] = []

    for _step in range(horizon):
        action = _option_action(option_name, branch_observation)
        mask = action_mask(branch_observation)
        action_index = tuple(Action).index(action)
        if mask[action_index] <= 0.0:
            action = Action.MOVE_FORWARD
            action_index = tuple(Action).index(action)
        branch_observation, reward, terminated, truncated, _info = branch.step(action)
        actions.append(action_index)
        next_observations.append(
            observation_vector(
                branch_observation,
                width=env.width,
                height=env.height,
                include_language=config.include_language_channel,
                mask_language=mask_language,
                include_object_kinds=config.include_object_kinds,
                interoception_mode=config.interoception_mode,
                body_dynamics_mode=config.body_dynamics_mode,
            )
        )
        next_needs.append(_needs_array(branch_observation.needs))
        rewards.append(float(reward))
        if terminated or truncated:
            break

    final_needs = next_needs[-1]
    return OptionBranchSample(
        observations=mx.array(np.stack(history), dtype=mx.float32),
        actions=mx.array(actions, dtype=mx.int32),
        next_observations=mx.array(np.stack(next_observations), dtype=mx.float32),
        next_needs=mx.array(np.stack(next_needs), dtype=mx.float32),
        rewards=mx.array(rewards, dtype=mx.float32),
        current_needs=mx.array(current_needs, dtype=mx.float32),
        option_label=option_index,
        trend_label=_trend(current_needs, final_needs),
    )


def _option_rollout_predictions(
    model: RecurrentActorCritic,
    observations: mx.array,
    step_masks: mx.array,
    observation_lengths: mx.array,
    actions: mx.array,
    action_masks: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    hidden = model.hidden_states(observations)
    hidden = mx.stop_gradient(
        _last_valid_hidden(hidden, observation_lengths, step_masks)
    )
    observation_predictions = []
    need_predictions = []
    reward_predictions = []
    state = hidden
    for step in range(actions.shape[1]):
        action_features = mx.eye(model.action_size)[actions[:, step]]
        x = mx.concatenate([state, action_features], axis=-1)
        x = nn.relu(model.transition_norm(model.transition(x)))
        next_state = x + nn.relu(model.transition_state(x))
        observation_predictions.append(model.next_observation(next_state))
        need_predictions.append(mx.sigmoid(model.next_needs(next_state)))
        reward_predictions.append(model.reward(next_state).squeeze(-1))
        state = mx.where(action_masks[:, step : step + 1] > 0.0, next_state, state)
    return (
        mx.stack(observation_predictions, axis=1),
        mx.stack(need_predictions, axis=1),
        mx.stack(reward_predictions, axis=1),
    )


def _option_world_loss(
    model: RecurrentActorCritic,
    observations: mx.array,
    step_masks: mx.array,
    observation_lengths: mx.array,
    actions: mx.array,
    action_masks: mx.array,
    action_lengths: mx.array,
    next_observations: mx.array,
    next_needs: mx.array,
    rewards: mx.array,
    step_needs_weight: float,
    final_needs_weight: float,
    observation_prediction_weight: float,
    reward_prediction_weight: float,
) -> mx.array:
    predicted_observations, predicted_needs, predicted_rewards = (
        _option_rollout_predictions(
            model,
            observations,
            step_masks,
            observation_lengths,
            actions,
            action_masks,
        )
    )
    step_needs_loss = _masked_mean(
        mx.mean((predicted_needs - next_needs) ** 2, axis=-1),
        action_masks,
    )
    final_predicted_needs = _last_action_value(predicted_needs, action_lengths)
    final_target_needs = _last_action_value(next_needs, action_lengths)
    final_needs_loss = mx.mean((final_predicted_needs - final_target_needs) ** 2)
    observation_loss = _masked_mean(
        mx.mean((predicted_observations - next_observations) ** 2, axis=-1),
        action_masks,
    )
    reward_loss = _masked_mean((predicted_rewards - rewards) ** 2, action_masks)
    return (
        step_needs_weight * step_needs_loss
        + final_needs_weight * final_needs_loss
        + observation_prediction_weight * observation_loss
        + reward_prediction_weight * reward_loss
    )


def _last_valid_hidden(
    hidden: mx.array,
    lengths: mx.array,
    step_masks: mx.array,
) -> mx.array:
    del step_masks
    selectors = mx.eye(hidden.shape[1])[lengths - 1]
    return mx.sum(hidden * selectors[..., None], axis=1)


def _last_action_value(values: mx.array, lengths: mx.array) -> mx.array:
    selectors = mx.eye(values.shape[1])[lengths - 1]
    while len(selectors.shape) < len(values.shape):
        selectors = selectors[..., None]
    return mx.sum(values * selectors, axis=1)


def _masked_mean(values: mx.array, masks: mx.array) -> mx.array:
    return mx.sum(values * masks) / mx.maximum(mx.sum(masks), 1.0)


def _format_result(phase: str, result: OptionWorldModelResult) -> str:
    return ",".join(
        [
            phase,
            str(result.samples),
            f"{result.average_horizon:.2f}",
            f"{result.step_need_mse:.6f}",
            f"{result.final_need_mse:.6f}",
            f"{result.final_observation_mse:.6f}",
            f"{result.reward_mse:.6f}",
            f"{result.trend_accuracy:.4f}",
            f"{result.lowest_need_accuracy:.4f}",
        ]
    )


def main() -> None:
    args = _parse_args()
    model, config = load_checkpoint(args.checkpoint)
    train_dataset = collect_option_branch_dataset(
        config,
        episodes=args.train_episodes,
        seed=args.seed,
        teacher_mode=args.teacher_mode,
        horizon=args.horizon,
        max_samples=args.max_train_samples,
    )
    eval_dataset = collect_option_branch_dataset(
        config,
        episodes=args.eval_episodes,
        seed=args.seed + 10_000,
        teacher_mode=args.teacher_mode,
        horizon=args.horizon,
        max_samples=args.max_eval_samples,
    )

    print(
        "phase,samples,avg_horizon,step_need_mse,final_need_mse,"
        "final_observation_mse,reward_mse,trend_accuracy,lowest_need_accuracy"
    )
    print(_format_result("before", evaluate_option_world_model(model, eval_dataset)))
    train_result = train_option_world_model(
        model,
        config,
        train_dataset,
        checkpoint_path=args.output_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        step_needs_weight=args.step_needs_weight,
        final_needs_weight=args.final_needs_weight,
        observation_prediction_weight=args.observation_prediction_weight,
        reward_prediction_weight=args.reward_prediction_weight,
    )
    print(f"# train_samples={train_result.samples}")
    print(f"# final_loss={train_result.final_loss:.6f}")
    if train_result.checkpoint_path is not None:
        print(f"# checkpoint={train_result.checkpoint_path}")
    print(_format_result("after", evaluate_option_world_model(model, eval_dataset)))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=9601)
    parser.add_argument("--train-episodes", type=int, default=600)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--max-train-samples", type=int, default=9000)
    parser.add_argument("--max-eval-samples", type=int, default=4500)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--step-needs-weight", type=float, default=2.0)
    parser.add_argument("--final-needs-weight", type=float, default=8.0)
    parser.add_argument("--observation-prediction-weight", type=float, default=0.03)
    parser.add_argument("--reward-prediction-weight", type=float, default=0.4)
    parser.add_argument("--teacher-mode", default="grounded")
    return parser.parse_args()


if __name__ == "__main__":
    main()
