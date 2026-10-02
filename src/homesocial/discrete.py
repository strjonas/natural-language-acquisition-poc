"""Discrete message operations with explicit surrogate gradients."""

import mlx.core as mx


def straight_through_one_hot(probabilities: mx.array) -> mx.array:
    """Select the largest probability, backpropagating as the identity.

    The integer choice has no derivative. Stop it before gathering so MLX
    never attempts to differentiate indices; only the soft path learns.
    Preserve the legacy arithmetic order, including its floating-point values.
    """
    indices = mx.stop_gradient(mx.argmax(probabilities, axis=-1))
    hard = mx.eye(probabilities.shape[-1])[indices]
    return hard + probabilities - mx.stop_gradient(probabilities)
