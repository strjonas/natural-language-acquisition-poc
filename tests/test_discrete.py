import mlx.core as mx
import numpy as np
import pytest

from homesocial.discrete import straight_through_one_hot


@pytest.mark.parametrize("shape", [(4,), (2, 4), (2, 3, 4)])
def test_hard_tokens_keep_forward_values_and_softmax_gradient(shape):
    logits = mx.array(np.arange(np.prod(shape)).reshape(shape) % 7 / 3, dtype=mx.float32)
    probabilities = mx.softmax(logits, axis=-1)
    weights = mx.array([-.7, .2, 1.3, -.1])
    expected_tokens = np.eye(4, dtype=np.float32)[np.argmax(np.asarray(probabilities), axis=-1)]

    actual = straight_through_one_hot(probabilities)
    np.testing.assert_allclose(np.asarray(actual), expected_tokens, atol=1e-7)
    # Exact legacy forward parity, in addition to the mathematical one-hot value.
    legacy = mx.eye(4)[mx.argmax(probabilities, axis=-1)] + probabilities - probabilities
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(legacy))

    gradient = mx.grad(lambda x: mx.sum(straight_through_one_hot(mx.softmax(x, axis=-1)) * weights))(logits)
    expected_gradient = probabilities * (weights - mx.sum(probabilities * weights, axis=-1, keepdims=True))
    np.testing.assert_allclose(np.asarray(gradient), np.asarray(expected_gradient), rtol=1e-5, atol=1e-7)
    assert np.any(np.asarray(gradient) != 0), "the sender must still learn through discrete messages"


def test_tied_probabilities_keep_first_token_and_identity_surrogate():
    probabilities = mx.array([[.5, .5, 0.], [.2, .4, .4]])
    np.testing.assert_allclose(np.asarray(straight_through_one_hot(probabilities)), [[1, 0, 0], [0, 1, 0]], atol=1e-7)
    weights = mx.array([2., -1., .5])
    gradient = mx.grad(lambda p: mx.sum(straight_through_one_hot(p) * weights))(probabilities)
    np.testing.assert_array_equal(np.asarray(gradient), np.broadcast_to(np.asarray(weights), probabilities.shape))
