"""The organism's single model.

One recurrent core consumes vision/interoception vectors plus heard creole
tokens, and every capability hangs off that shared state: acting (policy),
valuing (critic), and predicting the consequences of its own actions for the
world, its body, its reward, and what the caregiver will say. There are no
separately trained frozen heads; this is the integration the probe era never
built.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class OrganismModel(nn.Module):
    def __init__(
        self,
        *,
        vector_size: int,
        vocab_size: int,
        tokens_per_utterance: int,
        action_size: int,
        hidden_size: int = 256,
        token_embed_size: int = 32,
    ) -> None:
        super().__init__()
        self.vector_size = vector_size
        self.vocab_size = vocab_size
        self.tokens_per_utterance = tokens_per_utterance
        self.action_size = action_size
        self.hidden_size = hidden_size

        self.token_embedding = nn.Embedding(vocab_size, token_embed_size)
        self.token_rnn = nn.GRU(token_embed_size, token_embed_size)
        self.input = nn.Linear(vector_size + token_embed_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.core = nn.GRU(hidden_size, hidden_size)
        self.post = nn.Linear(hidden_size, hidden_size)
        self.post_norm = nn.LayerNorm(hidden_size)

        self.policy = nn.Linear(hidden_size, action_size)
        self.value = nn.Linear(hidden_size, 1)

        self.transition = nn.Linear(hidden_size + action_size, hidden_size)
        self.transition_norm = nn.LayerNorm(hidden_size)
        self.next_vector = nn.Linear(hidden_size, vector_size)
        self.next_needs = nn.Linear(hidden_size, 4)
        self.reward_head = nn.Linear(hidden_size, 1)
        self.next_tokens = nn.Linear(hidden_size, tokens_per_utterance * vocab_size)

    def _encode_tokens(self, tokens: mx.array) -> mx.array:
        """(B, T, L) int token ids -> (B, T, E) utterance encodings."""

        batch, steps, length = tokens.shape
        embedded = self.token_embedding(tokens.reshape(batch * steps, length))
        states = self.token_rnn(embedded)
        return states[:, -1, :].reshape(batch, steps, -1)

    def _features(self, vectors: mx.array, tokens: mx.array) -> mx.array:
        token_encoding = self._encode_tokens(tokens)
        x = mx.concatenate([vectors, token_encoding], axis=-1)
        return nn.relu(self.input_norm(self.input(x)))

    def core_states(
        self,
        vectors: mx.array,
        tokens: mx.array,
        hidden: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Run the recurrent core over a segment.

        Args:
            vectors: (B, T, vector_size) float observations.
            tokens: (B, T, tokens_per_utterance) int heard-token ids.
            hidden: (B, hidden_size) carry state or None.

        Returns:
            refined states (B, T, hidden_size) for the heads, and the raw
            final GRU state (B, hidden_size) to carry forward.
        """

        features = self._features(vectors, tokens)
        states = self.core(features, hidden)
        refined = states + nn.relu(self.post_norm(self.post(states)))
        return refined, states[:, -1, :]

    def policy_value(
        self,
        vectors: mx.array,
        tokens: mx.array,
        hidden: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        states, carry = self.core_states(vectors, tokens, hidden)
        logits = self.policy(states)
        values = self.value(states).squeeze(-1)
        return logits, values, carry

    def predict_consequences(
        self, states: mx.array, actions: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Action-conditioned predictions from core states.

        Args:
            states: (B, T, hidden_size) refined core states.
            actions: (B, T) int action indices taken at those states.

        Returns:
            next observation vectors (B, T, vector_size),
            next needs in [0, 1] (B, T, 4),
            reward deltas (B, T),
            caregiver-token logits (B, T, tokens_per_utterance, vocab_size).
        """

        action_features = mx.eye(self.action_size)[actions]
        x = mx.concatenate([states, action_features], axis=-1)
        h = nn.relu(self.transition_norm(self.transition(x)))
        next_vectors = self.next_vector(h)
        next_needs = mx.sigmoid(self.next_needs(h))
        rewards = self.reward_head(h).squeeze(-1)
        token_logits = self.next_tokens(h).reshape(
            *states.shape[:-1], self.tokens_per_utterance, self.vocab_size
        )
        return next_vectors, next_needs, rewards, token_logits
