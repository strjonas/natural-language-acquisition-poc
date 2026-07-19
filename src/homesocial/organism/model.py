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
        primitive_action_size: int | None = None,
        visible_slots: int = 0,
        object_feature_offset: int = 0,
        object_feature_size: int = 0,
    ) -> None:
        super().__init__()
        self.vector_size = vector_size
        self.vocab_size = vocab_size
        self.tokens_per_utterance = tokens_per_utterance
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.primitive_action_size = (
            action_size if primitive_action_size is None else primitive_action_size
        )
        self.visible_slots = visible_slots
        self.object_feature_offset = object_feature_offset
        self.has_object_options = action_size > self.primitive_action_size
        self.object_feature_size = object_feature_size if self.has_object_options else 0
        if self.has_object_options and (
            visible_slots <= 0 or object_feature_size <= 0
        ):
            raise ValueError(
                "Object options require visible slots and object features."
            )

        self.token_embedding = nn.Embedding(vocab_size, token_embed_size)
        self.token_rnn = nn.GRU(token_embed_size, token_embed_size)
        self.input = nn.Linear(vector_size + token_embed_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.core = nn.GRU(hidden_size, hidden_size)
        self.post = nn.Linear(hidden_size, hidden_size)
        self.post_norm = nn.LayerNorm(hidden_size)

        self.policy = nn.Linear(hidden_size, action_size)
        self.value = nn.Linear(hidden_size, 1)

        # Every visible-slot option is the same abstract act applied to a
        # different perceived object. Sharing its action identity makes the
        # consequence model generalize across slots; selected raw object
        # features provide the binding target without revealing hidden kind.
        transition_action_size = self.primitive_action_size + int(
            self.has_object_options
        )
        self.transition = nn.Linear(
            hidden_size + transition_action_size + self.object_feature_size,
            hidden_size,
        )
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

    def _transition_features(
        self, actions: mx.array, vectors: mx.array | None
    ) -> tuple[mx.array, mx.array]:
        transition_action_size = self.primitive_action_size + int(
            self.has_object_options
        )
        transition_actions = mx.where(
            actions < self.primitive_action_size,
            actions,
            self.primitive_action_size,
        )
        action_features = mx.eye(transition_action_size)[transition_actions]
        if not self.has_object_options:
            return action_features, mx.zeros((*actions.shape, 0))
        if vectors is None:
            raise ValueError("Object-option consequences require observation vectors.")
        start = self.object_feature_offset
        stop = start + self.visible_slots * self.object_feature_size
        slots = vectors[..., start:stop].reshape(
            *vectors.shape[:-1], self.visible_slots, self.object_feature_size
        )
        slot_indices = mx.clip(
            actions - self.primitive_action_size,
            0,
            self.visible_slots - 1,
        )
        indices = mx.broadcast_to(
            slot_indices[..., None, None],
            (*slot_indices.shape, 1, self.object_feature_size),
        )
        selected = mx.take_along_axis(slots, indices, axis=-2).squeeze(-2)
        is_option = (actions >= self.primitive_action_size)[..., None]
        return action_features, mx.where(is_option, selected, mx.zeros_like(selected))

    def predict_consequences(
        self,
        states: mx.array,
        actions: mx.array,
        vectors: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Action-conditioned predictions from core states.

        Args:
            states: (B, T, hidden_size) refined core states.
            actions: (B, T) int action indices taken at those states.
            vectors: (B, T, vector_size) observations used to bind an object
                option to its selected learner-visible slot.

        Returns:
            next observation vectors (B, T, vector_size),
            changes in the four bodily needs (B, T, 4),
            reward deltas (B, T),
            caregiver-token logits (B, T, tokens_per_utterance, vocab_size).
        """

        action_features, object_features = self._transition_features(
            actions, vectors
        )
        x = mx.concatenate([states, action_features, object_features], axis=-1)
        h = nn.relu(self.transition_norm(self.transition(x)))
        next_vectors = self.next_vector(h)
        # Most actions only incur metabolism, while the rare consequential
        # actions change one need sharply. Predicting a bounded residual keeps
        # the current sensed body on an exact identity path instead of asking
        # the model to reconstruct its absolute state from a latent vector.
        need_deltas = 0.5 * mx.tanh(self.next_needs(h))
        rewards = self.reward_head(h).squeeze(-1)
        token_logits = self.next_tokens(h).reshape(
            *states.shape[:-1], self.tokens_per_utterance, self.vocab_size
        )
        return next_vectors, need_deltas, rewards, token_logits
