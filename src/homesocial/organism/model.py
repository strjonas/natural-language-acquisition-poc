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

# Metabolic drift over the longest option in the implemented task tops out
# near 0.14, while consumption events begin near 0.225; the delta histogram is
# empty in between. The cut is a gap in the data, not a tuned threshold. It
# separates the two regimes in the loss and bounds the split drift path here.
DRIFT_REGIME_THRESHOLD = 0.175
# The per-tick metabolic scale of the world: food 0.010, water 0.014, energy
# 0.015 while walking. Dividing by its square measures the drift regime as a
# relative error instead of an absolute one.
DRIFT_ERROR_SCALE = 0.02
# A correct resource consumption is worth 0.4 on the demanded need relative
# to either wrong terminal action in the fixed delayed task. This is the
# measured physical scale of the sparse bodily-event regime, not a fitted
# prediction scale.
EVENT_ERROR_SCALE = 0.4


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
        object_option_types: int | None = None,
        episodic_binding_size: int = 0,
        episodic_binding_writes: bool = True,
        split_drift_head: bool = False,
        visible_radius: int = 2,
        pad_token_id: int = 0,
        referential_action_indices: tuple[int, ...] = (3, 4),
        report_slots: int = 0,
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
        extra_actions = action_size - self.primitive_action_size
        if extra_actions < 0:
            raise ValueError("action_size cannot be smaller than primitive_action_size.")
        if extra_actions == 0:
            inferred_option_types = 0
        elif object_option_types is None:
            if visible_slots <= 0 or extra_actions % visible_slots:
                raise ValueError(
                    "Object-option actions must form complete visible-slot blocks."
                )
            inferred_option_types = extra_actions // visible_slots
        else:
            inferred_option_types = object_option_types
            if inferred_option_types <= 0 or extra_actions != (
                inferred_option_types * visible_slots
            ):
                raise ValueError(
                    "action_size must equal primitive actions plus one visible-slot "
                    "block per object option type."
                )
        self.object_option_types = inferred_option_types
        self.has_object_options = self.object_option_types > 0
        self.object_feature_size = object_feature_size if self.has_object_options else 0
        if self.has_object_options and (
            visible_slots <= 0 or object_feature_size <= 0
        ):
            raise ValueError(
                "Object options require visible slots and object features."
            )
        if episodic_binding_size < 0:
            raise ValueError("episodic_binding_size must be nonnegative.")
        if episodic_binding_size > 0 and not self.has_object_options:
            raise ValueError("Episodic bindings require learner-visible object options.")
        if visible_radius <= 0:
            raise ValueError("visible_radius must be positive.")
        self.episodic_binding_size = episodic_binding_size
        self.episodic_binding_writes = episodic_binding_writes
        self.has_split_drift_head = split_drift_head
        self.surface_count = max(0, self.object_feature_size - 3)
        self.visible_radius = visible_radius
        self.pad_token_id = pad_token_id
        self.referential_action_indices = referential_action_indices
        self.has_episodic_bindings = episodic_binding_size > 0
        if self.has_episodic_bindings and self.surface_count <= 0:
            raise ValueError("Episodic bindings require surface identity features.")

        # The external bank is carried alongside, rather than compressed into,
        # the recurrent core.  Each learner-visible surface is an episodic key;
        # the value written there depends only on the heard utterance.  Validity
        # bits prevent an uninspected surface from reading a zero vector as if
        # it were a learned meaning.
        self.binding_bank_size = self.surface_count * episodic_binding_size
        self.binding_valid_size = self.surface_count if self.has_episodic_bindings else 0
        self.state_size = (
            hidden_size + self.binding_bank_size + self.binding_valid_size
        )
        self.carry_size = self.state_size

        self.token_embedding = nn.Embedding(vocab_size, token_embed_size)
        self.token_rnn = nn.GRU(token_embed_size, token_embed_size)
        binding_read_size = (
            visible_slots * (episodic_binding_size + 1)
            if self.has_episodic_bindings
            else 0
        )
        if self.has_episodic_bindings:
            self.binding_value = nn.Linear(token_embed_size, episodic_binding_size)
            self.binding_read = nn.Linear(episodic_binding_size, episodic_binding_size)
        self.input = nn.Linear(
            vector_size + token_embed_size + binding_read_size,
            hidden_size,
        )
        self.input_norm = nn.LayerNorm(hidden_size)
        self.core = nn.GRU(hidden_size, hidden_size)
        self.post = nn.Linear(hidden_size, hidden_size)
        self.post_norm = nn.LayerNorm(hidden_size)

        self.policy = nn.Linear(self.state_size, action_size)
        self.value = nn.Linear(self.state_size, 1)
        if self.has_episodic_bindings:
            option_policy_input = (
                hidden_size
                + self.object_feature_size
                + episodic_binding_size
                + 1
                + self.object_option_types
            )
            self.binding_option_policy = nn.Linear(option_policy_input, hidden_size)
            self.binding_option_policy_out = nn.Linear(hidden_size, 1)

        # Every visible-slot option is the same abstract act applied to a
        # different perceived object. Sharing its action identity makes the
        # consequence model generalize across slots; selected raw object
        # features provide the binding target without revealing hidden kind.
        transition_action_size = (
            self.primitive_action_size + self.object_option_types
        )
        self.transition = nn.Linear(
            hidden_size
            + transition_action_size
            + self.object_feature_size
            + (episodic_binding_size + 1 if self.has_episodic_bindings else 0),
            hidden_size,
        )
        self.transition_norm = nn.LayerNorm(hidden_size)
        self.next_vector = nn.Linear(hidden_size, vector_size)
        self.next_needs = nn.Linear(hidden_size, 4)
        if self.has_split_drift_head:
            # A second, deliberately range-limited path for slow metabolism.
            # It can express any drift the world produces and cannot express a
            # consumption jump, so the dense metabolic objective has somewhere
            # to live that is structurally incapable of overwriting the sparse
            # binding-conditioned one.
            self.drift_needs = nn.Linear(hidden_size, 4)
        self.reward_head = nn.Linear(hidden_size, 1)
        self.next_tokens = nn.Linear(hidden_size, tokens_per_utterance * vocab_size)

        # The mouth. It emits its own tokens over the same closed vocabulary the
        # organism hears, from the same state that acts, values and predicts.
        # Nothing supervises it: the only gradient it ever receives is the
        # advantage of the life that followed what it said.
        if report_slots < 0:
            raise ValueError("report_slots must be nonnegative.")
        self.report_slots = report_slots
        self.can_speak = report_slots > 0
        if self.can_speak:
            self.report_head = nn.Linear(self.state_size, report_slots * vocab_size)

    def _encode_tokens(self, tokens: mx.array) -> mx.array:
        """(B, T, L) int token ids -> (B, T, E) utterance encodings."""

        batch, steps, length = tokens.shape
        embedded = self.token_embedding(tokens.reshape(batch * steps, length))
        states = self.token_rnn(embedded)
        return states[:, -1, :].reshape(batch, steps, -1)

    def _features(
        self,
        vectors: mx.array,
        token_encoding: mx.array,
        binding_reads: mx.array | None = None,
    ) -> mx.array:
        parts = [vectors, token_encoding]
        if binding_reads is not None:
            parts.append(binding_reads)
        x = mx.concatenate(parts, axis=-1)
        return nn.relu(self.input_norm(self.input(x)))

    def _empty_carry_parts(
        self, vectors: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        batch = vectors.shape[0]
        core = mx.zeros((batch, self.hidden_size), dtype=vectors.dtype)
        memory = mx.zeros(
            (batch, self.surface_count, self.episodic_binding_size),
            dtype=vectors.dtype,
        )
        valid = mx.zeros((batch, self.surface_count), dtype=vectors.dtype)
        return core, memory, valid

    def split_carry(
        self, hidden: mx.array | None, vectors: mx.array
    ) -> tuple[mx.array | None, mx.array, mx.array]:
        """Return recurrent, lexical-value, and validity parts of a carry."""

        if not self.has_episodic_bindings:
            empty = mx.zeros((vectors.shape[0], 0), dtype=vectors.dtype)
            return hidden, empty.reshape(vectors.shape[0], 0, 0), empty
        if hidden is None:
            return self._empty_carry_parts(vectors)
        core = hidden[:, : self.hidden_size]
        memory_start = self.hidden_size
        memory_stop = memory_start + self.binding_bank_size
        memory = hidden[:, memory_start:memory_stop].reshape(
            hidden.shape[0], self.surface_count, self.episodic_binding_size
        )
        valid = hidden[:, memory_stop : memory_stop + self.binding_valid_size]
        return core, memory, valid

    def _visible_object_features(self, vectors: mx.array) -> mx.array:
        start = self.object_feature_offset
        stop = start + self.visible_slots * self.object_feature_size
        return vectors[..., start:stop].reshape(
            *vectors.shape[:-1], self.visible_slots, self.object_feature_size
        )

    def _attended_surface(self, vectors: mx.array) -> mx.array:
        """One-hot surface exactly one learner-visible cell ahead, if any."""

        slots = self._visible_object_features(vectors)
        presence = slots[..., 0]
        offsets = slots[..., 1:3]
        surfaces = slots[..., 3:]
        direction = vectors[..., 4:8]
        relative_by_direction = mx.array(
            [
                [0.0, -1.0 / self.visible_radius],
                [1.0 / self.visible_radius, 0.0],
                [0.0, 1.0 / self.visible_radius],
                [-1.0 / self.visible_radius, 0.0],
            ],
            dtype=vectors.dtype,
        )
        ahead = direction @ relative_by_direction
        matches = (
            (mx.max(mx.abs(offsets - ahead[..., None, :]), axis=-1) < 1e-5)
            * (presence > 0.5)
        ).astype(vectors.dtype)
        return mx.sum(matches[..., None] * surfaces, axis=-2)

    def _update_bindings(
        self,
        vectors: mx.array,
        tokens: mx.array,
        token_encoding: mx.array,
        memory: mx.array,
        valid: mx.array,
    ) -> tuple[mx.array, mx.array]:
        surface = self._attended_surface(vectors)
        action_start = self.object_feature_offset - self.primitive_action_size
        last_action = vectors[
            ..., action_start : action_start + self.primitive_action_size
        ]
        referential = mx.sum(
            last_action[..., list(self.referential_action_indices)], axis=-1
        )
        surface = surface * (referential > 0.5)[..., None]
        return self._write_binding_at_surface(
            surface,
            tokens,
            token_encoding,
            memory,
            valid,
        )

    def _write_binding_at_surface(
        self,
        surface: mx.array,
        tokens: mx.array,
        token_encoding: mx.array,
        memory: mx.array,
        valid: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Write heard language to an explicit learner-visible surface key."""

        heard = mx.any(tokens != self.pad_token_id, axis=-1).astype(memory.dtype)
        write = surface * heard[..., None]
        if not self.episodic_binding_writes:
            write = mx.zeros_like(write)
        value = mx.tanh(self.binding_value(token_encoding))
        memory = (
            memory * (1.0 - write[..., None])
            + value[..., None, :] * write[..., None]
        )
        valid = mx.maximum(valid, write)
        return memory, valid

    def _binding_reads(
        self, vectors: mx.array, memory: mx.array, valid: mx.array
    ) -> mx.array:
        slots = self._visible_object_features(vectors)
        presence = slots[..., 0]
        surfaces = slots[..., 3:]
        values = mx.einsum("...vs,...sd->...vd", surfaces, memory)
        values = mx.tanh(self.binding_read(values))
        slot_valid = mx.einsum("...vs,...s->...v", surfaces, valid) * presence
        values = values * slot_valid[..., None]
        return mx.concatenate([values, slot_valid[..., None]], axis=-1).reshape(
            *vectors.shape[:-1], -1
        )

    def _state_memory_parts(
        self, states: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        core = states[..., : self.hidden_size]
        if not self.has_episodic_bindings:
            empty = states[..., :0]
            return core, empty.reshape(*states.shape[:-1], 0, 0), empty
        memory_start = self.hidden_size
        memory_stop = memory_start + self.binding_bank_size
        memory = states[..., memory_start:memory_stop].reshape(
            *states.shape[:-1], self.surface_count, self.episodic_binding_size
        )
        valid = states[..., memory_stop : memory_stop + self.binding_valid_size]
        return core, memory, valid

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
            hidden: (B, carry_size) recurrent plus optional binding carry.

        Returns:
            states (B, T, state_size) for the heads, and the recurrent plus
            optional external-memory carry (B, carry_size).
        """

        token_encoding = self._encode_tokens(tokens)
        if not self.has_episodic_bindings:
            features = self._features(vectors, token_encoding)
            states = self.core(features, hidden)
            refined = states + nn.relu(self.post_norm(self.post(states)))
            return refined, states[:, -1, :]

        core_hidden, memory, valid = self.split_carry(hidden, vectors)
        raw_states: list[mx.array] = []
        state_memories: list[mx.array] = []
        state_validities: list[mx.array] = []
        for step in range(vectors.shape[1]):
            step_vector = vectors[:, step, :]
            step_tokens = tokens[:, step, :]
            step_token_encoding = token_encoding[:, step, :]
            # Read-before-write gives the external bank a clean causal clock.
            # The current utterance still enters the GRU through its ordinary
            # token encoding, while a newly acquired binding first influences
            # recurrent processing on the next observation. This lets a
            # post-label bank intervention remove the entire delayed-memory
            # pathway without erasing the transient raw-label core.
            reads = self._binding_reads(step_vector, memory, valid)
            features = self._features(
                step_vector,
                step_token_encoding,
                reads,
            )[:, None, :]
            step_states = self.core(features, core_hidden)
            core_hidden = step_states[:, -1, :]
            memory, valid = self._update_bindings(
                step_vector,
                step_tokens,
                step_token_encoding,
                memory,
                valid,
            )
            raw_states.append(core_hidden)
            state_memories.append(memory.reshape(memory.shape[0], -1))
            state_validities.append(valid)
        raw = mx.stack(raw_states, axis=1)
        refined = raw + nn.relu(self.post_norm(self.post(raw)))
        memory_states = mx.stack(state_memories, axis=1)
        validity_states = mx.stack(state_validities, axis=1)
        states = mx.concatenate([refined, memory_states, validity_states], axis=-1)
        carry = mx.concatenate(
            [core_hidden, memory_states[:, -1], validity_states[:, -1]], axis=-1
        )
        return states, carry

    def write_binding_into_state(
        self,
        states: mx.array,
        surfaces: mx.array,
        tokens: mx.array,
    ) -> mx.array:
        """Apply a surface-keyed lexical write with no observation step.

        This is the counterfactual "suppose my memory held this word for this
        surface". It touches only the external bank, never the recurrent core,
        so a planner can consider a hypothetical word without fabricating a
        sensory scene for its own decoder to misreconstruct. The surface
        one-hot comes from the learner's own observation and never contains or
        addresses hidden object kind.
        """

        if not self.has_episodic_bindings:
            raise ValueError("Binding writes require episodic bindings.")
        leading_shape = states.shape[:-1]
        flat_states = states.reshape(-1, self.state_size)
        flat_surfaces = surfaces.reshape(-1, self.surface_count)
        flat_tokens = tokens.reshape(-1, self.tokens_per_utterance)
        token_encoding = self._encode_tokens(flat_tokens[:, None, :])[:, 0, :]
        core, memory, valid = self._state_memory_parts(flat_states)
        memory, valid = self._write_binding_at_surface(
            flat_surfaces,
            flat_tokens,
            token_encoding,
            memory,
            valid,
        )
        written = mx.concatenate(
            [core, memory.reshape(memory.shape[0], -1), valid],
            axis=-1,
        )
        return written.reshape(*leading_shape, self.state_size)

    def observe_from_state(
        self,
        states: mx.array,
        vectors: mx.array,
        tokens: mx.array,
        *,
        binding_surfaces: mx.array | None = None,
    ) -> mx.array:
        """Apply one hypothetical observation update from planning states.

        ``binding_surfaces`` is an optional learner-visible surface one-hot
        selected by an inspect option. It lets a belief-space planner apply the
        same learned token-to-binding write as embodied observation processing
        without reconstructing an exact motor-level label pose. It never
        contains or addresses hidden object kind.
        """

        leading_shape = states.shape[:-1]
        flat_states = states.reshape(-1, self.state_size)
        flat_vectors = vectors.reshape(-1, self.vector_size)
        flat_tokens = tokens.reshape(-1, self.tokens_per_utterance)
        token_encoding = self._encode_tokens(flat_tokens[:, None, :])[:, 0, :]

        if not self.has_episodic_bindings:
            if binding_surfaces is not None:
                raise ValueError(
                    "Explicit binding surfaces require episodic bindings."
                )
            features = self._features(flat_vectors, token_encoding)
            raw = self.core(
                features[:, None, :],
                flat_states[:, : self.hidden_size],
            )[:, -1, :]
            refined = raw + nn.relu(self.post_norm(self.post(raw)))
            return refined.reshape(*leading_shape, self.state_size)

        core, memory, valid = self._state_memory_parts(flat_states)
        reads = self._binding_reads(flat_vectors, memory, valid)
        features = self._features(flat_vectors, token_encoding, reads)
        raw = self.core(features[:, None, :], core)[:, -1, :]
        refined = raw + nn.relu(self.post_norm(self.post(raw)))

        if binding_surfaces is None:
            memory, valid = self._update_bindings(
                flat_vectors,
                flat_tokens,
                token_encoding,
                memory,
                valid,
            )
        else:
            flat_surfaces = binding_surfaces.reshape(-1, self.surface_count)
            memory, valid = self._write_binding_at_surface(
                flat_surfaces,
                flat_tokens,
                token_encoding,
                memory,
                valid,
            )
        observed = mx.concatenate(
            [refined, memory.reshape(memory.shape[0], -1), valid],
            axis=-1,
        )
        return observed.reshape(*leading_shape, self.state_size)

    def policy_value(
        self,
        vectors: mx.array,
        tokens: mx.array,
        hidden: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        states, carry = self.core_states(vectors, tokens, hidden)
        logits = self.policy_logits(states, vectors)
        values = self.value(states).squeeze(-1)
        return logits, values, carry

    def policy_logits(self, states: mx.array, vectors: mx.array) -> mx.array:
        """Policy logits with object-local episodic reads when configured."""

        base = self.policy(states)
        if not self.has_episodic_bindings or not self.has_object_options:
            return base
        core, memory, valid = self._state_memory_parts(states)
        slots = self._visible_object_features(vectors)
        surfaces = slots[..., 3:]
        selected_values = mx.einsum("...vs,...sd->...vd", surfaces, memory)
        selected_valid = mx.einsum("...vs,...s->...v", surfaces, valid)
        option_scores: list[mx.array] = []
        for option_type in range(self.object_option_types):
            option_onehot = mx.broadcast_to(
                mx.eye(self.object_option_types, dtype=states.dtype)[option_type],
                (*states.shape[:-1], self.visible_slots, self.object_option_types),
            )
            expanded_core = mx.broadcast_to(
                core[..., None, :],
                (*core.shape[:-1], self.visible_slots, self.hidden_size),
            )
            features = mx.concatenate(
                [
                    expanded_core,
                    slots,
                    selected_values,
                    selected_valid[..., None],
                    option_onehot,
                ],
                axis=-1,
            )
            score = self.binding_option_policy_out(
                nn.relu(self.binding_option_policy(features))
            )[..., 0]
            option_scores.append(score)
        return mx.concatenate(
            [base[..., : self.primitive_action_size], *option_scores], axis=-1
        )

    def report_logits(self, states: mx.array) -> mx.array:
        """Per-slot logits over the whole vocabulary: (..., slots, vocab)."""

        if not self.can_speak:
            raise ValueError("This organism has no report head.")
        logits = self.report_head(states)
        return logits.reshape(*states.shape[:-1], self.report_slots, self.vocab_size)

    def _transition_features(
        self,
        states: mx.array,
        actions: mx.array,
        vectors: mx.array | None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        transition_action_size = (
            self.primitive_action_size + self.object_option_types
        )
        option_offsets = actions - self.primitive_action_size
        option_types = mx.clip(
            option_offsets // max(1, self.visible_slots),
            0,
            max(0, self.object_option_types - 1),
        )
        transition_actions = mx.where(
            actions < self.primitive_action_size,
            actions,
            self.primitive_action_size + option_types,
        )
        action_features = mx.eye(transition_action_size)[transition_actions]
        if not self.has_object_options:
            empty = mx.zeros((*actions.shape, 0))
            return action_features, empty, empty
        if vectors is None:
            raise ValueError("Object-option consequences require observation vectors.")
        start = self.object_feature_offset
        stop = start + self.visible_slots * self.object_feature_size
        slots = vectors[..., start:stop].reshape(
            *vectors.shape[:-1], self.visible_slots, self.object_feature_size
        )
        slot_indices = mx.clip(
            option_offsets % self.visible_slots,
            0,
            self.visible_slots - 1,
        )
        indices = mx.broadcast_to(
            slot_indices[..., None, None],
            (*slot_indices.shape, 1, self.object_feature_size),
        )
        selected = mx.take_along_axis(slots, indices, axis=-2).squeeze(-2)
        is_option = (actions >= self.primitive_action_size)[..., None]
        selected = mx.where(is_option, selected, mx.zeros_like(selected))
        if not self.has_episodic_bindings:
            return action_features, selected, mx.zeros((*actions.shape, 0))
        _, memory, valid = self._state_memory_parts(states)
        surfaces = selected[..., 3:]
        binding = mx.einsum("...s,...sd->...d", surfaces, memory)
        binding_valid = mx.einsum("...s,...s->...", surfaces, valid)[..., None]
        binding_features = mx.concatenate(
            [binding * binding_valid, binding_valid], axis=-1
        )
        return action_features, selected, binding_features

    def predict_consequences(
        self,
        states: mx.array,
        actions: mx.array,
        vectors: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Action-conditioned predictions from core states.

        Args:
            states: (B, T, state_size) refined core and optional bindings.
            actions: (B, T) int action indices taken at those states.
            vectors: (B, T, vector_size) observations used to bind an object
                option to its selected learner-visible slot.

        Returns:
            next observation vectors (B, T, vector_size),
            changes in the four bodily needs (B, T, 4),
            reward deltas (B, T),
            caregiver-token logits (B, T, tokens_per_utterance, vocab_size).
        """

        h = self.transition_state(states, actions, vectors)
        return self.decode_transition(h)

    def transition_state(
        self,
        states: mx.array,
        actions: mx.array,
        vectors: mx.array | None = None,
    ) -> mx.array:
        """Advance the learned latent dynamics by one agent decision."""

        action_features, object_features, binding_features = self._transition_features(
            states, actions, vectors
        )
        core_states = states[..., : self.hidden_size]
        x = mx.concatenate(
            [core_states, action_features, object_features, binding_features],
            axis=-1,
        )
        core = nn.relu(self.transition_norm(self.transition(x)))
        if not self.has_episodic_bindings:
            return core
        return mx.concatenate([core, states[..., self.hidden_size :]], axis=-1)

    def decode_transition(
        self, transition_states: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Decode observable and bodily consequences from imagined states."""

        h = transition_states[..., : self.hidden_size]
        next_vectors = self.next_vector(h)
        # Most actions only incur metabolism, while the rare consequential
        # actions change one need sharply. Predicting a bounded residual keeps
        # the current sensed body on an exact identity path instead of asking
        # the model to reconstruct its absolute state from a latent vector.
        need_deltas = 0.5 * mx.tanh(self.next_needs(h))
        if self.has_split_drift_head:
            # The drift path is bounded by the measured gap between metabolism
            # and consumption, so the two regimes cannot contend for the same
            # saturating output range.
            need_deltas = need_deltas + DRIFT_REGIME_THRESHOLD * mx.tanh(
                self.drift_needs(h)
            )
        rewards = self.reward_head(h).squeeze(-1)
        token_logits = self.next_tokens(h).reshape(
            *transition_states.shape[:-1],
            self.tokens_per_utterance,
            self.vocab_size,
        )
        return next_vectors, need_deltas, rewards, token_logits
