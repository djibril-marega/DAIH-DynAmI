from dataclasses import dataclass
from typing import Optional
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class PolicyOutput:
    policy: torch.Tensor
    temporal_mask: torch.Tensor
    G: torch.Tensor
    logits: Optional[torch.Tensor] = None
    step_scores: Optional[torch.Tensor] = None


class PolicyGenerator(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
        vocab_size: int,
        gumbel_tau: float = 1.0,
    ):
        super().__init__()

        if input_size <= 0 or hidden_dim <= 0 or vocab_size <= 0:
            raise ValueError("Dimensions must be positive.")

        self.vocab_size = int(vocab_size)
        self.eos_index = self.vocab_size
        self.gumbel_tau = float(gumbel_tau)

        # --------------------------------------------------
        # 1. Classical policy
        # --------------------------------------------------

        self.initial_hidden = nn.Linear(input_size, hidden_dim)

        self.rnn_cell = nn.GRUCell(
            input_size,
            hidden_dim,
        )

        self.output_layer = nn.Linear(
            hidden_dim,
            vocab_size + 1,
        )

        # --------------------------------------------------
        # 2. Tau encoding
        # --------------------------------------------------

        self.tau_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # FiLM modulation of the internal state.
        self.tau_film = nn.Linear(
            hidden_dim,
            hidden_dim * 2,
        )

        # Initially, tau does not affect the policy.
        nn.init.zeros_(self.tau_film.weight)
        nn.init.zeros_(self.tau_film.bias)

        # --------------------------------------------------
        # 3. Heuristic exploration
        # --------------------------------------------------

        self.exploration_layer = nn.Linear(
            hidden_dim,
            vocab_size + 1,
        )

        self.exploration_gate = nn.Sequential(
            nn.Linear(
                hidden_dim,
                max(1, hidden_dim // 2),
            ),
            nn.Tanh(),
            nn.Linear(
                max(1, hidden_dim // 2),
                1,
            ),
            nn.Sigmoid(),
        )

        # Initially, the exploratory head is neutral.
        nn.init.zeros_(self.exploration_layer.weight)
        nn.init.zeros_(self.exploration_layer.bias)

        # --------------------------------------------------
        # 4. Score and feedback
        # --------------------------------------------------

        self.score_layer = nn.Linear(hidden_dim, 1)

        self.feedback_layer = nn.Linear(
            vocab_size + 1,
            input_size,
        )

    def forward(
        self,
        context: torch.Tensor,
        max_steps: int = 20,
        *,
        temperature: Optional[float] = None,
        deterministic: Optional[bool] = None,
    ) -> PolicyOutput:

        if context.dim() == 1:
            context = context.unsqueeze(0)

        if context.dim() != 2:
            raise ValueError(
                "context must be of shape [B, input_size]."
            )

        if context.shape[-1] != self.initial_hidden.in_features:
            raise ValueError("Invalid context dimension.")

        if max_steps < 1:
            raise ValueError("max_steps must be >= 1.")

        if deterministic is None:
            deterministic = not self.training

        tau = (
            self.gumbel_tau
            if temperature is None
            else float(temperature)
        )

        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be positive and finite.")

        batch_size = context.shape[0]

        # --------------------------------------------------
        # 1. Tau encoding
        # --------------------------------------------------

        tau_input = context.new_full(
            (batch_size, 1),
            math.log(tau),
        )

        tau_embedding = self.tau_encoder(tau_input)

        # FiLM parameters.
        gamma, beta = self.tau_film(
            tau_embedding
        ).chunk(2, dim=-1)

        gamma = torch.tanh(gamma)

        # Heuristic exploration weight.
        exploration_weight = self.exploration_gate(
            tau_embedding
        )

        # --------------------------------------------------
        # 2. Recurrent initialization
        # --------------------------------------------------

        hidden = torch.tanh(
            self.initial_hidden(context)
        )

        current_input = context

        active = torch.ones(
            batch_size,
            dtype=torch.bool,
            device=context.device,
        )

        policies = []
        masks = []
        logits_history = []
        step_scores = []

        # --------------------------------------------------
        # 3. Autoregressive generation
        # --------------------------------------------------

        for step in range(max_steps):

            # Classical recurrent state.
            proposed_hidden = self.rnn_cell(
                current_input,
                hidden,
            )

            hidden = torch.where(
                active[:, None],
                proposed_hidden,
                hidden,
            )

            # --------------------------------------------------
            # 4. Tau awareness: FiLM modulation
            # --------------------------------------------------

            conditioned_hidden = (
                hidden * (1.0 + gamma) + beta
            )

            # --------------------------------------------------
            # 5. Classical policy
            # --------------------------------------------------

            policy_logits = self.output_layer(
                conditioned_hidden
            )

            # --------------------------------------------------
            # 6. Heuristic exploration
            # --------------------------------------------------

            exploration_logits = self.exploration_layer(
                conditioned_hidden
            )

            logits = (
                policy_logits
                + exploration_weight * exploration_logits
            )

            # EOS forbidden at the first step.
            if step == 0:
                logits = logits.clone()
                logits[:, self.eos_index] = float("-inf")

            # --------------------------------------------------
            # 7. Stochastic Gumbel exploration
            # --------------------------------------------------

            if deterministic:

                indices = logits.argmax(dim=-1)

                choice = F.one_hot(
                    indices,
                    num_classes=self.vocab_size + 1,
                ).to(context.dtype)

            else:

                # tau acts on the categorical distribution.
                choice = F.gumbel_softmax(
                    logits / tau,
                    tau=1.0,
                    hard=True,
                    dim=-1,
                )

            # --------------------------------------------------
            # 8. EOS handling
            # --------------------------------------------------

            if step == 0:

                eos_selected = torch.zeros(
                    batch_size,
                    dtype=torch.bool,
                    device=context.device,
                )

            else:

                eos_selected = (
                    choice[:, self.eos_index] > 0.5
                )

            step_active = active & (~eos_selected)

            action_vector = choice[:, :self.vocab_size]

            action_vector = (
                action_vector
                * step_active[:, None].to(context.dtype)
            )

            if step == 0:
                # Comblement de la fin tronquée avec la suite logique classique :
                if not bool(step_active.any()):
                    break

            # --------------------------------------------------
            # 9. Output tracking & Feedback loop
            # --------------------------------------------------
            
            # Predict the score for the current step
            score = self.score_layer(conditioned_hidden)
            step_scores.append(score)

            policies.append(action_vector.unsqueeze(1))
            masks.append(step_active.unsqueeze(1))
            logits_history.append(logits.unsqueeze(1))

            # Update active status for the next steps
            active = step_active

            # If all sequences are completed, stop early
            if not active.any():
                break

            # Project choice back to input space for the next GRU step
            current_input = self.feedback_layer(choice)

        # --------------------------------------------------
        # 10. Packaging outputs
        # --------------------------------------------------
        
        final_policy = torch.cat(policies, dim=1)
        final_mask = torch.cat(masks, dim=1)
        final_logits = torch.cat(logits_history, dim=1)
        final_scores = torch.cat(step_scores, dim=1)

        # Compute G (expected total return or target)
        total_return = final_scores.sum()

        return PolicyOutput(
            policy=final_policy,
            temporal_mask=final_mask,
            G=total_return,
            logits=final_logits,
            step_scores=final_scores,
        )
