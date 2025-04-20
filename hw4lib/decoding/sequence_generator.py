import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Callable
from ..data import H4Tokenizer

'''
TODO: Implement the `generate_greedy` and optionally the `generate_beam` methods of the `SequenceGenerator` class.

This file implements text generation strategies for transformer language models:

1. Greedy Search: Always selects the most likely next token
   - Simple but can lead to repetitive or suboptimal outputs
   - Useful for deterministic generation

2. Beam Search: Maintains top-k most likely sequences at each step
   - Explores multiple possible sequences in parallel
   - Often produces higher quality outputs than greedy search
   - More computationally intensive

3. Sampling with Filtering: Uses probabilistic sampling with constraints
   - Temperature: Controls randomness of sampling
   - Top-k: Limits sampling to k most likely tokens
   - Top-p (nucleus): Samples from minimal set of tokens comprising p probability mass
   - Useful for creative and diverse generation

Implementation Notes:
1. Helper Methods:
   - _apply_repeat_penalty: Penalizes repeated tokens
   - _filter_logits: Applies temperature and filtering
   - post_process_sequence: Handles EOS token truncation

2. Generation Methods:
   - generate_greedy: Implements basic greedy decoding
   - generate_beam: Implements beam search
   - generate_sample: Implements filtered sampling

3. Each generation method should:
   - Handle proper input validation
   - Track sequence scores
   - Handle EOS token detection
   - Support early stopping
'''

class SequenceGenerator:
    """
    A class for generating sequences using various decoding strategies.
    Supports greedy search, beam search, and sampling with top-k/nucleus filtering.
    """
    def __init__(
            self,
            score_fn: Callable,
            tokenizer: H4Tokenizer,
            max_length: int,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the sequence generator.
        
        Args:
            score_fn: Function that returns logits for next token prediction
            tokenizer: Tokenizer instance for handling token conversions
            max_length: Maximum sequence length to generate
            device: Device to run generation on
        """
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def _apply_repeat_penalty(
            self,
            logits: torch.Tensor,
            sequences: torch.Tensor,
            penalty: float = 1.0
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits based on tokens in sequences.
        Args:
            logits: Logits tensor of shape (batch_size, vocab_size) or (batch_size, beam_width, vocab_size)
            sequences: Sequences tensor of shape (batch_size, sequence_length) or (batch_size, beam_width, sequence_length)
            penalty: Repetition penalty value
        Returns:
            Logits tensor with repetition penalty applied
        """
        if penalty == 1.0:
            return logits
        
        # Handle both regular and beam search shapes
        if logits.dim() == 2:
            # Greedy search: (batch_size, vocab_size)
            for idx in range(sequences.size(0)):
                unique_tokens = torch.unique(sequences[idx])
                logits[idx, unique_tokens] = logits[idx, unique_tokens] / torch.where(
                    logits[idx, unique_tokens] > 0,
                    torch.full_like(logits[idx, unique_tokens], penalty),
                    torch.full_like(logits[idx, unique_tokens], 1.0/penalty)
                )
        else:
            # Beam search: (batch_size, beam_width, vocab_size)
            for batch_idx in range(sequences.size(0)):
                for beam_idx in range(sequences.size(1)):
                    unique_tokens = torch.unique(sequences[batch_idx, beam_idx])
                    logits[batch_idx, beam_idx, unique_tokens] = logits[batch_idx, beam_idx, unique_tokens] / torch.where(
                        logits[batch_idx, beam_idx, unique_tokens] > 0,
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], penalty),
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], 1.0/penalty)
                    )
        
        return logits

    def _filter_logits(
            self,
            logits: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> torch.Tensor:
        """Apply temperature, top-k, and top-p filtering to logits."""
        logits = logits / temperature

        if top_k > 0:
            top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            indices_to_remove = logits < top_k_logits[..., -1:]
            logits[indices_to_remove] = float('-inf')

        if top_p < 1.0:
            log_probs = torch.log_softmax(logits, dim=-1)
            sorted_log_probs, sorted_indices = torch.sort(log_probs, descending=True)
            cumulative_probs = torch.cumsum(torch.exp(sorted_log_probs), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        return logits

    def generate_greedy(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using greedy search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        # Add input validation

        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        
        # TODO: Implement greedy search
        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        current_seq = x.clone()  # We'll append new tokens to this
        initial_len = x.size(1)
        for _ in range(self.max_length - initial_len):
            # Stop if all sequences have already reached EOS
            if finished.all():
                break
            
            # (a) Obtain logits for next token (shape: (B, vocab_size))
            next_logits = self.score_fn(current_seq)

            # (b) Optionally apply repetition penalty
            if repeat_penalty != 1.0:
                next_logits = self._apply_repeat_penalty(next_logits, current_seq, penalty=repeat_penalty)

            # (c) Scale logits by temperature
            # NOTE: temperature should be > 0.0
            scaled_logits = next_logits / temperature

            # (d) Convert to log-probs
            log_probs = torch.log_softmax(scaled_logits, dim=-1)

            # (e) Greedy selection: argmax over the vocabulary
            next_tokens = torch.argmax(log_probs, dim=-1)  # shape: (B,)

            # (f) Extract token log-probs, update scores for sequences still active
            chosen_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)  # (B,)
            scores = torch.where(finished, scores, scores + chosen_scores)

            # (g) Append next tokens to each sequence
            current_seq = torch.cat([current_seq, next_tokens.unsqueeze(1)], dim=1)

            # (h) Mark sequences as finished if next token is EOS
            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos

        return current_seq, scores # Remove once implemented

    # def generate_beam(
    #         self,
    #         x: torch.Tensor,
    #         beam_width: int,
    #         temperature: float = 1.0,
    #         repeat_penalty: float = 1.0
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Generate sequences using beam search.
    #     Args:
    #         x: Input tensor of shape (batch_size, sequence_length)
    #         beam_width: Number of beams to use
    #         temperature: Temperature for logits scaling
    #         repeat_penalty: Penalty for repeated tokens
    #     Returns:
    #         Tuple of tensors: (sequences, scores)
    #          - sequences is of shape (batch_size, beam_width, sequence_length) where each sequence in a beam set is sorted by score
    #          - scores is of shape (batch_size, beam_width)
    #     """
    #     # Add input validation
    #     if not torch.is_tensor(x):
    #         raise TypeError("Input x must be a torch tensor")
    #     if x.dim() != 2:
    #         raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
    #     if beam_width < 1:
    #         raise ValueError("beam_width must be >= 1")
    #     if self.max_length < x.size(1):
    #         raise ValueError("max_length must be >= input sequence length")
        
    #     # TODO: Implement beam search
    #     B, cur_len = x.size()
    #     V = self.tokenizer.vocab_size  # or next_logits.size(-1) after score_fn
    #     device = x.device

    #     # (2) Initial scoring
    #     # logits: (B, V)
    #     logits = self.score_fn(x)
    #     if repeat_penalty != 1.0:
    #         logits = self._apply_repeat_penalty(logits, x, penalty=repeat_penalty)
    #     logits = logits / temperature
    #     log_probs = torch.log_softmax(logits, dim=-1)  # (B, V)

    #     # select top‐k initial tokens
    #     topk_scores, topk_tokens = log_probs.topk(beam_width, dim=-1)  # both (B, beam_width)

    #     # initialize beam sequences (B, beam, cur_len+1)
    #     # expand x to (B, beam, cur_len)
    #     x_beams = x.unsqueeze(1).expand(B, beam_width, cur_len).clone()
    #     # append the first tokens
    #     x_beams = torch.cat([x_beams, topk_tokens.unsqueeze(-1)], dim=-1)
    #     scores = topk_scores  # (B, beam_width)
    #     finished = topk_tokens == self.tokenizer.eos_id

    #     cur_len += 1

    #     # (3) Iterative expansion
    #     for _ in range(cur_len, self.max_length):
    #         if finished.all():
    #             break

    #         # for each beam: compute next‐step logits
    #         flat_beams = x_beams.reshape(B * beam_width, cur_len)
    #         next_logits = self.score_fn(flat_beams)  # (B*beam, V)
    #         next_logits = next_logits.view(B, beam_width, V)

    #         if repeat_penalty != 1.0:
    #             next_logits = self._apply_repeat_penalty(next_logits, x_beams, penalty=repeat_penalty)

    #         scaled = next_logits / temperature
    #         next_log_probs = torch.log_softmax(scaled, dim=-1)  # (B, beam, V)

    #         # accumulate scores: (B, beam, V)
    #         total_scores = scores.unsqueeze(-1) + next_log_probs

    #         # flatten beams: (B, beam*V)
    #         flat_scores, flat_indices = total_scores.view(B, -1).topk(beam_width, dim=-1)

    #         # decode beam & token indices
    #         beam_idx   = flat_indices // V       # (B, beam)
    #         token_idx  = flat_indices % V        # (B, beam)

    #         # gather corresponding sequences
    #         idx_expand = beam_idx.unsqueeze(-1).expand(-1, -1, cur_len)
    #         x_prev     = torch.gather(x_beams, 1, idx_expand)  # (B, beam, cur_len)

    #         # append new tokens
    #         x_beams    = torch.cat([x_prev, token_idx.unsqueeze(-1)], dim=-1)  # (B, beam, cur_len+1)
    #         scores     = flat_scores
    #         cur_len   += 1

    #         # update finished flags
    #         eos_hits = token_idx == self.tokenizer.eos_id
    #         finished = finished | eos_hits

    #     return x_beams,scores # Remove once implemented
    def generate_beam(
        self,
        x: torch.Tensor,
        beam_width: int,
        temperature: float = 1.0,
        repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Beam search decoding as described in the HW pseudocode.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            beam_width: Number of beams to keep per batch
            temperature: Temperature for logits scaling
            repeat_penalty: Repetition penalty factor
        
        Returns:
            A tuple (sorted_sequences, sorted_scores) where:
            - sorted_sequences has shape (batch_size, beam_width, max_length)
            - sorted_scores has shape (batch_size, beam_width)
        """
        print(">>> BEAM: called with x.shape=", x.shape, "max_length=", self.max_length, flush=True)
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")

        batch_size, seq_len = x.shape
        device = x.device
        eos_token_id = getattr(self, "eos_token_id", None)

        # Expand the initial input to have a beam dimension.
        # Initial shape: (batch_size, seq_len) -> (batch_size, beam_width, seq_len)
        x = x.unsqueeze(1).expand(-1, beam_width, -1).contiguous()

        # Create a tensor to hold the full generated sequences:
        # shape: (batch_size, beam_width, max_length)
        sequences = torch.full(
            (batch_size, beam_width, self.max_length),
            fill_value=self.tokenizer.pad_id,
            dtype=torch.long,
            device=device
        )
        sequences[:, :, :seq_len] = x

        # Initialize beam scores and a finished mask.
        beam_scores = torch.zeros(batch_size, beam_width, device=device)
        finished = torch.zeros(batch_size, beam_width, dtype=torch.bool, device=device)

        # --------------------------------------------------------------------------
        # --- Initial Step: Extend the starting sequence one token ---
        # Since all beams start with the same input sequence, we take the first beam.
        logits = self.score_fn(x[:, 0, :])  # (batch_size, vocab_size)
        logits = self._apply_repeat_penalty(logits, x[:, 0, :], repeat_penalty)
        logits = logits / temperature
        log_probs = torch.log_softmax(logits, dim=-1)

        # Select the top beam_width tokens for each batch element.
        topk_log_probs, topk_tokens = torch.topk(log_probs, beam_width, dim=-1)  # (batch_size, beam_width)
        beam_scores = topk_log_probs.clone()

        # Append these tokens as the next token in each beam.
        sequences[:, :, seq_len] = topk_tokens

        # Mark beams as finished if the selected token is EOS.
        if eos_token_id is not None:
            finished |= topk_tokens.eq(eos_token_id)

        current_length = seq_len + 1

        # --------------------------------------------------------------------------
        # --- Subsequent Steps: Extend sequences token-by-token ---
        while current_length < self.max_length:
            if finished.all():
                break

            # For each beam, we will compute the next-token log-probabilities.
            # Instead of flattening, we process each beam separately over the entire batch.
            next_log_probs_list = []
            for k in range(beam_width):
                # Extract the current sequence for beam k from all batch items.
                # Shape: (batch_size, current_length)
                seqs_k = sequences[:, k, :current_length]
                # Call score_fn on these sequences.
                logits_k = self.score_fn(seqs_k)  # (batch_size, vocab_size)
                logits_k = self._apply_repeat_penalty(logits_k, seqs_k, repeat_penalty)
                logits_k = logits_k / temperature
                log_probs_k = torch.log_softmax(logits_k, dim=-1)  # (batch_size, vocab_size)

                # For beams that are finished, force them to only generate EOS.
                if eos_token_id is not None:
                    mask = finished[:, k]  # (batch_size,)
                    if mask.any():
                        log_probs_k[mask, :] = float('-inf')
                        log_probs_k[mask, eos_token_id] = 0.0
                next_log_probs_list.append(log_probs_k.unsqueeze(1))  # (batch_size, 1, vocab_size)

            # Stack the results along the beam dimension: (batch_size, beam_width, vocab_size)
            next_log_probs = torch.cat(next_log_probs_list, dim=1)

            # Compute cumulative candidate scores by adding the previous beam scores.
            # beam_scores: (batch_size, beam_width) -> unsqueeze to (batch_size, beam_width, 1)
            cum_scores = beam_scores.unsqueeze(-1) + next_log_probs  # (batch_size, beam_width, vocab_size)

            # Flatten candidates to (batch_size, beam_width * vocab_size)
            flat_scores = cum_scores.view(batch_size, -1)

            # Select top beam_width candidates for each batch.
            topk_flat_scores, topk_indices = torch.topk(flat_scores, beam_width, dim=-1)

            vocab_size = next_log_probs.size(-1)
            # Decode the flat indices into beam index and token index.
            new_beam_indices = topk_indices // vocab_size
            new_token_indices = topk_indices % vocab_size

            # Prepare batch indices for advanced indexing.
            batch_indices = torch.arange(batch_size, device=device).unsqueeze(-1).expand(-1, beam_width)
            # Reorder sequences according to the newly chosen beam indices.
            sequences = sequences[batch_indices, new_beam_indices, :]
            # Append the new token at the current length position.
            sequences[:, :, current_length] = new_token_indices

            # Update the beam scores.
            beam_scores = topk_flat_scores

            # Update finished flags by gathering the previous finished mask and checking for EOS.
            if eos_token_id is not None:
                new_finished = new_token_indices.eq(eos_token_id)
                finished = finished[batch_indices, new_beam_indices] | new_finished
            else:
                finished = finished[batch_indices, new_beam_indices]

            current_length += 1

        # --------------------------------------------------------------------------
        # --- Final Reordering: Sort beams in descending order by score ---
        sorted_scores, sorted_indices = torch.sort(beam_scores, descending=True, dim=-1)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(-1)
        sorted_sequences = sequences[batch_indices, sorted_indices, :]

        return sorted_sequences, sorted_scores
    def generate_sample(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using sampling with top-k and nucleus filtering.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            top_k: Number of top-k tokens to sample from
            top_p: Proportion of top-p tokens to sample from
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < top_p <= 1.0:
            raise ValueError("top_p must be > 0 and <= 1.0")
        
        # Initialize scores and finished flag
        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            # Check if all sequences have finished
            if finished.all():
                break

            # Get logits and apply filtering
            next_scores = self.score_fn(x) # (batch_size, vocab_size)
            filtered_logits = self._filter_logits(next_scores, temperature, top_k, top_p)
            log_probs = torch.log_softmax(filtered_logits, dim=-1)
            
            # We need probabilities for multinomial sampling
            probs = torch.exp(log_probs)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1) # (batch_size,)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1) # (batch_size,)

            # Update scores only for unfinished sequences
            scores = torch.where(finished, scores, scores + token_scores)

            # Append next tokens
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1) # (batch_size, seq_len + 1)

            # Check if any sequence has reached EOS 
            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos

        return x, scores

    @staticmethod
    def post_process_sequence(seq: torch.Tensor, tokenizer: H4Tokenizer) -> torch.Tensor:
        """
        Post process sequences to remove content after EOS token.
        Args:
            seq: Input tensor of shape (batch_size, sequence_length) or (sequence_length)
            tokenizer: Tokenizer instance for handling token conversions
        Returns:
            if seq is a single sequence, return a tensor of same shape with sequence truncated at EOS
            if seq is a batch of sequences, return a list of tensors with each sequence truncated at first EOS
        """
        # Handle single sequence case
        if seq.dim() == 1:
            eos_indices = (seq == tokenizer.eos_id).nonzero()
            if len(eos_indices) > 0:
                end_idx = eos_indices[0].item() + 1
                return seq[:end_idx]
            return seq
        
        # Handle batched sequences
        eos_mask = seq == tokenizer.eos_id  # (batch_size, sequence_length)
        # Find first EOS token in each sequence
        eos_indices = eos_mask.float().cumsum(dim=1).eq(1) & eos_mask
        # Create sequence mask that includes everything up to and including first EOS
        seq_mask = eos_indices.cumsum(dim=1).eq(0) | eos_indices
        # Apply mask and pack sequences
        return [s[:m.sum()] for s, m in zip(seq, seq_mask)]