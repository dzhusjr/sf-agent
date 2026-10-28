r"""RecurrentPPO + auxiliary "threat prediction" head (the project's research contribution).

Idea: an LSTM agent that is *also* trained to predict whether it is about to be hit
should learn a representation of the opponent's intent (wind-ups, approach patterns),
which may improve win rate and sample efficiency.

Label (free, no opponent action needed -- works vs. the built-in CPU):
    y_t = 1[ the agent loses health during steps t, t+1, ..., t+k-1 ]
computed from info["hp_lost"] recorded during the rollout; windows that run past the
end of the rollout without a hit or episode end are masked out.

Training follows Phasic Policy Gradient (Cobbe et al., 2021): after the normal PPO
update we run an *auxiliary phase* on the same rollout:
    L_aux = BCE( sigma(w^T h_t), y_t ) + beta_clone * KL( pi_old(.|s_t) || pi(.|s_t) )
where h_t is the actor-LSTM output. The KL term keeps the policy from drifting while
the shared LSTM is shaped by the auxiliary loss. Setting aux.coef = 0 recovers plain
RecurrentPPO (useful as a control with identical code paths).
"""
from __future__ import annotations

import numpy as np
import torch as th
import torch.nn.functional as F
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList


class _HitRecorder(BaseCallback):
    """Records info['hp_lost'] and dones for every (step, env) of the current rollout."""

    def __init__(self, n_steps: int, n_envs: int):
        super().__init__()
        self.hit = np.zeros((n_steps, n_envs), np.float32)
        self.done = np.zeros((n_steps, n_envs), np.float32)

    def _on_rollout_start(self) -> None:
        self.hit[:] = 0
        self.done[:] = 0

    def _on_step(self) -> bool:
        pos = self.model.rollout_buffer.pos          # index of the step being added
        infos, dones = self.locals["infos"], self.locals["dones"]
        for e, info in enumerate(infos):
            self.hit[pos, e] = float(info.get("hp_lost", False))
        self.done[pos] = dones.astype(np.float32)
        return True


def threat_labels(hit: np.ndarray, done: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """y[t,e] = any hit in [t, t+k) before the episode ends; mask = label is certain."""
    T, E = hit.shape
    y = np.zeros((T, E), np.float32)
    mask = np.zeros((T, E), np.float32)
    for e in range(E):
        for t in range(T):
            label, certain = 0.0, False
            for j in range(t, min(t + k, T)):
                if hit[j, e] > 0:
                    label, certain = 1.0, True
                    break
                if done[j, e] > 0:
                    certain = True
                    break
            else:
                certain = certain or (t + k <= T)
            y[t, e], mask[t, e] = label, float(certain)
    return y, mask


class AuxRecurrentPPO(RecurrentPPO):
    def __init__(self, *args, aux_k: int = 8, aux_coef: float = 1.0, aux_epochs: int = 2,
                 aux_seq_len: int = 32, aux_batch_seqs: int = 64, aux_lr: float = 3e-4,
                 aux_clone_coef: float = 1.0, **kwargs):
        self.aux_k, self.aux_coef, self.aux_epochs = aux_k, aux_coef, aux_epochs
        self.aux_seq_len, self.aux_batch_seqs = aux_seq_len, aux_batch_seqs
        self.aux_lr, self.aux_clone_coef = aux_lr, aux_clone_coef
        super().__init__(*args, **kwargs)

    # ----------------------------------------------------------------- setup
    def _setup_model(self) -> None:
        super()._setup_model()
        hidden = self.policy.lstm_actor.hidden_size
        # registered on the policy -> saved/loaded with the model .zip
        self.policy.aux_head = th.nn.Linear(hidden, 1).to(self.device)
        self.aux_optimizer = th.optim.Adam(self.policy.parameters(), lr=self.aux_lr)
        self._recorder = _HitRecorder(self.n_steps, self.n_envs)

    def _excluded_save_params(self) -> list[str]:
        # recreated by _setup_model() on load; aux_head weights live in the policy state_dict
        return super()._excluded_save_params() + ["aux_optimizer", "_recorder"]

    def _init_callback(self, callback, progress_bar: bool = False):
        cb = super()._init_callback(callback, progress_bar)
        combined = CallbackList([self._recorder, cb])
        combined.init_callback(self)
        return combined

    # ----------------------------------------------------------------- train
    def train(self) -> None:
        buf = self.rollout_buffer
        # snapshot the (n_steps, n_envs, ...) rollout BEFORE super().train() flattens it
        obs = {k: v.copy() for k, v in buf.observations.items()}
        starts = buf.episode_starts.copy()
        h0, c0 = buf.hidden_states_pi.copy(), buf.cell_states_pi.copy()
        super().train()
        if self.aux_coef > 0:
            self._aux_phase(obs, starts, h0, c0)

    def _sequences(self, x: np.ndarray, L: int) -> np.ndarray:
        """(T, E, ...) -> (n_seq, L, ...) chunks per env (T must be divisible by L)."""
        T, E = x.shape[:2]
        x = np.swapaxes(x, 0, 1)                        # (E, T, ...)
        return x.reshape(E * (T // L), L, *x.shape[2:])

    def _aux_phase(self, obs, starts, h0, c0) -> None:
        T, E = starts.shape
        L = self.aux_seq_len if T % self.aux_seq_len == 0 else T
        y, mask = threat_labels(self._recorder.hit, self._recorder.done, self.aux_k)

        seq_obs = {k: self._sequences(v, L) for k, v in obs.items()}
        seq_starts = self._sequences(starts, L)
        seq_y, seq_mask = self._sequences(y, L), self._sequences(mask, L)
        # LSTM state at the start of each chunk: buffer stores (T, layers, E, H)
        # (n_chunks, layers, E, H) -> (E, n_chunks, layers, H) -> env-major like _sequences
        h_idx = np.arange(0, T, L)
        hs = h0[h_idx].transpose(2, 0, 1, 3).reshape(-1, h0.shape[1], h0.shape[3])
        cs = c0[h_idx].transpose(2, 0, 1, 3).reshape(-1, c0.shape[1], c0.shape[3])

        n_seq = seq_starts.shape[0]
        p = self.policy

        def run(idx):
            o = {k: th.as_tensor(v[idx].reshape(-1, *v.shape[2:])).to(self.device) for k, v in seq_obs.items()}
            feats = p.extract_features(o)
            pi_feats = feats if p.share_features_extractor else feats[0]
            st = (th.as_tensor(hs[idx]).swapaxes(0, 1).contiguous().to(self.device),
                  th.as_tensor(cs[idx]).swapaxes(0, 1).contiguous().to(self.device))
            ep = th.as_tensor(seq_starts[idx].reshape(-1)).float().to(self.device)
            lat, _ = p._process_sequence(pi_feats, st, ep, p.lstm_actor)
            logits = p.action_net(p.mlp_extractor.forward_actor(lat))
            return lat, logits

        # targets of the behavioural-cloning term: policy right after the PPO phase
        with th.no_grad():
            old_logits = {}
            for b in range(0, n_seq, self.aux_batch_seqs):
                idx = np.arange(b, min(b + self.aux_batch_seqs, n_seq))
                old_logits[b] = run(idx)[1]

        losses, accs = [], []
        p.set_training_mode(True)
        for _ in range(self.aux_epochs):
            for b in range(0, n_seq, self.aux_batch_seqs):
                idx = np.arange(b, min(b + self.aux_batch_seqs, n_seq))
                lat, logits = run(idx)
                tgt = th.as_tensor(seq_y[idx].reshape(-1)).to(self.device)
                m = th.as_tensor(seq_mask[idx].reshape(-1)).to(self.device)
                pred = p.aux_head(lat).squeeze(-1)
                bce = (F.binary_cross_entropy_with_logits(pred, tgt, reduction="none") * m).sum() / m.sum().clamp(min=1)
                kl = F.kl_div(F.log_softmax(logits, -1), F.log_softmax(old_logits[b], -1),
                              log_target=True, reduction="batchmean")
                loss = self.aux_coef * bce + self.aux_clone_coef * kl
                self.aux_optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(p.parameters(), self.max_grad_norm)
                self.aux_optimizer.step()
                with th.no_grad():
                    acc = (((pred > 0).float() == tgt).float() * m).sum() / m.sum().clamp(min=1)
                losses.append([bce.item(), kl.item()])
                accs.append(acc.item())

        l = np.mean(losses, axis=0)
        self.logger.record("aux/bce", float(l[0]))
        self.logger.record("aux/clone_kl", float(l[1]))
        self.logger.record("aux/accuracy", float(np.mean(accs)))
        self.logger.record("aux/positive_rate", float((y * mask).sum() / max(mask.sum(), 1)))
