"""
MD-SAM3: Memory-Decoupled SAM3 with Dual Memory and Diversity-Aware Frame Selection.

API based on SAM3-DMS (Sam3TrackerPredictor with SAM2-style inference):
  predictor.init_state(video_path)
  predictor.add_new_mask(state, frame_idx, obj_id, mask)
  predictor.propagate_in_video_preflight(state)
  for frame_idx, obj_ids, low_res, video_res, obj_scores in predictor.propagate_in_video(state, ...):
      # obj_scores: per-obj quality scores (SAM3-DMS extension)
      # Between yields: access/modify state["output_dict"]["non_cond_frame_outputs"]
"""

from __future__ import annotations

import logging
import os
import random
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class MDSam3Config:
    dms: bool = True           # per-object tracker states
    dm: bool = True            # prune low-quality non_cond_frames (rel partition)
    dfs: bool = True           # diversity-aware selection for occ partition

    tau_rel: float = 0.95      # DM: min eff_iou_score to keep a frame in rel_mem
    tau_occ: float = 0.65      # DFS: min score to add to unconditional buffer
    lambda_dfs: float = 0.5    # DFS: reliability vs diversity weight
    memory_size: int = 6       # M; each partition = M/2

    min_occ_frames: int = 2    # frames below occ_thresh to trigger occlusion state
    occ_score_thresh: float = 0.0   # obj_score_logit below this → occluded

    dfs_mode: str = "diversity"    # "diversity" | "recency" | "random"

    # SRG: self-calibrated relevance gating (per-instance adaptive tau_rel).
    # tau_eff = min(tau_rel, adaptive_beta * running median of the object's own
    # visible-frame reliability). Compensates the area-dependent reliability
    # bias that starves small instruments of memory under a fixed tau_rel.
    adaptive_tau: bool = False
    adaptive_beta: float = 1.0
    srg_quantile: float = 0.5   # q in Eq.(SRG); 0.5 = median

    # Deconfounding DMS: independent per-object propagation but the retention
    # decision uses the GROUP-MEAN reliability (as the coupled path does).
    # Isolates 'gating granularity' from 'propagation coupling'.
    group_gate: bool = False

    # Matched-retention controls: does DM's gain come from WHICH frames it
    # drops, or merely from HOW MANY? 'random'/'stride' discard the same
    # fraction of frames while ignoring reliability entirely.
    prune_mode: str = "reliability"   # 'reliability' | 'random' | 'stride' | 'perm' | 'shift' | 'even' | 'replay'
    prune_rate: float = 0.0            # target fraction pruned (random/stride)
    # Seed for the random controls. The old code hard-wired 12345, so every
    # 'random' run was the same single draw.
    prune_seed: int = 12345
    # 'perm'/'shift' replay a reference run's per-sequence decisions
    # (retention_decisions.json written by the reference run):
    #   perm  - delete exactly as many entries per sequence as the reference,
    #           at uniformly random positions (exact per-sequence rate match)
    #   shift - the reference deletion pattern circularly shifted by a random
    #           offset: same count and same run structure, but misaligned
    #           with the score
    #   replay - delete exactly the listed entries (oracle and score-ranked
    #           deletion lists built offline, scripts/oracle_decisions.py)
    prune_mask_file: str = ""
    # 'quantile': online rank rule. Delete entry t if its score is below the
    # prune_rate-quantile of the scores of earlier entries of this sequence
    # (the last quantile_window of them; 0 = all). Causal, single pass.
    quantile_window: int = 0
    # Input frames handed to SAM3. 'jpg95' reproduces all earlier runs (lossy
    # JPEG q95); 'png' is lossless (PNG bytes under a .jpg name: SAM3's tracker
    # loader filters by extension but PIL decodes by content).
    frame_format: str = "jpg95"

    # Direction C: DFS standalone (without DM)
    block_occ_pollution: bool = False  # delete occluded frames from nco to preserve pre-occ window
    store_mem_entries: bool = False    # cache full nco entries for re-injection on recovery


# ─────────────────────────────────────────────────────────────
#  Frame record
# ─────────────────────────────────────────────────────────────

class FrameRecord:
    __slots__ = ("frame_idx", "reliability", "features", "pred_mask", "mem_entry")

    def __init__(self, frame_idx, reliability, features=None, pred_mask=None, mem_entry=None):
        self.frame_idx = frame_idx
        self.reliability = reliability
        self.features = features    # (C, H, W) or None — SAM3 maskmem_features
        self.pred_mask = pred_mask  # (H, W) bool numpy for re-conditioning
        self.mem_entry = mem_entry  # deep-copied nco entry for re-injection (Direction C)


def _copy_nco_entry(entry: dict) -> dict:
    """CPU deep-copy of a SAM3 nco entry for buffer storage."""
    out = {}
    for k, v in entry.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().clone()
        elif isinstance(v, list):
            out[k] = [x.detach().cpu().clone() if isinstance(x, torch.Tensor) else x for x in v]
        elif k == "_occ_mem":
            continue  # don't copy sentinel flags
        else:
            out[k] = v
    return out


# ─────────────────────────────────────────────────────────────
#  Per-object occlusion state
# ─────────────────────────────────────────────────────────────

class ObjectTracker:
    def __init__(self, cfg: MDSam3Config):
        self.cfg = cfg
        self.buffer: List[FrameRecord] = []  # unconditional buffer
        self.occ_count = 0
        self.is_occluded = False

    def update(self, frame_idx: int, obj_score: float,
               eff_score: float, pred_mask, maskmem_feats=None,
               mem_entry: Optional[dict] = None) -> bool:
        """Returns True if this frame is a recovery event."""
        visible = obj_score > self.cfg.occ_score_thresh
        recovery = False

        if visible:
            if self.is_occluded and self.occ_count >= self.cfg.min_occ_frames:
                recovery = True
            self.is_occluded = False
            self.occ_count = 0
            if eff_score >= self.cfg.tau_occ:
                feat = maskmem_feats.cpu() if maskmem_feats is not None else None
                stored = (_copy_nco_entry(mem_entry)
                          if self.cfg.store_mem_entries and mem_entry is not None
                          else None)
                self.buffer.append(FrameRecord(frame_idx, eff_score, feat, pred_mask, stored))
        else:
            self.is_occluded = True
            self.occ_count += 1

        return recovery

    def select_occ_mem(self) -> List[FrameRecord]:
        n = self.cfg.memory_size // 2
        if not self.buffer:
            return []
        if len(self.buffer) <= n:
            return list(self.buffer)
        if not self.cfg.dfs or self.cfg.dfs_mode == "recency":
            return self.buffer[-n:]
        if self.cfg.dfs_mode == "random":
            return random.sample(self.buffer, n)
        return self._greedy(self.buffer, n)

    def _greedy(self, cands: List[FrameRecord], n: int) -> List[FrameRecord]:
        rels = np.array([c.reliability for c in cands])
        rmin, rmax = rels.min(), rels.max()
        norm = (rels - rmin) / (rmax - rmin + 1e-8)
        sel = [int(np.argmax(norm))]
        for _ in range(n - 1):
            rem = [i for i in range(len(cands)) if i not in sel]
            if not rem:
                break
            best, best_i = -1.0, rem[0]
            for i in rem:
                fi = cands[i].features
                dists = []
                for j in sel:
                    fj = cands[j].features
                    if fi is not None and fj is not None:
                        cs = F.cosine_similarity(
                            fi.float().flatten().unsqueeze(0),
                            fj.float().flatten().unsqueeze(0),
                        ).item()
                        dists.append((1.0 - cs) / 2.0)
                    else:
                        dists.append(
                            abs(cands[i].frame_idx - cands[j].frame_idx)
                            / max(len(cands), 1)
                        )
                min_d = min(dists)
                score = self.cfg.lambda_dfs * norm[i] + (1 - self.cfg.lambda_dfs) * min_d
                if score > best:
                    best, best_i = score, i
            sel.append(best_i)
        return [cands[i] for i in sorted(sel)]


# ─────────────────────────────────────────────────────────────
#  Inferencer
# ─────────────────────────────────────────────────────────────

class MDSam3Inferencer:
    """
    Wraps Sam3TrackerPredictor with MD-SAM3 memory management.

    Build:
        inferencer = build_md_sam3(cfg, checkpoint_path)

    Run:
        preds = inferencer.run_sequence(frame_files, init_masks)
        # preds: {local_frame_idx: (H,W) uint8 type mask}
    """

    def __init__(self, predictor, cfg: MDSam3Config):
        self.predictor = predictor
        self.cfg = cfg
        # DM retention accounting (reported so controls can be rate-matched)
        self.n_prune_seen = 0
        self.n_pruned = 0
        self._stride_acc = 0.0
        self._prune_rng = random.Random(cfg.prune_seed)
        self.seq_tag = ""
        # {seq_tag: {"tested": [frame_idx...], "deleted": [frame_idx...]}}
        self.decisions: Dict[str, Dict[str, List[int]]] = {}
        self._ref_decisions = None
        self._planned: Optional[set] = None
        # C7-store diagnostic: were re-inserted entries actually read?
        self.reinject_log: List[dict] = []

    def _read_window(self, nco: dict, t: int, n_frames: int) -> set:
        """Replicates SAM3's frame_filter (use_memory_selection=True): the
        non-conditioning entries that the memory read for frame t attends to.
        Read-only; used only for the re-injection diagnostic."""
        if t <= 0:
            return set()
        p = self.predictor
        max_num = min(n_frames, getattr(p, "max_obj_ptrs_in_encoder", 16))
        mf = getattr(p, "mf_threshold", 0.01)
        valid = []
        for i in range(t - 1, 0, -1):
            e = nco.get(i)
            if e is None or "eff_iou_score" not in e:
                continue
            if self._get_eff_score(e) > mf:
                valid.insert(0, i)
            if len(valid) >= max_num - 1:
                break
        if (t - 1) not in valid:
            valid.append(t - 1)
        k = getattr(p, "num_maskmem", 7) - 1
        return set(valid[-k:])

    def _plan_sequence(self):
        """For 'perm'/'shift', fix this sequence's deletion set up front from
        the reference run's decisions on the same sequence."""
        mode = self.cfg.prune_mode
        self._qhist = []          # per-sequence score history ('quantile')
        if mode not in ("perm", "shift", "even", "replay"):
            return
        if self._ref_decisions is None:
            import json
            if not self.cfg.prune_mask_file:
                raise ValueError(f"prune_mode={mode} needs prune_mask_file")
            with open(self.cfg.prune_mask_file) as f:
                self._ref_decisions = json.load(f)["sequences"]
        ref = self._ref_decisions[self.seq_tag]
        tested = list(ref["tested"])
        deleted = set(ref["deleted"])
        # String seed: deterministic across processes, independent per sequence
        rng = random.Random(f"{self.cfg.prune_seed}:{self.seq_tag}:{mode}")
        if mode == "replay":
            self._planned = set(deleted)
        elif mode == "perm":
            self._planned = set(rng.sample(tested, len(deleted)))
        elif mode == "even":
            # deterministic: k entries spaced as evenly as possible
            n, k = len(tested), len(deleted)
            self._planned = {tested[int((j + 0.5) * n / k)] for j in range(k)} if k else set()
        else:
            n = len(tested)
            off = rng.randrange(1, n) if n > 1 else 0
            self._planned = {tested[(i + off) % n]
                             for i, t in enumerate(tested) if t in deleted}
        logger.info(f"  {mode} control: {len(self._planned)}/{len(tested)} "
                    f"planned deletions for {self.seq_tag}")

    def _should_prune(self, eff_score: float, tau_eff: float,
                      frame_idx: Optional[int] = None) -> bool:
        """DM retention decision. 'reliability' is the method under study;
        'random'/'stride'/'perm'/'shift' are rate-matched controls that ignore
        eff_score."""
        self.n_prune_seen += 1
        mode = self.cfg.prune_mode
        if mode == "random":
            hit = self._prune_rng.random() < self.cfg.prune_rate
        elif mode in ("perm", "shift", "even", "replay"):
            if frame_idx is None or self._planned is None:
                raise RuntimeError(f"prune_mode={mode} is only supported on the joint path")
            hit = frame_idx in self._planned
        elif mode == "quantile":
            w = self.cfg.quantile_window
            h = self._qhist[-w:] if w else self._qhist
            hit = bool(h) and eff_score < float(np.quantile(h, self.cfg.prune_rate))
            self._qhist.append(float(eff_score))
        elif mode == "stride":
            self._stride_acc += self.cfg.prune_rate
            hit = self._stride_acc >= 1.0
            if hit:
                self._stride_acc -= 1.0
        else:
            hit = eff_score < tau_eff
        if hit:
            self.n_pruned += 1
        if frame_idx is not None:
            d = self.decisions.setdefault(self.seq_tag, {"tested": [], "deleted": []})
            d["tested"].append(int(frame_idx))
            # score of the entry, logged for every mode (does not affect the decision)
            d.setdefault("scores", []).append(round(float(eff_score), 6))
            if hit:
                d["deleted"].append(int(frame_idx))
        return hit

    def run_sequence(
        self,
        frame_files: List[str],
        init_masks: Dict[int, np.ndarray],
        init_frame_local_idx: int = 0,
    ) -> Dict[int, np.ndarray]:
        """
        Args:
            frame_files: sorted list of image paths (full sequence)
            init_masks: {class_id: binary HxW uint8 mask} for init frame
            init_frame_local_idx: position in frame_files used as init (default 0)
        Returns:
            {local_idx: (H,W) uint8 type-mask}
        """
        tmpdir = tempfile.mkdtemp(prefix="mdsam3_")
        try:
            for i, fp in enumerate(frame_files):
                img = cv2.imread(fp)
                out = os.path.join(tmpdir, f"{i:05d}.jpg")
                if self.cfg.frame_format == "png":
                    ok, buf = cv2.imencode(".png", img)
                    assert ok
                    with open(out, "wb") as fh:
                        fh.write(buf.tobytes())
                else:
                    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            H, W = cv2.imread(frame_files[0]).shape[:2]
            n = len(frame_files)

            self._plan_sequence()
            if self.cfg.prune_mode in ("perm", "shift", "even", "replay") and (
                    self.cfg.group_gate or self.cfg.dms or self.cfg.block_occ_pollution):
                raise ValueError("perm/shift controls are implemented for the joint path only")
            if self.cfg.group_gate:
                return self._run_groupgate(tmpdir, H, W, n, init_masks, init_frame_local_idx)
            if self.cfg.dms or self.cfg.block_occ_pollution:
                return self._run_decoupled(tmpdir, H, W, n, init_masks, init_frame_local_idx)
            else:
                return self._run_coupled(tmpdir, H, W, n, init_masks, init_frame_local_idx)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ── Coupled (C0, C2, C3, C6) ─────────────────────────────

    def _run_coupled(self, video_dir, H, W, n_frames, init_masks, init_idx):
        state = self.predictor.init_state(video_path=video_dir)
        added = []
        for cls_id, mask in init_masks.items():
            if not mask.any():
                continue
            self.predictor.add_new_mask(
                state, init_idx, cls_id, torch.from_numpy(mask.astype(bool))
            )
            added.append(cls_id)
        if not added:
            return {}

        preds = {}
        obj_trackers: Dict[int, ObjectTracker] = {c: ObjectTracker(self.cfg) for c in added}
        grp_rel_history: List[float] = []  # group reliability, for coupled-path SRG

        for frame_idx, obj_ids, _, video_masks, obj_scores in self.predictor.propagate_in_video(
            state, start_frame_idx=0, max_frame_num_to_track=n_frames,
            reverse=False, tqdm_disable=True, propagate_preflight=True
        ):
            pred_type = np.zeros((H, W), dtype=np.uint8)
            nco = state["output_dict"]["non_cond_frame_outputs"]

            for idx, oid in enumerate(obj_ids):
                mask_prob = video_masks[idx, 0]
                binary = (mask_prob > 0.0).cpu().numpy()
                pred_type[binary] = oid

            # DM: prune non_cond memory for low-quality frames. The gating
            # quantity is the group mean over objects (SAM3's own policy).
            # Hoisted out of the object loop: the entry is shared across
            # objects, so the decision was already frame-level.
            if self.cfg.dm and frame_idx in nco:
                eff = self._get_eff_score(nco[frame_idx])
                tau_eff = self.cfg.tau_rel
                if self.cfg.adaptive_tau:
                    if self._get_obj_score(nco[frame_idx]) > self.cfg.occ_score_thresh:
                        grp_rel_history.append(eff)
                    if grp_rel_history:
                        tau_eff = min(tau_eff, self.cfg.adaptive_beta * float(
                            np.quantile(grp_rel_history, self.cfg.srg_quantile)))
                if self._should_prune(eff, tau_eff, frame_idx):
                    del nco[frame_idx]

            preds[frame_idx] = pred_type
        del state
        return preds

    # ── Decoupled (C1, C4, C5, C7, C3-fix-*) ───────────────

    def _run_decoupled(self, video_dir, H, W, n_frames, init_masks, init_idx):
        per_obj_preds: Dict[int, Dict[int, np.ndarray]] = {}

        for cls_id, mask in init_masks.items():
            if not mask.any():
                continue
            state = self.predictor.init_state(video_path=video_dir)
            self.predictor.add_new_mask(
                state, init_idx, cls_id, torch.from_numpy(mask.astype(bool))
            )
            ot = ObjectTracker(self.cfg)
            obj_preds = {}
            self._propagate_obj(state, ot, cls_id, n_frames, H, W, obj_preds)
            del state
            per_obj_preds[cls_id] = obj_preds

        # Merge type masks
        preds = {}
        for fi in range(n_frames):
            t = np.zeros((H, W), dtype=np.uint8)
            for cls_id in sorted(per_obj_preds):
                m = per_obj_preds[cls_id].get(fi, None)
                if m is not None:
                    t[m > 0] = cls_id
            preds[fi] = t
        return preds

    def _propagate_obj(self, state, ot: ObjectTracker, cls_id: int,
                        n_frames: int, H: int, W: int,
                        obj_preds: Dict[int, np.ndarray]):
        last_recovery_frame = -99
        device = next(self.predictor.parameters()).device if hasattr(self.predictor, 'parameters') else torch.device('cuda')
        rel_history: List[float] = []  # visible-frame reliability, for SRG
        score_log = os.environ.get("MDSAM3_SCORE_LOG")
        pending_reads: List[dict] = []   # re-injections awaiting the read check

        for frame_idx, obj_ids, _, video_masks, obj_scores in self.predictor.propagate_in_video(
            state, start_frame_idx=0, max_frame_num_to_track=n_frames,
            reverse=False, tqdm_disable=True, propagate_preflight=True
        ):
            nco = state["output_dict"]["non_cond_frame_outputs"]

            # Diagnostic (C7-store): the read for this frame has already used
            # the pool as it stood before this frame's entry was added.
            if pending_reads:
                win = self._read_window(nco, frame_idx, n_frames)
                for rec in pending_reads:
                    rec["read"] |= set(rec["injected"]) & win
                    rec["checked"] += 1
                done = [r for r in pending_reads if r["checked"] >= 5]
                for r in done:
                    self.reinject_log.append({"seq": self.seq_tag, "obj": int(cls_id), "t": r["t"],
                                              "injected": sorted(r["injected"]),
                                              "read_within_5": sorted(r["read"])})
                pending_reads = [r for r in pending_reads if r["checked"] < 5]

            if len(obj_ids) == 0:
                obj_preds[frame_idx] = np.zeros((H, W), dtype=np.uint8)
                ot.update(frame_idx, -1.0, 0.0, None)
                # Block pollution even when obj_ids is empty
                if self.cfg.block_occ_pollution and frame_idx in nco:
                    del nco[frame_idx]
                continue

            mask_prob = video_masks[0, 0]
            pred_bin = (mask_prob > 0.0).cpu().numpy().astype(np.uint8)
            obj_preds[frame_idx] = pred_bin

            # Get quality scores
            score_entry = nco.get(frame_idx) or state["output_dict"]["cond_frame_outputs"].get(frame_idx, {})
            obj_logit = self._get_obj_score(score_entry)
            eff_score = self._get_eff_score(score_entry)
            maskmem_feats = score_entry.get("maskmem_features", None)

            if obj_logit > self.cfg.occ_score_thresh:
                rel_history.append(eff_score)
            if score_log:
                with open(score_log, "a") as f:
                    f.write(f"{getattr(self, 'seq_tag', '')},{cls_id},{frame_idx},"
                            f"{int(pred_bin.sum())},{obj_logit:.4f},{eff_score:.4f}\n")

            # Update occlusion tracker (pass full nco entry for Direction C storage)
            recovery = ot.update(
                frame_idx, obj_logit, eff_score, pred_bin,
                maskmem_feats.cpu() if maskmem_feats is not None else None,
                mem_entry=score_entry if self.cfg.store_mem_entries else None,
            )

            # DM: prune low-quality frames from relevance-aware memory
            if self.cfg.dm and frame_idx in nco:
                tau_eff = self.cfg.tau_rel
                if self.cfg.adaptive_tau and rel_history:
                    tau_eff = min(tau_eff, self.cfg.adaptive_beta *
                                  float(np.quantile(rel_history, self.cfg.srg_quantile)))
                if self._should_prune(eff_score, tau_eff):
                    del nco[frame_idx]

            # Direction C: block occluded frames from memory to preserve pre-occ window
            if self.cfg.block_occ_pollution and ot.is_occluded and frame_idx in nco:
                del nco[frame_idx]

            # DFS: on recovery, inject diverse pre-occlusion frames into memory.
            # Requires either DM (for the occ partition) or block_occ_pollution (standalone DFS).
            if (self.cfg.dfs and recovery and
                    (self.cfg.dm or self.cfg.block_occ_pollution) and
                    frame_idx - last_recovery_frame > 5):
                selected = ot.select_occ_mem()
                if selected:
                    before = set(nco.keys())
                    self._inject_occ_mem(state, nco, selected, cls_id, device)
                    if self.cfg.store_mem_entries:
                        new = sorted(set(nco.keys()) - before)
                        pending_reads.append({"t": int(frame_idx), "injected": new,
                                              "read": set(), "checked": 0})
                last_recovery_frame = frame_idx

    def _inject_occ_mem(self, state, nco: dict, frames: List[FrameRecord],
                         cls_id: int, device=None):
        """
        Inject DFS-selected frames into non_cond memory as occ-aware partition.

        With DM active: just protect existing nco entries from DM pruning.
        With block_occ_pollution (standalone DFS): re-insert from stored copies
        if the frame has been evicted from nco.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cond_out = state["output_dict"]["cond_frame_outputs"]
        for rec in frames:
            fi = rec.frame_idx
            if fi in cond_out:
                continue  # conditioning frames need no intervention
            if fi in nco:
                # Protect from DM pruning (or just mark for clarity)
                nco[fi]["_occ_mem"] = True
                if "eff_iou_score" in nco[fi]:
                    nco[fi]["eff_iou_score"] = torch.tensor(self.cfg.tau_rel + 0.01)
            elif rec.mem_entry is not None:
                # Re-inject from stored CPU copy (Direction C standalone path)
                entry = {}
                for k, v in rec.mem_entry.items():
                    if isinstance(v, torch.Tensor):
                        entry[k] = v.to(device)
                    elif isinstance(v, list):
                        entry[k] = [x.to(device) if isinstance(x, torch.Tensor) else x
                                     for x in v]
                    else:
                        entry[k] = v
                entry["_occ_mem"] = True
                nco[fi] = entry


    # ── Decoupled propagation with GROUP-LEVEL gating (DMS deconfound) ──

    def _run_groupgate(self, video_dir, H, W, n_frames, init_masks, init_idx):
        """Independent per-object propagation (as DMS) but the DM retention test
        uses the group-mean reliability over all objects at that frame (as the
        coupled path). Objects are advanced in lockstep so the group mean at
        frame t is exact rather than taken from a previous pass.
        """
        states, gens, trackers, per_obj_preds = {}, {}, {}, {}
        for cls_id, mask in init_masks.items():
            if not mask.any():
                continue
            st = self.predictor.init_state(video_path=video_dir)
            self.predictor.add_new_mask(
                st, init_idx, cls_id, torch.from_numpy(mask.astype(bool)))
            states[cls_id] = st
            gens[cls_id] = self.predictor.propagate_in_video(
                st, start_frame_idx=0, max_frame_num_to_track=n_frames,
                reverse=False, tqdm_disable=True, propagate_preflight=True)
            trackers[cls_id] = ObjectTracker(self.cfg)
            per_obj_preds[cls_id] = {}

        ids = sorted(states)
        while True:
            step = {}
            exhausted = False
            for c in ids:
                try:
                    step[c] = next(gens[c])
                except StopIteration:
                    exhausted = True
                    break
            if exhausted or not step:
                break

            # Pass 1: read every object's own reliability at this frame
            eff, obj_lg, frame_idx = {}, {}, None
            for c in ids:
                fi, obj_ids, _, video_masks, _ = step[c]
                frame_idx = fi
                out = states[c]["output_dict"]
                nco = out["non_cond_frame_outputs"]
                entry = nco.get(fi) or out["cond_frame_outputs"].get(fi, {})
                if len(obj_ids) == 0:
                    per_obj_preds[c][fi] = np.zeros((H, W), dtype=np.uint8)
                    eff[c], obj_lg[c] = 0.0, -1.0
                    continue
                per_obj_preds[c][fi] = (
                    (video_masks[0, 0] > 0.0).cpu().numpy().astype(np.uint8))
                eff[c] = self._get_eff_score(entry)
                obj_lg[c] = self._get_obj_score(entry)

            # Group-mean reliability: the quantity the coupled path gates on
            group_eff = float(np.mean([eff[c] for c in ids])) if ids else 0.0

            # Pass 2: apply the SAME group decision to every object's memory
            for c in ids:
                trackers[c].update(frame_idx, obj_lg[c], eff[c],
                                   per_obj_preds[c].get(frame_idx))
                nco = states[c]["output_dict"]["non_cond_frame_outputs"]
                if self.cfg.dm and frame_idx in nco and \
                        self._should_prune(group_eff, self.cfg.tau_rel):
                    del nco[frame_idx]

        for c in ids:
            del states[c]

        preds = {}
        for fi in range(n_frames):
            t = np.zeros((H, W), dtype=np.uint8)
            for cls_id in sorted(per_obj_preds):
                m = per_obj_preds[cls_id].get(fi)
                if m is not None:
                    t[m > 0] = cls_id
            preds[fi] = t
        return preds

    # ── Helpers ─────────────────────────────────────────────

    @staticmethod
    def _get_obj_score(entry: dict) -> float:
        """Get objectness score logit from output entry."""
        v = entry.get("object_score_logits", None)
        if v is None:
            return 0.0
        if isinstance(v, torch.Tensor):
            return float(v.mean().item())
        return float(v)

    @staticmethod
    def _get_eff_score(entry: dict) -> float:
        """Get effective IOU score from output entry."""
        v = entry.get("eff_iou_score", None)
        if v is None:
            v = entry.get("iou_score", None)
        if v is None:
            return 0.0
        if isinstance(v, torch.Tensor):
            return float(v.mean().item())
        return float(v)


# ─────────────────────────────────────────────────────────────
#  Config presets
# ─────────────────────────────────────────────────────────────

CONFIG_PRESETS = {
    # Original 2^3 ablation
    "C0": MDSam3Config(dms=False, dm=False, dfs=False),
    "C1": MDSam3Config(dms=True,  dm=False, dfs=False),
    "C2": MDSam3Config(dms=False, dm=True,  dfs=False),
    "C3": MDSam3Config(dms=False, dm=False, dfs=True,  dfs_mode="diversity"),
    "C4": MDSam3Config(dms=True,  dm=True,  dfs=False),
    "C5": MDSam3Config(dms=True,  dm=False, dfs=True,  dfs_mode="diversity"),
    "C6": MDSam3Config(dms=False, dm=True,  dfs=True,  dfs_mode="diversity"),
    "C7": MDSam3Config(dms=True,  dm=True,  dfs=True,  dfs_mode="diversity"),
    # Table 4: DFS strategy ablation (M2 = DMS+DM base + different selection modes)
    "M2-recency": MDSam3Config(dms=True, dm=True, dfs=True, dfs_mode="recency"),
    "M2-random":  MDSam3Config(dms=True, dm=True, dfs=True, dfs_mode="random"),
    "M2-dfs":     MDSam3Config(dms=True, dm=True, dfs=True, dfs_mode="diversity"),
    # SRG: self-calibrated relevance gating on top of DMS+DM
    "C4-SRG": MDSam3Config(dms=True, dm=True, dfs=False, adaptive_tau=True),
    # SRG applied to the COUPLED path (group-level self-calibration):
    # tests whether SRG needs per-instance decoupling to pay off.
    "C2-SRG": MDSam3Config(dms=False, dm=True, dfs=False, adaptive_tau=True),
    # Exact per-sequence rate-matched controls for C2 (need --prune-mask-file
    # pointing at a C2 run's retention_decisions.json, and --prune-seed)
    "C2-perm":  MDSam3Config(dms=False, dm=True, dfs=False, prune_mode="perm"),
    "C2-shift": MDSam3Config(dms=False, dm=True, dfs=False, prune_mode="shift"),
    "C2-even":  MDSam3Config(dms=False, dm=True, dfs=False, prune_mode="even"),
    # Deletes exactly the entries listed in --prune-mask-file
    "C2-replay": MDSam3Config(dms=False, dm=True, dfs=False, prune_mode="replay"),
    # Online rank rule (needs --prune-rate; optional --quantile-window)
    "C2-quant": MDSam3Config(dms=False, dm=True, dfs=False, prune_mode="quantile"),
    "C7-SRG": MDSam3Config(dms=True, dm=True, dfs=True, dfs_mode="diversity",
                            adaptive_tau=True),
    # DMS deconfound: independent propagation + group-level gating
    "C4-GG": MDSam3Config(dms=True, dm=True, dfs=False, group_gate=True),
    # C7 with the occlusion partition ACTUALLY re-injected. In C7 the
    # injection only flags entries already in memory (a no-op); with
    # store_mem_entries the evicted frames are re-inserted for real.
    "C7-store": MDSam3Config(dms=True, dm=True, dfs=True, dfs_mode="diversity",
                              store_mem_entries=True),
    # Direction C: DFS standalone via occlusion-pollution blocking
    # C3-fix-a: block occlusion pollution only (no DFS injection); tests the blocking mechanism alone
    "C3-fix-a": MDSam3Config(dms=True, dm=False, dfs=False, block_occ_pollution=True),
    # C3-fix-b: block + DFS injection with re-insertion from stored entries; full Direction C
    "C3-fix-b": MDSam3Config(dms=True, dm=False, dfs=True, dfs_mode="diversity",
                              block_occ_pollution=True, store_mem_entries=True),
}


# ─────────────────────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────────────────────

def build_md_sam3(cfg: MDSam3Config, checkpoint_path: str,
                  device: str = "cuda") -> MDSam3Inferencer:
    """Load SAM3 tracker with visual backbone and wrap with MD-SAM3."""
    from src.paths import ensure_on_syspath
    ensure_on_syspath()   # SAM3-DMS first

    from sam3.model_builder import build_sam3_video_model  # noqa: must come from SAM3-DMS

    # Build full SAM3 model (detector + tracker)
    sam3_model = build_sam3_video_model(
        checkpoint_path=checkpoint_path,
        device=device,
        apply_temporal_disambiguation=True,   # enable use_memory_selection
        use_decoupled_selection=False,         # we handle DMS ourselves via per-obj states
    )

    # Use tracker predictor directly; give it the detector backbone for image encoding
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    predictor = predictor.to(device).eval()

    return MDSam3Inferencer(predictor, cfg)
