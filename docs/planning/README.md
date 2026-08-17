# TSPN Planning Documents

This directory holds the project's planning history, copied here (2026-08-17)
because they previously lived outside git — the 5 original docs in the
user's `~/Downloads/`, the approved phase-by-phase plan in Claude Code's own
local plan storage (`~/.claude/plans/`) — meaning they could be lost or
become inaccessible without ever being version-controlled.

## How these relate to `PROJECT_STATE.md` (repo root)

**`PROJECT_STATE.md` is the source of truth whenever it conflicts with
anything here.** These 5 files describe the *original, intended* design;
real-world data availability forced many deviations during implementation
(see `PROJECT_STATE.md` §1 for the full, itemized list — 37 findings and
counting). `PROJECT_STATE.md` is a *delta* document: it assumes you still
have these originals for everything that *wasn't* deviated from (most
formulas, most checkpoint definitions), and only calls out what changed.
Read both together, not `PROJECT_STATE.md` alone, for full context.

## Files

| File | What it is |
|---|---|
| `TSPN_Research_Brief.md` | The original research problem statement, related work, architecture overview, evaluation design, and target-venue framing. Read this first for *why* the project exists. |
| `TSPN_Implementation_Masterplan.md` | The most detailed, prompt-style walkthrough of the full build, organized by phase. |
| `TSPN_Locked_Implementation_Plan.md` | The "do not deviate" version of the spec — exact formulas, exact hyperparameters, exact file schemas as originally intended. |
| `TSPN_Risk_Checkpoints.md` | Numbered checkpoints (CP01, CP02, ...) describing specific known failure modes and how to detect/fix each. Referenced throughout `PROJECT_STATE.md` and in code comments (e.g. "CP21", "CP37") — this is what those references mean. |
| `TSPN_Complete_Implementation_Guide.md` | A more narrative, day-by-day implementation guide covering the same ground as the Masterplan from a different angle. |
| `TSPN_Approved_Phase4-13_Plan.md` | The plan this session's work actually followed, written after reconciling the above 5 docs against the real codebase state as of 2026-08-13 (Phase 4 was mid-flight and partially broken at that point — see `PROJECT_STATE.md` §1.2). Covers Phase 4-fix through Phase 13 at decreasing levels of detail the further out the phase. |

## Reading order for a new session / new collaborator

1. `TSPN_Research_Brief.md` — understand the goal.
2. `PROJECT_STATE.md` (repo root) — understand what actually exists today and how it differs from plan.
3. Whichever of the 5 original docs covers the phase you're working on, for the original formula/spec detail `PROJECT_STATE.md` doesn't repeat.
4. `TSPN_Approved_Phase4-13_Plan.md` for the phase-by-phase task breakdown of what's left.
