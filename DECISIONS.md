# Decisions

Append-only record of *why the project changed course*. Read this before proposing anything
that looks like a resolved question — these are closed unless the "Revisit if" clause fires.

---

## D001 — Repo is the single source of truth; four knowledge files, fixed roles

Date: 2026-08-31

Context:
A ~20 hour sprint run across multiple agent sessions and possibly multiple agent tools.
Chat history is not durable and does not transfer between tools.

Decision:
`PROJECT_BRIEF.md` (stable contract) / `DECISIONS.md` (reasoning) / `RESULTS.md` (append-only
reality) / `STATE.md` (compressed current view, rewritten, <~100 lines). Agents read all four
before acting and update `RESULTS.md` + propose `STATE.md` after every meaningful result.

Reason:
Separates *what the project is* from *what we currently believe*. Without this an agent
reconstructs the scientific goal from filenames and drifts.

Revisit if:
Never, within this sprint.

---

## D002 — Agent instructions live in `AGENTS.md`; `CLAUDE.md` is a one-line import

Date: 2026-08-31

Context:
Claude Code reads `CLAUDE.md`; most other coding agents read `AGENTS.md`. Options were
(a) duplicate the content, (b) symlink `CLAUDE.md -> AGENTS.md`, (c) a one-line `@AGENTS.md`
import in `CLAUDE.md`.

Decision:
(c). `AGENTS.md` is canonical. `CLAUDE.md` contains only `See @AGENTS.md`.

Reason:
Duplication drifts silently, which is the exact failure mode this repo exists to prevent.
A symlink is a single file on disk but degrades invisibly wherever symlinks are not preserved
(Windows checkouts without `core.symlinks`, zip/tarball copies, web IDEs, some sandboxes) —
it becomes a text file whose entire content is the string `AGENTS.md`, and the agent silently
gets no instructions. Claude Code expands `@path` imports into context, so the import costs one
indirection and fails loudly rather than silently.

Revisit if:
The instructions grow large enough that per-tool sections are needed. Then keep `AGENTS.md`
canonical and add tool-specific content *below* the import in `CLAUDE.md`, never instead of it.

---

## D003 — Upstream dataset is a pinned, gitignored, read-only clone

Date: 2026-08-31

Context:
`cot-proxy-tasks/` is a 1.1 GB clone of `Centrattic/cot-proxy-tasks` with its own `.git`.

Decision:
Do not vendor or submodule it. Gitignore the directory, pin the commit (`4482324`,
"tasks 1 and 2", 2026-05-04) here and in every run `config.json`, and treat it as read-only.

Reason:
Committing 46k JSON files or nesting a git repo buys nothing; the commit hash is the entire
reproducibility requirement. Writing into the clone would make provenance unrecoverable.

Revisit if:
Upstream publishes a revision that changes Task 1 labels — then pin the new hash in a new
DECISIONS entry and mark affected results as superseded, not invalidated.
