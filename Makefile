# ══════════════════════════════════════════════════════════════
# Longleaf Workflow — Qwen3 Analysis
# ══════════════════════════════════════════════════════════════
#
# Prerequisites:
#   1. Connected to UNC VPN (Cisco AnyConnect → vpn.unc.edu)
#   2. Authenticated once: `make ssh` (opens shell, exit with ctrl-d)
#      After that, all commands reuse the SSH tunnel for 2 hours.
#
# First time:
#   make ssh       → authenticate
#   make setup     → create dirs + venv on Longleaf
#   make test      → submit GPU smoke test
#   make status    → check if test job is queued/running
#   make logs      → see output when done
#
# Daily:
#   make ssh       → authenticate (once per session)
#   make submit JOB=pretrain   → sync code + submit training job
#   make status    → check queue
#   make logs      → tail output
#   make pull      → bring checkpoints/logs home
#
# ══════════════════════════════════════════════════════════════

# ── Configuration ────────────────────────────────────────────
REMOTE      := longleaf
REMOTE_DIR  := /work/users/t/r/treese20/qwen3-analysis
REMOTE_WORK := /work/users/t/r/treese20
LOCAL_DIR   := $(shell pwd)

# Default SLURM job (override with JOB=<name>)
JOB         ?= pretrain

# ── Default target ───────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "  Longleaf Workflow"
	@echo "  ════════════════════════════════════════"
	@echo ""
	@echo "  Setup (first time):"
	@echo "    make ssh           Open SSH session (authenticate once)"
	@echo "    make setup         Create dirs + venv on Longleaf"
	@echo ""
	@echo "  Code sync:"
	@echo "    make sync          Push code to Longleaf"
	@echo "    make sync-dry      Preview what would sync (dry run)"
	@echo ""
	@echo "  Jobs:"
	@echo "    make submit        Sync + submit (default: pretrain.slurm)"
	@echo "    make submit JOB=x  Sync + submit longleaf/x.slurm"
	@echo "    make test          Sync + submit GPU smoke test"
	@echo "    make status        Check job queue"
	@echo "    make cancel JOB=x  Cancel job by ID"
	@echo ""
	@echo "  Logs:"
	@echo "    make logs          Tail the most recent log"
	@echo "    make logs JOB=x    Tail a specific job ID's log"
	@echo ""
	@echo "  Results:"
	@echo "    make pull          Pull checkpoints, snapshots, logs"
	@echo ""
	@echo "  Remote shell:"
	@echo "    make ssh           Interactive Longleaf session"
	@echo "    make remote-ls     List files on Longleaf"
	@echo ""

# ── SSH ──────────────────────────────────────────────────────
.PHONY: ssh
ssh:
	@echo "Connecting to Longleaf..."
	@echo "(After authenticating, all make commands reuse this tunnel for 2h)"
	@echo ""
	ssh $(REMOTE)

# ── Sync code to Longleaf ───────────────────────────────────
.PHONY: sync sync-dry
sync:
	@echo "Syncing $(LOCAL_DIR) → $(REMOTE):$(REMOTE_DIR)"
	rsync -avz \
		--exclude-from=.rsyncignore \
		"$(LOCAL_DIR)/" "$(REMOTE):$(REMOTE_DIR)/"
	@echo "Sync complete."

sync-dry:
	@echo "DRY RUN — previewing what would sync:"
	rsync -avzn \
		--exclude-from=.rsyncignore \
		"$(LOCAL_DIR)/" "$(REMOTE):$(REMOTE_DIR)/"

# ── Submit SLURM jobs ───────────────────────────────────────
.PHONY: submit test
submit: sync
	@echo ""
	@echo "Submitting longleaf/$(JOB).slurm..."
	ssh $(REMOTE) "cd $(REMOTE_DIR) && sbatch longleaf/$(JOB).slurm"

test: sync
	@echo ""
	@echo "Submitting GPU smoke test..."
	ssh $(REMOTE) "cd $(REMOTE_DIR) && sbatch longleaf/test-gpu.slurm"

# ── Job management ──────────────────────────────────────────
.PHONY: status cancel
status:
	@ssh $(REMOTE) "squeue -u \$$USER -o '%.10i %.20j %.8T %.10M %.6D %R'"

cancel:
	@echo "Cancelling job $(JOB)..."
	ssh $(REMOTE) "scancel $(JOB)"

# ── Logs ─────────────────────────────────────────────────────
.PHONY: logs
logs:
ifeq ($(JOB),pretrain)
	@echo "Tailing most recent log in $(REMOTE_DIR)/longleaf/logs/"
	@echo "(Ctrl-C to stop)"
	@echo ""
	ssh $(REMOTE) "ls -t $(REMOTE_DIR)/longleaf/logs/*.out 2>/dev/null | head -1 | xargs tail -f"
else
	@echo "Tailing log for job $(JOB)..."
	@echo "(Ctrl-C to stop)"
	@echo ""
	ssh $(REMOTE) "tail -f $(REMOTE_DIR)/longleaf/logs/$(JOB).out"
endif

# ── Pull results back ───────────────────────────────────────
.PHONY: pull
pull:
	@echo "Pulling checkpoints..."
	@mkdir -p checkpoints
	rsync -avz "$(REMOTE):$(REMOTE_DIR)/checkpoints/" "$(LOCAL_DIR)/checkpoints/"
	@echo ""
	@echo "Pulling snapshots..."
	@mkdir -p snapshots
	rsync -avz "$(REMOTE):$(REMOTE_DIR)/snapshots/" "$(LOCAL_DIR)/snapshots/"
	@echo ""
	@echo "Pulling logs..."
	@mkdir -p longleaf/logs
	rsync -avz "$(REMOTE):$(REMOTE_DIR)/longleaf/logs/" "$(LOCAL_DIR)/longleaf/logs/"
	@echo ""
	@echo "Pull complete."

# ── Setup (one-time) ────────────────────────────────────────
.PHONY: setup
setup: sync
	@echo ""
	@echo "Running one-time setup on Longleaf..."
	ssh $(REMOTE) "cd $(REMOTE_DIR) && bash longleaf/setup.sh"

# ── Utility ──────────────────────────────────────────────────
.PHONY: remote-ls
remote-ls:
	@ssh $(REMOTE) "ls -la $(REMOTE_DIR)/"
