# EviTrace v1 Pilot Preference Deployment Receipt

Deployment date: 2026-07-24

Scope: pilot preference only. Formal preference and all transition projects
remain undeployed.

## Frozen input

- Task manifest:
  `task_manifest.json`
- Manifest SHA-256:
  `22f3862370a34ba101c951cefd23b8ce6e49ad147314d307927269f2cb7bf574`
- Pilot preference artifact rows: 30
- Pilot preference artifact SHA-256:
  `5b57188659524253382e882c8f7eddcfdc189343cd8d06acd96a130f96d74daf`

## Initial apply and backup

The initial launcher apply returned `state=created` at
`2026-07-24T10:16:16.496473+00:00`. Before the transaction it created:

- `label_studio_data/backups/pre_trace_alignment_20260724_101616_046634.sqlite3`
- bytes: `7401472`
- SHA-256:
  `6cb39a7fb160f1cb0193265628083ce9cd2250c95eb800c789da229ee3c927c0`
- SQLite `quick_check`: `ok`
- foreign-key violations: `0`
- pre-deployment counts: 12 projects, 2,200 tasks, 1,303 completions

The original launcher used one default filename for dry-run and apply reports,
so a subsequent postcheck dry-run replaced that first JSON receipt. The
launcher and regression tests now use mode-specific receipt names. This note
records the initial apply facts retained in the confirmed launcher output and
reverified directly against the backup and live database.

## Live projects

| Project | Title | Tasks | Completions | `maximum_annotations` | Task-order SHA-256 |
|---:|---|---:|---:|---:|---|
| 23 | `[YULIN ONLY] Pilot-Exp3-Trace-Preference-v1` | 30 | 0 | 1 | `9e179532e0757747eaae5ce588e3432404ba5321f79845ebd062c902f96c6ca1` |
| 24 | `[ZHIQIANG ONLY] Pilot-Exp3-Trace-Preference-v1` | 30 | 0 | 1 | `ea07481ddfb5916289573e55e5dc4735d7629b75a667b4d364ee9b27f656ddbe` |

Post-deployment checks found SQLite `quick_check=ok`, zero foreign-key
violations, and no changes to projects 1--22 or their 2,200 tasks and 1,303
existing completions. The two projects have different frozen task orders and
the same 30-task universe.

Current mode-specific receipts:

- `launch_v1_pilot_preference_dry_run.json`: transactional postcheck,
  `state=already_applied`, `rolled_back=true`;
- `launch_v1_pilot_preference_apply.json`: idempotent apply validation,
  `state=already_applied`.
