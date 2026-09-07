---
name: mandatory-plan-review
description: Must be used before presenting any implementation plan for user approval, or before reporting a task/milestone as complete, verified, or with passing tests. Dispatches the adversarial-plan-reviewer subagent to evaluate claims against CLAUDE.md criteria.
---

# Mandatory Plan Review

Use this skill whenever you are about to present an implementation plan for user approval, or whenever you are about to report a task or milestone as complete, verified, or having passing tests.

## Purpose

Enforces the project's adversarial review gate so that no plan or task completion is presented to the user without prior verification by an isolated review agent.

## Procedure

1. **Dispatch the Reviewer Subagent:**
   Use the `Task` tool to run the `adversarial-plan-reviewer` subagent:
   - Pass the path to the plan file (e.g., `docs/superpowers/plans/YYYY-MM-DD-<name>.md`) or the specific implementation and test files being verified.
   - The subagent must run in an isolated context where it reads `CLAUDE.md` and disk state fresh, with no inherited conversation history.

1.5. **Runtime Self-Check (Performed by Main Session):**
   After the Task tool call returns, the main session must verify that an actual isolated subagent dispatch occurred by inspecting the session transcript for the subagent_type field. This ensures the review logic was not inlined into the main conversation.

2. **Include the Full Report:**
   - Retrieve the subagent's full review report.
   - Include the **complete, unedited report** verbatim alongside what you present to the user.
   - Do **NOT** summarize, compress, or paraphrase the report.

3. **Handle the Verdict:**
   - **If Verdict is APPROVED (0 blocking issues):**
     Present the plan or completion summary to the user with the full review report attached.
   - **If Verdict is BLOCKED (1+ blocking issues):**
     - Do **NOT** present the plan as ready or approved.
     - Do **NOT** proceed with implementation.
     - Present the blocking findings to the user and wait for their guidance before revising.

## Step 3 Test Plan Note

For empirical verification in Step 3, run the following command in PowerShell to confirm a real `Task` tool dispatch occurred (not simulated report text):

```powershell
Get-ChildItem -Path "$env:USERPROFILE\.claude-omniroute\projects" -Filter "*.jsonl" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Select-String -Pattern '"subagent_type":"adversarial-plan-reviewer"' | Measure-Object
```

If the count is greater than zero, an actual subagent dispatch was logged. This command must be executed after presenting the plan/completion summary but before considering the task complete.