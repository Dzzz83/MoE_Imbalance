# Doublecheck spec

## Goal
Transform 13 docs/ Markdown files into 6 lean, high-utility files (AGENTs.md + 5 merged documents) per the approved audit plan.

## Scope
In scope: docs/ folder only. Create 4 new merged files (project_overview.md, research-findings.md, results.md, future-directions.md). Modify AGENTs.md (fill 3 template placeholders, add Section 12 pointer). Delete 10 source files after migration. Out of scope: any code changes, new research, analysis, or content beyond consolidation.

## Acceptance criteria
1. AGENTs.md updated with filled placeholders (lines 4-6, Section 5 glossary) and Section 12 pointer to project_overview.md. 2. project_overview.md created from project-context.md + refactor-plan.md + stage0-data-pipeline.md + redo-plan.md. 3. research-findings.md created from problem.md + experiments.md + final-report.md (body). 4. results.md created from proper-split-results.md + short-report-proper-split.md + final-report.md (addendum). 5. future-directions.md created from PLAN.md + novel-routing-ideas-analysis.md. 6. All 10 source files deleted: project-context.md, refactor-plan.md, stage0-data-pipeline.md, redo-plan.md, proper-split-results.md, short-report-proper-split.md, final-report.md, problem.md, experiments.md, PLAN.md, novel-routing-ideas-analysis.md. 7. Each new file has a standalone first-paragraph summary and uses hierarchical H1/H2/H3 headings.

## Failure modes
File write error: abort and report. Missing content during merge: the author has all source content in context from prior reads. Partial file creation: roll back and retry the failed file. Delete of non-existent file: warn and continue.

## Priorities
Correctness of merged content > conciseness. AGENTs.md sanctity is non-negotiable (no structural/behavioral changes). Cache-hit optimization (atomic topics, standalone summaries) is preferred over file count reduction.

## Non-goals
Not adding new research, analysis, or findings beyond what exists in source files. Not modifying code or project structure. Not changing AGENTs.md behavioral rules or workflow. Not adding images, diagrams, or non-text assets.
