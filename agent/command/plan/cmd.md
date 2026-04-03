Use this tool to read the competition plan and track your progress through it.

## Commands

| Command | Usage |
|---|---|
| `read` | Read the full plan.md |
| `status` | Show progress count and next pending step |
| `mark_done <step text>` | Mark a completed step as done |

## Workflow

1. At the **start of every new turn**, call `status` to know your next step
2. Execute that step using python/file/web_search tools
3. After completing the step, call `mark_done <step description>`
4. Continue with the next step

## Plan format (markdown checkboxes)

```markdown
## Competition Plan

- [ ] EDA: load and describe dataset
- [ ] Baseline: logistic regression, CV score
- [x] Setup: verified data paths and metric
- [ ] Feature engineering v1
- [ ] LightGBM model + hyperparameter search
- [ ] Final submission
```

## Creating a plan

If no plan exists yet, create one using the `file` tool:

```
write /kaggle/working/KaggleClaw/plan.md
## Competition Plan

- [ ] EDA: explore dataset shape, dtypes, missing values
- [ ] Baseline model
- [ ] Feature engineering
- [ ] Model iteration
- [ ] Final submission
```

Always keep the plan updated so you can resume from any interruption.
