Use this tool to read, write, and manage files in the Kaggle working directory.

## Commands

| Command | Usage |
|---|---|
| `read <path>` | Read a file's contents |
| `write <path>\n<content>` | Write (or overwrite) a file |
| `list [path]` | List files in a directory |
| `delete <path>` | Delete a file or directory |
| `mkdir <path>` | Create a directory |
| `exists <path>` | Check if a path exists |
| `move <src> <dst>` | Move or rename a file |

## Allowed paths
- **Read**: `/kaggle/working/`, `/kaggle/input/`, `/tmp/`
- **Write/Delete**: `/kaggle/working/`, `/tmp/`

## Examples

```
read /kaggle/working/KaggleClaw/plan.md
write /kaggle/working/KaggleClaw/train.py
import pandas as pd
df = pd.read_csv('/kaggle/input/train.csv')
list /kaggle/working/KaggleClaw/run/
mkdir /kaggle/working/KaggleClaw/run/exp1/
```

## Tips
- For large edits to existing files, prefer `apply_patch` over `write`
- Always verify paths with `exists` before reading
- Create output directories with `mkdir` before writing results
