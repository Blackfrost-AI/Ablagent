---
description: Run the offline synthetic derisk pipeline (no network, no torch)
---

Run the fully offline synthetic pipeline and summarize the evidence:

```bash
ablagent run --pack derisk
```

Then run `ablagent status <run_dir>` on the printed run directory. Report
completed stages and the release state. This path never touches the network.