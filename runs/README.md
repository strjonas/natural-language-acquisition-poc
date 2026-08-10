# Public experiment bundle

Most local run output is intentionally ignored. This directory allow-lists only
the files needed to inspect and smoke-reproduce the strongest current result:

- `organism/probe63_individual_self/*.json`: treatment, controls, ceiling
  survey, and closed-loop summaries for Probe63 (about 70 KB total);
- `organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz` and
  its JSON configuration: the inherited parent model required by the Probe63
  executable (about 3 MB).

The parent checkpoint is a limitation, not an independent replication: all
Probe63 seed blocks share this learned parent. The full childhood-to-adult
training lifecycle must still be repeated from independent initializations.

Run `python scripts/summarize_probe63.py` from the repository root to recalculate
the headline aggregate directly from these files.
