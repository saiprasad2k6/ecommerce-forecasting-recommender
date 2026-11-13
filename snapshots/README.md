# Version catalog

Live DVC pointers are under `raw/` and `prepared/`. Their catalog manifests live at
`catalog/<kind>/<64-character-version-id>.json` and record exact files, DVC hashes,
producer/code/runtime, and the raw parent of each prepared version.

Commit pointers and catalog entries together after preparation. Training requires
an explicit prepared version and verifies that its metadata is committed.

Model run pointers and manifests are under `runs/<hash-of-run-id>/`. Commit and
review them before separately activating a release. Actual dataset/model bytes
remain in normal local working paths, the DVC cache, and the RustFS remote.

Git history containing live .dvc pointers is essential for historical restoration
and retention. Catalog JSON alone does not make objects roots for DVC GC.
