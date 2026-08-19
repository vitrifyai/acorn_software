# Retiring the old ACORN

This is the checklist for making this tree the one everybody uses, and deleting
`/home/vnw/acorn`. Nothing here has been done yet — the shared install is
untouched and still what the desktop launcher starts.

Read the "Before you flip the switch" section first. The cutover itself is three
commands; the testing before it is the part that matters.

---

## Where things stand

| | Shared install | This tree |
|---|---|---|
| Path | `/home/vnw/acorn` | `/home/vnw/acorn-sim-playground` |
| Version | acorn 0.2.0 | acorn 0.3.0 + 7 plugin packages |
| Started by | `/usr/share/applications/ACORN.desktop` → `acorn2.sh` | `acorn_sim_playground.sh` |
| Simulators | no | yes |
| Workspaces | no | yes |

**Nothing in the shared install is missing from this one.** Every source file was
compared: the shared tree has no function, class, or feature that this tree
lacks. Its differences are all older versions of code that has since moved on
here — fewer supported file formats, an earlier YOLO/UNet path, an LLM client
without timeouts or retries. There is nothing to port back.

---

## Before you flip the switch

The automated checks pass — 60 tests, and the app launches and switches
workspaces under a display server. What software cannot check for you is whether
it still does *your* work. Run this tree as your daily driver for a week first:

```bash
/home/vnw/acorn-sim-playground/acorn_sim_playground.sh
```

Specifically worth exercising, because these are the paths the reorganization
touched:

- [ ] Open a real DM4 and a real MRC stack; check contrast, frames, pixel size
- [ ] Annotate with SAM — point, box, and scribble prompts, then accept and reject
- [ ] Run YOLO and UNet, including a batch across a folder
- [ ] Queue images, finalize a dataset, and train a model end to end
- [ ] Particle measurements and a plot; spatial analysis; tracking
- [ ] Generate a TEM simulation, a 4D-STEM scan, and a reference match
- [ ] CryoBLOB, on a folder as well as a single image
- [ ] Ask CLU for something from each workspace, including something out of scope
      (it should switch workspaces first rather than pick the wrong tool)
- [ ] Save a session, quit, reopen it

If anything is wrong, it is almost certainly in one of the five extracted mixins
(`sam_controller`, `detector_controller`, `export_controller`, `movie`,
`threads`) — those moved code, so that is where a mistake would live.

---

## The cutover

**1. Point the shared desktop entry here.** This is the switch that moves every
user at once, and it needs sudo:

```bash
sudo cp /usr/share/applications/ACORN.desktop /usr/share/applications/ACORN.desktop.bak
sudo sed -i \
  -e 's|/home/vnw/acorn/acorn2.sh|/home/vnw/acorn-sim-playground/acorn_sim_playground.sh|' \
  -e 's|/home/vnw/acorn/src|/home/vnw/acorn-sim-playground/src|' \
  /usr/share/applications/ACORN.desktop
```

**2. Rename the tree**, so the name stops describing a playground:

```bash
mv /home/vnw/acorn-sim-playground /home/vnw/acorn-next
```

Then fix the absolute paths inside `acorn_sim_playground.sh` (it hard-codes its
own directory in six places) and re-run `uv pip install -e .` plus the seven
`packages/*` installs, because editable installs record absolute paths.

**3. Leave the old tree in place for one more week**, renamed but not deleted:

```bash
mv /home/vnw/acorn /home/vnw/acorn-retired
```

If nobody reports a problem, delete it. Its git history is worth keeping — push
the repo somewhere first, or keep `acorn-retired/.git`.

---

## Rolling back

At any point before deleting the old tree:

```bash
sudo cp /usr/share/applications/ACORN.desktop.bak /usr/share/applications/ACORN.desktop
```

Within this tree, the state before the reorganization is the `rollback-baseline`
branch:

```bash
git checkout rollback-baseline
uv pip install --python .venv-py312/bin/python -e .
```

---

## Three things that are shared, and will bite you

**1. The CryoBLOB plugin is one directory, installed into both venvs.**
`/home/vnw/acorn-cryoblob-plugin` is editable-installed into the shared venv and
this one. Editing it changes both applications at once. Nothing in this
reorganization touched it, deliberately. After the old tree is gone this stops
mattering — until then, treat it as live.

**2. CLU's API key is in `$HOME`, not in either tree.**
`~/.acorn/llm_config.json` is shared by both installs. That is convenient here
(no reconfiguration after cutover) but it means a provider change in one shows up
in the other.

**3. This tree redirects its caches and config; the shared one does not.**
`acorn_sim_playground.sh` sets `XDG_CACHE_HOME`, `XDG_CONFIG_HOME` and
`ACORN_EMBEDDING_CACHE` into `.runtime/`, which is what kept the two installs
from interfering. After cutover that isolation is no longer needed and costs
users their existing SAM embedding cache. Either drop those five `export` lines
from the launcher, or copy the cache across:

```bash
cp -r ~/.cache/acorn/embeddings/. /home/vnw/acorn-next/.runtime/cache/acorn/embeddings/
```

Workspace preferences live at `$XDG_CONFIG_HOME/acorn/workspace.json`, so
dropping the redirect also means each user starts on the welcome screen once —
which for a first release of the workspace layer is the behaviour you want.
