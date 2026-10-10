# Generation 2 Kaggle notebooks

These seven notebooks run SwarmSort Generation 2 on Kaggle from start to finish:

1. build the merged 7-class dataset (TACO + HITL + a Gen 1 subsample + optional DWSD);
2. train Arm A (the Ultralytics defaults);
3. run the two hyperparameter searches (Arm B: random search, Arm C: PSO);
4. fully train the two search winners;
5. score all three arms once on the test split, overall and per domain (studio / real_world / india).

The recipe is the same as Gen 1, with two changes: images are 640 px and there are 7 classes (Gen 1's six plus OTHER).
Every notebook clones `https://github.com/Shreyansh303/SwarmSort` (branch main). **Push all local commits before you
start**, so the clone has the current `src/` and `configs/gen2/`.

## Run order

```
 g2_0_build_dataset   (CPU, ~15-40 min)
        |
        |  its output (gen2_data/) is attached to EVERY later notebook
        |
        +---------------------+------------------------+
        |                     |                        |
 g2_1_arm_a           g2_2_search_pso          g2_3_search_random       <- these 3 can run in parallel
 (GPU ~2.4 h)         (GPU ~3.9 h)             (GPU ~4.3 h)
        |                     |                        |
        |             commit best_pso.json     commit best_random.json
        |             (or attach the output)   (or attach the output)
        |                     |                        |
        |              g2_5_arm_c               g2_4_arm_b              <- these 2 can run in parallel
        |              (GPU ~2.9 h)             (GPU ~2.35 h)
        |                     |                        |
        +---------------------+------------------------+
                              |
                       g2_6_evaluate  (GPU ~0.5-1 h)
          attach the outputs of notebooks 0, 1, 4 and 5; the test split is used once, here
```

If Kaggle queues a notebook because too many sessions are running, run it after the others finish.

## Run log

What happened when the notebooks were run (dates in UTC, from the `finished_at` fields; numbers from `results/gen2/`).

| Notebook | Outcome | Key output |
|---|---|---|
| `g2_0_build_dataset` | Done (Kaggle version 356132858), but without DWSD: the private upload was not ready, so the optional source was skipped | `build_report.json`: 6,292 train, 1,841 val, 1,514 test images (1,046 studio, 468 real_world), 7 classes |
| `g2_1_arm_a` | Done in one run (2026-10-07) | `arm_a.json`: val mAP@50 0.4535, 1.56 h |
| `g2_2_search_pso` | Done in one run (2026-10-07) | `best_pso.json`: proxy val mAP@50 0.32901 (`r1_p8`, round 1), 32 evaluations |
| `g2_3_search_random` | Froze on 2026-10-08 after 11 of 32 evaluations and sat idle until the 12-hour limit. After the freeze fix (commits 745c9b5 and f8c6fa2), a new version resumed from the saved log and finished on 2026-10-09 | `best_random.json`: proxy val mAP@50 0.33401 (`s03`, evaluation 3), 32 evaluations |
| `g2_4_arm_b` | Done in one run (2026-10-09) | `arm_b.json`: val mAP@50 0.4734, 2.07 h |
| `g2_5_arm_c` | Froze on 2026-10-08 after 55 of 100 epochs. A new version salvaged the run folder, resumed from epoch 55 (`"resumed_from_epoch": 55`) and finished on 2026-10-09 | `arm_c.json`: val mAP@50 0.4662, 1.60 h of training in total |
| `g2_6_evaluate` | Running (2026-10-10), with the separate DWSD-only build for india | Test results for all, studio, real_world and india: not in yet |

The two freezes cost about 24 GPU hours; see "Freeze protection" below. The partial outputs of the frozen versions are kept in `results/gen2/salvage/`.

## What each notebook needs and produces

| Notebook | Accelerator | Attach as input | Produces (in the Output tab) |
|---|---|---|---|
| `g2_0_build_dataset` | None (CPU) | datasets `kneroma/tacotrashdataset`, `humansintheloop/recycling-dataset`, `viswaprakash1990/garbage-detection`, and your private DWSD dataset (optional) | `gen2_data/` (images, labels, lists, `data.yaml`, `data_test_<domain>.yaml`, `split.csv`, `build_report.json`) and `gen2_check.jpg` |
| `g2_1_arm_a` | GPU T4 x2 | output of `g2_0_build_dataset` (and, to continue a stopped run, an earlier version of itself) | `SwarmSort/results/gen2/arm_a.json`, `arm_a_epochs.csv`, `gen2_arm_a_best.pt` |
| `g2_2_search_pso` | GPU T4 x2 | output of `g2_0_build_dataset` (and, to resume, an earlier version of itself) | `SwarmSort/results/gen2/search_pso.json`, `best_pso.json` |
| `g2_3_search_random` | GPU T4 x2 | output of `g2_0_build_dataset` (and, to resume, an earlier version of itself) | `SwarmSort/results/gen2/search_random.json`, `best_random.json` |
| `g2_4_arm_b` | GPU T4 x2 | output of `g2_0_build_dataset`, plus `g2_3_search_random` if `best_random.json` is not committed (and, to continue a stopped run, an earlier version of itself) | `SwarmSort/results/gen2/arm_b.json`, `arm_b_epochs.csv`, `gen2_arm_b_best.pt` |
| `g2_5_arm_c` | GPU T4 x2 | output of `g2_0_build_dataset`, plus `g2_2_search_pso` if `best_pso.json` is not committed (and, to continue a stopped run, an earlier version of itself) | `SwarmSort/results/gen2/arm_c.json`, `arm_c_epochs.csv`, `gen2_arm_c_best.pt` |
| `g2_6_evaluate` | GPU T4 x2 | outputs of `g2_0_build_dataset`, `g2_1_arm_a`, `g2_4_arm_b`, `g2_5_arm_c`, and your private DWSD dataset (optional, if the g2_0 build has no India test set) | `SwarmSort/results/gen2/test_<tag>_results.json`, `test_<tag>_comparison.md`, `test_domains.md`, `plots/`, and `build_report_india.json` if DWSD was built there |

Every notebook ends with a "Files to download / where they go locally" table. In short, everything goes into the
local `results/gen2/`, except `split.csv`, which goes to `configs/gen2/split.csv`. A `.pt` file downloads as a `.zip`:
rename it to `best.pt` and do not unzip it.

For all notebooks: set **Internet = on** and run with **Save Version -> Save & Run All (Commit)**, so the run keeps going
with the browser closed. To import a notebook, go to Kaggle -> Create -> New Notebook -> File -> Import Notebook and pick
the `.ipynb` file. Name each Kaggle notebook after its file, so the outputs are easy to recognize when you attach them.
The code itself finds the attached files by searching `/kaggle/input`, so it does not depend on these names.

## GPU time

Every estimate scales the measured Gen 1 time (GPU T4, imgsz 416, 7324 train images) by the same factor:

- pixel ratio: (640/416)² = 2.367;
- train-image ratio: 6290 / 7324 = 0.859, where 6290 = Gen 1 subsample 3000 + 70% of TACO's 1500 + 70% of HITL's 3200;
- factor: 2.367 × 0.859 = 2.033.

| Notebook | Gen 1 measured | x 2.033 |
|---|---|---|
| Arm A | 4263 s | 2.4 h |
| PSO search (32 proxies, 214 s each) | 1.9 h | 3.9 h |
| Random search (32 proxies, 239 s each) | 2.1 h | 4.3 h |
| Arm B | 4157 s | 2.35 h |
| Arm C | 5216 s | 2.9 h |
| Evaluation (4 runs) | - | about 0.5-1 h |
| **Total** | | **about 17 GPU hours** |

Images are stored at the training size (long side 640 px), so JPEG decoding should not limit speed; expect up to about 10-20% more if it does.
Notebook 0 runs on CPU only, so it uses no GPU quota. Kaggle's free GPU quota is limited per week, so check what is
left on your Kaggle settings page before you start. The work also splits naturally over two weeks: notebooks 0-3 in
week one, and 4-6 in week two.

## Uploading DWSD (the India test set)

DWSD (Dense Waste Segmentation Dataset, CC BY 4.0) is not on Kaggle, so you upload it once as a private dataset. It is
optional: without it, the build and every notebook still run, but there is no India test set.

1. Open https://data.mendeley.com/datasets/gr99ny6b8p and download the dataset zip (`DSWD.zip`, about 250 MB).
2. On Kaggle, go to **Datasets -> New Dataset** and upload the zip. Kaggle unpacks zip files by itself. You can also
   unzip it locally and upload the folder.
3. Give the dataset a title, for example `dwsd-india-waste` (the current upload is
   `shreyanshjain30/dwsd-india-waste`). `configs/gen2/sources.yaml` finds the source by its `DSWD` (or `DWSD`) folder,
   which holds `Train/Image`, `Train/Mask`, `Test/Image` and `Test/Mask`.
4. Keep the visibility on **Private** and click **Create**.
5. In `g2_0_build_dataset`, use **Add Input -> Datasets -> Your Datasets** and attach it, together with the three
   public datasets.

DWSD comes as grey-level semantic masks (`format: semantic_mask`): each pixel value 1-15 is a class, and every
connected region of one value becomes one box (touching objects of the same class merge into one box). The value ->
class names in the config are inferred from the images, because the names published with the dataset do not match
what the masks show (see `configs/gen2/sources.yaml`). All DWSD images with at least one box go to the test split (the
india domain); masks with no class pixels are left out. They are never used for training. If the build stops with an
unknown mask value, its message names the value and a mask file. Add the value to the `dwsd` `mask_values` (and its
name to the class map) in `configs/gen2/sources.yaml`, push, and run the notebook again.

**DWSD can also be attached at evaluation time instead.** Because DWSD is test-only, it does not change the training
data, so the main build does not need it. If `g2_0_build_dataset` ran without DWSD (as the current Gen 2 build did),
attach your DWSD dataset to `g2_6_evaluate` instead. That notebook then builds a separate, DWSD-only test set with
`build_dataset.py --only dwsd` (same config, so the same 7 classes in the same order, and the same `--seed` and
`--max-side` as the main build) into `/kaggle/working/gen2_india`, and scores it with tag `india`. It also runs an MD5
cross-check against every image of the main build and reports any identical files in `build_report_india.json`.
Without DWSD, `g2_6_evaluate` skips india with a message. India never stops the notebook: if DWSD is attached but its
build or its evaluation fails (for example an unknown mask value), the error is printed with a clear warning, india is
left out of the tables, and all, studio and real_world are scored as usual. Fix the config, push and run again to add
india.

## If something fails

- **"The Generation 2 dataset is not attached"**: attach the output of `g2_0_build_dataset` (Add Input ->
  Notebook Output).
- **"no --relocate-from mode yet"**, **"no --only option yet"** or a missing config: the GitHub repository is older
  than these notebooks. Push your local commits and run again.
- **A dead or frozen search session**: attach the stopped version to a new run of the same notebook. The search
  continues where it stopped, and at most one proxy training is lost.
- **A dead or frozen training session** (`g2_1`, `g2_4`, `g2_5`): attach the stopped version to a new run of the same
  notebook. Its salvage cell copies `results/gen2/runs/arm_X/` (with `weights/last.pt` and `results.csv`) from it,
  and `train_final.py --resume` continues from the last finished epoch.
- **A dead or frozen evaluation session**: rerun it with the same inputs. The evaluation is seeded and deterministic.

In short: **if a version stopped or froze, attach its output and run again: it continues.**

## Freeze protection

On 2026-10-08 two runs, `g2_3_search_random` and `g2_5_arm_c`, froze about 55 minutes after training started: no
error, no output, until Kaggle's 12-hour limit stopped them (about 24 GPU hours lost). The most likely cause is the
attached notebook-output mount under `/kaggle/input` (where the dataset was read from) stopping to answer after about an
hour, so the dataloader workers waited forever. Every GPU notebook now has three protections:

1. **Local copy of the dataset.** Before relocating, the attached `gen2_data/` is copied to local disk (`/kaggle/tmp`,
   else `/tmp`; never `/kaggle/working`, so the output does not grow). File names and sizes are checked, and the copy
   time and size are printed. The copy reads the mount too, so it stops with a `TimeoutError` if no file arrives for
   2 minutes, instead of hanging. Training then reads only the local copy. The images are the same bytes, so the results
   are the same as before.
2. **Stall watchdog** (`src/watchdog_run.py`). Every script runs through `run()`, which streams its output and watches
   the output and the run folder (or search log). After 20 minutes with neither, it kills the script and its dataloader
   workers (the whole process group) and starts it again, at most 3 times. A normal error still stops the cell at once.
   The setup cell first runs two watchdog tests (about 10 s) to check that killing works on the Kaggle machine.
3. **Resume.** A restarted `search.py` replays its log; `train_final.py --resume` continues `weights/last.pt` with
   Ultralytics' resume, after checking that the checkpoint's training arguments (data, epochs, imgsz, batch, seed,
   optimizer, the six hyperparameters, ...) equal the ones of this run. Without a `last.pt` it is a normal run. The JSON
   then records `"resumed": true` and `"resumed_from_epoch"`. A resumed training continues the same weights, optimizer
   state and epoch counter, but it is not bit-identical to an uninterrupted one (the data order after the restart
   differs), which matters only for exact reproduction.

Not changed on purpose: the dataloader `workers` (Ultralytics' default 8) and every other training argument, so the
results stay comparable with Arm A and the PSO search, which already ran. Passing a lower `workers` value is a possible
further safeguard for future work.
