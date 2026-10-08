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

## What each notebook needs and produces

| Notebook | Accelerator | Attach as input | Produces (in the Output tab) |
|---|---|---|---|
| `g2_0_build_dataset` | None (CPU) | datasets `kneroma/tacotrashdataset`, `humansintheloop/recycling-dataset`, `viswaprakash1990/garbage-detection`, and your private DWSD dataset (optional) | `gen2_data/` (images, labels, lists, `data.yaml`, `data_test_<domain>.yaml`, `split.csv`, `build_report.json`) and `gen2_check.jpg` |
| `g2_1_arm_a` | GPU T4 x2 | output of `g2_0_build_dataset` | `SwarmSort/results/gen2/arm_a.json`, `arm_a_epochs.csv`, `gen2_arm_a_best.pt` |
| `g2_2_search_pso` | GPU T4 x2 | output of `g2_0_build_dataset` (and, to resume, an earlier version of itself) | `SwarmSort/results/gen2/search_pso.json`, `best_pso.json` |
| `g2_3_search_random` | GPU T4 x2 | output of `g2_0_build_dataset` (and, to resume, an earlier version of itself) | `SwarmSort/results/gen2/search_random.json`, `best_random.json` |
| `g2_4_arm_b` | GPU T4 x2 | output of `g2_0_build_dataset`, plus `g2_3_search_random` if `best_random.json` is not committed | `SwarmSort/results/gen2/arm_b.json`, `arm_b_epochs.csv`, `gen2_arm_b_best.pt` |
| `g2_5_arm_c` | GPU T4 x2 | output of `g2_0_build_dataset`, plus `g2_2_search_pso` if `best_pso.json` is not committed | `SwarmSort/results/gen2/arm_c.json`, `arm_c_epochs.csv`, `gen2_arm_c_best.pt` |
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
- **A dead search session**: attach the failed version to a new run of the same notebook. The search continues where
  it stopped, and at most one proxy training is lost.
- **A dead training or evaluation session**: rerun it with the same inputs. Every run is seeded and deterministic.
