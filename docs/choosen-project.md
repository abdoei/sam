#### 23. Prompt-Driven Interactive Segmentation as a Dataset-Annotation Tool

**Objectives**

- Build a small interactive labeling tool around SAM2 (click/box prompts → mask).
- Measure annotation-time savings vs. manual polygon labeling; quantify mask quality vs. ground truth.

**References**

- Ravi et al., "SAM 2: Segment Anything in Images and Videos" (as above).

**Dataset:** a few hundred images from COCO, scored against existing COCO ground-truth masks.

**Baseline:** manual polygon annotation time, or the stock SAM2 demo UI. **Compute tier:** CPU-OK (inference-only).

| M   | Wk  | Student A — Data & Baseline                 | Student B — Model/Algorithm                    | Student C — Evaluation                              | Deliverable                        |
| --- | --- | ------------------------------------------- | ---------------------------------------------- | --------------------------------------------------- | ---------------------------------- |
| 1   | 3   | COCO subset + GT masks prepared             | Design minimal click/box-prompt UI around SAM2 | Define time-to-annotate + mask-IoU protocol         | Proposal + SAM2 inference demo     |
| 2   | 6   | Manual-annotation time measured on a sample | Working interactive tool, first version        | Preliminary IoU-vs-GT comparison                    | Progress report, working tool demo |
| 3   | 9   | Final manual-baseline timing                | Polished tool with usability fixes             | Full time-savings + IoU comparison, small user test | Final report, repo, demo           |
