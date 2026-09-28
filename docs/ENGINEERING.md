# Engineering decisions

## Problem and user journey
Make an existing fire/smoke checkpoint inspectable through a small browser product. A user uploads an image, chooses a threshold, reviews labeled boxes, and exports the annotated result. This is an image-inspection prototype, not an emergency monitoring service.

## Decisions and trade-offs
| Decision | Reason | Trade-off |
|---|---|---|
| One detector shared by API and CLI | Avoid inconsistent preprocessing and thresholds | Both entry points depend on the same model interface |
| CPU-first, cached model, serialized inference | Make local use predictable and avoid concurrent shared-model access | Throughput is limited; no load-testing claim |
| Explicit missing-model errors | Distinguish unavailable inference from an empty detection result | Fresh clones require an authorized checkpoint |
| File-size, pixel and image-format checks | Reject unsuitable uploads early | Public hosting would still need authentication and rate limits |
| Normalize EXIF orientation | Keep rendered boxes aligned with the displayed image | Additional decode/preprocessing work |
| Exclude unverified redistributable weights | Keep published source separate from model/data rights | Public clone is not an out-of-the-box detection demo |

## Evidence and next experiments
The local checkpoint produced three fire boxes on one existing sample at threshold 0.35. That validates an execution path, not precision or recall. The API tests exercise request behavior; they do not validate model quality. Next: establish dataset/checkpoint provenance; evaluate a held-out set with hard negatives; report precision/recall across thresholds and latency distributions; inspect overlapping boxes; test camera/video handling only if that becomes the product requirement.

## Three-minute walkthrough
1. Follow model setup in the README, start the application and upload an authorized image.
2. Change the threshold and explain the sensitivity trade-off without claiming calibrated confidence.
3. Export the annotated result; inspect the API response.
4. Show the missing-model and invalid-input behavior covered by tests.
5. State precisely what remains unvalidated before a field deployment.
