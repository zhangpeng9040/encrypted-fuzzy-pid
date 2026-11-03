# Data columns and units

## Main full-course trajectories

`step` is the zero-based sample number, with nominal interval 0.1 s. `x` and `y` are metres; `speed` and `ref` are actual/reference speed in m/s. `error=ref-speed` is the same-sample tracking error. `command` is the applied normalized longitudinal command. `applied_seq` identifies the earlier input that generated it; -1 denotes initialization. `request_sent` indicates a request. `scheduling_lateness_ms` measures client scheduling lateness. Startup samples are included in metrics.

## Requests

`mode` identifies the preserved execution path: `pid_plain`, `pid_packed`, `chen`, `fuzzy`, `poly` or `ckks_poly`, depending on the file. `seq` is the input sample index. `total_ms` measures request latency; encryption/serialization/server/decryption sub-times are available where recorded. `exchange_ms` includes server work and is not pure network RTT. Message byte counts are application-frame sizes. Local monotonic timestamps are not comparable across machines. Main `requests.csv` files contain formal measurements.

## Partition comparison

Only `global_1x1x1`, `coarse_4x4x3`, `base_8x8x6` and `fine_16x16x12` are included, plus a fuzzy reference for tracking. `fit_summary.csv` reports normalized Output RMSE on 12,000 common inputs; `summary.csv` reports closed-loop speed-tracking metrics. `coefficient_bytes` determines coefficient storage in KiB. Retained numeric strings are unchanged from the original records.

## Operating conditions

Twelve files cover `nominal`, `noise`, `actuator` (command limit) and `disturbance`, with repetitions r0–r2 and 1200 samples each. Seeds are 1900, 1901 and 1902. `e_input`, `de_input` and `I_input` describe processed controller inputs. `error` uses true speed, even when measurements contain noise. `applied` is the command. The request CSV includes scenario labels.

## Control-output errors

`data/control_output_errors/matched_errors.csv` reconstructs 1228 matched pairs from the 1229-sample encrypted full-course trajectory. `fitting_error` is saturated plaintext polynomial output minus fuzzy output. `ckks_raw_error` is the decrypted result minus the unsaturated plaintext polynomial output. All matched outputs are unsaturated, allowing reconstruction from the stored commands. Both errors are in normalized control-command units, not m/s.
