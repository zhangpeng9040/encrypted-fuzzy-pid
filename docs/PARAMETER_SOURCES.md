# Controller parameter sources

This file records the source and effective settings of both fuzzy controllers.

The comparator reference is:

Hao Chen, Xiuyang Wang, Chongfeng Wei, Chuan Hu and Xi Zhang. A Dynamic Path Planning and Tracking Control of Autonomous Vehicles: An Integrated Approach Using Improved A*, Fuzzy DWA, and Fuzzy PID. **IEEE Transactions on Automation Science and Engineering**, 23, 2800-2813, 2026. https://doi.org/10.1109/TASE.2026.3653413

Chen et al.'s Section V, Tables III-IV and Eqs. (11)-(12), specify the fuzzy rules, activation-weighted gain corrections and additive base-plus-correction structure. The discussion on printed page 2807 explains that a new vehicle needs baseline PID gains and membership-function scales adapted to its dynamics. The numerical settings below are the effective values used by this study's benchmark runner.

| Item | Proposed fuzzy PID | Chen-based comparator | Source |
|---|---|---|---|
| Rule structure | 7-by-7 scheduling rule table in the released `fuzzy_controller.py` | 5-by-5 correction rule tables in `tase2026_fuzzy_controller.py` | Proposed: this study. Chen: Tables III-IV of Chen et al. |
| Gain update | Multiplicative scheduling of Kp/Ki and error-rate adjustment of Kd | Additive baseline plus corrections | Proposed: this study. Chen: Eqs. (11)-(12). |
| Base gains Kp, Ki, Kd | 1.0, 0.3, 0.001 | 1.0, 0.3, 0.001 | Common platform values used in this study. |
| Input treatment | PID input clipped to e [-15,15], de [-3,3], I [-20,20]; memberships evaluated on e [-10,10], de [-2,2] | Fuzzy inputs clip(e, -16,16)/8 and (2/3)clip(de,-3,3); raw error and derivative in PID products | Present platform implementation. |
| Membership sets | Seven trapezoidal sets per variable, parameters in released code and manuscript Tables 3-5 | Five triangular sets with centers -2,-1,0,1,2 and half-width 1 | Numerical shapes/scales fixed for this platform. |
| Scheduling strengths | alpha_p=0.5, alpha_i=0.3, alpha_d=0.2 | Correction centers listed below | Present platform settings. |
| Correction centers for Kp | Not applicable | -4,-2,0,2,4 | Present platform setting. |
| Correction centers for Ki | Not applicable | -0.04,-0.02,0,0.02,0.04 | Present platform setting. |
| Correction centers for Kd | Not applicable | -1.5,-0.75,0,0.75,1.5 | Present platform setting. |
| Gain limits | 0.5 to 1.5 times each base gain | Kp [0.1,5], Ki [0,1], Kd [0,5] | Present platform setting. |
| Shared execution | dt=0.1 s; 20-sample integral; zero derivative initially; command [-1,1]; same course and Stanley function | Same | Present platform protocol. |

The complete Chen rule tables are implemented in `code/tase2026_fuzzy_controller.py`. Effective platform overrides are applied in `code/run_original_four_twohost.py`; use the settings in the table above together with those runner overrides when reproducing the comparison.


## Vehicle and experiment settings

These values document the released implementation. Most are defined in the Python runners; editing this document does not change an experiment. The runtime partition definitions are in `configs/partitions.json`.

| Setting | Value |
|---|---|
| Sample interval | 0.1 s |
| Integral window | 20 samples; limits [-20, 20] |
| First derivative | Zero at initialization |
| Nominal command range | [-1, 1] |
| Wheelbase | 2.9 m |
| Nominal longitudinal acceleration gain | 3.0 |
| Initial state | x=-180.34 m, y=79.54 m, heading=20 degrees, speed=0 m/s |
| Stanley settings | gain=0.3; softening=20; steering limits ±30 degrees |
| PID base gains Kp, Ki, Kd | [1, 0.3, 0.001] |
| Fuzzy scheduling strengths | [0.5, 0.3, 0.2] |
| Polynomial input coordinate order | ["e", "de", "I"] |
| Processed input ranges | [[-15, 15], [-3, 3], [-20, 20]] |
| Membership input domains | e: [-10, 10]; de: [-2, 2] |
| Training grid per cell | 10 × 8 × 7, 560 samples including interval endpoints |
| Independent approximation evaluation | 12,000 shared uniform inputs; seed 20261020 |
| Full-course reference preparation | 0.01 m interpolation; 20 m lookahead; at most 3000 samples |
| Geometric completion | nearest waypoint index >= last five waypoints and endpoint distance <4m |
| Applied-command scheduling | available response at next sampling boundary; one outstanding request; hold previous command while waiting |

## RNS-CKKS implementation and timing

TenSEAL 0.3.16 uses polynomial modulus degree 8192, coefficient-modulus bit sizes [60, 40, 40, 60], global scale 2^40 and 1 context thread. CKKS-PID packs [e,I,de] against encrypted [Kp,Ki,Kd]. CKKS-Fuzzy-PID packs six features [e,de,I,e+de,e+I,de+I], performs one ciphertext squaring, plaintext weighting and a cloud slot sum. The vehicle retains the secret key. Fuzzy polynomial coefficients and the transmitted region index are public.

CKKS-PID performs 20 warmup requests before formal measurement; startup checks and warmup requests are excluded from the paper's latency statistics. Request latency includes input preparation, encryption where used, serialization, LAN communication, cloud evaluation and response decoding/decryption. Setup/key provisioning and the wait for sampled actuation are excluded. The saved CKKS-PID and CKKS-Fuzzy-PID latency records come from their respective reported runs.

## Operating conditions

Each condition uses 3 repetitions of 1200 samples. Seeds are [1900, 1901, 1902]. Conditions are evaluated separately with frozen controller parameters and coefficients.

| Condition | Change from nominal |
|---|---|
| Nominal | Noise-free measurements; command range [-1,1] |
| Measurement noise | Gaussian speed noise, sigma=0.05 m/s |
| Command limit | [-0.6, 0.6] |
| Disturbance | -0.3 m/s² during 30 ≤ t < 60 s |

## Recorded software environment

| Component | Client / analysis | Server |
|---|---|---|
| Python | 3.13.9 | 3.10.16 |
| NumPy | 2.3.5 | 2.2.6 |
| TenSEAL | 0.3.16 | 0.3.16 |
| Matplotlib | 3.10.8 | Not needed for server evaluation |

Use the versions listed above when reproducing the recorded environment, and an operating system/Python combination with a matching TenSEAL wheel.
