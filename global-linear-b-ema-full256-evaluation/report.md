# G-L+EMA and Official PDE-B Full-256 Evaluation

## Scope

This report evaluates the completed 100-epoch **Global Linear TTT PDE-B** run
using its independently selected raw and EMA checkpoints:

- branch: `experiment/global-linear-b-ema-dual-checkpoints-v1`
- training config: `pdes_global-linear-ttt-b-ema_256_full.yaml`
- training data: `pde-transformer-ape2d-full`
- resolution: 256 x 256
- parameters: 130,968,608 (130.97M)
- token mixer: Global Linear TTT
- EMA decay: 0.999
- raw checkpoint: `raw-best.ckpt`, source epoch 96
- EMA checkpoint: `ema-best.ckpt`, source epoch 99
- raw validation MSE: 0.000427677
- EMA validation MSE: 0.000345411

Both evaluations use one GTX 1080 Ti, batch 8, FP32, and one 29-transition
autoregressive rollout per trajectory. Fast weights are recreated from the
learned initial weights on every model call; no temporal cache is used.

The official Attention baseline is the public
[PDE-B `mc-b` checkpoint](https://huggingface.co/thuerey-group/pde-transformer/tree/main/mc-b),
with 130,591,568 parameters. It is evaluated with the same evaluator, data,
batch size, precision, GPU type, and rollout protocol as our model.

## Evaluation Status

| Evaluation | Status | Time |
|---|---|---:|
| Raw weights, Full-256 strict test | Complete | 1117.6 s |
| EMA weights, Full-256 strict test | Complete | 1115.9 s |
| Raw weights, generated ID/OOD test | Complete | 364.5 s |
| EMA weights, generated ID/OOD test | Complete | 370.3 s |
| Official PDE-B, Full-256 strict test | Complete | 1071.8 s |
| Official PDE-B, generated ID/OOD test | Complete | 358.6 s |

## Full-256 Strict Test

The strict Full-256 test contains 16 PDE families and 850 held-out
trajectories from `datasets_ape2d_full`.

| Aggregate | Weights | @1 | @10 | @20 | @29 |
|---|---|---:|---:|---:|---:|
| Macro | Raw | 0.039935 | 0.321817 | 0.627383 | 0.760342 |
| Macro | **EMA** | **0.037259** | **0.290256** | **0.559227** | **0.676671** |
| Micro | Raw | 0.045353 | 0.309973 | 0.549783 | 0.666280 |
| Micro | **EMA** | **0.042551** | **0.271492** | **0.478128** | **0.574911** |

Relative to raw-best, EMA-best reduces Full macro nRMSE by 6.70%, 9.81%,
10.86%, and 11.00% at the four horizons. EMA wins on 14/16 PDEs at @1 and
13/16 PDEs at @10, @20, and @29, so the aggregate gain is broad but not
universal.

### Full-256 Per-PDE Results

| PDE | Raw @1 | EMA @1 | Raw @10 | EMA @10 | Raw @20 | EMA @20 | Raw @29 | EMA @29 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| diff | 0.036945 | 0.034747 | 0.164627 | 0.145730 | 0.233928 | 0.197761 | 0.282052 | 0.230441 |
| burgers | 0.021435 | 0.017487 | 0.107687 | 0.047403 | 0.237510 | 0.088434 | 0.373787 | 0.128061 |
| kdv | 0.052584 | 0.049566 | 0.222693 | 0.188714 | 0.383717 | 0.306282 | 0.541133 | 0.412168 |
| ks | 0.014064 | 0.011473 | 0.234742 | 0.234389 | **0.762253** | 0.764868 | **1.152410** | 1.164000 |
| fisher | 0.022280 | 0.021874 | 0.389994 | 0.344191 | **0.591427** | 0.618099 | **0.540850** | 0.577934 |
| gs_alpha | 0.018544 | 0.017284 | **0.232307** | 0.256578 | 0.650066 | 0.486494 | 0.936681 | 0.726642 |
| gs_beta | **0.022242** | 0.024801 | 0.369481 | 0.338162 | 0.729245 | 0.562253 | 0.875378 | 0.610159 |
| gs_gamma | 0.026966 | 0.018209 | 0.351748 | 0.317604 | 0.703494 | 0.672374 | 0.891921 | 0.875365 |
| gs_delta | 0.015569 | 0.013481 | **0.470351** | 0.479795 | 1.019660 | 1.013800 | 0.966438 | 0.959185 |
| gs_epsilon | 0.012151 | 0.009546 | 0.161534 | 0.078853 | 0.279010 | 0.167909 | 0.444420 | 0.306797 |
| gs_theta | 0.010022 | 0.007271 | 0.518322 | 0.516022 | 0.954474 | 0.944445 | 0.984549 | 0.969552 |
| gs_iota | 0.011086 | 0.007733 | 0.156505 | 0.147443 | 0.619054 | 0.609828 | 0.671096 | 0.644384 |
| gs_kappa | 0.017706 | 0.015065 | 0.204107 | 0.186720 | 0.539944 | 0.525474 | 0.787164 | 0.782588 |
| sh | 0.063395 | 0.057624 | 0.491073 | 0.439685 | 0.686210 | 0.581706 | 0.728374 | 0.619163 |
| decay_turb | **0.237360** | 0.237533 | **0.673707** | 0.678764 | **0.733671** | 0.756314 | **0.785848** | 0.794140 |
| kolm_flow | 0.056607 | 0.052445 | 0.400185 | 0.244046 | 0.914469 | 0.651587 | 1.203370 | 1.026160 |

Bold raw values identify the PDE/horizon combinations where raw-best is
better; all other pairs favor EMA-best.

## Generated ID/OOD Test

The generated matrix contains 17 PDE families, three parameter conditions,
and three unseen simulation seeds per condition: 153 trajectories in total.
Because each PDE-condition pair has the same number of trajectories, macro
and micro values are equal.

| Condition | Weights | @1 | @10 | @20 | @29 |
|---|---|---:|---:|---:|---:|
| ID | Raw | 0.046283 | 0.395229 | 0.647423 | 0.759736 |
| ID | **EMA** | **0.044338** | **0.339774** | **0.562416** | **0.678587** |
| OOD-low | Raw | 0.076776 | 0.483909 | 0.771718 | 0.890334 |
| OOD-low | **EMA** | **0.074204** | **0.472991** | **0.730902** | **0.864799** |
| OOD-high | Raw | 0.063353 | 0.420342 | 0.646357 | 0.784225 |
| OOD-high | **EMA** | **0.062215** | **0.403013** | **0.574061** | **0.729087** |
| All | Raw | 0.062137 | 0.433160 | 0.688499 | 0.811432 |
| All | **EMA** | **0.060253** | **0.405259** | **0.622460** | **0.757491** |

EMA improves every condition-level aggregate. Per-PDE EMA win counts at
@1/@10/@20/@29 are:

| Condition | EMA wins |
|---|---|
| ID | 13/17, 14/17, 16/17, 14/17 |
| OOD-low | 12/17, 9/17, 11/17, 10/17 |
| OOD-high | 12/17, 11/17, 12/17, 14/17 |

### ID Per-PDE Results

| PDE | Raw @1 | EMA @1 | Raw @10 | EMA @10 | Raw @20 | EMA @20 | Raw @29 | EMA @29 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| diff | 0.025407 | 0.026249 | 0.072745 | 0.063791 | 0.108629 | 0.103692 | 0.120182 | 0.128050 |
| hyp | 0.112919 | 0.107967 | 0.427999 | 0.383962 | 0.645773 | 0.551403 | 1.036210 | 0.846599 |
| burgers | 0.032167 | 0.030807 | 0.204610 | 0.187104 | 0.361068 | 0.322133 | 0.478583 | 0.400771 |
| kdv | 0.029696 | 0.027423 | 0.178563 | 0.133405 | 0.450659 | 0.348600 | 0.635853 | 0.493855 |
| ks | 0.012360 | 0.010086 | 0.335263 | 0.152606 | 0.929249 | 0.625752 | 1.240830 | 1.005050 |
| fisher | 0.013816 | 0.018479 | 0.197489 | 0.237757 | 0.225226 | 0.288224 | 0.183395 | 0.279308 |
| gs_alpha | 0.029511 | 0.028737 | 0.493572 | 0.293340 | 0.837836 | 0.519964 | 1.059380 | 0.744247 |
| gs_beta | 0.100189 | 0.095979 | 1.334970 | 1.271620 | 1.051010 | 0.972738 | 1.008920 | 0.939450 |
| gs_gamma | 0.045417 | 0.039173 | 0.388574 | 0.410042 | 0.786333 | 0.773545 | 0.925184 | 0.981052 |
| gs_delta | 0.018655 | 0.020546 | 0.559799 | 0.548687 | 1.096570 | 1.095050 | 1.009430 | 1.000360 |
| gs_epsilon | 0.010364 | 0.009458 | 0.155375 | 0.066967 | 0.281876 | 0.172585 | 0.400537 | 0.302056 |
| gs_theta | 0.016242 | 0.015726 | 0.591921 | 0.580006 | 1.022290 | 1.014490 | 1.039600 | 1.003550 |
| gs_iota | 0.043028 | 0.046168 | 0.174647 | 0.179195 | 0.590844 | 0.577521 | 0.615970 | 0.601714 |
| gs_kappa | 0.025936 | 0.022700 | 0.186326 | 0.143803 | 0.488747 | 0.462816 | 0.731793 | 0.712985 |
| sh | 0.082521 | 0.075060 | 0.483541 | 0.426005 | 0.820446 | 0.690894 | 0.895682 | 0.733715 |
| decay_turb | 0.140081 | 0.134809 | 0.408757 | 0.397453 | 0.365676 | 0.325428 | 0.345556 | 0.280501 |
| kolm_flow | 0.048500 | 0.044388 | 0.524746 | 0.300417 | 0.943959 | 0.716246 | 1.188410 | 1.082720 |

### OOD-Low Per-PDE Results

| PDE | Raw @1 | EMA @1 | Raw @10 | EMA @10 | Raw @20 | EMA @20 | Raw @29 | EMA @29 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| diff | 0.127785 | 0.128961 | 0.393209 | 0.401686 | 0.471887 | 0.487375 | 0.529969 | 0.550426 |
| hyp | 0.107203 | 0.095897 | 0.410412 | 0.378498 | 0.636557 | 0.550028 | 0.953550 | 0.747683 |
| burgers | 0.041776 | 0.040709 | 0.217238 | 0.202345 | 0.372194 | 0.336963 | 0.488134 | 0.415368 |
| kdv | 0.029715 | 0.027443 | 0.178588 | 0.133449 | 0.450684 | 0.348677 | 0.635820 | 0.493901 |
| ks | 0.014568 | 0.014402 | 0.260847 | 0.348894 | 1.029660 | 0.908375 | 1.337910 | 1.569160 |
| fisher | 0.063719 | 0.062906 | 0.336564 | 0.349855 | 0.414144 | 0.457795 | 0.375212 | 0.443277 |
| gs_alpha | 0.023141 | 0.023289 | 0.240283 | 0.207424 | 0.544524 | 0.637768 | 0.788395 | 0.852745 |
| gs_beta | 0.150418 | 0.148165 | 0.981972 | 1.010580 | 1.153580 | 1.147020 | 1.218280 | 1.161880 |
| gs_gamma | 0.091748 | 0.087586 | 0.779848 | 0.789025 | 1.136310 | 1.134110 | 1.261230 | 1.256720 |
| gs_delta | 0.046085 | 0.047127 | 0.832408 | 0.831965 | 1.202770 | 1.197220 | 1.090290 | 1.086400 |
| gs_epsilon | 0.015993 | 0.016342 | 0.177211 | 0.246285 | 0.297389 | 0.363077 | 0.512977 | 0.591809 |
| gs_theta | 0.047983 | 0.047833 | 0.875909 | 0.888332 | 1.140400 | 1.157410 | 1.109460 | 1.133900 |
| gs_iota | 0.062933 | 0.065616 | 0.394137 | 0.397939 | 0.831457 | 0.835092 | 0.870159 | 0.877132 |
| gs_kappa | 0.043825 | 0.037870 | 0.386541 | 0.312335 | 0.710854 | 0.661507 | 0.860565 | 0.820696 |
| sh | 0.119216 | 0.110189 | 0.667127 | 0.582409 | 1.195810 | 0.842444 | 1.366940 | 1.116880 |
| decay_turb | 0.208763 | 0.199227 | 0.629236 | 0.617379 | 0.587315 | 0.579226 | 0.531457 | 0.498893 |
| kolm_flow | 0.110317 | 0.107903 | 0.464927 | 0.342447 | 0.943683 | 0.781243 | 1.205330 | 1.084710 |

### OOD-High Per-PDE Results

| PDE | Raw @1 | EMA @1 | Raw @10 | EMA @10 | Raw @20 | EMA @20 | Raw @29 | EMA @29 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| diff | 0.064207 | 0.058630 | 0.259570 | 0.226764 | 0.370930 | 0.297689 | 0.463976 | 0.358331 |
| hyp | 0.122917 | 0.119421 | 0.500677 | 0.453706 | 0.721413 | 0.613636 | 1.167210 | 0.952793 |
| burgers | 0.027229 | 0.025493 | 0.198251 | 0.177761 | 0.354664 | 0.310854 | 0.473798 | 0.389493 |
| kdv | 0.029674 | 0.027400 | 0.178536 | 0.133356 | 0.450632 | 0.348515 | 0.635890 | 0.493805 |
| ks | 0.021135 | 0.020608 | 0.358469 | 0.366009 | 0.851966 | 0.854072 | 1.073100 | 1.040730 |
| fisher | 0.031544 | 0.036467 | 0.266141 | 0.311007 | 0.204841 | 0.256803 | 0.160603 | 0.226832 |
| gs_alpha | 0.037667 | 0.036589 | 0.711026 | 0.525718 | 1.103760 | 0.867177 | 1.268650 | 1.064000 |
| gs_beta | 0.216352 | 0.211667 | 0.621557 | 0.878276 | 0.608670 | 0.359457 | 0.588245 | 0.866097 |
| gs_gamma | 0.118777 | 0.114719 | 0.565862 | 0.568767 | 0.881468 | 0.891808 | 0.956515 | 0.954411 |
| gs_delta | 0.036946 | 0.037473 | 0.916096 | 0.908509 | 0.956688 | 0.940308 | 0.965673 | 0.950244 |
| gs_epsilon | 0.021378 | 0.019685 | 0.353757 | 0.243324 | 0.495786 | 0.427822 | 0.826717 | 0.727628 |
| gs_theta | 0.052935 | 0.051054 | 0.745616 | 0.728253 | 1.039460 | 1.050560 | 1.039940 | 1.036450 |
| gs_iota | 0.044015 | 0.048654 | 0.216075 | 0.222442 | 0.622041 | 0.611773 | 0.655287 | 0.621615 |
| gs_kappa | 0.033952 | 0.037408 | 0.214621 | 0.273217 | 0.541283 | 0.602976 | 0.806646 | 0.851954 |
| sh | 0.072290 | 0.071172 | 0.409253 | 0.363703 | 0.691778 | 0.576138 | 0.770291 | 0.638147 |
| decay_turb | 0.110762 | 0.112135 | 0.336497 | 0.306232 | 0.304067 | 0.242070 | 0.311364 | 0.236533 |
| kolm_flow | 0.035228 | 0.029088 | 0.293802 | 0.164178 | 0.788612 | 0.507384 | 1.167930 | 0.985419 |

The source values remain available in each checkpoint's
`results_conditions_cache_off.csv` and `results_trajectories_cache_off.csv`.

## PDE-S to PDE-B Scale Comparison

The previous PDE-S EMA run used the same Full-256 and generated ID/OOD
protocol. This comparison therefore measures the observed result of scaling
the G-L architecture from 33.36M to 130.97M parameters, while still reflecting
independent training runs rather than a controlled multi-seed scale study.

| Test | Model | Parameters | @1 | @10 | @20 | @29 |
|---|---|---:|---:|---:|---:|---:|
| Full macro | PDE-S EMA | 33.36M | 0.041823 | 0.350419 | 0.633739 | 0.762153 |
| Full macro | **PDE-B EMA** | **130.97M** | **0.037259** | **0.290256** | **0.559227** | **0.676671** |
| Generated all | PDE-S EMA | 33.36M | 0.062968 | 0.437566 | 0.701277 | 0.859343 |
| Generated all | **PDE-B EMA** | **130.97M** | **0.060253** | **0.405259** | **0.622460** | **0.757491** |

PDE-B improves Full macro nRMSE by 10.91%, 17.17%, 11.76%, and 11.22%
relative to PDE-S at @1/@10/@20/@29. On the generated aggregate, the
reductions are 4.31%, 7.38%, 11.24%, and 11.85%.

## Official PDE-B Checkpoint: Matched Evaluation

This section uses the actual public `mc-b` checkpoint rather than copying the
paper table. The official model has the original shifted-window Attention
mixer and 130.59M parameters. Lower nRMSE is better.

### Aggregate Comparison

| Test | Model | @1 | @10 | @20 | @29 |
|---|---|---:|---:|---:|---:|
| Full macro | Official PDE-B | 0.038883 | 0.328779 | 0.603770 | 0.725117 |
| Full macro | **Our G-L PDE-B EMA** | **0.037259** | **0.290256** | **0.559227** | **0.676671** |
| Full micro | Official PDE-B | 0.043389 | 0.297896 | 0.509100 | 0.612404 |
| Full micro | **Our G-L PDE-B EMA** | **0.042551** | **0.271492** | **0.478128** | **0.574911** |
| Generated all | Official PDE-B | 0.063953 | 0.451980 | 0.669548 | 0.778043 |
| Generated all | **Our G-L PDE-B EMA** | **0.060253** | **0.405259** | **0.622460** | **0.757491** |

On Full macro, G-L EMA reduces nRMSE relative to the official checkpoint by
4.18%, 11.72%, 7.38%, and 6.68% at @1/@10/@20/@29. It also improves all four
aggregate horizons on the generated test matrix.

### Full-256 Per-PDE Comparison

| PDE | Official @1 | G-L EMA @1 | Official @10 | G-L EMA @10 | Official @20 | G-L EMA @20 | Official @29 | G-L EMA @29 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| diff | 0.034819 | **0.034747** | 0.151643 | **0.145730** | 0.205008 | **0.197761** | 0.238603 | **0.230441** |
| burgers | **0.016209** | 0.017487 | 0.057597 | **0.047403** | 0.108812 | **0.088434** | 0.155055 | **0.128061** |
| kdv | **0.045624** | 0.049566 | **0.185887** | 0.188714 | 0.308251 | **0.306282** | 0.418382 | **0.412168** |
| ks | 0.014528 | **0.011473** | 0.275380 | **0.234389** | 0.803680 | **0.764868** | 1.179920 | **1.164000** |
| fisher | 0.022960 | **0.021874** | 0.372422 | **0.344191** | **0.610057** | 0.618099 | 0.603012 | **0.577934** |
| gs_alpha | 0.026946 | **0.017284** | 0.262141 | **0.256578** | 0.584065 | **0.486494** | 0.857750 | **0.726642** |
| gs_beta | 0.029801 | **0.024801** | 0.358429 | **0.338162** | 0.585985 | **0.562253** | 0.658579 | **0.610159** |
| gs_gamma | 0.025579 | **0.018209** | 0.441532 | **0.317604** | 0.738177 | **0.672374** | 0.911252 | **0.875365** |
| gs_delta | 0.014123 | **0.013481** | 0.579079 | **0.479795** | 1.039060 | **1.013800** | 0.967337 | **0.959185** |
| gs_epsilon | 0.014039 | **0.009546** | 0.152414 | **0.078853** | 0.337578 | **0.167909** | 0.571584 | **0.306797** |
| gs_theta | 0.009220 | **0.007271** | 0.583492 | **0.516022** | 0.981926 | **0.944445** | 1.001700 | **0.969552** |
| gs_iota | 0.009491 | **0.007733** | 0.167839 | **0.147443** | 0.641672 | **0.609828** | 0.654200 | **0.644384** |
| gs_kappa | 0.016525 | **0.015065** | 0.223978 | **0.186720** | 0.610877 | **0.525474** | 0.843423 | **0.782588** |
| sh | 0.064033 | **0.057624** | 0.454652 | **0.439685** | 0.624690 | **0.581706** | 0.673593 | **0.619163** |
| decay_turb | **0.220611** | 0.237533 | 0.690879 | **0.678764** | **0.729807** | 0.756314 | **0.747844** | 0.794140 |
| kolm_flow | 0.057620 | **0.052445** | 0.303099 | **0.244046** | 0.750680 | **0.651587** | 1.119650 | **1.026160** |

G-L EMA wins 14/16 PDEs at @20. The official checkpoint is better on
`fisher` and `decay_turb` at that horizon.

### Generated ID/OOD Condition Comparison

| Condition | Model | @1 | @10 | @20 | @29 |
|---|---|---:|---:|---:|---:|
| ID | Official PDE-B | 0.046006 | 0.374603 | 0.578076 | 0.704193 |
| ID | **Our G-L PDE-B EMA** | **0.044338** | **0.339774** | **0.562416** | **0.678587** |
| OOD-low | Official PDE-B | 0.078611 | 0.485216 | **0.720920** | **0.811696** |
| OOD-low | **Our G-L PDE-B EMA** | **0.074204** | **0.472991** | 0.730902 | 0.864799 |
| OOD-high | Official PDE-B | 0.067242 | 0.496122 | 0.709649 | 0.818241 |
| OOD-high | **Our G-L PDE-B EMA** | **0.062215** | **0.403013** | **0.574061** | **0.729087** |

G-L EMA improves every ID and OOD-high aggregate horizon. OOD-low is mixed:
G-L EMA is better at @1 and @10, while the official Attention checkpoint is
better at @20 and @29. Each condition contains only three trajectories per
PDE, so these generated-condition comparisons have higher uncertainty than
the 850-trajectory Full test.

At the representative @20 horizon, the per-PDE comparison is:

| PDE | ID Official | ID G-L EMA | OOD-low Official | OOD-low G-L EMA | OOD-high Official | OOD-high G-L EMA |
|---|---:|---:|---:|---:|---:|---:|
| diff | 0.105416 | 0.103692 | 0.504173 | 0.487375 | 0.278702 | 0.297689 |
| hyp | 0.576095 | 0.551403 | 0.522480 | 0.550028 | 0.640557 | 0.613636 |
| burgers | 0.337177 | 0.322133 | 0.351668 | 0.336963 | 0.325761 | 0.310854 |
| kdv | 0.417334 | 0.348600 | 0.417424 | 0.348677 | 0.417235 | 0.348515 |
| ks | 0.606551 | 0.625752 | 0.334173 | 0.908375 | 0.927538 | 0.854072 |
| fisher | 0.155671 | 0.288224 | 0.434482 | 0.457795 | 0.174802 | 0.256803 |
| gs_alpha | 0.592217 | 0.519964 | 0.616926 | 0.637768 | 0.974045 | 0.867177 |
| gs_beta | 0.942270 | 0.972738 | 1.135440 | 1.147020 | 2.355150 | 0.359457 |
| gs_gamma | 0.786218 | 0.773545 | 1.117520 | 1.134110 | 0.917660 | 0.891808 |
| gs_delta | 1.096330 | 1.095050 | 1.202230 | 1.197220 | 0.938493 | 0.940308 |
| gs_epsilon | 0.303498 | 0.172585 | 0.511093 | 0.363077 | 0.421955 | 0.427822 |
| gs_theta | 1.049990 | 1.014490 | 1.155120 | 1.157410 | 1.055600 | 1.050560 |
| gs_iota | 0.589050 | 0.577521 | 0.845611 | 0.835092 | 0.623437 | 0.611773 |
| gs_kappa | 0.500666 | 0.462816 | 0.658775 | 0.661507 | 0.637626 | 0.602976 |
| sh | 0.647833 | 0.690894 | 1.104660 | 0.842444 | 0.577259 | 0.576138 |
| decay_turb | 0.320055 | 0.325428 | 0.569748 | 0.579226 | 0.253570 | 0.242070 |
| kolm_flow | 0.800918 | 0.716246 | 0.774121 | 0.781243 | 0.544647 | 0.507384 |

All @1/@10/@20/@29 per-PDE values and trajectory-level samples are retained
in the linked condition and trajectory CSV files below.

## Artifacts

### Raw-best

- [Full CSV](raw/full_test/results_cache_off.csv)
- [Full JSON](raw/full_test/results_cache_off.json)
- [Full summary](raw/full_test/summary.json)
- [Full log](raw/full_test.log)
- [ID/OOD aggregate CSV](raw/id_ood/results_cache_off.csv)
- [ID/OOD condition CSV](raw/id_ood/results_conditions_cache_off.csv)
- [ID/OOD trajectory CSV](raw/id_ood/results_trajectories_cache_off.csv)
- [ID/OOD summary](raw/id_ood/summary.json)
- [ID/OOD log](raw/id_ood.log)

### EMA-best

- [Full CSV](ema/full_test/results_cache_off.csv)
- [Full JSON](ema/full_test/results_cache_off.json)
- [Full summary](ema/full_test/summary.json)
- [Full log](ema/full_test.log)
- [ID/OOD aggregate CSV](ema/id_ood/results_cache_off.csv)
- [ID/OOD condition CSV](ema/id_ood/results_conditions_cache_off.csv)
- [ID/OOD trajectory CSV](ema/id_ood/results_trajectories_cache_off.csv)
- [ID/OOD summary](ema/id_ood/summary.json)
- [ID/OOD log](ema/id_ood.log)

### Official PDE-B (`mc-b`)

- [Full CSV](official_mc_b/full_test/results_cache_off.csv)
- [Full JSON](official_mc_b/full_test/results_cache_off.json)
- [Full summary](official_mc_b/full_test/summary.json)
- [Full log](official_mc_b/full_test.log)
- [ID/OOD aggregate CSV](official_mc_b/id_ood/results_cache_off.csv)
- [ID/OOD condition CSV](official_mc_b/id_ood/results_conditions_cache_off.csv)
- [ID/OOD trajectory CSV](official_mc_b/id_ood/results_trajectories_cache_off.csv)
- [ID/OOD summary](official_mc_b/id_ood/summary.json)
- [ID/OOD log](official_mc_b/id_ood.log)

## Conclusion

For this PDE-B run, EMA-best is the correct default inference checkpoint: it
improves every aggregate metric, and the advantage grows toward longer Full
rollout horizons. Scaling G-L from PDE-S to PDE-B also improves all matched
aggregate horizons, although the result should not yet be presented as a
multi-seed scaling law. Under the same evaluator, our G-L PDE-B EMA also
outperforms the public official PDE-B checkpoint on every Full macro and micro
horizon, while the generated OOD-low long-horizon result remains an important
exception.

## Comparison with Published PDE-B at Step 20

The following table compares our Full-256 strict-test EMA result with the
published PDE-B row in Table 10 of the
[PDE-Transformer paper](https://openreview.net/pdf/db8006d123d010d7f615b94cde6b8e5d0fd49166.pdf).
Both rows report `nRMSE20` after 20 autoregressive steps on the 16 pre-training
PDE test datasets. Lower is better; the better value in each column is bold.

| Method | diff | burgers | kdv | ks | fisher | gs-alpha | gs-beta | gs-gamma | gs-delta | gs-epsilon | gs-theta | gs-iota | gs-kappa | sh | decay-turb | kolm-flow | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Published PDE-B Attention | 0.2090 | 0.1142 | 0.3204 | 0.8599 | 0.6131 | 0.6186 | 0.6110 | 0.7558 | 1.0253 | 0.3636 | 0.9902 | 0.6432 | 0.6435 | 0.6354 | 0.7333 | 0.8005 | 0.621063 |
| Official PDE-B checkpoint, our evaluator | 0.205008 | 0.108812 | 0.308251 | 0.803680 | **0.610057** | 0.584065 | 0.585985 | 0.738177 | 1.039060 | 0.337578 | 0.981926 | 0.641672 | 0.610877 | 0.624690 | **0.729807** | 0.750680 | 0.603770 |
| **Our G-L PDE-B EMA** | **0.197761** | **0.088434** | **0.306282** | **0.764868** | 0.618099 | **0.486494** | **0.562253** | **0.672374** | **1.013800** | **0.167909** | **0.944445** | **0.609828** | **0.525474** | **0.581706** | 0.756314 | **0.651587** | **0.559227** |

Our G-L PDE-B EMA is better on 14 of 16 PDEs and reduces the unweighted macro
`nRMSE20` by 9.96% (`0.621063` to `0.559227`). The published PDE-B remains
better on `fisher` and `decay-turb`. This is a result comparison under the
matched public test protocol, not a claim that every training implementation
detail is identical to the paper's original run. The public checkpoint's
re-evaluated macro value is 0.603770 rather than the paper table's 0.621063,
which confirms that copied paper numbers and checkpoint re-evaluation should
be reported separately.
