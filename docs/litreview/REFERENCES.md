# References — Crowd Crush / Stampede Precursor Detection

**Verification key**
- `[F]` full text retrieved and read locally (PDF in `pdfs/`)
- `[M]` metadata only (title/venue/year confirmed from a search result or landing page; abstract-level claims only)
- `[G]` gated — needs institutional access before citing in the paper
- `[A]` abstract-level only, full text unobtainable — must not carry a load-bearing claim

> Every entry below was surfaced from a live search, not from recall. **Re-verify each
> `[M]`/`[G]` entry against the publisher record before submission** — author lists and page
> numbers here are not authoritative.

## A. Foundational crowd physics (pre-2019, cite as foundational)

| # | Reference | V |
|---|---|---|
| A1 | Johansson, A., Helbing, D., Al-Abideen, H.Z., Al-Bosta, S. *From Crowd Dynamics to Crowd Safety: A Video-Based Analysis.* Advances in Complex Systems, 2008. arXiv:0810.4590 — **defines crowd pressure = density x variance of VELOCITIES; critical 0.02 s^-2; flow < 0.8 ped/m/s**. NB: the body text says "variance of speeds" loosely, but the conclusions give the formal definition as "the density times the variance of velocities" — use the vector form; the two differ sharply under counterflow | [F] |
| A2 | Helbing, D., Johansson, A., Al-Abideen, H.Z. *Crowd turbulence: The physics of crowd disasters.* 2007 | [M] |
| A3 | Mehran, R., Oyama, A., Shah, M. *Abnormal Crowd Behavior Detection using Social Force Model.* CVPR 2009. https://ieeexplore.ieee.org/document/5206641/ — canonical baseline; UMN dataset | [M] |
| A4 | Chan, A.B., Liang, Z.S.J., Vasconcelos, N. *Privacy preserving crowd monitoring: Counting people without people models or tracking.* CVPR 2008 | [M] |
| A5 | Helbing, D., et al. *How simple rules determine pedestrian behavior and crowd disasters.* PNAS. https://pmc.ncbi.nlm.nih.gov/articles/PMC3084058/ | [M] |
| A6 | Golas, A., et al. *Continuum Modeling of Crowd Turbulence.* Physical Review E, 2014 | [M] |
| A7 | *Crowd Behavior Analysis: A Review where Physics meets Biology.* arXiv:1511.06586 | [F] |

## B. Crowd disasters — post-incident analyses (2023-2026)

| # | Reference | V |
|---|---|---|
| B1 | *Unraveling the causes of the Seoul Halloween crowd-crush disaster.* PLOS One, 2024. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0306764 — density 7.57 ped/m^2 avg, 9.95 max; pressure 1063 N/m | [M] |
| B2 | Hwang et al. *Virtual Reenactment of the Itaewon Crowd Crush Using Kinodynamic Simulation.* Computer Animation and Virtual Worlds, 2025 | [M] |
| B3 | *158 Deaths at Halloween Night: An AcciMap analysis of the 2022 Itaewon crowd crush.* 2025 | [M] |
| B4 | *Reviewing the Itaewon Halloween crowd crush, Korea 2022: Qualitative content analysis.* 2023. PMC10687381 | [M] |
| B5 | *Prediction and prevention of crowd-crush accidents using crowd-density simulation based on Unity engine.* Discover Applied Sciences, 2024 | [M] |
| B6 | *Forward propagation of a push through a row of people.* arXiv:2503.19104, 2025 | [M] |
| B7 | *Modelling crowd pressure and turbulence through a mixed-type continuum approach.* Transportmetrica B, 2024 | [M] |
| B8 | *Can high-density human collective motion be forecasted by spatiotemporal fluctuations?* arXiv:1809.07875 | [M] |

## C. Stampede / crowd-disaster detection systems (direct competitors)

| # | Reference | V |
|---|---|---|
| C1 | *Stampede detector based on deep learning models using dense optical flow.* Engineering Applications of AI, 2025. https://www.sciencedirect.com/science/article/pii/S0952197624020992 | [G] |
| C2 | *Stampede detection and crowd analysis using CNN-LSTM and Farneback optical flow.* Scientific Reports, 2026 | [G] |
| C3 | *Stampede Alert Clustering Algorithmic System Based on Tiny-Scale Strengthened DETR.* arXiv:2404.10359 | [F] |
| C4 | *Development of a Risk Space Prediction Model Based on CCTV Images Using Deep Learning: Crowd Collapse.* IJASEIT 14(1), 2024 | [M] |
| C5 | *Computer vision based crowd disaster avoidance system: A survey.* Int. J. Disaster Risk Reduction | [M] |
| C6 | *Drishti AI-Event Guardian: Intelligent Real-Time Crowd Monitoring and Emergency Response for Mass Gatherings.* arXiv:2606.05185 | [M] |
| C7 | *A Review of Crowd Abnormal Behavior Recognition Technology Based on Computer Vision.* Preprints, 2024 | [M] |

## D. Crowd anomaly detection (2019-2026)

| # | Reference | V |
|---|---|---|
| D1 | *Deep crowd anomaly detection: state-of-the-art, challenges, and future research directions.* Artificial Intelligence Review, 2024. doi:10.1007/s10462-024-11092-8 | [G] |
| D2 | *Crowd anomaly estimation and detection: A review.* 2024. ScienceDirect S2773186324000999 | [G] |
| D3 | *VelocityNet: Real-Time Crowd Anomaly Detection via Person-Specific Velocity Analysis.* arXiv:2510.18187 | [F] |
| D4 | *Robust crowd anomaly detection via hybrid ensemble learning for real-world surveillance.* Scientific Reports, 2025 | [M] |
| D5 | *Anomaly detection in crowd scenes via cross trajectories.* Applied Intelligence, 2025 | [M] |
| D6 | *Detecting abnormal crowd behaviors based on the div-curl characteristics of flow fields.* Pattern Recognition. S003132031830414X | [G] |
| D7 | *Understanding crowd flow patterns using active-Langevin model.* Pattern Recognition, 2021 | [M] |
| D8 | *Measuring Crowd Collectiveness via Global Motion Correlation.* ICCVW 2019 | [M] |
| D9 | *Crowd Density Estimation via Global Crowd Collectiveness Metric.* Drones 8(11), 2024 | [M] |
| D10 | *An Adaptive Training-less System for Anomaly Detection in Crowd Scenes.* arXiv:1906.00705 | [M] |
| D11 | *Spatio-temporal Texture Modelling for Real-time Crowd Anomaly Detection.* CVIU | [M] |

## E. Video anomaly detection — surveys and methods (2021-2026)

| # | Reference | V |
|---|---|---|
| E1 | Abdalla, M., Javed, S., Al Radi, M., Ulhaq, A., Werghi, N. *Video Anomaly Detection in 10 Years: A Survey and Outlook.* arXiv:2405.19387, 2024 | [F] |
| E2 | *Networking Systems for Video Anomaly Detection: A Tutorial and Survey.* arXiv:2405.10347 | [F] |
| E3 | *Deep Learning for Video Anomaly Detection: A Review.* arXiv:2409.05383 | [F] |
| E4 | *Anomaly Detection using Edge Computing in Video Surveillance System: Review.* arXiv:2107.02778 | [F] |
| E5 | *Privacy-Preserving Video Anomaly Detection: A Survey.* arXiv:2411.14565 | [M] |
| E6 | *Generalized Video Anomaly Event Detection: Systematic Taxonomy and Comparison of Deep Models.* arXiv:2302.05087 | [M] |
| E7 | Sultani, W., Chen, C., Shah, M. *Real-world anomaly detection in surveillance videos.* CVPR 2018 (UCF-Crime) | [M] |
| E8 | *Language-guided Open-world Video Anomaly Detection under Weak Supervision.* arXiv:2503.13160 | [M] |
| E9 | *Weakly Supervised VAD and Localization with Spatio-Temporal Prompts.* arXiv:2408.05905 | [M] |
| E10 | Ullah, W., et al. *TransCNN: Hybrid CNN and transformer for surveillance anomaly detection.* Eng. Appl. AI 123, 2023 | [M] |
| E11 | *Scene-dependent video anomaly detection: A benchmark and weakly supervised model.* 2025 | [M] |

## F. Crowd counting / density estimation (2019-2026)

| # | Reference | V |
|---|---|---|
| F1 | Deng, et al. *Deep learning in crowd counting: A survey.* CAAI Trans. Intelligence Technology, 2024 | [M] |
| F2 | Wang, et al. *A comprehensive survey of crowd density estimation and counting.* IET Image Processing, 2025 | [M] |
| F3 | *A Survey on Deep Learning-based Single Image Crowd Counting.* arXiv:2012.15685 | [M] |
| F4 | Sindagi, Yasarla, Patel. *JHU-CROWD++: Large-Scale Crowd Counting Dataset and A Benchmark Method.* TPAMI | [M] |
| F5 | Wang, Q., et al. *NWPU-Crowd: A Large-Scale Benchmark for Crowd Counting and Localization.* TPAMI 2020 | [M] |
| F6 | *AutoScale: Learning to Scale for Crowd Counting and Localization.* arXiv:1912.09632 | [M] |
| F7 | *Boosting Crowd Counting via Multifaceted Attention.* CVPR 2022. arXiv:2203.02636 | [M] |
| F8 | *Congested Crowd Instance Localization with Dilated Convolutional Swin Transformer.* arXiv:2108.00584 | [M] |
| F9 | *LDC-Net: A Unified Framework for Localization, Detection and Counting in Dense Crowds.* arXiv:2110.04727 | [M] |
| F10 | *RCCFormer: A Robust Crowd Counting Network Based on Transformer.* arXiv:2504.04935, 2025 | [M] |
| F11 | *A Comprehensive Survey on Occlusion-Robust Crowd Density Estimation: Techniques and Applications.* 2025 | [M] |
| F12 | *HAJJv2-CrowdCount: Zero-Shot Benchmark for Dense Crowd Counting.* arXiv:2607.07322 | [M] |
| F13 | *Counting Crowds in Bad Weather.* arXiv:2306.01209 | [M] |

## G. Face detection / landmarks (the on-device sensor)

| # | Reference | V |
|---|---|---|
| G1 | Guo, J., Deng, J., Lattas, A., Zafeiriou, S. *Sample and Computation Redistribution for Efficient Face Detection (SCRFD).* arXiv:2105.04714, ICLR 2022 | [F] |
| G2 | Deng, J., et al. *RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild.* CVPR 2020 | [M] |
| G3 | *Crowd Density Estimation Based on Face Detection Under Significant Occlusions and Head Pose Variations.* MCSS 2020, Springer LNCS. doi:10.1007/978-3-030-59000-0_16 — **abstract-level only, full text unobtainable.** Trains a detector robust to <=90 deg out-of-plane rotation and 25/50/75% occlusion; 48k training images, 109 test images (21-905 faces, mean 145). Cite for framing only; NOT load-bearing (see G10, G11). | [A] |
| G4 | Marzani, F., van Ede, T., Heijenk, G., van Steen, M. *Head Count: Privacy-Preserving Face-Based Crowd Monitoring.* arXiv:2604.14250, 2026 | [F] |
| G5 | *B-FPGM: Lightweight Face Detection via Bayesian-Optimized Soft FPGM Pruning.* arXiv:2501.16917, 2025 | [M] |
| G6 | *EResFD: Rediscovery of the Effectiveness of Standard Convolution for Lightweight Face Detection.* arXiv:2204.01209 | [M] |
| G7 | *A Lightweight and Accurate Face Detection Algorithm Based on RetinaFace.* arXiv:2308.04340 | [M] |
| G8 | *EfficientFace: An Efficient Deep Network with Feature Enhancement for Accurate Face Detection.* arXiv:2302.11816 | [M] |
| G9 | *Filter-Pruning of Lightweight Face Detectors Using a Geometric Median Criterion.* arXiv:2311.16613 | [M] |
| G10 | *DAFE-FD: Density Aware Feature Enrichment for Face Detection.* arXiv:1901.05375 — uses an estimated **density map to improve face detection**; establishes the density/detectability coupling in the *opposite* direction to Contribution 2 | [F] |
| G11 | Wang, X., et al. *Repulsion Loss: Detecting Pedestrians in a Crowd.* CVPR 2018. arXiv:1711.07752 — quantifies experimentally how "the detector is harmed by crowd occlusion"; quantitative grounding for the occlusion index | [F] |

## H. Tracking (considered and rejected — see gap analysis)

| # | Reference | V |
|---|---|---|
| H1 | Zhang, Y., et al. *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* ECCV 2022. arXiv:2110.06864 | [F] |
| H2 | Cao, J., et al. *Observation-Centric SORT (OC-SORT).* CVPR 2023 | [M] |
| H3 | *Deep OC-SORT: Multi-Pedestrian Tracking by Adaptive Re-Identification.* arXiv:2302.11813 | [M] |
| H4 | *UCMCTrack: Multi-Object Tracking with Uniform Camera Motion Compensation.* AAAI 2024. arXiv:2312.08952 | [M] |
| H5 | *DeNoising-MOT: Towards Multiple Object Tracking with Severe Occlusions.* arXiv:2309.04682 | [M] |
| H6 | *Head Anchor Enhanced Detection and Association for Crowded Pedestrian Tracking.* arXiv:2508.05514 | [M] |
| H7 | *Repulsion Loss: Detecting Pedestrians in a Crowd.* CVPR 2018. arXiv:1711.07752 | [M] |

## I. Edge / TinyML deployment

| # | Reference | V |
|---|---|---|
| I1 | *MiCrowd: Vision-Based Deep Crowd Counting on MCU.* Sensors, 2023. PMC10098830 | [M] |
| I2 | *An Ultra-low Power TinyML System for Real-time Visual Processing at Edge.* arXiv:2207.04663 | [M] |
| I3 | *Tiny Machine Learning and On-Device Inference: A Survey of Applications, Challenges, and Future Directions.* 2025. PMC12115890 | [M] |
| I4 | *Toward an Integrated IoT-Edge Computing Framework for Smart Stadium Development.* J. Sensor and Actuator Networks 15(1), 2026 | [M] |
| I5 | *Efficient People Counting in Thermal Images: Benchmark of Resource-Constrained Hardware.* 2022 | [M] |

## J. Aerial / multi-view crowd monitoring

| # | Reference | V |
|---|---|---|
| J1 | *DroneNet: Crowd Density Estimation using Self-ONNs for Drones.* arXiv:2211.07137 | [M] |
| J2 | *Detection, Tracking, and Counting Meets Drones in Crowds: A Benchmark.* CVPR 2021. arXiv:2105.02440 | [M] |
| J3 | *DenseTrack: Drone-based Crowd Tracking via Density-aware Motion-appearance Synergy.* arXiv:2407.17272 | [M] |
| J4 | *Density-based clustering with fully-convolutional networks for crowd flow detection from drones.* Neurocomputing, 2023 | [M] |
| J5 | *A Multi-Drone Multi-View Dataset and Deep Learning Framework for Pedestrian Detection and Tracking.* arXiv:2511.08615 | [M] |

## K. Prior art on head orientation as a signal (NOVELTY THREAT — read before claiming)

| # | Reference | V |
|---|---|---|
| K1 | US Patent 11,509,831 — *Synchronous head movement (SHMOV) detection systems and methods* — alerts when >=50% of head directions align within 35% | [M] |
| K2 | US Patent 11,175,733 — *Method of view frustum detection* | [M] |
| K3 | *Head pose estimation with uncertainty and an application to dyadic interaction detection.* CVIU, 2024 | [M] |
| K4 | *Deep Head Pose: Gaze-Direction Estimation in Multimodal Video.* IEEE TMM | [M] |
| K5 | *Anomaly Detection in Surveillance Video using Motion-Direction Model.* IEEE, 2018 | [M] |

## L. Optical flow (the motion sensor)

| # | Reference | V |
|---|---|---|
| L1 | Farneback, G. *Two-Frame Motion Estimation Based on Polynomial Expansion.* SCIA 2003 — dense flow baseline | [M] |
| L2 | Teed, Z., Deng, J. *RAFT: Recurrent All-Pairs Field Transforms for Optical Flow.* ECCV 2020 | [M] |
| L3 | *Learning Optical Flow, Depth, and Scene Flow without Real-World Labels.* arXiv:2203.15089 | [M] |
| L4 | *Sparse Optical Flow Implementation Using a Neural Network for Low-Resolution Thermal Aerial Imaging.* J. Imaging 8(10), 2022 | [M] |
