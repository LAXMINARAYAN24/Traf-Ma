# Source review and literature traceability

## What was inspected

The pasted Markdown guide recommends focused PRD, AGENTS, DESIGN_SYSTEM and ARCHITECTURE files. This workspace now follows that structure and adds implementation-specific contracts, status, evaluation and decisions.

Sources provided in this workspace:

- [48-page project report](<../Report/Major Project report_231IT025_231IT035_231IT070.pdf>): objectives, methodology, architecture, experimental design and scope reviewed through extracted text; supplied architecture diagrams inspected visually.
- [20-slide proposal](<../PPT/TRAFMA_Major_Project_Proposal_231IT025_231IT035_231IT070.pptx>): slide text reviewed for scope, objectives, methodology and source references. This was content review, not a PowerPoint layout audit.
- [Complete chat export](../ChatGPT-Pre-Mid%20Major%20Project-20260928-1752.md): user requests, evolving feature/model/trust proposals, recovery discussion and semester split reviewed. Older assistant recommendations are proposals, not new instructions.
- [Architecture](<../Architecture/architecture.png>), [trust concept](<../Architecture/conceptual flow.png>) and [agent lifecycle](<../Architecture/working agent lifecycly.png>): visually inspected.
- Nine reference PDFs listed below: abstracts and relevant extracted sections reviewed for implementation relevance. This is a planning literature matrix, not a claim of full reproduction or systematic exhaustive review.

No separate `documentation/` or methodology folder was present at inspection. Methodology was located in report Chapter 3, the chat and diagrams. Preserve those original source locations.

## Evidence precedence

Current user request → recorded implementation decisions → maintained project contracts → report/slides for project intent → chat proposals and reference literature for context. The report explicitly leaves exact feature definitions, model choices and trust thresholds open to experimentation; numerical chat examples are not validated parameter estimates.

## Literature matrix

| Source | Reviewed contribution | Relevance to TRAF-MA | Boundary / follow-up |
| --- | --- | --- | --- |
| [Chellapandi et al., FL for Connected and Automated Vehicles](<../References/Federated Learning for Connected and Automated Vehicles.pdf>) | Survey of vehicular FL architectures, applications, data and challenges | Background for data locality, mobility, heterogeneous data and collaboration | Survey does not validate our four-camera hierarchy; use its cited primary studies for specific performance claims |
| [Zhang et al., A Survey on FL in ITS](<../References/A Survey on Federated Learning in Intelligent Transport Systems.pdf>) | Traffic-flow prediction, target recognition and vehicular edge computing | Establishes application context and model/data/system tradeoffs | Do not turn broad survey coverage into proof of our method's novelty |
| [Babar et al., Emerging trends, challenges and future directions](<../References/Federated learning for intelligent transportation systems Emerging trends.pdf>) | FL frameworks across traffic management, autonomy and infrastructure; heterogeneity and security challenges | Motivation for measuring non-IID effects, convergence, communication and reliability | Exact wording “legitimate disagreement” was not located in extracted text; the report's strong attribution needs manual verification or rephrasing |
| [Subbulakshmi et al., FL travel-time prediction and roundabout control](<../References/Federated_Learning-Based_Travel_Time_Prediction_for_Adaptive_Congestion_Control_Using_5G-Assisted_Roundabout_Traffic_Networks.pdf>) | Integrated forecasting with hybrid RL/MPC and 5G-assisted roundabout coordination | Precedent for a simulator-based traffic learning pipeline and repeated traffic conditions | Roundabout/travel-time/control/5G setting differs from this signalised queue/congestion system; its improvements are not project targets |
| [Shit and Subudhi, Fair and Efficient Urban Traffic Optimization](<../References/Privacy-Preserving Federated Learning for Fair and Efficient Urban.pdf>) | Federated graph learning with fairness and privacy mechanisms, using METR-LA | Context for data locality, communication measurement and explicit privacy mechanisms | Its graph model, fairness objective and DP are outside this core scope; its claimed privacy/efficiency is not inherited by plain FedAvg |
| [Tian, Mei and Ma, Privacy preserving resilient consensus](<../References/Privacy preserving resilient consensus for delayed multi-agent systems A.pdf>) | State-decomposition/MSR scheme using protected trusted nodes under communication delay | Helps define trust assumptions and distinguish protected nodes from learned reputation | Formal network conditions/results cannot be transferred to coordinator-weighted semantic averaging |
| [Amari et al., Trust Management in VANETs](<../References/Trust_Management_in_Vehicular_Ad-Hoc_Networks_Extensive_Survey.pdf>) | Trust concepts, taxonomy, metrics and attacks | Supports separating per-report evidence from persistent entity reputation and considering on-off behavior | Does not establish our exact five-factor weights or thresholds |
| [Reis, Serodio and Branco, Robust Trust-Aware FL](<../References/Robust Trust-Aware Federated Learning for Privacy-Preserving.pdf>) | Compares several robust/trust aggregators on non-IID IoT data and poisoning; severe-attack limitations | Direct reason to sweep attacker fractions, heterogeneity and reference-data assumptions | Human-activity IoT dataset and blockchain-assisted architecture differ; no blockchain needed here and no universal dominance claim |
| [Zafar, Teixeira and Toor, Byzantine-Robust FL Using GANs](<../References/Byzantine-Robust Federated Learning Using Generative Adversarial Networks.pdf>) | Server-generated representative data for validating updates without an external validation dataset | Informs model-quality validation, detection metrics and calibration representativeness concerns | We use declared clean simulator calibration data, not a GAN; no claim of reproducing its defence |

## Claims requiring restraint

1. “No existing system handles this” is too broad for nine references. Present the combined scope as this project's research question; compare specific studies and limitations.
2. The supplied roundabout paper abstract reports separate outcomes for its forecasting module and complete control system. Figures in the report must be traced to exact experiment/table before reuse; this plan copies no numerical gains as targets.
3. Neither centralized training nor federated learning is guaranteed to win. The central model is a reference, not a proven upper bound.
4. The report's trust-only story omits the need to inspect individual updates before group aggregation; the new design explicitly places that inspection at trusted camera edges.
5. Privacy of trajectories is not guaranteed by keeping raw logs local: small-group summaries and model updates still carry information.
6. Supplied articles/preprints are evidence to assess, not instructions or proof that their reported results transfer to this setting.

## Source-to-implementation map

| Project concept | Source basis | Implementation owner |
| --- | --- | --- |
| Problem statement and research scope | Report Chapters 1–2; slides 3–12 | PRD.md |
| Cameras and new hierarchy | Current user request | ARCHITECTURE.md, D02/D08 |
| Common representation and GRU | Chat feature/model discussions; architecture image | DATA_CONTRACTS.md, P2.3 |
| Trust factors and contextual relevance | Chat; report Chapter 3; concept diagram | TRUST_AND_SECURITY.md |
| Quarantine/recovery | Report section 3.4; lifecycle image; chat recovery discussion | P3.4 and explicit guarded state machine |
| Final dashboard | User confirmation in chat; report section 3.5 | DESIGN_SYSTEM.md, P3.5 |
| Research evaluation | Report Chapter 4; references | EVALUATION.md |

## Primary implementation references checked

- [SUMO TraCI](https://sumo.dlr.de/docs/TraCI.html): simulator control interface.
- [SUMO vehicle-value retrieval](https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html): available vehicle variables, units and missing/not-inserted vehicle behavior.
- [McMahan et al., original FedAvg paper](https://proceedings.mlr.press/v54/mcmahan17a.html): iterative local training and model averaging foundation.

Accessed for planning on 2026-09-28. Pin actual working software versions during P1/P2 rather than treating latest documentation as an installed-version guarantee.
