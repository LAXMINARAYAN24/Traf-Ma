# Explain Project Implementation

**User:** LAXMINARAYAN
**Created:** 11/5/2026 21:17:08  
**Updated:** 11/5/2026 23:21:13  

## Change summary: original 20-window run → 350v250 follow-up (6 November 2026)

This explanation originally focused on the 350-vehicle experiment with 20 windows per vehicle. The new 350v250 experiment is added below as a separate follow-up. The original numbers remain here as the historical baseline; the two experiments must not be described as a controlled, same-data comparison.

| Item | Original 20-window experiment | New 350v250 experiment |
|---|---:|---:|
| Vehicle clients | 350 | 350 |
| Vehicle windows per client | 20 | 250 |
| Vehicle training windows | 7,000 | 87,500 |
| RSU training windows in self-update-on | 4 × 1,750 = 7,000 | 4 × 1,750 = 7,000 |
| Total training windows, central / HFL self-update-on | 14,000 | 94,500 |
| Total training windows, flat / HFL self-update-off | 7,000 | 87,500 |
| Dataset and training seed | `phase2_compact_1500`, seed 20261031 | `phase2_350clients_1000windows_feasibility`, seed 20261105 |
| Test cohort | 14,000 windows | 14,000 windows |

The feasibility dataset was collected in three separate 1,800-second SUMO runs for train, validation and test, each configured for 500 vehicles. The training split had 361 vehicle clients with at least 250 windows, so 350 were selected; the maximum for any one trajectory was 412 windows. No client had 1,000 windows. Each RSU had 1,761 eligible windows, enough for the 1,750 cap.

| Method | Old MAE / macro-F1 | New MAE / macro-F1 |
|---|---:|---:|
| Central GRU | 14.905 m / 0.8357 | **9.281 m / 0.8733** |
| Flat FedAvg | 57.391 m / 0.2669 | **38.241 m / 0.6784** |
| Hierarchical, RSU self-update off | 57.782 m / 0.2524 | **38.277 m / 0.6781** |
| Hierarchical, RSU self-update on | 27.352 m / 0.7078 | **15.113 m / 0.8344** |

The new results are better on these reported metrics, but the dataset split and training seed changed. The improvement cannot be attributed to the larger per-client budget alone. Results are from one seed and overlapping windows; they are descriptive, not run-level confidence estimates.

All four new runs and their report are under `runs/phase2_350v250_20261105_223954/`. The comparison is `runs/phase2_350v250_20261105_223954/report/report.md` (HTML is alongside it). The detailed follow-up section below records the run settings and breakdowns.

## Prompt:
05/11/2026, 21:20:13

[https://github.com/LAXMINARAYAN24/Traf-MA-350](https://github.com/LAXMINARAYAN24/Traf-MA-350) this repo has the complete implementaion for now now tel me the complete story of the project from start to end with examples with actual numerical values configuration parameters formulas ans setting on each step what is hteinput being taken what hapend to that input and what is the out and where is that output then trandferred agaain  and diagrams and examples so give me numerical values for the example it has complete data , dataset , and report .

and keep the explaination simple words&#x20;
and also the implementaiton is divided between teammates for sumo simulatin shubro
, GRU harshul, and FL laxminaryan .&#x20;
but its not necessary that they will only know their part every one has to know wach every part&#x20;
and in the last keep the list of question a ml , cv , deep  learning professor can ask for our project with its answer &#x20;
and give the ocmplete numerical values used in the impllementaion only for the C:\Users\sahul\Downloads\Traf-Ma\trafma_leftside_gru_final\trafma_project\runs\phase2_compact_350_vehicle_rsu_7000each
and ignore all other phase 2 runs and artifacts &#x20;

## Response:
05/11/2026, 21:20:25 · gpt-6-t-mini

I’ll trace the repository’s actual code, documentation, dataset, and the one specified `phase2_compact_350_vehicle_rsu_7000each` run. Then I’ll turn that evidence into a step-by-step story with data flow, numerical examples, formulas, diagrams, teammate responsibilities, and viva questions—leaving every other Phase 2 run out.

The repository is public and its default branch contains the report, implementation guide, and a clearly named compact 350-vehicle/RSU experiment. I’m now isolating the requested run’s own config, selected-data manifest, training outputs, and source code so the numerical explanation is tied to that run—not to similarly named experiments elsewhere in the repository.

One important distinction has emerged: the repository’s top-level report still describes an earlier 100-vehicle experiment, while your requested target is a 350-vehicle compact run with 7,000 examples per client. I will not copy the old report’s metrics into this explanation. I’m checking the new run’s own manifest and metrics, and I’ll call out any repository/local-folder mismatch explicitly.

> **Retrieved GitHub repository metadata and directory structure**
>
> 
>
> **Reviewed Traf-MA Phase 2 documentation, configurations, reports, and experiment manifest**
>
> 
>
> **Reviewed repository, configurations, simulation artifacts, implementations, datasets, and training performance**
>
> 
>
> **Reviewed Traf-MA traffic simulation, coordination, modeling, and federated training implementations**
>
> 
>
> **Analyzed Phase Two traffic training results, metrics, predictions, and implementation documentation**
>
> 
>
> Worked for 8m 21s

# TRAF-MA: Complete Project Story - From SUMO Simulation to GRU and Hierarchical Federated Learning

Repository: LAXMINARAYAN24/Traf-MA-350 · Analysis based on the repository's current default branch and the specified 350-vehicle experiment

(350-vehicle experiment identified)

The objective here is to understand the project as one complete system, not as three separate assignments. By the end, every team member should be able to explain\:

1. How the road network is created and how vehicles move in it.
2. How SUMO generates vehicle information.
3. How vehicles and RSUs observe traffic.
4. How messages travel from vehicles to RSUs and then to the coordinator.
5. How the dataset is generated, labelled and converted into ML input.
6. How the GRU predicts queue length and congestion.
7. How federated learning trains multiple clients without pooling their original training histories.
8. How four RSUs aggregate vehicle updates and how the global model is produced.
9. How to interpret the saved experiment's actual numerical results.
10. What a professor can ask about SUMO, ML, computer vision, deep learning and federated learning.

## First: the exact experiment we are explaining

I located the following experiment package in the repository\:

`artifacts/phase2_compact_350_hfl_rsu_on_7000each/`

It points to your requested local experiment\:

```text
C:\Users\sahul\Downloads\Traf-MA\
trafma_leftside_gru_final\trafma_project\
runs\phase2_compact_350_vehicle_rsu_7000each\
```

Its saved run is\:

```text
hierarchical_fedavg_phase2_compact_1500_seed20261031_
20261104T194135Z
```

The package contains the run-specific configuration, selected feature tensors, selected-data manifest, model checkpoints, client updates, predictions, metrics and report.

**Original numerical boundary\:** The walkthrough below preserves this experiment as the 20-window baseline. A separate 350v250 follow-up and comparison have since been added before Chapter 1; the new results do not replace or get mixed into the original run's artifacts.

### The most important numbers

# 350

**Vehicle learners**

20 training windows per vehicle

# 4

**RSU learners**

1,750 training windows per RSU

# 14,000

**Training windows**

7,000 vehicle + 7,000 RSU

# 100

**Federated rounds**

5 local epochs per client per round

# 27.352 m

**Test queue MAE**

Average absolute prediction error

# 0.7078

**Test macro-F1**

Three-class classification score

The validation cohort also has 14,000 windows, and the test cohort has 14,000 windows.

| Data category | Vehicle | RSU | Total |
|--------|--------|--------|--------|
| Training | 7,000 | 7,000 | 14,000 |
| Validation | 7,000 | 7,000 | 14,000 |
| Testing | 7,000 | 7,000 | 14,000 |
| **All selected windows** | **21,000** | **21,000** | **42,000** |

There are 354 learners in the federated experiment: 350 vehicle learners plus four persistent RSU learners.

Notice something important: 7,000 vehicle training windows does not mean every vehicle produces 7,000 examples.

$$
350 \times 20 = 7,000
$$

Similarly\:

$$
4 \times 1,750 = 7,000
$$

That is how the 14,000-example training budget is formed.

The results are from one training seed and overlapping one-second windows. Therefore, they are not confidence estimates from independent simulations.

## Follow-up experiment: 350 clients × 250 windows

This section updates the original walkthrough with the completed 350v250 experiment. It uses the same broad pipeline—SUMO observations, agent-local feature windows, GRU prediction, then local client training and aggregation—but changes the dataset, sampling budget and training seed.

### What changed in the data and training setup

- **Dataset:** `data/phase2_350clients_1000windows_feasibility/`, generated from three separate 1,800-second, 500-vehicle SUMO runs for train, validation and test. The runs completed with 500 arrivals each, with zero collisions and zero teleports.
- **Vehicle selection:** 350 distinct train-split trajectory clients, capped at 250 windows each: `350 × 250 = 87,500` vehicle training windows. The pilot had 361 clients meeting that cap and a maximum of 412 windows for one client, so the requested 1,000 per client was not achievable from these trajectories.
- **RSU selection:** In runs that include RSU self-updates, each of `rsu_N`, `rsu_E`, `rsu_S`, and `rsu_W` contributes up to 1,750 of its own sensor windows: `4 × 1,750 = 7,000` RSU windows.
- **Budgets by method:** Central GRU and hierarchical FedAvg with RSU self-update on use 94,500 training windows. Flat FedAvg and hierarchical FedAvg with RSU self-update off use 87,500 vehicle windows. The training budgets intentionally differ by method, so interpret the method comparison with that limitation.
- **Evaluation:** Each run uses 7,000 vehicle and 7,000 RSU windows for validation and 14,000 for test. The four new runs share the new comparison's held-out identities; these are not the same test examples as the old compact run.
- **Training settings:** Seed 20261105; 100 FL rounds and five local epochs. Central GRU ran on CUDA for 26 epochs, with a 150-epoch maximum, `min_epochs: 1`, and early-stopping patience 15. The earlier central run used `min_epochs: 50` and ran 83 epochs. FedAvg is CPU multi-threaded.
- **Unchanged task definition:** A window is still 10 consecutive one-second records with 22 features per step, and the targets remain queue length and congestion class 30 seconds later. This follow-up changes how much eligible data is selected, not the feature schema or target horizon.

### New run results

All scores below are from the new 14,000-window held-out test cohort.

| Method | Training windows | Duration | Test queue MAE | Test queue RMSE | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Central GRU | 94,500 | 4.5 min; 26 epochs | **9.281 m** | **19.667 m** | **0.8733** | **0.8562** |
| Flat FedAvg | 87,500 | 136.3 min; 100 rounds | 38.241 m | 58.847 m | 0.6784 | 0.6696 |
| Hierarchical FedAvg, RSU self-update off | 87,500 | 131.6 min; 100 rounds | 38.277 m | 58.902 m | 0.6781 | 0.6698 |
| Hierarchical FedAvg, RSU self-update on | 94,500 | 148.6 min; 100 rounds | 15.113 m | 26.388 m | 0.8344 | 0.8106 |

For the RSU self-update-on run, test macro-F1 was 0.8618 for vehicle windows and 0.8042 for RSU windows. Its queue MAE was 16.851 m for vehicle windows and 13.376 m for RSU windows. The central model's vehicle and RSU queue MAEs were 11.047 m and 7.514 m respectively.

Compared with the 20-window report, every method's aggregate macro-F1 is higher and queue MAE is lower. This is a descriptive comparison only: the new runs use a different dataset split and seed, and the methods have different training budgets. We cannot conclude that raising the cap from 20 to 250, by itself, caused the gains. Multiple paired seeds and a deliberately matched data comparison would be needed to isolate that effect.

### Where the new artifacts are saved

```text
runs/phase2_350v250_20261105_223954/
├── configs/                  # snapshots of the four configs used
├── central/                  # central GRU run and checkpoints
├── flat/                     # flat FedAvg run and round artifacts
├── hierarchical_rsu_off/     # hierarchical run without RSU self-update
├── hierarchical_rsu_on/      # hierarchical run with RSU self-update
└── report/
    ├── report.md
    └── report.html
```

The report contains the aggregate and type-specific metrics, plots, run durations and manifest hashes. It only includes these four runs.

---

## Chapter 1 - What problem are we solving?

Imagine a four-way intersection in an Indian city.

There are four incoming roads\:

- North
- South
- East
- West

Vehicles are approaching the junction. Some are moving freely, some are slowing down, and some are waiting in a queue because the traffic signal is red.

The traffic management system needs to answer two questions\:

**Question 1\:** How long will the traffic queue become in the future?

**Question 2\:** Will the approach be free, moderately congested, or heavily congested?

Our project attempts to predict these conditions 30 seconds before they happen.

For example, consider the North approach.

At simulation time 100 seconds, the current queue might be 35 metres. The system receives ten seconds of local traffic history and predicts the queue at 130 seconds.

An illustrative prediction could be\:

| Quantity | Value |
|--------|--------|
| Current queue | 35 m |
| Prediction horizon | 30 seconds |
| Actual future queue | 55 m |
| Predicted future queue | 49 m |
| Absolute error | 6 m |

The actual values in this small illustration are not claimed to be a particular saved prediction row. They show how the forecasting task works.

The model also predicts the congestion category at the future time.

This is a **multi-task learning problem** because we predict two outputs simultaneously.

1. A continuous numerical output: queue length in metres.
2. A discrete output: congestion class.

## Chapter 2 - The complete system architecture

Before understanding individual files, understand this complete flow.

## TRAF-MA end-to-end data flow

## SUMO simulator

Four-arm signalised intersection · vehicle movement · traffic signals

**350 vehicle learners**

Local observations

**4 RSUs**

Roadside observations

Logs, messages and future labels

**Feature engineering**

## 10 × 22 input tensor

Ten observations, 22 features per observation

## GRU predictor

Queue regression + three-class congestion prediction

**Local training**

Each client trains on its selected local windows

## Hierarchical FedAvg

Vehicle updates → parent RSUs → global coordinator

**Global GRU model**

Predictions, evaluation metrics, checkpoints and report

There are two important flows in this architecture\:

**Traffic observation flow\:** vehicle information and RSU summaries move through the communication architecture.

**Learning flow\:** trained GRU parameters move through the federated architecture.

These are related but are not the same operation. This distinction is particularly important in the viva.

---

## Chapter 3 - SUMO: Creating the traffic world

This is Shubro's major implementation area, but everyone must understand it.

[![SUMO simulation of the intersection | Download Scientific Diagram](https://images.openai.com/static-rsc-4/E6M1HbJBWynCtjLf5N4QGoGZqLNjTj0UhG1OFEutx10GsuQWbO8EJidRwzGM2nqjVVGEXD2fX0IuW4X25AOsch2yWCWsAtzfYFidcLxyeU9BpF7Od3yLVbmXzj4nZhSw-XqIuhu-3PhtGp19kd6RWfFgM_4oWv8Vqw_8F4V96exEOfLVjoihpJFs7FjoMCa1?purpose=fullsize)](https://www.researchgate.net/figure/SUMO-simulation-of-the-intersection_fig1_378999357)

### 3.1 What is SUMO?

SUMO means **Simulation of Urban MObility**\.

It is a traffic simulator. It creates virtual roads, vehicles, signals, vehicle movement and traffic conditions.

It answers questions such as\:

- Where is every vehicle?
- How fast is it moving?
- Is it waiting?
- Which road is it travelling on?
- Has it reached the junction?
- Has it completed its route?

SUMO does not train our GRU. Its main job is to create the traffic environment and its changing state.

Our Python program communicates with SUMO through **TraCI**, the Traffic Control Interface.

### 3.2 The road network

The network contains four approaches arranged around one central junction.

## Four-arm intersection

```svg
<svg viewBox="0 0 340 285" width="100%" xmlns="http://www.w3.org/2000/svg"><rect x="131" y="0" width="78" height="285" fill="#737B83" /><rect x="0" y="103" width="340" height="78" fill="#737B83" /><rect x="131" y="103" width="78" height="78" fill="#545D66" /><path d="M170 0 V103 M170 181 V285 M0 142 H131 M209 142 H340" stroke="#F5F5F5" stroke-width="2" stroke-dasharray="9 7" /><path d="M131 103 L209 103 M131 181 L209 181" stroke="#D8DCE0" stroke-width="1" /><g fill="#E9F1FE" stroke="#245D98" stroke-width="1.5"><rect x="152" y="44" width="15" height="27" rx="3" /><rect x="172" y="214" width="15" height="27" rx="3" /><rect x="39" y="124" width="27" height="15" rx="3" /><rect x="273" y="147" width="27" height="15" rx="3" /></g><g fill="#FFFFFF" font-size="12" font-weight="bold"><text x="170" y="16" text-anchor="middle">NORTH</text><text x="170" y="274" text-anchor="middle">SOUTH</text><text x="10" y="94">WEST</text><text x="329" y="94" text-anchor="end">EAST</text><text x="170" y="145" text-anchor="middle">JUNCTION</text></g><g fill="#FFD166" stroke="#865C00"><circle cx="119" cy="112" r="4" /><circle cx="221" cy="112" r="4" /><circle cx="119" cy="172" r="4" /><circle cx="221" cy="172" r="4" /></g></svg>
```

Conceptual layout: four approaches converge on the signalised junction. Each approach has one incoming lane and one outgoing lane.

The geometry is\:

| Parameter | Implementation value |
|--------|--------|
| Number of arms | 4 |
| Arm length | 300 m |
| Lanes per direction | 1 |
| Junction location | \(0, 0) |
| North endpoint | \(0, 300) |
| South endpoint | \(0, -300) |
| East endpoint | \(300, 0) |
| West endpoint | \(-300, 0) |
| Driving side | Left |

The repository's `trafma/netgen.py` generates the road network.

It creates three important inputs for SUMO\:

- `nodes.nod.xml`\: road endpoints and central junction.
- `edges.edg.xml`\: the incoming and outgoing roads.
- `connections.con.xml`\: the allowed turn connections.

The generated network contains\:

$$
4 \text{ approaches}\times 3 \text{ turn choices}=12
$$

signalised movement connections.

### 3.3 Understanding the Indian left-hand traffic setting

This point is worth remembering.

In India, vehicles travel on the left side of the road.

Our repository has\:

```yaml
network:
  drive_side: left
```

The network generator passes\:

```bash
--lefthand true
```

to SUMO's `netconvert`\.

It also checks the resulting lane geometry. If SUMO builds a mirrored network, the code raises an error rather than silently accepting incorrect geometry.

### 3.4 Vehicle configuration

The standard vehicle configuration in `trafma/config.py` is\:

| Parameter | Value | Meaning |
|--------|--------|--------|
| Vehicle length | 5 m | Physical vehicle length |
| Minimum gap | 2.5 m | Desired inter-vehicle gap |
| Acceleration | 2.6 m/s² | Maximum acceleration setting |
| Deceleration | 4.5 m/s² | Maximum deceleration setting |
| Tau | 1 s | Driver reaction/headway behaviour parameter |
| Sigma | 0.5 | Driver imperfection |
| Maximum speed | 13.89 m/s | Approximately 50 km/h |

The maximum speed conversion\:

$$
13.89\times3.6=50.004\text{ km/h}
$$

A vehicle is therefore configured for approximately 50 km/h maximum speed.

These are simulator settings, not measured real-world driving statistics.

### 3.5 How are 350 vehicles generated?

The demand generator is implemented in\:

`trafma/demand.py`

The process is\:

1. Read the configured vehicle count.
2. Create reproducible random departure times.
3. Select an incoming road.
4. Select a turn.
5. Calculate the destination road.
6. Write a route file.

Turn probabilities are\:

| Movement | Probability |
|--------|--------|
| Straight | 0.60 |
| Left | 0.20 |
| Right | 0.20 |

The probability sum must equal one\:

$$
0.60+0.20+0.20=1
$$

For a hypothetical 350 independently sampled vehicles, expected movement counts are\:

$$
N_{\text{straight}}=350(0.60)=210
$$

$$
N_{\text{left}}=350(0.20)=70
$$

$$
N_{\text{right}}=350(0.20)=70
$$

These are expected counts, not claims that the random saved route file contains exactly 210, 70 and 70.

The random generator uses a seed\:

```python
random.Random(f"trafma-demand:{cfg.run.seed}")
```

This makes demand generation reproducible for a given configuration and seed.

### 3.6 Traffic signal configuration

We use split phasing.

That means only one approach receives a protected green signal at a time.

The standard settings are\:

| Signal stage | Duration |
|--------|--------|
| Green | 25 seconds |
| Yellow | 3 seconds |
| All-red | 2 seconds |

There are four approaches, so\:

$$
T_{\text{cycle}}=4(25+3+2)
$$

$$
\boxed{T_{\text{cycle}}=120\text{ seconds}}
$$

**One complete 120-second cycle**

**North approach**

25 s green · 3 s yellow · 2 s all-red

**East approach**

25 s green · 3 s yellow · 2 s all-red

**South approach**

25 s green · 3 s yellow · 2 s all-red

**West approach**

25 s green · 3 s yellow · 2 s all-red

---

**Total cycle**

# 120 s

Why use split phasing?

Because opposing or crossing traffic movements do not receive green simultaneously. It is a simple way of preventing conflicting approaches from being released together.

The implemented signal is fixed-time, not AI-controlled. Our GRU predicts traffic conditions; it does not currently decide the signal timings.

### 3.7 How does the simulator actually execute?

The main loop is in `trafma/runner.py`\.

The sequence is\:

```text
Load configuration
       |
       v
Build SUMO network
       |
       v
Generate routes for vehicles
       |
       v
Start SUMO through TraCI
       |
       v
Advance simulation by 1 second
       |
       v
Read vehicle and traffic-light state
       |
       v
Generate observations
       |
       v
Send messages and collect labels
       |
       v
Repeat until simulation ends
```

TraCI reads vehicle properties such as\:

- Position
- Speed
- Acceleration
- Road ID
- Lane position
- Waiting time

For each vehicle, `TraciWorld.snapshot()` collects these values into a Python world snapshot.

This is an important separation\:

**SUMO produces the simulated reality; Python reads that reality and implements the agents.**

---

## Chapter 4 - Vehicle agents and RSUs

This is where the simulated world becomes a distributed sensing system.

We use two types of agents.

### 4.1 Vehicle agent

A vehicle agent represents an individual car that is capable of observing a limited amount of information.

For example\:

```text
Vehicle: veh_0042
Origin: North
Parent: rsu_N
```

Its parent RSU is assigned using its incoming road.

The assignment stays fixed during that vehicle's simulated journey.

A vehicle does not get complete access to the entire road network.

It has a local observation radius of 100 metres.

[![Getting Started with Vehicle-to-Everything Tech: Use Cases to Maximize Benefits | Government Fleet](https://images.openai.com/static-rsc-4/zFTfRBpoDbutuEZ5sOoZxgkwn4eN9YXs55QE0eF0hPO_iIk-2qnb0ezaDndGLesk-_AJ8WaXcSM8a6CGft1JuH-owAj-YPRPM0RUNUXOoyWIrWEwmkdb6JsRrCxPBI0UobU5CbsXQwNY2TNFtmsW2KGKAfw4fbJfVG8GhIU6_7tKE7HKWrNpRjnSL2LmNCkH?purpose=fullsize)](https://www.government-fleet.com/blogposts/getting-started-with-vehicle-to-everything-tech-use-cases-to-maximize-benefits)

It can observe its own\:

- Speed
- Acceleration
- Position
- Waiting time
- Distance to the stop line
- Nearby vehicles on the same incoming road
- Nearby vehicles' average speed
- Nearby halted vehicles

Signal information is also limited.

With the default `signal_broadcast: false`, vehicle signal-state information is available only when the vehicle is within its sensing radius; the countdown timer is unavailable.

When a vehicle is in the junction or outside its incoming-approach scope, its model observation history is not treated as valid incoming-local data.

### 4.2 RSU agent

RSU means **Roadside Unit**\.

An RSU is a roadside infrastructure agent responsible for an approach.

There are four persistent RSUs\:

| RSU | Responsibility |
|--------|--------|
| `rsu_N` | North approach |
| `rsu_E` | East approach |
| `rsu_S` | South approach |
| `rsu_W` | West approach |

Each RSU has a sensing zone of 150 metres.

[![How Traffic Light Sensors Detect Your Car](https://images.openai.com/static-rsc-4/-IOTtx_csq_ehl3aC5ARJPiRAoRkDiDIziCuPqqutCdRL_ZNwdjkF3zuJFTCZnrzTnfYNqzPzL9we4WZvBgI_bGcTDwgC8POU_d17eYEenzqj7upNio6ejGquWMDnlOmZmiUmakW8RGhlUQOxJ3SxTUFFRBOWyGBrGOsg7QF5l9Whes6khu4VvIjLQwEtQHs?purpose=fullsize)](https://autos.yahoo.com/policy-and-environment/articles/traffic-light-sensors-detect-car-172500992.html)

An RSU observes its entire configured approach zone.

It collects\:

- Number of vehicles in the zone
- Number of halted vehicles
- Average speed
- Average acceleration
- Average distance to the stop line
- Average and maximum waiting time
- Queue length
- Traffic signal state
- Remaining signal time
- Number of vehicles on its outbound zone

An RSU is not an actual physical camera in this implementation. It is a software agent that receives SUMO-derived sensor observations.

### 4.3 Vehicle and RSU are not the same sensor

Consider this situation\:

| Observation | Vehicle | RSU |
|--------|--------|--------|
| Own speed | Yes | Included in average |
| Nearby vehicles | Limited to 100 m | Whole 150 m zone |
| Average road speed | Local neighbour-based | Zone average |
| Queue length | Not directly exposed as target | Directly observed as local measurement |
| Signal information | Limited availability | Direct approach signal |
| Number of vehicles | Local visible neighbourhood | Zone count |

This creates a natural difference in what the learners observe.

A vehicle is a local observer.

An RSU is an approach-level observer.

That difference is central to the project's federated-learning motivation.

---

## Chapter 5 - Communication: V2I and RSU-to-coordinator

This chapter explains how information travels between agents.

V2I means Vehicle-to-Infrastructure communication.

I2C in our implementation means the RSU-to-coordinator link.

## Message transmission architecture

**Vehicle agent**

`veh_0042` sends a local observation

(V2I link)

**Parent RSU**

`rsu_N` validates and stores its child's report

(I2C link)

**Global coordinator**

Checks provenance, duplicate messages and freshness

**Traffic-state snapshot**

Separate direct RSU evidence from relayed vehicle evidence

### 5.1 V2I transmission

Suppose\:

```text
Vehicle: veh_0042
Parent RSU: rsu_N
Simulation time: 100 seconds
```

The vehicle creates a report containing its allowed observation fields.

The envelope includes fields such as\:

| Field | Example |
|--------|--------|
| `msg_id` | `veh_0042#0` |
| `msg_type` | `vehicle_report` |
| `origin_id` | `veh_0042` |
| `sender_id` | `veh_0042` |
| `receiver_id` | `rsu_N` |
| `seq` | 0 |
| `sim_time` | 100 |
| `hops` | `[veh_0042]` |
| `evidence` | `relayed_child` |

These are illustrative message values, not a claimed historical message captured from the experiment.

### 5.2 Why do we need message validation?

A communication system must not blindly accept every report.

Our RSU checks\:

1. Is the message structurally valid?
2. Is the message actually from one of its own children?
3. Has it already received the same message?
4. Is its sequence number newer than the previous one?
5. Is the report too old?

The RSU keeps track of accepted, duplicate, stale, out-of-order, invalid and incorrectly addressed reports.

### 5.3 RSU-to-coordinator transmission

RSUs send two kinds of information\:

- Their own direct RSU summary.
- Bundles containing vehicle reports received from their children.

The coordinator performs its own validation.

It also checks the route of the message using `hops`\.

For example, a valid relayed message should show\:

```text
[veh_0042, rsu_N]
```

The coordinator must not trust a payload's claimed approach by itself. It uses sender identity and provenance.

### 5.4 What happens when a message is lost?

The channel code supports\:

- Message loss
- Communication delay
- Message duplication

The link configuration has\:

```yaml
comms:
  v2i:
    loss_prob: 0.0
    delay_s: 0.0
    duplicate_prob: 0.0
  i2c:
    loss_prob: 0.0
    delay_s: 0.0
    duplicate_prob: 0.0
```

These are the baseline configuration defaults, not proof of fault-free wireless hardware.

The current experiment's offline learning replay does not simulate real wireless transmission between physical devices.

A particularly important point for the viva\:

**Communication-layer data transfer and federated model-update transfer are two different things.**

---

## Chapter 6 - Ground truth: How do we know what the correct answer is?

Before training a model, we need to know what it should predict.

That correct answer is called a **label** or ground truth.

Suppose the model is given ten seconds of observations.

It should predict the queue 30 seconds in the future.

How do we know that future queue?

We use the SUMO simulator's evaluator.

The evaluator can inspect the full simulated traffic state. The live agents cannot.

This is deliberate.

The evaluator creates the target labels separately from agent observations.

### 6.1 What is a traffic queue?

A queue is not simply the total number of stopped vehicles.

Our implementation calculates a contiguous chain of halted vehicles.

Its rules are\:

| Parameter | Value |
|--------|--------|
| Halted speed threshold | ≤ 0.1 m/s |
| First vehicle must be within | 15 m of stop line |
| Maximum gap within chain | 10 m |
| Maximum measured zone | 150 m |

The queue starts with a halted vehicle close to the stop line and continues backward while subsequent vehicles remain close enough.

The measurement is taken to the rear bumper of the last vehicle in the chain, capped at 150 metres.

### 6.2 A numerical queue calculation

Consider a hypothetical North approach.

Vehicle positions are measured as distance from the stop line.

| Vehicle | Distance | Speed | Gap from previous queue member |
|--------|--------|--------|--------|
| A | 5 m | 0.0 m/s | Starting vehicle |
| B | 12 m | 0.0 m/s | 2 m |
| C | 20 m | 0.05 m/s | 3 m |
| D | 32 m | 0.0 m/s | 7 m |
| E | 48 m | 0.0 m/s | 11 m |

Assume every vehicle has length 5 metres.

Vehicles A through D satisfy the queue condition.

Vehicle E does not because its bumper gap exceeds 10 metres.

The measured queue length is therefore\:

$$
Q=32+5=37\text{ m}
$$

The queue is 37 metres long, not four vehicles long.

That distinction matters because our target is a distance measured in metres.

### 6.3 Congestion classification

The queue is divided by the 150-metre observation zone.

$$
r=\frac{Q}{Z}
$$

where\:

- $Q$ \= queue length
- $Z$ \= zone length

The class thresholds are\:

| Class ID | Name | Queue ratio |
|--------|--------|--------|
| 0 | Free | Less than 0.10 |
| 1 | Moderate | 0.10 to less than 0.35 |
| 2 | Heavy | 0.35 or above |

For the 37-metre example\:

$$
r=\frac{37}{150}=0.2467
$$

Therefore\:

$$
\boxed{\text{Class}=1\text{ (Moderate)}}
$$

Consider another example with queue length 60 metres\:

$$
r=\frac{60}{150}=0.40
$$

The class becomes\:

$$
\boxed{\text{Class}=2\text{ (Heavy)}}
$$

### 6.4 The 30-second prediction horizon

The history length is ten observations.

The simulation and reporting step are both one second.

Therefore, one observation window represents ten consecutive timestamps.

For example\:

| Observation | Time |
|--------|--------|
| 1 | 100 s |
| 2 | 101 s |
| 3 | 102 s |
| 4 | 103 s |
| 5 | 104 s |
| 6 | 105 s |
| 7 | 106 s |
| 8 | 107 s |
| 9 | 108 s |
| 10 | 109 s |

The prediction target occurs at\:

$$
t_{\text{target}}=109+30
$$

$$
\boxed{t_{\text{target}}=139\text{ s}}
$$

The model sees observations up to 109 seconds and must predict the evaluator's queue at 139 seconds.

This is future prediction, not simply current queue detection.

### 6.5 Preventing data leakage

Suppose a model receives a feature called `zone_queue_m`, which directly reveals the answer it is supposed to predict.

That would be data leakage.

Our feature adapter explicitly excludes target-related fields, including\:

- `zone_queue_m`
- `zone_queue_ratio`
- `congestion_class`
- `edge_queue_m`
- `zone_queue_count`
- `label_t`

The evaluator's truth is kept separate from the learning inputs.

This is an important strength of the data design.

---

## Chapter 7 - Dataset generation and the exact data used

Now we have\:

- Traffic observations.
- Vehicle agents.
- RSU observations.
- Future queue measurements.
- Future congestion labels.

We must combine these into an ML dataset.

The dataset used in the saved experiment is\:

```text
phase2_compact_1500
```

The run-specific package contains an exported selection of the exact data used for its experiment.

It includes\:

```text
artifacts/
└── phase2_compact_350_hfl_rsu_on_7000each/
    └── selected_data/
        ├── manifest.json
        ├── selected_windows.csv
        └── selected_windows.npz
```

### 7.1 Why do we need three data files?

| File | Purpose |
|--------|--------|
| `manifest.json` | Provenance, learner counts, dimensions, hashes and split sizes |
| `selected_windows.csv` | Human-readable window metadata and labels |
| `selected_windows.npz` | Actual numerical input tensors and target arrays |

Think of the CSV as the index of the experiment and the NPZ as the numerical training material.

The raw dataset was not copied into the experiment package. Its recorded manifest hash is used for provenance.

Therefore, the selected tensors are available in the experiment package, but the entire original raw simulation dataset is not.

### 7.2 Training data allocation

**Vehicle training data**

## 7,000

350 clients × 20 windows each

---

**RSU training data**

## 7,000

4 RSUs × 1,750 windows each

---

**Total training windows**

# 14,000

### 7.3 What does a client mean?

A client is an independent learning participant.

For example\:

```text
veh_0003
veh_0015
veh_0019
...
veh_0366
```

These are vehicle learner identifiers drawn from eligible training histories.

The selected-data manifest records 350 vehicle learners and four RSU learners.

Each vehicle learner contributes 20 sampled training windows.

Each persistent RSU learner has its own 1,750-window training budget.

One important detail: the selected RSU windows are sampled from the corresponding approach's RSU source histories. The four RSU learners do not each receive the entire road network's history.

### 7.4 Is the dataset balanced?

Not necessarily.

The original observations come from different traffic scenarios\:

- Free
- Moderate
- Heavy
- Asymmetric

The number of free, moderate and heavy examples differs.

The selected data can therefore be non-IID.

Non-IID means that different learners have different data distributions.

For example, a vehicle spending most of its simulated time in a heavy-demand scenario may observe a different proportion of congestion classes from an RSU observing an entire road.

This is one reason federated learning is interesting for this problem.

---

## Chapter 8 - Feature engineering: Converting traffic into numbers

This is the bridge between SUMO and deep learning.

The neural network cannot understand messages such as\:

```text
The car is approaching the junction.
The nearby vehicles are slow.
The signal is red.
```

We convert these observations into numerical values.

Our implementation is in\:

`trafma/features.py`

### 8.1 Why 22 features?

The saved model uses a common feature schema containing 22 values per timestep.

It has four groups\:

1. Ten measured values.
2. Ten availability masks.
3. Two client-type indicators.

The feature schema is frozen as\:

```text
trafma.features/1
```

### 8.2 All 22 input features

| No. | Feature | Meaning |
|--------|--------|--------|
| 1 | `local_mean_speed_mps` | Average local speed |
| 2 | `local_mean_accel_mps2` | Average acceleration |
| 3 | `local_mean_dist_to_stop_m` | Distance to stop line |
| 4 | `local_density_ratio` | Local density |
| 5 | `local_halted_count_ratio` | Proportion halted |
| 6 | `local_mean_waiting_time_s` | Average waiting time |
| 7 | `signal_remaining_s` | Remaining signal time |
| 8 | `signal_red` | Red signal indicator |
| 9 | `signal_yellow` | Yellow signal indicator |
| 10 | `signal_green` | Green signal indicator |
| 11 | `local_mean_speed_mps_available` | Speed availability |
| 12 | `local_mean_accel_mps2_available` | Acceleration availability |
| 13 | `local_mean_dist_to_stop_m_available` | Distance availability |
| 14 | `local_density_ratio_available` | Density availability |
| 15 | `local_halted_count_ratio_available` | Halting availability |
| 16 | `local_mean_waiting_time_s_available` | Waiting availability |
| 17 | `signal_remaining_s_available` | Signal countdown availability |
| 18 | `signal_red_available` | Signal-state availability |
| 19 | `signal_yellow_available` | Signal-state availability |
| 20 | `signal_green_available` | Signal-state availability |
| 21 | `client_vehicle` | Vehicle indicator |
| 22 | `client_rsu` | RSU indicator |

This means the model knows not just the recorded values, but also which measurements were actually available.

### 8.3 Normalisation

Neural networks generally work better when inputs are on comparable scales.

Our implementation uses several normalisation rules.

**Speed\:**

$$
x_{\text{speed}}=
\operatorname{clip}
\left(\frac{v}{13.89},0,1\right)
$$

For a speed of 6.945 m/s\:

$$
x_{\text{speed}}=\frac{6.945}{13.89}=0.5
$$

**Acceleration\:**

$$
x_{\text{accel}}=
\operatorname{clip}
\left(\frac{a}{2\max(a_{\max},d_{\max})},-1,1\right)
$$

With the configuration\:

$$
2\max(2.6,4.5)=9.0
$$

For an acceleration of 1.8 m/s²\:

$$
x_{\text{accel}}=\frac{1.8}{9}=0.2
$$

**Distance to stop line\:**

$$
x_{\text{distance}}=
\operatorname{clip}\left(\frac{d}{300},0,1\right)
$$

If a vehicle is 90 metres from the stop line\:

$$
x_{\text{distance}}=\frac{90}{300}=0.3
$$

**Waiting time\:**

$$
x_{\text{waiting}}=
\operatorname{clip}\left(\frac{w}{300},0,1\right)
$$

If the waiting time is 60 seconds\:

$$
x_{\text{waiting}}=\frac{60}{300}=0.2
$$

**Signal time\:**

$$
x_{\text{signal}}=
\operatorname{clip}\left(\frac{t_{\text{remaining}}}{120},0,1\right)
$$

If 15 seconds remain\:

$$
x_{\text{signal}}=\frac{15}{120}=0.125
$$

### 8.4 How is vehicle density calculated?

For the RSU, the implementation calculates an approximate capacity using\:

$$
C=\frac{Z}{L+G}
$$

where\:

- $Z=150$ m is the RSU zone.
- $L=5$ m is vehicle length.
- $G=2.5$ m is minimum gap.

$$
C=\frac{150}{5+2.5}
$$

$$
\boxed{C=20}
$$

So the zone has a reference capacity of 20 vehicle-length-plus-gap units.

For 10 detected vehicles\:

$$
\text{density ratio}=\frac{10}{20}=0.5
$$

If four vehicles are halted\:

$$
\text{halted ratio}=\frac{4}{20}=0.2
$$

These values are clipped to the range 0-1.

For a vehicle learner, density is calculated differently because it has a limited local observation region. Its visible capacity depends on its position and 100-metre radius.

This is a useful viva question: *Why don't vehicle and RSU density use exactly the same denominator?*

Because their coverage regions are different.

### 8.5 What are missing-value masks?

Imagine a car has no access to the signal countdown.

There are two possible situations\:

- The signal countdown is truly zero.
- The signal countdown is not available to that car.

These are not the same thing.

If the value is missing, we represent the normalised number as zero and its availability mask as zero.

If the measurement is available and its value is genuinely zero, the mask is one.

This prevents the network from confusing missing information with a measured zero.

### 8.6 A complete illustrative feature vector

Suppose a vehicle observes\:

- Speed = 6.945 m/s
- Acceleration = 1.8 m/s²
- Distance = 90 m
- Local density = 0.5
- Halted ratio = 0.2
- Waiting time = 60 s
- No signal countdown
- No signal state

Its numerical input could look like this\:

```python
[
  0.50,  # speed
  0.20,  # acceleration
  0.30,  # distance
  0.50,  # density
  0.20,  # halted ratio
  0.20,  # waiting
  0.00,  # signal time (unavailable)
  0.00,  # red
  0.00,  # yellow
  0.00,  # green

  1.00,  # speed available
  1.00,  # acceleration available
  1.00,  # distance available
  1.00,  # density available
  1.00,  # halted available
  1.00,  # waiting available
  0.00,  # signal time unavailable
  0.00,  # red unavailable
  0.00,  # yellow unavailable
  0.00,  # green unavailable

  1.00,  # vehicle
  0.00   # not RSU
]
```

Illustrative feature row calculated using the saved schema; not asserted to be a particular row in the selected NPZ file.

The equivalent RSU vector has the two client-type indicators reversed.

### 8.7 How does this become a model input?

Each observation has 22 numbers.

We collect ten consecutive observations.

Therefore\:

$$
X\in\mathbb{R}^{10\times22}
$$

The complete batch format is\:

$$
X\in\mathbb{R}^{B\times10\times22}
$$

For a batch size of 64\:

$$
\boxed{X\text{ shape}=(64,10,22)}
$$

The number 64 is the batch-size setting in the saved learning configuration.

This tensor is passed to the GRU.

---

## Chapter 9 - GRU: How does the model learn from traffic history?

Harshul's major implementation area is the GRU, but the whole team should understand the architecture.

### 9.1 Why GRU?

GRU means **Gated Recurrent Unit**\.

It is a recurrent neural network used for sequential data.

Our traffic records are sequential\:

```text
Time 1 → Time 2 → Time 3 → ... → Time 10
```

A regular feed-forward network can process a row of numbers, but a GRU is designed to process a sequence and maintain useful information from earlier timesteps.

For example, consider\:

| Time | Speed |
|--------|--------|
| 1 | 12 m/s |
| 2 | 11 m/s |
| 3 | 9 m/s |
| 4 | 7 m/s |
| 5 | 5 m/s |
| 6 | 3 m/s |
| 7 | 1 m/s |
| 8 | 0 m/s |
| 9 | 0 m/s |
| 10 | 0 m/s |

The GRU can learn that the vehicle's speed is decreasing and the traffic condition is deteriorating.

It does not just receive the final speed of zero.

It processes the entire ten-step sequence.

### 9.2 Exact model architecture

The saved configuration gives us\:

| Parameter | Value |
|--------|--------|
| Input size | 22 |
| Hidden size | 32 |
| GRU layers | 1 |
| Dropout | 0.0 |
| Shared layer size | 16 |
| Output regression neurons | 1 |
| Output classification neurons | 3 |
| Trainable parameters | 5,972 |

The architecture in `trafma/models.py` is\:

## GRU model architecture

**Input tensor**

## 10 × 22

Ten timesteps and 22 features

**GRU encoder**

# 32

Hidden units · one layer

**Shared dense layer**

# 16

Linear transformation + ReLU

**Queue head**

# 1

Queue ratio

**Class head**

# 3

Free / moderate / heavy logits

### 9.3 What happens inside the GRU?

A GRU uses two main gates.

**Update gate\:** decides how much previous information to retain.

**Reset gate\:** decides how much previous information to ignore when forming new information.

The mathematical equations are\:

$$
z_t=\sigma(W_zx_t+U_zh_{t-1}+b_z)
$$

$$
r_t=\sigma(W_rx_t+U_rh_{t-1}+b_r)
$$

The candidate hidden state is\:

$$
\tilde h_t=\tanh(W_hx_t+U_h(r_t\odot h_{t-1})+b_h)
$$

The final hidden state is\:

$$
h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t
$$

Here\:

- $x_t$ \= input at the current timestep.
- $h_{t-1}$ \= information retained from the previous timestep.
- $z_t$ \= update gate.
- $r_t$ \= reset gate.
- $\sigma$ \= sigmoid function.
- $\odot$ \= element-wise multiplication.

You do not need to memorise every matrix dimension first. Understand the intuition: the GRU decides which historical information is important and which new observation should change its internal state.

### 9.4 What happens after the GRU?

The network takes the final timestep's encoded hidden representation.

$$
h_{10}\in\mathbb{R}^{32}
$$

Then the shared dense layer transforms it\:

$$
32\longrightarrow16
$$

It applies ReLU\:

$$
\operatorname{ReLU}(x)=\max(0,x)
$$

The resulting 16-dimensional representation feeds two separate heads.

### 9.5 Output 1: Queue regression

The queue head has one output neuron.

It produces a queue ratio using sigmoid\:

$$
\hat r=\sigma(W_qh+b_q)
$$

The ratio is between zero and one.

The predicted queue length is\:

$$
\boxed{\hat Q=Z\hat r}
$$

where $Z=150$ metres.

Suppose the model produces\:

$$
\hat r=0.32
$$

Then\:

$$
\hat Q=150(0.32)
$$

$$
\boxed{\hat Q=48\text{ metres}}
$$

### 9.6 Output 2: Congestion classification

The classification head has three outputs\:

```text
Output 0 → Free
Output 1 → Moderate
Output 2 → Heavy
```

These are called logits.

The model converts the logits into class probabilities using softmax\:

$$
P(c_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

For example, suppose\:

```text
Free logits     = 0.5
Moderate logits = 1.5
Heavy logits    = 0.2
```

The corresponding probabilities are approximately\:

| Class | Probability |
|--------|--------|
| Free | 22.4% |
| Moderate | 61.1% |
| Heavy | 16.5% |

The predicted class is Moderate because it has the highest probability.

This is an illustrative neural-network output, not a particular saved prediction.

### 9.7 How are both outputs trained together?

The model has two tasks\:

- Predict queue ratio.
- Predict congestion class.

We combine their losses\:

$$
L=L_{\text{regression}}+L_{\text{classification}}
$$

The saved weights are\:

$$
\lambda_r=1.0
$$

$$
\lambda_c=1.0
$$

The regression uses Smooth L1 loss.

The classification uses cross-entropy loss.

Therefore\:

$$
L_{\text{total}}=
1.0L_{\text{SmoothL1}}+
1.0L_{\text{CrossEntropy}}
$$

Why not use only classification?

Because predicting a category loses information.

For example, predicting Heavy does not tell us whether the queue will be 55 metres or 130 metres.

The regression head tells us the expected queue distance; the classification head gives a simpler traffic-state category.

### 9.8 How many parameters does the GRU contain?

The repository's model has 5,972 trainable parameters.

We can verify that from the layer dimensions.

For a single GRU layer\:

$$
P_{\text{GRU}}=3H(I+H+2)
$$

With $I=22$, $H=32$\:

$$
P_{\text{GRU}}=3(32)(22+32+2)
$$

$$
=96\times56
$$

$$
=5,376
$$

Shared dense layer\:

$$
P_{\text{shared}}=(32\times16)+16=528
$$

Queue head\:

$$
P_{\text{queue}}=(16\times1)+1=17
$$

Classification head\:

$$
P_{\text{class}}=(16\times3)+3=51
$$

Total\:

$$
5,376+528+17+51
$$

$$
\boxed{5,972\text{ parameters}}
$$

That is the same size used by central and federated training.

### 9.9 Training settings

The run's resolved configuration contains\:

| Parameter | Value |
|--------|--------|
| Learning rate | 0.001 |
| Batch size | 64 |
| Configured epochs | 150 |
| Minimum epochs | 50 |
| Early stopping patience | 15 |
| Local FL epochs | 5 |
| Regression loss weight | 1.0 |
| Classification loss weight | 1.0 |
| Deterministic mode | True |
| Device setting | Auto |
| CPU thread setting | 0 |

Important distinction\:

`epochs: 150` is the configured epoch budget. It is not the number of local epochs used inside each federated round.

Federated clients train for five local epochs per round.

The central model's training schedule and federated clients' training schedules must not be confused.

---

## Chapter 10 - Federated Learning: Why do we need it?

This is Laxminarayan's main implementation area.

Suppose we have 350 vehicles and four RSUs.

One possible approach would be to collect all the observations from every vehicle and RSU and put them into one central computer.

That is called **centralised learning**\.

But our project investigates another approach.

Instead of transferring every original training example to one central location, each learner trains locally and transfers its model update.

That is the idea behind **Federated Learning (FL)**\.

### 10.1 A simple example

Imagine three vehicle clients\:

```text
Vehicle A → 20 windows
Vehicle B → 20 windows
Vehicle C → 20 windows
```

Each starts with the same initial GRU model.

They train it using their own examples.

After local training, they have updated model parameters.

They send those parameters to the aggregation system.

The coordinator combines them into an improved model.

The important distinction\:

| Centralised training | Federated training |
|--------|--------|
| Original examples are pooled | Original examples remain in their assigned client datasets |
| One training process sees the pooled data | Many local training processes |
| One central optimiser | Local optimisers and a global aggregation step |
| Simple reference approach | Distributed learning approach |

In our implementation, model-update payloads contain tensors and limited training metadata. They do not contain the original input histories, individual labels or evaluator truth.

However, we must be careful with the privacy claim.

**Federated learning is not automatically differential privacy or guaranteed privacy.**

Model updates can still leak information through various attacks. Our implementation does not establish a formal privacy guarantee.

### 10.2 What is FedAvg?

FedAvg means Federated Averaging.

It is an algorithm for combining the models produced by different learners.

Suppose three clients produce these updates for one model weight\:

| Client | Number of samples | Updated weight |
|--------|--------|--------|
| A | 20 | 0.50 |
| B | 20 | 0.60 |
| C | 20 | 0.70 |

The weighted average is\:

$$
w_{\text{new}}=
\frac{n_Aw_A+n_Bw_B+n_Cw_C}
{n_A+n_B+n_C}
$$

Substitute\:

$$
w_{\text{new}}=
\frac{20(0.50)+20(0.60)+20(0.70)}{60}
$$

$$
=\frac{10+12+14}{60}
$$

$$
\boxed{w_{\text{new}}=0.60}
$$

That is a simplified example of FedAvg.

Our implementation performs this calculation for every tensor in the model, not just one weight.

The model contains 5,972 trainable parameters, so the aggregation is performed over the complete set of model tensors.

### 10.3 What happens during one federated round?

Every round follows this pattern\:

1. Start with the current global model.
2. Distribute its parameters to clients.
3. Each client trains locally for five epochs.
4. Each client produces an updated model.
5. The updates are validated.
6. Vehicle updates go to their parent RSU.
7. Each RSU produces a group-level model update.
8. The coordinator combines the RSU updates.
9. The global model is updated.
10. The new global model is evaluated using validation data.

The same procedure is repeated for 100 rounds in our experiment.

### 10.4 Hierarchical Federated Learning

Our project uses three logical tiers.

## Three-tier HFL

**Tier 3 - Global coordinator**

Final aggregation of four RSU model updates

**North RSU**

Tier 2 · Edge aggregation

**East RSU**

Tier 2 · Edge aggregation

**South RSU**

Tier 2 · Edge aggregation

**West RSU**

Tier 2 · Edge aggregation

**Tier 1 - Vehicle learners**

350 vehicle clients, assigned to their original parent RSUs

Why hierarchical?

Because a four-way intersection naturally has four groups.

North vehicles belong to the North group.

East vehicles belong to the East group.

South vehicles belong to the South group.

West vehicles belong to the West group.

Each RSU first aggregates its own group's updates.

Then the global coordinator aggregates the RSU group results.

### 10.5 Actual client allocation in this experiment

| Group | Vehicle learners | Vehicle windows | RSU's own windows | Total group mass |
|--------|--------|--------|--------|--------|
| North | Approach-specific selected clients | - | 1,750 | - |
| East | Approach-specific selected clients | - | 1,750 | - |
| South | Approach-specific selected clients | - | 1,750 | - |
| West | Approach-specific selected clients | - | 1,750 | - |
| Total | 350 | 7,000 | 7,000 | 14,000 |

The number of selected vehicles per approach is determined by the stratified client selection procedure. It should not be assumed to be exactly 87 or 88 for each direction.

The four RSUs each have exactly 1,750 selected windows.

The four vehicle groups together contribute 7,000 windows.

The four RSU learners contribute another 7,000.

### 10.6 Why does an RSU train its own GRU?

An RSU is both\:

- An infrastructure aggregator.
- A learning client in this particular experiment.

These are two separate responsibilities.

The RSU collects its own sensor histories from the corresponding approach.

It trains its own model update.

It also receives model updates from the vehicles assigned to it.

This is called **RSU self-update enabled**\.

The setting is\:

```yaml
federation:
  rsu_self_update: true
```

If this were disabled, an RSU would still perform its aggregation role, but would not contribute its own separate sensor-history model update.

This is one of the important design choices in your final experiment.

### 10.7 Example of hierarchical aggregation

Let's use simplified invented values for one model parameter.

Suppose the North group has\:

- North RSU: 1,750 samples, weight 0.60.
- Vehicle A: 20 samples, weight 0.50.
- Vehicle B: 20 samples, weight 0.70.

The North RSU first combines the three updates\:

$$
w_N=
\frac{1750(0.60)+20(0.50)+20(0.70)}
{1750+20+20}
$$

$$
w_N=\frac{1050+10+14}{1790}
$$

$$
\boxed{w_N\approx0.599}
$$

This becomes the North group's model update.

The same is done for East, South and West.

Finally, the global coordinator combines the four RSU-level updates.

The actual implementation uses each group's total sample mass as its aggregation weight.

This preserves the total example contribution through the two aggregation levels.

### 10.8 The critical distinction: Do not add the RSU update twice

A common mistake is\:

1. Add all vehicle updates to the global aggregation.
2. Add RSU updates to the global aggregation.
3. Then add the RSU's vehicle aggregate again.

That would double-count vehicle contributions.

Our implementation instead performs\:

```text
Vehicle + RSU self-update
          |
          v
   Parent RSU aggregation
          |
          v
  One group-level update
          |
          v
 Global aggregation
```

The parent RSU's own update is included once in its group.

The global coordinator receives one resulting update for each RSU group.

This matters when explaining the algorithm to an FL professor.

### 10.9 How does the model update travel?

The update is represented by the `ModelUpdate` structure.

It includes\:

| Field | Purpose |
|--------|--------|
| `client_id` | Identity of the learning client |
| `client_type` | Vehicle or RSU |
| `round_id` | Current federated round |
| `base_model_version` | Version of the model used locally |
| `base_model_hash` | Confirms the base model |
| `feature_schema_hash` | Confirms identical feature definitions |
| `sample_count` | Number of examples used locally |
| `tensors` | Updated model parameters |
| `group_id` | Parent RSU group |
| `training_time_s` | Local training duration |
| `training_metadata` | Allowlisted scalar training information |

Why do we use model hashes?

Imagine a client accidentally trains an outdated global model.

Its update should not be combined with updates derived from another model version.

The aggregator checks the base-model version and hash.

It also checks the feature schema, tensor names, dimensions, numerical validity, round identity, deadline and duplicate client identities.

An invalid update can be rejected instead of silently contaminating the global model.

### 10.10 What happens after 100 rounds?

Our experiment stores\:

```text
training_run/
├── checkpoints/
│   ├── round_001.pt
│   ├── round_002.pt
│   ├── ...
│   └── round_100.pt
├── rounds.csv
├── rounds.json
├── client_rounds.csv
├── updates.jsonl
├── predictions.csv
├── metrics.json
├── manifest.json
├── resolved_config.yaml
└── summary.md
```

The most important checkpoint is\:

```text
checkpoints/round_100.pt
```

It stores the model state, model configuration, feature schema hash, model version, round number and model hash.

The prediction file stores both validation and test predictions.

The metrics file stores the corresponding calculated results.

The experiment's wall-clock duration is recorded as approximately 40.3 minutes.

### 10.11 Was this a live federated network?

This needs a careful answer.

The specified 350-client result is described in its saved run summary as\:

`offline_saved_trajectory_replay`

That means the algorithm trains on previously collected simulation trajectories.

The same fixed replay cohort participates in every round.

The system is not actually deploying 350 physical learning devices connected through wireless infrastructure.

There is a separate optional live SUMO federation hook in the code, but it should not be confused with the experiment whose metrics we are discussing.

Similarly, `transient_vehicle_participation_rate: 0.75` appears in the configuration. The recorded run is not a transient-participation experiment; its summary explicitly says the same fixed cohort participates every round.

---

## Chapter 11 - Exact experiment configuration

This is the numerical configuration reference for the requested experiment.

The source file is\:

`artifacts/phase2_compact_350_hfl_rsu_on_7000each/config.yaml`

### 11.1 Learning configuration

| Setting | Exact value |
|--------|--------|
| Dataset | `data/phase2_compact_1500` |
| Training seed | 20261031 |
| Evaluation seed | 20261031 |
| Device | `auto` |
| Threads | 0 |
| Deterministic | True |
| Input size | 22 |
| Sequence length | 10 |
| Hidden size | 32 |
| GRU layers | 1 |
| Dropout | 0.0 |
| Shared dense size | 16 |
| Learning rate | 0.001 |
| Batch size | 64 |
| Epoch budget | 150 |
| Minimum epochs | 50 |
| Early stopping patience | 15 |
| Regression loss weight | 1.0 |
| Classification loss weight | 1.0 |
| Maximum training windows | 14,000 |
| Maximum validation windows | 14,000 |
| Local epochs | 5 |

`device: auto` means the training environment selects the available PyTorch device. It does not, by itself, establish which hardware was used.

### 11.2 Federated configuration

| Setting | Exact value |
|--------|--------|
| Algorithm | Hierarchical FedAvg |
| Communication rounds | 100 |
| Maximum clients | 354 |
| Cohort seed | 20261031 |
| Vehicle windows per client | 20 |
| RSU windows per group | 1,750 |
| Require full client budget | True |
| RSU self-update | True |
| Aggregation | Sample-weighted model weights |
| Deadline setting | 300 seconds |
| Live round interval setting | 60 seconds |
| Transient participation setting | 0.75 |
| IID seed | 20261031 |

The deadline and live-round interval are configuration settings. Because this run is an offline replay, do not interpret the live-round interval as evidence of actual wireless communication latency.

### 11.3 Dataset and experiment identifiers

| Property | Recorded value |
|--------|--------|
| Dataset | `phase2_compact_1500` |
| Feature schema | `trafma.features/1` |
| Feature dimension | 22 |
| Sequence length | 10 |
| Vehicle learners | 350 |
| RSU learners | 4 |
| Total clients | 354 |
| Training windows | 14,000 |
| Validation windows | 14,000 |
| Test windows | 14,000 |
| Training rounds | 100 |
| Local epochs per round | 5 |
| Wall duration | 40.3 minutes |
| Training seed | 20261031 |

### 11.4 Selected-data manifest fingerprints

The manifest records\:

```text
dataset_id:
phase2_compact_1500

feature_dimension:
22

sequence_length:
10

training_learners:
350 vehicles + 4 RSUs

selected training:
14,000 windows

selected validation:
14,000 windows

selected test:
14,000 windows
```

The feature schema hash is\:

```text
46196acbcc933a9b60f0a3151cf3a5822350c6988b7894a50f5722ac80f4abe2
```

The run manifest hash recorded by the selected-data manifest is\:

```text
2efe03a811a2474b7f1246d921cb1ba1bdc0ed25003c8df500202fd123cac86f
```

The selected tensor file is separately hashed, as are the CSV metadata and source dataset manifest.

This makes it possible to check whether an exported dataset or experiment package has been changed.

---

## Chapter 12 - Experimental results: What did we achieve?

Now we reach the most important question\:

**Did the trained model actually predict traffic correctly?**

The final selected run's test results are\:

| Metric | Result |
|--------|--------|
| Test rows | 14,000 |
| Queue MAE | 27.352 m |
| Queue RMSE | 42.185 m |
| Macro-F1 | 0.7078 |
| Balanced accuracy | 0.6925 |
| Total training windows | 14,000 |
| Training rounds | 100 |

### 12.1 Understanding MAE

MAE means Mean Absolute Error.

Formula\:

$$
MAE=\frac{1}{N}\sum_{i=1}^{N}|y_i-\hat y_i|
$$

Where\:

- $y_i$ \= actual queue length.
- $\hat y_i$ \= predicted queue length.
- $N$ \= number of test examples.

Our result is\:

$$
\boxed{MAE=27.352\text{ m}}
$$

This means that across the 14,000 evaluated samples, the average absolute queue forecast error is about 27.35 metres.

It does not mean every prediction is wrong by 27 metres.

Some errors are smaller and some are considerably larger.

### 12.2 Understanding RMSE

RMSE means Root Mean Squared Error.

$$
RMSE=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2}
$$

Our result\:

$$
\boxed{RMSE=42.185\text{ m}}
$$

Why is RMSE larger than MAE?

Because RMSE squares the individual errors.

A large error therefore has a much stronger effect.

For our experiment\:

$$
42.185>27.352
$$

This shows that larger misses contribute noticeably to the overall prediction error.

### 12.3 Understanding macro-F1

F1 combines precision and recall.

$$
F1=\frac{2PR}{P+R}
$$

Where\:

- $P$ \= precision.
- $R$ \= recall.

Macro-F1 calculates the F1 score for each class and averages the three class scores\:

$$
MacroF1=\frac{F1_{\text{free}}+F1_{\text{moderate}}+F1_{\text{heavy}}}{3}
$$

Our saved result\:

$$
\boxed{MacroF1=0.7078}
$$

This is approximately 70.78% on a 0-100 scale.

It is not the same thing as 70.78% accuracy.

### 12.4 Understanding balanced accuracy

Balanced accuracy calculates the recall for each class and averages them.

$$
BA=\frac{R_0+R_1+R_2}{3}
$$

The saved result\:

$$
\boxed{BA=0.6925}
$$

Therefore, the average class recall is around 69.25%.

This is useful because our congestion categories are not equally represented.

### 12.5 Performance by learner type

One of the most interesting findings is the large difference between the RSU and vehicle test cohorts.

| Metric | RSU | Vehicle |
|--------|--------|--------|
| Test examples | 7,000 | 7,000 |
| MAE | 6.995 m | 47.709 m |
| RMSE | 11.707 m | 58.498 m |
| Macro-F1 | 0.8880 | 0.5282 |
| Balanced accuracy | 0.8886 | 0.5449 |
| Accuracy | 0.9164 | 0.5521 |

Why is the RSU score much better?

An RSU has a broader view of its approach.

It observes the configured 150-metre zone and receives a more representative aggregate picture of that road's condition.

A vehicle sees a smaller region, has position-dependent visibility and may not have signal information available.

The model therefore has less information for vehicle-level prediction.

This is a plausible explanation consistent with the observation design. The score difference alone does not prove a particular causal mechanism.

### 12.6 Performance by approach

| Approach | Test rows | Queue MAE | RMSE | Macro-F1 |
|--------|--------|--------|--------|--------|
| East | 3,226 | 25.331 m | 40.826 m | 0.7163 |
| North | 3,892 | 29.782 m | 44.124 m | 0.6993 |
| South | 3,933 | 32.229 m | 47.141 m | 0.6942 |
| West | 2,949 | 19.851 m | 33.040 m | 0.7183 |
| **Total** | **14,000** | **27.352 m** | **42.185 m** | **0.7078** |

Notice that the North, South, East and West test row counts are not equal.

The report aggregates all test predictions; each approach's metrics are calculated on its own available test rows.

### 12.7 Confusion matrix: What classes did the model confuse?

For the RSU cohort, the saved confusion matrix is\:

| Actual \\ Predicted | Free | Moderate | Heavy |
|--------|--------|--------|--------|
| Free | 3,873 | 189 | 40 |
| Moderate | 209 | 924 | 88 |
| Heavy | 0 | 59 | 1,618 |

For the vehicle cohort\:

| Actual \\ Predicted | Free | Moderate | Heavy |
|--------|--------|--------|--------|
| Free | 2,186 | 135 | 180 |
| Moderate | 445 | 355 | 105 |
| Heavy | 2,178 | 92 | 1,324 |

The diagonal numbers are correct classifications.

For example, RSU Free predictions\:

$$
\text{Correct free predictions}=3873
$$

The model also predicted some actual Free examples as Moderate or Heavy.

This is called misclassification.

The vehicle cohort has a particularly difficult heavy-class problem.

It misclassified 2,178 actual Heavy examples as Free.

That is an important weakness to be prepared to discuss.

### 12.8 Why can the model have good RSU and weak vehicle performance?

Think about the difference in observation\:

**RSU**

- Sees a larger section of the approach.
- Has zone-level vehicle counts.
- Calculates mean speed and queue information from its zone.
- Has a dedicated sensor-history dataset.

**Vehicle**

- Sees its own local information.
- Uses nearby same-approach vehicles.
- Has a smaller visible segment.
- Does not have direct access to the full queue length.

A vehicle can be moving freely while a larger queue exists farther ahead, or it might observe a local slowdown that does not represent the full approach.

That is one reason vehicle predictions are more difficult.

### 12.9 A same-cohort comparison with other 350-vehicle methods

The repository's status document records a matched four-experiment comparison, together with a Ridge baseline.

These values are useful for understanding where our chosen method sits\:

| Method | Queue MAE | Macro-F1 |
|--------|--------|--------|
| Signal-aware Ridge | 29.905 m | 0.5335 |
| Central GRU | 14.905 m | 0.8357 |
| Flat FedAvg | 57.391 m | 0.2669 |
| Hierarchical FedAvg, RSU self-update off | 57.782 m | 0.2524 |
| **Hierarchical FedAvg, RSU self-update on** | **27.352 m** | **0.7078** |

This gives us a very important conclusion.

The central GRU performed best on this particular cohort.

The self-update-on HFL experiment performed substantially better than the other saved federated comparisons.

But its result did not outperform central training.

### 12.10 How much better was RSU self-update-on than self-update-off?

Macro-F1\:

$$
0.7078-0.2524=0.4554
$$

The difference is 0.4554 absolute macro-F1 points.

Queue MAE\:

$$
57.782-27.352=30.430\text{ m}
$$

The enabled run had 30.43 metres lower MAE.

However, there is a confounding factor\:

| Condition | Training windows |
|--------|--------|
| RSU self-update off | 7,000 |
| RSU self-update on | 14,000 |

The enabled experiment adds four learning clients and 7,000 additional RSU samples.

Therefore, we cannot claim that the entire improvement was caused exclusively by RSU self-update.

We can say\:

> The saved self-update-enabled experiment achieved considerably better results than the self-update-disabled experiment under the tested configurations. The additional RSU learners and training budget are part of that comparison, so a matched-budget experiment is needed to isolate the effect of RSU self-training.

That is a stronger academic answer because it does not overclaim.

### 12.11 Why is central GRU better?

Central training can learn directly from the pooled 14,000 training examples.

The federated model learns through repeated local updates and aggregation.

Natural non-IID data, local optimisation, limited examples per vehicle and the hierarchy can all influence convergence.

We cannot conclude from one run that federated learning itself is inherently worse.

We can conclude that the implemented federation settings did not match central performance in this experiment.

### 12.12 Are 14,000 test rows 14,000 independent observations?

No.

The model creates overlapping sliding windows.

For example\:

```text
Window 1: times 100-109
Window 2: times 101-110
Window 3: times 102-111
```

These windows share nine out of ten observations in adjacent pairs.

Therefore, they are correlated.

The evaluation also uses one training seed.

So we should describe the results as single-seed simulation results, not as an independent-sample statistical guarantee.

---

## Chapter 13 - What happens to every output file?

This chapter connects the mathematics to the actual project folders.

| Output | What it contains | Why it matters |
|--------|--------|--------|
| `selected_data/manifest.json` | Data provenance and counts | Verifies the selection |
| `selected_data/selected_windows.csv` | Metadata for each selected window | Human-readable inspection |
| `selected_data/selected_windows.npz` | Numerical sequences and targets | Actual ML input |
| `training_run/resolved_config.yaml` | Final configuration | Reproducibility |
| `training_run/updates.jsonl` | Model update records | Audit federation |
| `training_run/client_rounds.csv` | Per-client round activity | Check accepted and rejected clients |
| `training_run/rounds.csv` | Per-round metrics and update summaries | Understand convergence |
| `training_run/rounds.json` | Detailed round records | Inspect individual aggregation |
| `training_run/checkpoints/round_001.pt` through `round_100.pt` | Model state at each round | Reconstruct model progression |
| `training_run/predictions.csv` | Validation and test predictions | Compare actual versus predicted |
| `training_run/metrics.json` | Regression and classification results | Quantitative evaluation |
| `report/report.md` | Experiment explanation | Easy-to-read report |
| `report/report.html` | Graphical experiment report | Presentation |

### 13.1 Follow one example through the entire system

Let's follow an illustrative North-approach vehicle.

```text
veh_0042
```

**Step 1 - SUMO**

SUMO says the vehicle is at a particular position, travelling at a particular speed.

**Step 2 - Vehicle agent**

The agent receives its permitted local observations.

For example\:

```text
Speed = 6.945 m/s
Acceleration = 1.8 m/s²
Waiting = 60 seconds
```

**Step 3 - Message layer**

It sends its vehicle report to `rsu_N`\.

**Step 4 - RSU layer**

The North RSU processes that report and separately calculates its own local-zone measurements.

**Step 5 - Dataset layer**

The ten consecutive valid observations are transformed into a 10 × 22 feature window.

**Step 6 - Evaluator**

The label is obtained from the future queue at 30 seconds after the end of the window.

**Step 7 - GRU**

The model receives the 10 × 22 tensor.

**Step 8 - Local training**

The vehicle learner uses its selected 20 examples and trains for five epochs within each federated round.

**Step 9 - Hierarchical aggregation**

Its updated model parameters go to `rsu_N`\.

**Step 10 - Global aggregation**

The North, East, South and West RSU group updates reach the coordinator.

**Step 11 - Prediction**

The final GRU produces\:

```text
Predicted queue ratio
Predicted congestion logits
```

**Step 12 - Output**

The prediction is stored in the prediction file and used for evaluation.

That is the complete information-to-learning journey.

---

## Chapter 14 - Team responsibilities: Shubro, Harshul and Laxminarayan

Your division of work is\:

| Team member | Main responsibility | Major project area |
|--------|--------|--------|
| Shubro | SUMO simulation | Network generation, demand, traffic signals, simulation |
| Harshul | GRU | Feature handling, architecture, neural training and outputs |
| Laxminarayan | Federated Learning | Client selection, FedAvg, hierarchical aggregation and experiment records |

However, in the viva, all three must understand the entire pipeline.

### 14.1 Shubro - SUMO and simulation

Must be able to explain\:

- What SUMO is.
- What TraCI does.
- What netconvert does.
- Why four incoming approaches are required.
- Why the simulator uses split phasing.
- How vehicle routes are generated.
- How left-hand traffic is configured.
- How speed and positions are obtained.
- How the RSUs and vehicle agents receive observations.
- How simulation outputs become logs.

**Questions Shubro should be particularly prepared for\:**

What if SUMO teleports a vehicle?

What is the difference between simulation time and wall-clock time?

How do you generate reproducible demand?

How does a signal phase affect the queue?

### 14.2 Harshul - GRU and deep learning

Must be able to explain\:

- What the feature adapter does.
- Why there are 22 features.
- Why the input is 10 × 22.
- What normalisation does.
- How missing values are handled.
- Why GRU was selected.
- What hidden size means.
- What update and reset gates do.
- Why there are two output heads.
- What Smooth L1 and cross-entropy do.
- How model performance is evaluated.

**Questions Harshul should be particularly prepared for\:**

Why GRU instead of LSTM?

What is vanishing gradient?

Why not use a transformer?

What is softmax?

Why is RMSE higher than MAE?

### 14.3 Laxminarayan - Federated Learning

Must be able to explain\:

- What a federated client is.
- Why we do local training.
- What a model update contains.
- What FedAvg does.
- What sample-weighted aggregation means.
- Why there are four RSUs.
- How the hierarchy works.
- Why the RSU has its own independent local training data.
- Why self-update-on and off differ.
- Why we have 100 rounds.
- Why the FL experiment is offline replay.
- Why the central GRU outperformed HFL here.
- Why FL is not automatically a privacy guarantee.

**Questions Laxminarayan should be particularly prepared for\:**

What happens if a client sends an outdated model?

How do you handle non-IID data?

Why use hierarchical rather than flat FedAvg?

What happens if one RSU receives no updates?

Why does each RSU get one group contribution rather than sending all its children's updates directly to the global aggregator?

---

## Chapter 15 - The complete formula sheet

These are the formulas worth keeping together for your examination.

| Concept | Formula |
|--------|--------|
| Signal cycle | $T=4(G+Y+A)$ |
| Queue ratio | $r=Q/Z$ |
| RSU zone capacity | $C=Z/(L+G_{\min})$ |
| Speed normalisation | $v'=v/v_{\max}$ |
| Acceleration normalisation | $a'=a/[2\max(a_{\max},d_{\max})]$ |
| Distance normalisation | $d'=d/L_{\text{arm}}$ |
| Waiting normalisation | $w'=w/300$ |
| GRU hidden state | $h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t$ |
| Queue prediction | $\hat Q=150\hat r$ |
| Softmax | $p_i=e^{z_i}/\sum_j e^{z_j}$ |
| Total loss | $L=L_r+L_c$ |
| FedAvg | $w_g=\sum_i n_iw_i/\sum_i n_i$ |
| MAE | $\sum\|y-\hat y\|/N$ |
| RMSE | $\sqrt{\sum(y-\hat y)^2/N}$ |
| Precision | $TP/(TP+FP)$ |
| Recall | $TP/(TP+FN)$ |
| F1 | $2PR/(P+R)$ |
| Macro-F1 | Mean of the three class F1 scores |
| Balanced accuracy | Mean of the three class recalls |

### Quick numerical revision

$$
\text{Signal cycle}=120\text{ seconds}
$$

$$
\text{RSU zone}=150\text{ metres}
$$

$$
\text{Feature input}=10\times22
$$

$$
\text{GRU parameters}=5,972
$$

$$
\text{Vehicle training}=350\times20=7,000
$$

$$
\text{RSU training}=4\times1,750=7,000
$$

$$
\text{Total}=14,000
$$

$$
\text{Rounds}=100
$$

$$
\text{Local epochs}=5
$$

$$
MAE=27.352\text{ m}
$$

$$
RMSE=42.185\text{ m}
$$

$$
MacroF1=0.7078
$$

$$
BA=0.6925
$$

---

# Chapter 16 - Professor's viva question bank with answers

These questions are divided into machine learning, computer vision, deep learning, SUMO, datasets, and federated learning.

The answers are written in simple language, but the technical vocabulary is retained so you can answer at an engineering level.

## A. General project questions

**1\. What is the main objective of TRAF-MA?**

To predict future traffic queue length and congestion class using distributed vehicle and roadside observations, a GRU prediction model and hierarchical federated learning.

**2\. What does TRAF-MA stand for?**

TRAF-MA is the project's name. The implemented focus is traffic simulation, machine learning and hierarchical federated learning.

**3\. What is the input to the complete system?**

Vehicle and RSU observations generated from SUMO, including speed, acceleration, distance, waiting time, traffic density and signal information.

**4\. What is the final output?**

A future queue-length prediction in metres and a congestion category: Free, Moderate or Heavy.

**5\. Why predict 30 seconds in advance?**

It gives a future traffic estimate rather than only identifying the current situation. It is a chosen experimental forecast horizon, not a proven optimal horizon.

**6\. Why are there two prediction outputs?**

Queue regression predicts an exact continuous value, while classification provides an easier-to-interpret traffic category.

**7\. Why use both vehicle and RSU learners?**

Vehicles provide local observations, whereas RSUs provide approach-level observations. Combining the two lets us study different sensing capabilities in distributed learning.

**8\. Is this a real-world traffic deployment?**

No. It is a SUMO-based simulation experiment with software agents and an offline federated training evaluation. It is a research prototype, not a deployed traffic-control system.

**9\. Does the GRU control traffic signals?**

No. The current implementation predicts traffic conditions. Signal timings are configured in the SUMO environment and are not controlled by the trained GRU.

**10\. What is the main research challenge?**

Learning useful future traffic predictions from distributed, limited and naturally different observations while aggregating the models across vehicle and infrastructure groups.

## B. Machine learning questions

**11\. What is machine learning?**

Machine learning allows a computer to learn patterns from examples instead of requiring every decision to be manually programmed.

**12\. What type of ML problem is this?**

It is a multi-task supervised learning problem involving regression and classification.

**13\. Why is it supervised learning?**

Because every input window is associated with a known future queue value and congestion label generated from the SUMO evaluator.

**14\. What is regression?**

Regression predicts a numerical quantity. Here it predicts queue length in metres.

**15\. What is classification?**

Classification predicts a category. Here the possible categories are Free, Moderate and Heavy.

**16\. What is a feature?**

A feature is an input variable used by a machine learning model, such as speed, waiting time or traffic density.

**17\. What is a target variable?**

It is the correct output the model must learn to predict. Our targets are future queue ratio and future congestion class.

**18\. What is normalisation?**

Normalisation scales input numbers into comparable ranges so a feature with large numerical values does not dominate merely because of its units.

**19\. What is data leakage?**

Data leakage occurs when information from the target or evaluation dataset unintentionally enters the model's training inputs or selection process.

**20\. How do you prevent data leakage?**

We exclude future queue and congestion-label fields from the feature schema and keep complete simulation runs separate between training, validation and testing.

**21\. What is overfitting?**

Overfitting occurs when a model memorises training examples but performs poorly on unseen data.

**22\. How can overfitting be reduced?**

Using validation-based checkpoint selection, regularisation, suitable model capacity, early stopping and independent test evaluation.

**23\. Why do we need a validation set?**

It helps select the model configuration and checkpoint without using the final test set to make those decisions.

**24\. What is a test set?**

It is the held-out data used to evaluate the final model after training and model selection.

**25\. What is a hyperparameter?**

A configuration value selected before or during model development, such as learning rate, batch size and hidden size.

## C. Deep learning and GRU questions

**26\. What is deep learning?**

Deep learning is a part of machine learning that uses neural networks containing layers of trainable parameters to learn patterns.

**27\. Why did you choose GRU?**

Our data is sequential. A GRU can retain useful information from previous traffic observations through its recurrent hidden state and gates.

**28\. What does GRU stand for?**

Gated Recurrent Unit.

**29\. Why GRU instead of a normal feed-forward network?**

A feed-forward network does not inherently maintain a recurrent hidden state across timesteps. GRU is designed to process sequence information.

**30\. What are the two GRU gates?**

The reset gate and update gate.

**31\. What is the update gate?**

It controls how information from the previous hidden state and current candidate state is combined.

**32\. What is the reset gate?**

It controls how much of the previous hidden state is used when calculating the candidate hidden state.

**33\. What is hidden size?**

It is the number of values in the GRU hidden representation. Our implementation uses 32.

**34\. Why ten timesteps?**

It is the configured history length. At one-second sampling, ten timesteps represent ten consecutive observations.

**35\. Why 22 input features?**

Ten normalised traffic/signal measurements, ten availability masks and two client-type indicators.

**36\. What is the purpose of the availability masks?**

They indicate which measurements were actually available. This helps differentiate an unavailable value from a genuinely measured zero.

**37\. Why is dropout zero?**

That is the configured value in this experiment. It should not be interpreted as evidence that dropout is universally unnecessary.

**38\. What is ReLU?**

Rectified Linear Unit\:

$$
ReLU(x)=\max(0,x)
$$

It introduces non-linearity into the shared dense layer.

**39\. Why use sigmoid for queue regression?**

The model predicts a queue ratio between zero and one. Sigmoid constrains the predicted ratio to this range.

**40\. Why use softmax?**

Softmax converts classification logits into a probability distribution across the three congestion categories.

**41\. What is backpropagation?**

Backpropagation calculates gradients of the loss with respect to model parameters so the optimiser can update them.

**42\. What is an optimiser?**

An algorithm that updates model parameters using their gradients. The local training implementation uses Adam.

**43\. What is a learning rate?**

It controls the step size of parameter updates. Our saved setting is 0.001.

**44\. What is batch size?**

The number of examples processed together in one training batch. Our configured batch size is 64.

**45\. What is an epoch?**

One complete pass through the examples assigned to a training process. Federated clients use five local epochs per round in this experiment.

**46\. Why is there more than one output head?**

Because queue regression and congestion classification solve related but different prediction tasks. They can share learned traffic representations.

**47\. What is Smooth L1 loss?**

A regression loss that behaves quadratically for small errors and approximately linearly for larger errors. It is less sensitive to large outliers than ordinary squared error.

**48\. What is cross-entropy loss?**

A loss that penalises incorrect probability assignments for classification.

**49\. What is vanishing gradient?**

It occurs when gradients become very small while propagating through many network steps, making earlier layers or timesteps difficult to train.

**50\. Does GRU solve vanishing gradients completely?**

No. Gates help preserve useful information and make training easier, but they do not eliminate every gradient problem.

**51\. Why not use LSTM?**

LSTM is also suitable for sequences. GRU has a simpler gated structure, making it a reasonable choice for this implementation. A controlled LSTM comparison would be needed to establish which performs better.

**52\. Why not use a transformer?**

A transformer is another possible sequence architecture, but our implemented baseline is GRU. A transformer could be evaluated under a matched dataset, feature, training and compute budget.

**53\. How many trainable parameters does the model have?**

5,972 for the specified architecture.

**54\. What are the two heads' dimensions?**

One output neuron for queue ratio and three output neurons for congestion logits.

**55\. What does a model checkpoint contain?**

The saved model state and associated configuration, model version, round and hash metadata.

## D. Computer vision questions

This section is particularly important because you mentioned a professor who may teach computer vision.

Be precise here: **our current implementation is not an image-based computer vision model.**

**56\. Is your project a computer vision project?**

Not in its current implementation. It uses structured numerical observations generated by SUMO rather than images or video frames.

**57\. Why is this relevant to computer vision?**

Traffic monitoring is a common computer vision application. A future version could use traffic-camera images to detect vehicles, estimate vehicle positions and derive traffic density.

**58\. What is object detection?**

Object detection identifies objects in an image and predicts bounding boxes and class labels. For traffic, the objects could be cars, buses and motorcycles.

**59\. Which object detection model could you use?**

YOLO is one option. An object detector could detect vehicles from traffic-camera frames.

**60\. How would you calculate vehicle count from a camera?**

Run object detection on each frame and count unique detected vehicles inside a predefined road region. Tracking is needed to avoid repeatedly counting the same vehicle across frames.

**61\. Why is vehicle tracking needed?**

Object detection detects vehicles in individual frames. Tracking connects those detections across consecutive frames, allowing us to estimate their movement and speed.

**62\. What is a bounding box?**

A rectangular region around an identified object, usually represented by its coordinates and width and height.

**63\. What is Intersection over Union (IoU)?**

It measures overlap between a predicted bounding box and its ground-truth box\:

$$
IoU=\frac{\text{Area of intersection}}{\text{Area of union}}
$$

**64\. What is precision in object detection?**

It measures the fraction of predicted detections that are correct.

$$
Precision=\frac{TP}{TP+FP}
$$

**65\. What is recall in object detection?**

It measures the fraction of actual objects successfully detected.

$$
Recall=\frac{TP}{TP+FN}
$$

**66\. What is non-maximum suppression?**

It removes overlapping duplicate detections, keeping the most suitable bounding box for an object.

**67\. How would computer vision integrate with TRAF-MA?**

A camera pipeline could\:

1. Capture traffic frames.
2. Detect vehicles.
3. Track vehicles across frames.
4. Estimate speed and spatial position.
5. Calculate queue length and density.
6. Convert measurements into the existing feature schema.
7. Feed them to the trained prediction model.

**68\. Why not directly train on images?**

Image-based training would require additional data, a visual encoder and substantially different preprocessing. The current saved model expects structured 22-value feature sequences, not images.

**69\. What is an optical flow?**

It estimates apparent pixel movement between video frames and could help estimate motion, although camera calibration and vehicle tracking would be important for actual speed estimation.

**70\. What is perspective transformation?**

It maps image coordinates into a road-plane coordinate system using a calibrated homography. This could help convert image positions into approximate physical distances.

## E. SUMO and simulation questions

**71\. What does SUMO stand for?**

Simulation of Urban MObility.

**72\. Why did you use SUMO?**

It provides a reproducible environment to generate vehicle movement, road traffic and traffic signal states.

**73\. What is TraCI?**

Traffic Control Interface. It allows external programs, such as our Python project, to interact with a running SUMO simulation.

**74\. What does netconvert do?**

It converts road-network input descriptions into SUMO network files.

**75\. Why four approaches?**

The project models a four-arm intersection with North, South, East and West approaches.

**76\. What is split phasing?**

A signal configuration in which each approach receives its own protected green phase, rather than allowing conflicting approaches to receive green simultaneously.

**77\. How do you calculate the signal cycle?**

$$
T=4(25+3+2)=120\text{ seconds}
$$

**78\. What is a vehicle route?**

The sequence of road edges that a vehicle follows through the network.

**79\. What is a simulation seed?**

A value used to initialise pseudo-random processes. The same seed and configuration make the simulator's random choices reproducible.

**80\. What is the difference between simulation time and wall time?**

Simulation time advances according to the virtual traffic model. Wall time measures how many real seconds or minutes the computer takes to execute that simulation or training process.

**81\. What is a SUMO teleport?**

A simulator event where a vehicle is temporarily removed from its normal location to resolve situations such as severe blocking or gridlock.

**82\. What is an RSU zone?**

The configured region of an approach that an RSU is allowed to observe. Our standard setting is 150 metres.

**83\. What is a vehicle sensing radius?**

It defines the local neighbourhood available to a vehicle learner. Our standard setting is 100 metres.

**84\. Do the RSUs change when a vehicle crosses the junction?**

No. The current vehicle-parent relationship remains fixed for the vehicle's journey. There is no route-aware handoff in the current implementation.

**85\. Are the RSUs physical cameras?**

No. They are software agents in the simulation, not deployed physical cameras or radar devices.

## F. Dataset and evaluation questions

**86\. Why are the observations sampled at one-second intervals?**

That is the configured report interval, giving a consistent time series for the sequence model.

**87\. What is the future horizon?**

Thirty seconds after the last observation in a sequence.

**88\. What is a sliding window?**

A fixed-length subset of consecutive observations used as one training example. A new window moves forward by one observation.

**89\. Why do windows overlap?**

Moving one second at a time increases the number of available sequential examples and captures gradual traffic changes.

**90\. What is the disadvantage of overlapping windows?**

Adjacent windows share most of their observations, so they cannot be treated as independent samples for statistical confidence calculations.

**91\. How many training examples do we use?**

14,000: 7,000 vehicle and 7,000 RSU windows.

**92\. How many test examples do we use?**

14,000, consisting of 7,000 vehicle and 7,000 RSU windows.

**93\. Why split by complete simulation runs?**

It reduces the risk that windows from the same trajectory appear in training and testing.

**94\. What is a confusion matrix?**

A table showing the actual class against predicted class, allowing us to inspect correct and incorrect classifications by category.

**95\. What is macro-F1?**

The arithmetic mean of the individual class F1 scores, giving each class equal importance.

**96\. Why is RMSE higher than MAE in our experiment?**

Large errors contribute more strongly to RMSE because individual errors are squared before averaging.

**97\. Why is vehicle performance lower than RSU performance?**

Vehicle observations are local and incomplete, whereas the RSU has a broader approach-level view.

**98\. What does balanced accuracy tell us?**

It gives the average recall across the classes, helping when the class distribution is unequal.

**99\. Is 0.7078 macro-F1 the same as 70.78% accuracy?**

No. Macro-F1 averages the three class F1 scores. Accuracy measures the percentage of all predictions that are correct.

**100\. How can you improve the experiment's evaluation?**

Use multiple independent training seeds and additional held-out simulation runs, with run-level confidence intervals and balanced comparisons.

## G. Federated learning questions

**101\. What is federated learning?**

A learning method where separate clients train locally and send model updates for aggregation instead of sending their original training examples to a central training dataset.

**102\. What is FedAvg?**

Federated Averaging. It combines local model parameters using sample-weighted averaging.

**103\. What is a federated client in your project?**

An individual vehicle learner or one of the four persistent RSU learners.

**104\. Why 354 clients?**

The chosen experiment contains 350 vehicle learners and four RSU learners.

**105\. Why hierarchical federated learning?**

It organises vehicle clients into four geographical approach groups. Their parent RSUs aggregate local updates before the global coordinator performs the final aggregation.

**106\. What is a global model?**

The shared neural network state produced after aggregating the participating clients' updates.

**107\. What is a local model?**

A client-specific copy of the global model that is trained on the client's own examples.

**108\. What is non-IID data?**

Non-independent and identically distributed data. Different clients can see different types and proportions of examples.

**109\. Why is non-IID data expected in this project?**

Vehicles and RSUs have different observation sizes and histories. Their approaches and simulated traffic conditions also vary.

**110\. What is IID data?**

Data where examples follow the same underlying distribution and are independent. It is often used as a diagnostic or simplified assumption.

**111\. What is the purpose of FedAvg weighting by sample count?**

It gives clients with more training samples a proportionally larger contribution to the weighted average.

**112\. What is RSU self-update?**

The RSU trains on its own sensor histories in addition to aggregating updates from vehicle learners assigned to it.

**113\. What happens when RSU self-update is disabled?**

The RSU remains an aggregation point but does not add its own independent sensor-history model update.

**114\. Why did the RSU self-update-on configuration perform better in the saved comparison?**

In the new 350v250 comparison, RSU self-update-on reached macro-F1 0.8344, versus 0.6781 with it off. The on run also had 7,000 extra RSU training windows (94,500 versus 87,500 total), so the difference cannot be attributed to self-update alone. The older 20-window comparison had the same budget confound at 14,000 versus 7,000 windows.

**115\. Why did central GRU outperform HFL?**

In the new comparison, central GRU had macro-F1 0.8733 and queue MAE 9.281 m; hierarchical FedAvg with RSU self-update on had 0.8344 and 15.113 m. Both used 94,500 training windows and the same held-out identities. Central training pools the selected examples directly, while HFL trains locally and aggregates. This single-seed result does not establish how the methods generalise across traffic runs.

**116\. What is a federated round?**

One complete iteration of local training, update submission and global model aggregation.

**117\. Why 100 rounds?**

That is the configured experimental training schedule. It allows repeated updates from the same fixed client cohort.

**118\. Why five local epochs?**

It is the chosen configuration controlling how many local dataset passes each client performs before producing an update.

**119\. What is a model update?**

The set of model parameters after a client completes local training, along with the required identity, sample-count and validation metadata.

**120\. Do clients send their entire dataset to the server?**

Not in the saved federated training implementation. The update messages contain model tensors and permitted metadata, not raw training histories or individual labels.

**121\. Does FL guarantee privacy?**

No. It reduces the need to centralise raw training examples but does not guarantee that model updates cannot reveal information. Additional privacy mechanisms can be required.

**122\. What happens if a client submits a duplicate update?**

The aggregation code detects duplicate client identities and rejects the conflicting duplicate contribution.

**123\. What happens if a client sends an outdated update?**

The update is checked against the current model version and model hash. An outdated update can be rejected.

**124\. What is the purpose of the feature-schema hash?**

It ensures clients use the same feature definition, order and shape. Otherwise, their model updates might not represent equivalent inputs.

**125\. What is the difference between flat and hierarchical FedAvg?**

Flat FedAvg aggregates client updates directly into one global model. Hierarchical FedAvg first aggregates them at their parent RSU groups and then aggregates the four group updates.

**126\. Does the current experiment use transient client participation?**

No. Its saved run summary states that the same fixed replay cohort participates in every round.

**127\. Is the 350-client experiment a live network experiment?**

No. It is offline saved-trajectory replay. A separate optional live SUMO federation mechanism exists in the repository.

**128\. What is one limitation of the current FL comparison?**

The self-update-enabled experiment uses 94,500 training examples, compared with 87,500 in the vehicle-only self-update-disabled experiment. The extra 7,000 are RSU sensor windows, so this comparison does not isolate the effect of RSU self-training. The same issue existed in the 20-window comparison, where the totals were 14,000 versus 7,000.

**129\. What is another limitation?**

There is one training seed and overlapping test windows, so the scores are descriptive rather than a statistically established generalisation estimate.

**130\. How would you improve federated learning performance?**

Test matched training budgets, vary local epochs and client participation separately, repeat experiments with multiple seeds, examine client label distributions and compare alternative aggregation methods.

## H. Research and advanced questions

**131\. What is multi-task learning?**

Learning multiple related prediction tasks with a shared representation. Our model jointly predicts queue ratio and congestion category.

**132\. Why is multi-task learning useful here?**

Queue length and congestion class are related. Both use information about the same traffic condition, so a shared representation can support both tasks.

**133\. What is concept drift?**

A change in the underlying data relationship over time. For example, a traffic model trained on one traffic pattern may become less useful when traffic demand or driving behaviour changes.

**134\. What is model drift?**

A deterioration in model performance as new data distributions differ from the training environment.

**135\. How would you detect concept drift?**

Monitor changing input distributions and prediction errors over time, using matured ground-truth labels where they are available.

**136\. What is an ablation study?**

An experiment in which one component is removed or changed to understand its contribution to the overall system.

**137\. What ablation would you perform next?**

Compare hierarchical FL with and without RSU self-updates while holding the number of training examples constant.

**138\. What is weighted averaging?**

Combining parameters or values using weights proportional to their importance or sample count.

**139\. What is model convergence?**

The process of a model's training objective stabilising or improving toward a relatively steady solution.

**140\. Does decreasing training loss guarantee good generalisation?**

No. Training loss can decrease even when the model overfits. Held-out evaluation is necessary.

**141\. What is reproducibility?**

The ability to repeat an experiment using the same data, configuration, seeds and implementation and obtain consistent results.

**142\. How do you support reproducibility?**

The implementation records seeds, configurations, model checkpoints, experiment manifests, hashes, predictions and metrics.

**143\. What is one practical limitation before deployment?**

The simulator uses one intersection and synthetic observations. The system would need real sensor data and external validation before being used on actual roads.

**144\. Can the model work with traffic cameras?**

Potentially, but a camera-to-feature pipeline is needed to convert images into measurements compatible with the model's 22-value feature contract.

**145\. What if one RSU stops sending updates?**

The hierarchical aggregator can still work with the valid groups that submit accepted updates, provided at least one group contributes. Performance and coverage may deteriorate.

**146\. What is secure aggregation?**

A cryptographic approach that allows a server to combine updates without seeing every client's individual update in plaintext.

**147\. Does this project implement secure aggregation?**

No. The current update aggregation is explicit sample-weighted model-weight averaging with update validation, not cryptographic secure aggregation.

**148\. What is differential privacy?**

A mathematical privacy framework that bounds how much a released result can reveal about one person's contribution, often through carefully calibrated noise.

**149\. Does this project implement differential privacy?**

No. It should be described as federated training without a formal differential-privacy mechanism.

**150\. What is the next meaningful research improvement?**

Evaluate the system on matched-budget, multi-seed experiments and eventually real traffic sensor data, while studying client drift, privacy protections and route-aware RSU handoff.

---

# Chapter 17 - Final limitations and what you should never claim in the viva

This is perhaps the most important chapter for defending research honestly.

| Topic | What is implemented | What is not established |
|--------|--------|--------|
| SUMO | Four-arm traffic simulation | Real-world traffic deployment |
| Vehicle agents | Local numerical observations | Real vehicle hardware |
| RSU | Software roadside sensing agents | Physical camera/radar installation |
| GRU | Two-head queue and congestion prediction | Generalisation to all intersections |
| Federated learning | Flat and hierarchical FedAvg | Formal privacy guarantee |
| Communication | Message validation and simulated channel | Real wireless network reliability |
| Evaluation | Held-out simulated trajectories | Independent real-world benchmark |
| Signal control | Fixed-time split phasing | Learned adaptive traffic-signal control |
| Computer vision | Not part of current model pipeline | Image-based vehicle detection |
| RSU handoff | Fixed parent assignment | Dynamic cross-intersection handoff |

### Five weaknesses worth preparing for

**1\. Vehicle model performance still needs more evaluation.**

In the new RSU-self-update-on run, the vehicle test slice has queue MAE 16.851 metres and macro-F1 0.8618, compared with 47.709 metres and 0.5282 in the previous 20-window report. These slices come from different held-out datasets and one seed each, so do not present the change as a controlled measurement of the window cap.

**2\. RSU self-update comparison has a data-budget confound.**

In the 350v250 comparison, the enabled run has 94,500 training windows and the disabled run has 87,500. The enabled run includes 7,000 RSU sensor windows that are absent from the disabled run.

**3\. Central GRU scored higher than HFL in the new comparison.**

Central macro-F1 was 0.8733 versus 0.8344 for hierarchical FedAvg with self-update on, with queue MAE 9.281 m versus 15.113 m. Both had 94,500 training windows and the same test identities. Do not claim that HFL is the best-performing method merely because it is the main research focus.

**4\. Results use one training seed and overlapping windows.**

The metrics are descriptive. A multi-seed, run-level uncertainty analysis is needed.

**5\. This is not a computer vision implementation.**

Explain how camera data could enter the existing feature adapter, but do not claim YOLO, CNN or image processing has already been implemented.

---

# Chapter 18 - How all three of you should explain the project together

Here is the story you should be able to deliver in approximately five minutes.

**Opening - The problem**

"Our project, TRAF-MA, focuses on predicting traffic congestion at a four-way intersection. Instead of only monitoring current traffic, we predict the queue length and congestion category 30 seconds into the future."

---

**Part 1 - Shubro: Simulation**

"We create the four-arm intersection using SUMO. We configure 300-metre arms, one incoming lane per direction, left-hand traffic and a 120-second split-phase traffic signal cycle. SUMO generates vehicle movement, and our Python application communicates with it through TraCI. We then create vehicle and RSU agents to collect their limited observations."

---

**Part 2 - Harshul: Machine learning**

"We process the observations into ten consecutive timesteps, with 22 numerical features at each timestep. This creates a 10 by 22 tensor. We pass it into a GRU with 32 hidden units, a 16-dimensional shared layer and two output heads. One predicts future queue ratio; the other predicts Free, Moderate or Heavy congestion."

---

**Part 3 - Laxminarayan: Federated learning**

"Rather than simply pooling all the selected learning histories, we train clients locally. We use 350 vehicle learners and four RSU learners. The vehicle models send their updates to their parent RSUs. The four RSUs aggregate their own group and send group-level updates to the global coordinator, which creates the global model using hierarchical FedAvg."

---

**Closing - Results**

"In the newer 350v250 RSU-self-update-enabled experiment, we used 350 vehicle clients with 250 windows each and four RSUs with up to 1,750 windows each. That is 94,500 training windows, 100 rounds and five local epochs. On 14,000 held-out windows, HFL achieved a queue MAE of 15.113 metres, RMSE of 26.388 metres and macro-F1 of 0.8344. The earlier 20-window run achieved 27.352 metres MAE and 0.7078 macro-F1. The newer metrics are better, but the dataset split and seed changed, so they do not isolate the effect of increasing the window cap."

## Final revision: The 15 numbers you should memorise

# 4

Road approaches

# 300 m

Arm length

# 120 s

Signal cycle

# 150 m

RSU zone

# 100 m

Vehicle radius

# 30 s

Prediction horizon

# 10

Sequence steps

# 22

Features

# 32

GRU hidden size

# 5,972

Model parameters

# 350

Vehicle learners

# 4

RSU learners

# 94,500 / 87,500

Training windows: central and HFL self-update-on / flat and HFL self-update-off

# 100

FL rounds

# 5

Local epochs

---

**New 350v250 HFL self-update-on test results**

Queue MAE
**15.113 m**

Queue RMSE
**26.388 m**

Macro-F1
**0.8344**

Balanced accuracy
**0.8106**

For context, the new central GRU scored queue MAE 9.281 m and macro-F1 0.8733. The older 20-window HFL self-update-on values remain 27.352 m and 0.7078.

## Repository references for revision

These are the most useful files to open when revising specific chapters\:

| Subject | Repository file |
|--------|--------|
| Overall architecture | [github.com](https://github.com/LAXMINARAYAN24/Traf-MA-350/blob/main/README.md?utm_source=chatgpt.com) |
| Actual system architecture | [github.com](https://github.com/LAXMINARAYAN24/Traf-MA-350/blob/main/docs/ARCHITECTURE_AS_BUILT.md?utm_source=chatgpt.com) |
| SUMO configuration | [github.com](https://github.com/LAXMINARAYAN24/Traf-MA-350/blob/main/trafma/config.py?utm_source=chatgpt.com) |
| SUMO network | [github.com](https://github.com/LAXMINARAYAN24/Traf-MA-350/blob/main/trafma/netgen.py?utm_source=chatgpt.com) |
| Traffic observation | [github.com](https://github.com/LAXMINARAYAN24/Traf-MA-350/blob/main/trafma/sensors.py?utm_source=chatgpt.com) |
| Feature engineering | [github.com](https://github.com/LAXMINARAYAN24/Traf-MA-350/blob/main/trafma/features.py?utm_source=chatgpt.com) |
| GRU model | [github.com](https://github.com/LAXMINARAYAN24/Traf-MA-350/blob/main/trafma/models.py?utm_source=chatgpt.com) |
| Federated aggregation | [github.com](https://github.com/LAXMINARAYAN24/Traf-MA-350/blob/main/trafma/federation.py?utm_source=chatgpt.com) |
| FL training | [github.com](https://github.com/LAXMINARAYAN24/Traf-MA-350/blob/main/trafma/federated_training.py?utm_source=chatgpt.com) |
| Selected experiment config | [github.com](https://github.com/LAXMINARAYAN24/Traf-MA-350/blob/main/artifacts/phase2_compact_350_hfl_rsu_on_7000each/config.yaml?utm_source=chatgpt.com) |
| Selected experiment report | [github.com](https://github.com/LAXMINARAYAN24/Traf-MA-350/blob/main/artifacts/phase2_compact_350_hfl_rsu_on_7000each/report/report.md?utm_source=chatgpt.com) |
| Selected data manifest | [github.com](https://github.com/LAXMINARAYAN24/Traf-MA-350/blob/main/artifacts/phase2_compact_350_hfl_rsu_on_7000each/selected_data/manifest.json?utm_source=chatgpt.com) |
| New 350v250 comparison report | `runs/phase2_350v250_20261105_223954/report/report.md` |
| New 350v250 config snapshots and run artifacts | `runs/phase2_350v250_20261105_223954/` |

**My strongest recommendation\:** Before the viva, all three of you should be able to draw the complete SUMO → observation → feature tensor → GRU → local training → RSU aggregation → global model pipeline on a whiteboard, explain one numerical example at every stage, and defend why central GRU performed better than the current HFL setup.

That is the difference between knowing your assigned module and actually understanding the project.



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
