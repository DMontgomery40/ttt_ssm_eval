# Claude Code Prompt: TTT-SSM Phase 0 Dashboard

## CRITICAL CONTEXT — READ THIS FIRST

You are building a UI for a **completely novel system that does not exist anywhere else**. This is NOT a standard ML training dashboard. This is NOT TensorBoard. This is NOT Weights & Biases. **Forget everything you think you know about ML dashboards.**

This system implements **Test-Time Training (TTT)** — a paradigm where a neural network **updates its own weights during inference**, not just during training. The weights change while the model is running. Those changes persist across sessions. Different sessions diverge into different weight configurations. This is fundamentally different from any ML system you've seen before.

The core thesis: **"The weights ARE the memory."** Instead of a context window that flushes, the model literally rewires itself to remember. This is closer to how biological learning works than standard deep learning.

---

## SYSTEM ARCHITECTURE OVERVIEW

### What This System Does

1. **Diagonal Stable SSM**: A State Space Model with stability guarantees. Hidden state evolves via `h_{t+1} = decay * h_t + input`. The `decay` is parameterized to always be in (0,1), preventing explosion.

2. **Online Weight Updates**: Every N steps (a "chunk"), the model:
   - Snapshots its current weights
   - Computes a loss on recent observations
   - Runs a gradient update using the **Muon optimizer** (not AdamW!)
   - Checks if the update was safe (gradient norm, loss regression, state explosion)
   - Either **commits** the update or **rolls back** to the snapshot

3. **Session Persistence**: Each "session" has its own evolving weights. Sessions are saved to disk and can be resumed. Different sessions (with different environments) develop different weight configurations.

4. **Hidden-μ Physics Environment**: A 2D point mass with friction coefficient μ that the model cannot observe directly. The model must infer μ from how the physics behaves and encode that knowledge into its weights.

5. **Baseline vs Adaptive Comparison**: Every run compares:
   - **Baseline**: Frozen weights, no updates (standard inference)
   - **Adaptive**: Online weight updates enabled (TTT)
   
   Both see the exact same trajectory. The gap between them shows what TTT learns.

---

## DATA STRUCTURES YOU WILL VISUALIZE

### 1. Per-Step Metrics (from `per_step.csv`)
```
t,baseline_mse,adaptive_mse,did_update,update_ok
0,0.00123,0.00123,0,0
1,0.00456,0.00445,0,0
...
32,0.00789,0.00234,1,1  // Update happened and committed
...
64,0.00567,0.00189,1,0  // Update happened but rolled back
```

### 2. Update Events (from `update_events.json`)
```json
[
  {
    "t": 32,
    "status": "commit",
    "pre_loss": 0.00789,
    "post_loss": 0.00456,
    "grad_norm": 2.34,
    "pre_max_h": 1.23,
    "post_max_h": 1.45
  },
  {
    "t": 64,
    "status": "rollback_loss_regression",
    "pre_loss": 0.00567,
    "post_loss": 0.00890,
    "grad_norm": 15.67,
    "pre_max_h": 2.34,
    "post_max_h": 8.90
  },
  {
    "t": 96,
    "status": "rollback_grad_norm",
    "pre_loss": 0.00456,
    "post_loss": null,
    "grad_norm": 25.00,
    "pre_max_h": 1.89
  }
]
```

### 3. Session Metadata (from `session/meta.json`)
```json
{
  "schema_version": 1,
  "session_id": "session_001",
  "created_at_unix": 1704307200,
  "mu": 0.1234,
  "env_mode": "linear",
  "model_signature": "abc123...",
  "base_ckpt_hash": "def456...",
  "model_cfg": {
    "obs_dim": 4,
    "act_dim": 2,
    "z_dim": 4,
    "u_dim": 16,
    "n_state": 32,
    "dt": 1.0
  },
  "plasticity_cfg": {
    "lr": 0.005,
    "weight_decay": 0.0,
    "momentum": 0.95,
    "nesterov": true,
    "ns_steps": 5,
    "chunk": 32,
    "buffer_len": 32,
    "rollback_tol": 0.20,
    "grad_norm_max": 20.0,
    "state_norm_max": 1000000.0
  }
}
```

### 4. Session Metrics (from `session/metrics.json`)
```json
{
  "baseline_mse_mean": 0.00456,
  "adaptive_mse_mean": 0.00123,
  "baseline_mse_last100_mean": 0.00789,
  "adaptive_mse_last100_mean": 0.00089,
  "updates_attempted": 18,
  "updates_committed": 15,
  "updates_rolled_back": 3
}
```

### 5. Plastic Weights (from `session/plastic_state.pt`)
Three 2D tensors:
- `W_u`: shape [u_dim, z_dim + act_dim] = [16, 6]
- `B`: shape [n_state, u_dim] = [32, 16]
- `W_o`: shape [z_dim, n_state] = [4, 32]

These are the weights that change during TTT. Visualize their evolution, distributions, spectral properties.

---

## UI REQUIREMENTS

### Technology Stack
- **React 18+** with TypeScript
- **Tailwind CSS** for styling
- **Recharts** or **D3.js** for visualizations (prefer Recharts for simplicity, D3 for complex custom viz)
- **Framer Motion** for animations
- Single-page app with tab navigation
- Dark mode by default (this is a research tool, researchers work at night)

### Design Philosophy
- **Information-dense but not cluttered**: Every pixel should communicate something
- **Real-time feel**: Even though data may be static, use subtle animations to make it feel alive
- **Scientific accuracy**: No chartjunk, no 3D pie charts, no gratuitous gradients
- **Researcher-friendly**: Assume the user is a PhD student or ML engineer who wants raw numbers accessible

---

## TAB STRUCTURE

### Tab 1: Overview Dashboard
**Purpose**: At-a-glance summary of the entire run

**Components**:

1. **Hero Metrics Row** (top of page):
   - Large stat cards showing:
     - "Baseline MSE (final 100)" with value and sparkline
     - "Adaptive MSE (final 100)" with value and sparkline
     - "Improvement %" = (baseline - adaptive) / baseline * 100
     - "Updates Committed / Attempted" as fraction with circular progress
     - "Hidden μ" = the ground truth friction coefficient
     - "Session Age" = time since creation

2. **Primary MSE Comparison Chart**:
   - Full-width line chart
   - X-axis: timestep (0 to N)
   - Y-axis: MSE (log scale option toggle)
   - Two lines: baseline (gray/muted) and adaptive (vibrant blue/green)
   - Vertical markers at each update attempt:
     - Green tick = committed update
     - Red X = rolled back update
   - Hover shows exact values + update event details if applicable
   - Shaded region between curves to emphasize the "learning gap"

3. **Update Transaction Timeline**:
   - Horizontal timeline below the MSE chart
   - Each update attempt is a node
   - Node color: green (commit), red (rollback_loss), orange (rollback_grad), purple (rollback_state)
   - Click node to see full details in a slide-out panel
   - Show running totals: "15 commits, 2 loss rollbacks, 1 grad rollback"

4. **Session Identity Card**:
   - Shows: session_id, env_mode, mu, creation time
   - Model signature (truncated hash with copy button)
   - Plasticity config summary (lr, chunk size, rollback tolerance)

---

### Tab 2: Weight Evolution
**Purpose**: Visualize how the plastic weights change over the session

**Components**:

1. **Weight Matrix Heatmaps**:
   - Three heatmaps side by side: W_u, B, W_o
   - Color scale: diverging (blue-white-red) centered at 0
   - Show current values
   - If historical snapshots available, add a time slider to scrub through evolution
   - Hover on cell shows exact value and position

2. **Weight Statistics Over Time**:
   - Line charts showing per-matrix statistics at each update:
     - Frobenius norm
     - Max absolute value
     - Mean
     - Standard deviation
   - Separate subplot for each matrix, or overlaid with legend

3. **Spectral View** (advanced):
   - For each 2D matrix, show singular value distribution
   - Bar chart of top-10 singular values
   - Condition number (ratio of largest to smallest SV)
   - This reveals if updates are making the matrices ill-conditioned

4. **Weight Delta Visualization**:
   - Show the CHANGE in weights between updates
   - Heatmap of (W_after - W_before) for each committed update
   - Helps identify which weights are being modified most aggressively

---

### Tab 3: Update Transactions
**Purpose**: Deep dive into every update attempt with full transaction semantics

**Components**:

1. **Transaction Table**:
   - Sortable, filterable table of all update events
   - Columns: timestep, status, pre_loss, post_loss, Δloss, grad_norm, pre_max_h, post_max_h
   - Status column has colored badges (commit=green, rollback variants=red/orange/purple)
   - Click row to expand with full details

2. **Transaction Detail Panel** (when row selected):
   - Shows all fields from the update event
   - Visual diff: "pre_loss → post_loss" with arrow and color (green if improved, red if regressed)
   - Explains WHY rollback happened in plain English:
     - "Rolled back because post_loss (0.089) exceeded pre_loss * 1.20 (0.068)"
     - "Rolled back because grad_norm (25.0) exceeded threshold (20.0)"
   - If committed: "Update improved loss by 42% and passed all safety checks"

3. **Rollback Analysis**:
   - Pie chart or bar chart: breakdown of rollback reasons
   - "Loss regression: 2, Gradient explosion: 1, State norm: 0"
   - Time series: when do rollbacks tend to happen? Early? Late? Randomly?

4. **Gate Threshold Visualization**:
   - Show the thresholds as horizontal lines on relevant charts
   - grad_norm_max = 20.0 (show on grad_norm time series)
   - rollback_tol = 0.20 (show as ±20% band around pre_loss on loss chart)
   - state_norm_max = 1e6 (show on max_h time series)

---

### Tab 4: Environment & Physics
**Purpose**: Understand the hidden-μ physics simulation

**Components**:

1. **Trajectory Visualization**:
   - 2D plot of the point mass trajectory (x vs y position)
   - Animated replay option (play/pause/scrub)
   - Color gradient along path indicates time (start=blue, end=red)
   - Velocity vectors as small arrows at sampled points

2. **State Time Series**:
   - Four subplots stacked: x, y, vx, vy over time
   - Shows how position and velocity evolve
   - Overlay action inputs as stem plot or secondary axis

3. **Physics Parameter Display**:
   - Large display of the hidden μ value
   - Explanation: "This session's friction coefficient is μ = 0.1234"
   - Visual metaphor: slider showing where μ falls in [μ_min, μ_max] range
   - If nonlinear mode: show the static/dynamic friction thresholds

4. **Prediction Overlay**:
   - On the trajectory plot, show model predictions vs ground truth
   - Ground truth = solid line
   - Baseline predictions = dashed gray
   - Adaptive predictions = dashed colored
   - Error magnitude as shaded region or separate error plot

---

### Tab 5: Model Architecture
**Purpose**: Explain and visualize the SSM architecture

**Components**:

1. **Architecture Diagram**:
   - Visual flow diagram of the SSM:
   ```
   [z_t, a_t] → W_u → u_t → B → (decay * h_t + Bu) → h_{t+1} → W_o → ẑ_{t+1}
   ```
   - Highlight which components are plastic (W_u, B, W_o) vs frozen (decay/a_raw)
   - Show tensor shapes at each stage

2. **Stability Visualization**:
   - Show the decay values: A = -softplus(a_raw), decay = exp(A * dt)
   - Histogram of decay values (should all be in (0, 1))
   - Explain: "All decay values are < 1, guaranteeing stable dynamics"

3. **Parameter Count**:
   - Breakdown of parameters:
     - W_u: 16 × 6 = 96 (plastic)
     - B: 32 × 16 = 512 (plastic)
     - W_o: 4 × 32 = 128 (plastic)
     - a_raw: 32 (frozen)
   - Total plastic: 736, Total frozen: 32
   - Visual: stacked bar or treemap

4. **Muon Optimizer Explainer**:
   - Brief explanation of Muon vs AdamW
   - "Muon orthogonalizes gradients via Newton-Schulz iteration"
   - Show the configured hyperparameters: lr, momentum, ns_steps

---

### Tab 6: Session Management
**Purpose**: Handle multiple sessions, compare them, manage persistence

**Components**:

1. **Session List**:
   - Table of all sessions in the run directory
   - Columns: session_id, μ value, env_mode, created_at, updates_committed, final MSE improvement
   - Select multiple sessions to compare

2. **Session Comparison**:
   - When 2+ sessions selected, show overlay charts:
     - MSE curves for each session on same axes
     - Bar chart comparing final metrics
   - Useful for comparing different μ values or linear vs nonlinear

3. **Session Resume Info**:
   - For current session, show:
     - "This session can be resumed with: --resume_session_dir=./runs/.../session"
     - Copy button for the command
   - List of files in session directory with sizes

4. **Base Model Info**:
   - Show base_ckpt_hash (truncated with copy)
   - model_signature
   - "All sessions with this signature are compatible for comparison"

---

## VISUALIZATION REQUIREMENTS — MAKE THESE EXCEPTIONAL

### For All Charts:
- Smooth animations on data load and updates
- Tooltips that are information-rich but not overwhelming
- Axis labels with units where applicable
- Grid lines subtle (10% opacity)
- Legend should not overlap data
- Responsive to container size

### The MSE Comparison Chart (this is the hero visualization):
- **Must show the "learning gap" dramatically**
- Consider: area fill between baseline and adaptive curves with gradient
- Adaptive curve should feel "alive" — subtle glow or thicker stroke
- Update markers should be visually distinct without overwhelming the data
- Log scale toggle is critical (MSE can span orders of magnitude)
- Add a "smoothed" toggle (rolling average) for noisy runs

### The Transaction Timeline:
- Think of it like a Git commit history visualization
- Each node should have a clear visual state
- Connections between nodes show time progression
- Clicking a node should feel like inspecting a commit
- Consider a "diff view" for what changed in the weights

### The Weight Heatmaps:
- Must handle negative and positive values symmetrically
- Consider using a perceptually uniform colormap (viridis for magnitude, RdBu for signed)
- Cell borders should be subtle
- Add value annotations on hover, not permanently (too cluttered)
- For large matrices, consider a zoom/pan interface

### The Trajectory Plot:
- This is where the physics becomes tangible
- Animated replay is important for understanding
- Consider trail effects (fading path behind current position)
- Action vectors could be shown as force arrows at each timestep
- Add a "ghost" comparison: what would've happened with different μ?

---

## INTERACTIVITY REQUIREMENTS

1. **Cross-Chart Linking**:
   - Hovering on a point in the MSE chart should highlight the corresponding position in the trajectory plot
   - Clicking an update event should scroll/highlight it in all views

2. **Time Scrubbing**:
   - A global time slider that syncs across all time-series visualizations
   - Play/pause button for animated replay

3. **Filter/Focus**:
   - Ability to zoom into a time range across all charts
   - Filter update events by status (show only rollbacks, etc.)

4. **Export**:
   - Download chart as PNG/SVG
   - Export data as CSV
   - Copy values to clipboard

5. **Keyboard Navigation**:
   - Arrow keys to step through time
   - Tab to switch tabs
   - Escape to close panels

---

## IMPLEMENTATION NOTES

### Data Loading
- Assume data is provided via props or fetched from local JSON files
- Create mock data generators for development
- Handle missing fields gracefully (some runs may not have all data)

### State Management
- Use React Context or Zustand for global state (selected session, current time, etc.)
- Keep visualization state (zoom, filters) in URL params for shareability

### Performance
- Virtualize long lists (update events can be hundreds of items)
- Use canvas for large heatmaps if DOM becomes slow
- Debounce resize handlers
- Memoize expensive computations

### Accessibility
- All charts should have aria labels
- Color should not be the only differentiator (use patterns/shapes too)
- Keyboard navigable

---

## FILE STRUCTURE

```
src/
├── components/
│   ├── charts/
│   │   ├── MSEComparisonChart.tsx
│   │   ├── TransactionTimeline.tsx
│   │   ├── WeightHeatmap.tsx
│   │   ├── TrajectoryPlot.tsx
│   │   ├── SpectralView.tsx
│   │   └── ...
│   ├── layout/
│   │   ├── TabNavigation.tsx
│   │   ├── Header.tsx
│   │   └── Panel.tsx
│   ├── cards/
│   │   ├── MetricCard.tsx
│   │   ├── SessionCard.tsx
│   │   └── ConfigCard.tsx
│   └── ...
├── hooks/
│   ├── useSessionData.ts
│   ├── useTimeSync.ts
│   └── ...
├── types/
│   ├── session.ts
│   ├── metrics.ts
│   └── ...
├── utils/
│   ├── formatting.ts
│   ├── colorScales.ts
│   └── ...
├── data/
│   └── mockData.ts
└── App.tsx
```

---

## REMEMBER

- This is a **research tool for a novel paradigm**. The user is trying to understand if their model is actually learning during inference.
- The key insight to convey: **"The adaptive curve drops below baseline because the weights are changing to encode the hidden physics parameter."**
- Every visualization should help answer: "Is TTT working? Is it safe? What did it learn?"
- When in doubt, show more data, not less. Researchers want access to raw numbers.
- Make it beautiful, but make it functional first.

---

## FINAL CHECK

Before submitting, verify:
- [ ] All 6 tabs are implemented
- [ ] MSE comparison chart shows learning gap clearly
- [ ] Update transactions are explorable with full details
- [ ] Weight evolution is visualized (heatmaps + statistics)
- [ ] Trajectory plot works with animation
- [ ] Session metadata is displayed
- [ ] Dark mode is default
- [ ] Responsive to different screen sizes
- [ ] Mock data works for development
- [ ] No console errors or warnings
