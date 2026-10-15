# Option 1 Implementation Plan: Hierarchical Control for Sequential Reaching

## 🎯 Project Goal

Implement a **minimal but complete** hierarchical control system that demonstrates:
1. Long-horizon task capability (3-stage sequential reaching)
2. Combination of planning (IK) and learning (PPO)
3. Measurable improvement over baseline methods
4. **Honest, reproducible results** for CV/PhD application

---

## 📋 Task Definition: Sequential Reaching

### Why Sequential Reaching (not Pick-Place)?

**Strategic reasons:**
- ✅ Avoids contact dynamics complexity (no grasping physics)
- ✅ Still demonstrates "long-horizon" concept
- ✅ Fast to train (hours, not days)
- ✅ Reproducible results
- ❌ Pick-place is too hard and unpredictable for 1-week timeline

### Task Description

**Scenario**: Robot must reach 3 target positions in sequence with both arms

**Episode structure:**
```
Stage 1: Left arm  → Target A (0.4, 0.3, 1.2)
         Right arm → Target B (0.4, -0.3, 1.2)
         
Stage 2: Left arm  → Target C (0.6, 0.2, 1.4)
         Right arm → Target D (0.6, -0.2, 1.0)
         
Stage 3: Left arm  → Target E (0.3, 0.4, 1.5)
         Right arm → Target F (0.3, -0.4, 1.1)
```

**Success criteria:**
- Both arms within 5cm of current targets → advance to next stage
- Complete all 3 stages → episode success

**Difficulty factors:**
- Sparse reward: only get reward when completing each stage
- Long horizon: 3 stages × ~100 steps = 300 steps max
- Coordination: both arms must succeed to advance

---

## 🏗️ Architecture Design

### Three-Tier System

```
┌─────────────────────────────────────────┐
│ High-Level: Stage Manager               │
│ - Tracks current stage (1/2/3)          │
│ - Provides target positions             │
│ - Checks stage completion               │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ Mid-Level: IK Planner (Optional)        │
│ - Generates "suggested" joint angles    │
│ - Used by Residual PPO as reference     │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ Low-Level: PPO Controller               │
│ - Pure PPO: learns from scratch         │
│ - Residual PPO: learns delta from IK    │
└─────────────────────────────────────────┘
```

---

## 🧪 Experiments to Run

### Experiment Matrix

| Method | Description | Expected Result |
|--------|-------------|----------------|
| **Baseline 1: Pure PPO** | Standard PPO on sequential task | Slow/no convergence |
| **Baseline 2: + Reward Shaping** | Dense distance-based rewards | Moderate success (40-60%) |
| **Method 3: + Curriculum** | Start with 1 stage, add more | Better success (60-75%) |
| **Method 4: + Residual PPO** | PPO learns delta from IK | Best (70-85%) |

### Training Configuration

```python
common_config = {
    "timesteps": 500_000,  # 500k steps per experiment
    "n_envs": 4,           # 4 parallel environments
    "n_seeds": 3,          # 3 random seeds for statistics
    "eval_freq": 10_000,   # Evaluate every 10k steps
}
```

**Total training time estimate:**
- 4 methods × 3 seeds × 500k steps = 6M steps total
- ~1-2 hours per seed (with 4 parallel envs)
- **Total: 6-12 hours** (can run overnight)

---

## 📁 File Structure

```
BimanBot/
├── backend/
│   ├── gym_env_sequential.py          [NEW] Sequential reaching environment
│   ├── curriculum_wrapper.py          [NEW] Curriculum learning wrapper
│   └── residual_policy.py             [NEW] Residual PPO wrapper
│
├── examples/
│   ├── train_sequential_baseline.py   [NEW] Pure PPO + Reward Shaping
│   ├── train_sequential_curriculum.py [NEW] Curriculum learning
│   └── train_sequential_residual.py   [NEW] Residual PPO
│
├── experiments/
│   ├── run_all_experiments.sh         [NEW] Automated experiment runner
│   ├── plot_results.py                [NEW] Generate learning curves
│   └── results/                       [NEW] Experiment logs and data
│
└── docs/
    └── sequential_reaching_results.md [NEW] Experiment report
```

---

## 🔨 Implementation Steps

### Phase 1: Core Environment (Day 1, ~4-6 hours)

**File**: `backend/gym_env_sequential.py`

```python
class SequentialReachEnv(DualArmReachEnv):
    """
    3-stage sequential reaching task.
    Inherits from existing DualArmReachEnv to minimize code.
    """
    
    def __init__(self):
        super().__init__()
        self.current_stage = 0  # 0, 1, 2
        self.stage_targets = self._define_stages()
        
    def _define_stages(self):
        # Fixed targets for reproducibility
        return [
            ([0.4, 0.3, 1.2], [0.4, -0.3, 1.2]),  # Stage 0
            ([0.6, 0.2, 1.4], [0.6, -0.2, 1.0]),  # Stage 1
            ([0.3, 0.4, 1.5], [0.3, -0.4, 1.1]),  # Stage 2
        ]
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Check stage completion
        if self._stage_completed():
            reward += 10.0  # Stage bonus
            self.current_stage += 1
            
            if self.current_stage >= 3:
                terminated = True  # Episode success
                reward += 50.0     # Task completion bonus
        
        return obs, reward, terminated, truncated, info
```

**Implementation checklist:**
- [ ] Extend DualArmReachEnv
- [ ] Add stage tracking logic
- [ ] Add stage completion detection
- [ ] Test with random actions
- [ ] Verify reward structure

---

### Phase 2: Reward Shaping Baseline (Day 1, ~2 hours)

**File**: `examples/train_sequential_baseline.py`

Two variants in one script:
1. **Sparse reward**: Only stage completion gives reward
2. **Shaped reward**: Add distance-based intermediate rewards

```python
def compute_reward(self, reward_type="sparse"):
    if reward_type == "sparse":
        # Original: only stage completion
        return reward
    else:  # "shaped"
        # Add distance penalty
        dist = self._distance_to_current_targets()
        return reward - 0.01 * dist
```

---

### Phase 3: Curriculum Learning (Day 2, ~3-4 hours)

**File**: `backend/curriculum_wrapper.py`

```python
class CurriculumWrapper(gym.Wrapper):
    """
    Gradually increase task difficulty:
    - Start: Only stage 1
    - After 80% success: Enable stage 2
    - After 80% success: Enable stage 3
    """
    
    def __init__(self, env, initial_stages=1):
        super().__init__(env)
        self.max_stages = initial_stages
        self.success_buffer = deque(maxlen=100)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Track success rate
        if terminated or truncated:
            self.success_buffer.append(info.get('success', False))
            
            # Curriculum progression
            if len(self.success_buffer) == 100:
                success_rate = sum(self.success_buffer) / 100
                if success_rate > 0.8 and self.max_stages < 3:
                    self.max_stages += 1
                    print(f"📈 Curriculum: Now using {self.max_stages} stages")
        
        return obs, reward, terminated, truncated, info
```

---

### Phase 4: Residual PPO (Day 2-3, ~4-6 hours)

**File**: `backend/residual_policy.py`

```python
class ResidualPPOPolicy:
    """
    PPO learns small corrections on top of IK suggestions.
    
    action_final = action_ik + ppo_output * 0.1
    """
    
    def __init__(self, ik_solver, ppo_policy):
        self.ik_solver = ik_solver
        self.ppo_policy = ppo_policy
    
    def predict(self, obs, state=None):
        # 1. Get IK suggestion (if available)
        try:
            ik_action = self.ik_solver.solve_for_targets(obs)
        except:
            ik_action = np.zeros(14)  # Fallback
        
        # 2. PPO learns residual
        ppo_residual, state = self.ppo_policy.predict(obs, state)
        
        # 3. Combine (small residual)
        final_action = ik_action + 0.1 * ppo_residual
        
        return final_action, state
```

**Key design choice:**
- IK gives "nominal" solution
- PPO learns small corrections (±0.1 rad)
- Even if IK fails, PPO can still work

---

### Phase 5: Experiments (Day 3-4, overnight)

**File**: `experiments/run_all_experiments.sh`

```bash
#!/bin/bash

# Run all experiments with 3 seeds each
for seed in 42 123 456; do
    # Baseline 1: Pure PPO (sparse reward)
    python examples/train_sequential_baseline.py \
        --reward-type sparse --seed $seed &
    
    # Baseline 2: Shaped reward
    python examples/train_sequential_baseline.py \
        --reward-type shaped --seed $seed &
    
    # Method 3: Curriculum
    python examples/train_sequential_curriculum.py \
        --seed $seed &
    
    # Method 4: Residual PPO
    python examples/train_sequential_residual.py \
        --seed $seed &
    
    wait  # Wait for this batch to finish
done
```

---

### Phase 6: Analysis & Visualization (Day 4-5, ~4 hours)

**File**: `experiments/plot_results.py`

Generate:
1. **Learning curves**: Success rate vs timesteps
2. **Bar chart**: Final performance comparison
3. **Stage progression**: How fast agents reach each stage
4. **Statistics table**: Mean ± std across seeds

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(results_dict):
    """
    results_dict = {
        "Pure PPO": [(timesteps, success_rate), ...],
        "Shaped": [...],
        "Curriculum": [...],
        "Residual": [...]
    }
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method, data in results_dict.items():
        timesteps, mean, std = compute_mean_std(data)
        ax.plot(timesteps, mean, label=method, linewidth=2)
        ax.fill_between(timesteps, mean-std, mean+std, alpha=0.2)
    
    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Success Rate (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Sequential Reaching: Method Comparison")
    
    plt.savefig("results/learning_curves.png", dpi=300)
```

---

## 📊 Expected Realistic Results

Based on task complexity, here's what we can **honestly expect**:

| Method | Success Rate (mean ± std) | Convergence Speed | Notes |
|--------|---------------------------|-------------------|-------|
| Pure PPO (sparse) | 10-25% | Slow/never | Baseline, might not converge |
| + Reward Shaping | 40-60% | 300k steps | Moderate improvement |
| + Curriculum | 60-75% | 200k steps | Consistent improvement |
| + Residual PPO | 65-80% | 150k steps | Best sample efficiency |

**Important**: These are **estimates**. Real results might be ±15%. 
**We will use whatever we actually get** - honest results are more valuable than fake numbers.

---

## ✅ Success Criteria

### Minimum Viable Result (for CV)
- ✅ All 4 methods successfully trained
- ✅ At least 3 seeds per method
- ✅ Learning curves generated
- ✅ Clear improvement: Residual/Curriculum > Baseline

### Ideal Result
- ✅ Statistical significance (t-test, p < 0.05)
- ✅ Residual PPO converges 2×+ faster
- ✅ Final success rate > 70% for best method

### Acceptable Result (still publishable)
- Even if numbers are lower (e.g., 50% vs 30%)
- As long as there's **measurable improvement**
- And we can **explain why** (task is hard, sparse reward, etc.)

---

## 🚫 What We're NOT Doing

To keep scope realistic:

❌ LLM integration (too complex for 1 week)
❌ Pick-place with contact (too unpredictable)
❌ Vision-based control (not needed for reaching)
❌ Real robot transfer (simulation only)
❌ Hierarchical RL with learned high-level (state machine is enough)

---

## 📝 Final Deliverables

1. **Code**: 4 new Python files (~800 lines total)
2. **Experiments**: 12 trained models (4 methods × 3 seeds)
3. **Results**: Learning curves, statistics table
4. **Report**: `sequential_reaching_results.md` (2-3 pages)
5. **CV Description**: Honest, data-backed project summary

---

## ⏱️ Time Estimate

| Phase | Time | Can Run Overnight? |
|-------|------|-------------------|
| Environment | 4-6h | No |
| Baselines | 2-4h | No |
| Curriculum | 3-4h | No |
| Residual PPO | 4-6h | No |
| **Training** | 8-12h | ✅ **Yes** |
| Analysis | 3-4h | No |
| Report | 2-3h | No |
| **Total** | **26-39h** | |

**Realistic timeline**: 4-6 days if working full-time, 1-2 weeks part-time.

---

## 🎯 Next Steps

1. **Review this plan** - Any concerns or changes?
2. **Start Phase 1** - Implement SequentialReachEnv
3. **Test basic training** - Verify one method works before scaling up
4. **Run experiments** - Overnight batch job
5. **Analyze & write** - Generate CV-ready materials

---

**Ready to proceed?** Let me know if you want to:
- A) Start implementing Phase 1 now
- B) Adjust the plan first
- C) Ask questions about any part
