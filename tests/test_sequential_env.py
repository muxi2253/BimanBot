#!/usr/bin/env python3
"""
Unit tests for Sequential Reaching Environment

Run with: pytest tests/test_sequential_env.py
Or: python -m pytest tests/test_sequential_env.py -v
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

try:
    import numpy as np
    import pytest
    from gym_env_sequential import SequentialReachEnv
except ImportError as e:
    print(f"⚠️  Skipping tests: {e}")
    print("Run in conda environment with: conda activate dual-arm")
    sys.exit(0)


class TestSequentialReachEnv:
    """Test suite for Sequential Reaching Environment"""

    def test_env_creation(self):
        """Test basic environment creation"""
        env = SequentialReachEnv(reward_type="sparse")
        assert env is not None
        assert env.observation_space.shape == (21,)  # 14 joints + 6 TCP + 1 stage
        assert env.action_space.shape == (14,)
        env.close()

    def test_reward_types(self):
        """Test different reward types"""
        for reward_type in ["sparse", "shaped", "dense"]:
            env = SequentialReachEnv(reward_type=reward_type)
            assert env.reward_type == reward_type
            env.close()

    def test_max_stages(self):
        """Test curriculum learning with different max_stages"""
        for max_stages in [1, 2, 3]:
            env = SequentialReachEnv(max_stages=max_stages)
            assert env.max_stages == max_stages
            env.close()

    def test_reset(self):
        """Test environment reset"""
        env = SequentialReachEnv()
        obs, info = env.reset(seed=42)

        assert obs.shape == (21,)
        assert info['current_stage'] == 0
        assert info['stages_completed'] == 0
        assert not info['episode_success']

        env.close()

    def test_stage_targets(self):
        """Test that stage targets are properly defined"""
        env = SequentialReachEnv(randomize_targets=False)
        obs, info = env.reset(seed=42)

        # Should have 3 stages defined
        assert len(env.stage_targets) == 3

        # Each stage has left and right targets
        for left, right in env.stage_targets:
            assert left.shape == (3,)
            assert right.shape == (3,)

        env.close()

    def test_observation_includes_stage(self):
        """Test that observation includes stage indicator"""
        env = SequentialReachEnv()
        obs, _ = env.reset(seed=42)

        # Last element should be stage indicator (0.0 for stage 0)
        stage_indicator = obs[-1]
        assert 0.0 <= stage_indicator <= 1.0

        env.close()

    def test_step_execution(self):
        """Test basic step execution"""
        env = SequentialReachEnv(reward_type="shaped")
        obs, info = env.reset(seed=42)

        # Take random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (21,)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert 'current_stage' in info

        env.close()

    def test_reproducibility(self):
        """Test that same seed produces same results"""
        env1 = SequentialReachEnv()
        obs1, _ = env1.reset(seed=42)

        env2 = SequentialReachEnv()
        obs2, _ = env2.reset(seed=42)

        assert np.allclose(obs1, obs2)

        env1.close()
        env2.close()


if __name__ == "__main__":
    print("Running Sequential Environment Tests...")
    print("=" * 60)

    # Run tests manually if pytest not available
    test = TestSequentialReachEnv()

    tests = [
        ("Environment Creation", test.test_env_creation),
        ("Reward Types", test.test_reward_types),
        ("Max Stages", test.test_max_stages),
        ("Reset", test.test_reset),
        ("Stage Targets", test.test_stage_targets),
        ("Observation Includes Stage", test.test_observation_includes_stage),
        ("Step Execution", test.test_step_execution),
        ("Reproducibility", test.test_reproducibility),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            print(f"✅ {name}")
            passed += 1
        except Exception as e:
            print(f"❌ {name}: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {failed} test(s) failed")
        sys.exit(1)
