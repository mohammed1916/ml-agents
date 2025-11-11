import os
import numpy as np
import pandas as pd
from collections import defaultdict

class AgentTrajectoryLogger:
    """
    Logs per-agent 3D trajectories and velocities over time,
    using structure compatible with the visualization script.
    """
    

    def __init__(self, out_dir="trajectory_logs", max_steps=2000):
        self.out_dir = out_dir
        self.positions = defaultdict(list)
        # self.velocities = defaultdict(list)
        # self.rotations = defaultdict(list)
        # self.captured_flags = defaultdict(list)
        self.current_episode = 0
        self.step = 0
        self.max_steps = max_steps
        os.makedirs(out_dir, exist_ok=True)

    def record(self, global_agent_id, mate_entry):
        """
        mate_entry = [x, y, z, rot, vx, vy, vz, wasCaptured]
        """
        x, y, z, rot, vx, vy, vz, cap = mate_entry
        self.positions[global_agent_id].append([x, y, z])
        # self.velocities[global_agent_id].append([vx, vy, vz])
        # self.rotations[global_agent_id].append(rot)
        # self.captured_flags[global_agent_id].append(cap)
        self.step += 1

    def end_episode(self):
        """Consolidate all logged data and save."""
        all_agents = sorted(self.positions.keys())
        T = min(len(self.positions[a]) for a in all_agents)
        num_agents = len(all_agents)

        pos = np.zeros((num_agents, T, 3))
        vel = np.zeros_like(pos)
        goals = np.zeros((num_agents, 3))  # if known, otherwise placeholder

        for i, aid in enumerate(all_agents):
            pos[i] = np.array(self.positions[aid][:T])
            # vel[i] = np.array(self.velocities[aid][:T])

        np.savez_compressed(
            os.path.join(self.out_dir, f"episode_{self.current_episode}.npz"),
            positions=pos,
            # velocities=vel,
            goals=goals,
        )

        # Reset
        self.positions.clear()
        # self.velocities.clear()
        # self.rotations.clear()
        # self.captured_flags.clear()
        self.current_episode += 1
        self.step = 0
        print(f"[TrajectoryLogger] Saved episode {self.current_episode-1} -> {self.out_dir}")

