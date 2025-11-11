import os
import numpy as np
import pandas as pd
from collections import defaultdict


class AgentTrajectoryLogger:
    """
    Logs per-agent 3D trajectories and velocities over time,
    using structure compatible with the visualization script.
    Each agent’s trajectory is now saved separately per episode
    to prevent overwriting between agents.
    """

    def __init__(self, out_dir="trajectory_logs", max_steps=2000):
        self.out_dir = out_dir
        self.positions = defaultdict(list)
        self.velocities = defaultdict(list)
        self.rotations = defaultdict(list)
        self.captured_flags = defaultdict(list)
        self.current_episode = 0
        self.step = 0
        self.max_steps = max_steps
        os.makedirs(out_dir, exist_ok=True)

    def record(self, global_agent_id, mate_entry):
        """
        Records a single timestep entry for the given agent.
        mate_entry = [x, y, z, rot, vx, vy, vz, wasCaptured]
        """
        x, y, z, rot, vx, vy, vz, cap = mate_entry
        self.positions[global_agent_id].append([x, y, z])
        self.velocities[global_agent_id].append([vx, vy, vz])
        self.rotations[global_agent_id].append(rot)
        self.captured_flags[global_agent_id].append(cap)
        self.step += 1

    def end_episode(self):
        """
        Consolidate all logged data and save.
        Each agent gets its own NPZ file: episode_<ep>_agent_<id>.npz
        """
        if not self.positions:
            print("[TrajectoryLogger] No data recorded this episode.")
            return

        all_agents = sorted(self.positions.keys())
        T = min(len(self.positions[a]) for a in all_agents)
        num_agents = len(all_agents)

        print(f"[TrajectoryLogger] Saving {num_agents} agent trajectories for episode {self.current_episode}")

        for i, aid in enumerate(all_agents):
            pos = np.array(self.positions[aid][:T])
            vel = np.array(self.velocities[aid][:T])
            rot = np.array(self.rotations[aid][:T])
            cap = np.array(self.captured_flags[aid][:T])
            goals = np.zeros((1, 3))  # placeholder if unknown

            filename = f"episode_{self.current_episode}_agent_{aid}.npz"
            filepath = os.path.join(self.out_dir, filename)
            np.savez_compressed(
                filepath,
                positions=pos[np.newaxis, :, :],  # shape (1, T, 3)
                velocities=vel[np.newaxis, :, :],
                rotations=rot[np.newaxis, :],
                captured_flags=cap[np.newaxis, :],
                goals=goals,
            )
            print(f"[TrajectoryLogger] Saved {filename} -> {self.out_dir}")

        # Reset
        self.positions.clear()
        self.velocities.clear()
        self.rotations.clear()
        self.captured_flags.clear()
        self.current_episode += 1
        self.step = 0
