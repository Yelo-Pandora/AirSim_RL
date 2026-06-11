import numpy as np

import config


class RuntimeSafetyShield:
    """Test-time action shield for the lower TD3 policy."""

    def __init__(self, env):
        self.env = env
        self.reset()

    def reset(self):
        self.interventions = 0
        self.emergency_interventions = 0
        self.recovery_interventions = 0
        self.recovery_steps = 0
        self.recovery_steps_remaining = 0
        self.recovery_step_index = 0
        self.recovery_side = 1.0
        self.last_reason = "none"

    def filter_action(self, action, obs):
        action = np.asarray(action, dtype=np.float32).copy()
        diagnostics = self._diagnostics(obs)
        reason = self._reason(diagnostics)
        self.last_reason = reason

        if self._should_start_recovery(reason):
            self._start_recovery(reason, diagnostics)

        if self.recovery_steps_remaining > 0:
            self.interventions += 1
            self.recovery_steps += 1
            shielded = self._recovery_action(diagnostics)
            self.recovery_steps_remaining -= 1
            self.recovery_step_index += 1
            self.last_reason = f"recovery_{reason}" if reason != "none" else "recovery"
            shielded = self._apply_vertical_limits(shielded, diagnostics)
            return np.clip(shielded, -1.0, 1.0).astype(np.float32), diagnostics

        if reason == "none":
            return np.clip(action, -1.0, 1.0), diagnostics

        self.interventions += 1
        if reason.startswith("emergency"):
            self.emergency_interventions += 1
            shielded = self._brake_action(vertical_only=False)
        else:
            brake = self._brake_action(vertical_only=False)
            shielded = (
                (1.0 - float(config.SAFETY_SHIELD_WARN_BLEND)) * action
                + float(config.SAFETY_SHIELD_WARN_BLEND) * brake
            )

        shielded = self._apply_vertical_limits(shielded, diagnostics)
        return np.clip(shielded, -1.0, 1.0).astype(np.float32), diagnostics

    def _should_start_recovery(self, reason):
        if not bool(getattr(config, "SAFETY_SHIELD_RECOVERY_ENABLED", True)):
            return False
        if self.recovery_steps_remaining > 0:
            return False
        return reason in set(getattr(config, "SAFETY_SHIELD_RECOVERY_TRIGGER_REASONS", ()))

    def _start_recovery(self, reason, diagnostics):
        self.recovery_interventions += 1
        if reason.startswith("emergency"):
            self.emergency_interventions += 1
        self.recovery_steps_remaining = int(getattr(config, "SAFETY_SHIELD_RECOVERY_STEPS", 6))
        self.recovery_step_index = 0
        # Positive side means body-left; negative means body-right.
        self.recovery_side = 1.0 if diagnostics["side_left_min"] >= diagnostics["side_right_min"] else -1.0

    def _diagnostics(self, obs):
        lidar_norm_max = float(getattr(self.env, "lidar_norm_max", 20.0))
        lidar = np.asarray(obs.get("lidar", []), dtype=np.float32).reshape(-1)
        lidar = lidar * lidar_norm_max if lidar.size else np.ones(105, dtype=np.float32) * lidar_norm_max

        front = lidar[:45] if lidar.size >= 45 else lidar
        side_left = lidar[45:60] if lidar.size >= 60 else lidar
        side_right = lidar[60:75] if lidar.size >= 75 else lidar
        any_min = float(np.min(lidar)) if lidar.size else lidar_norm_max

        depth_norm_max = float(getattr(self.env, "depth_norm_max", 20.0))
        depth = np.asarray(obs.get("depth", []), dtype=np.float32)
        if depth.size:
            depth = depth.reshape(depth.shape[-2], depth.shape[-1]) * depth_norm_max
            h, w = depth.shape
            center = depth[max(0, h // 2 - 1): min(h, h // 2 + 2), max(0, w // 2 - 2): min(w, w // 2 + 3)]
            depth_center_min = float(np.min(center))
        else:
            depth_center_min = depth_norm_max

        return {
            "front_min": float(np.min(front)) if front.size else lidar_norm_max,
            "side_left_min": float(np.min(side_left)) if side_left.size else lidar_norm_max,
            "side_right_min": float(np.min(side_right)) if side_right.size else lidar_norm_max,
            "any_lidar_min": any_min,
            "depth_center_min": depth_center_min,
            "bottom_dist": float(getattr(self.env, "current_dis_z_bottom", np.inf)),
            "top_dist": float(getattr(self.env, "current_dis_z_top", np.inf)),
        }

    def _reason(self, diagnostics):
        if diagnostics["bottom_dist"] <= float(config.SAFETY_SHIELD_EMERGENCY_BOTTOM_DIST):
            return "emergency_bottom"
        if diagnostics["top_dist"] <= float(config.SAFETY_SHIELD_EMERGENCY_TOP_DIST):
            return "emergency_top"
        if diagnostics["any_lidar_min"] <= float(config.SAFETY_SHIELD_ANY_STOP_DIST):
            return "emergency_lidar_any"
        if diagnostics["front_min"] <= float(config.SAFETY_SHIELD_FRONT_STOP_DIST):
            return "emergency_lidar_front"
        if diagnostics["depth_center_min"] <= float(config.SAFETY_SHIELD_DEPTH_STOP_DIST):
            return "emergency_depth_front"
        if (
            diagnostics["front_min"] <= float(config.SAFETY_SHIELD_FRONT_WARN_DIST)
            or diagnostics["depth_center_min"] <= float(config.SAFETY_SHIELD_DEPTH_WARN_DIST)
        ):
            return "warn_front"
        if diagnostics["bottom_dist"] <= float(config.SAFETY_SHIELD_MIN_BOTTOM_DIST):
            return "warn_bottom"
        if diagnostics["top_dist"] <= float(config.SAFETY_SHIELD_MIN_TOP_DIST):
            return "warn_top"
        return "none"

    def _brake_action(self, vertical_only=False):
        action = np.zeros(3, dtype=np.float32)
        velocity = np.asarray(getattr(self.env, "current_velocity", np.zeros(3)), dtype=np.float32)
        max_velocity = max(float(getattr(self.env, "max_velocity", 10.0)), 1e-6)
        accel_scale = max(float(getattr(self.env, "accel_scale", 2.0)), 1e-6)
        brake = -velocity / max_velocity * float(config.SAFETY_SHIELD_BRAKE_GAIN) * (max_velocity / accel_scale)
        if vertical_only:
            brake[:2] = 0.0
        action[:] = brake
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def _recovery_action(self, diagnostics):
        brake_steps = int(getattr(config, "SAFETY_SHIELD_RECOVERY_BRAKE_STEPS", 2))
        brake = self._brake_action(vertical_only=False)

        if self.recovery_step_index < brake_steps:
            brake[2] = min(float(brake[2]), float(config.SAFETY_SHIELD_RECOVERY_CLIMB_ACTION))
            return brake

        yaw_rad = np.deg2rad(float(getattr(self.env, "current_yaw_deg", 0.0)))
        body_forward = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0.0], dtype=np.float32)
        body_left = np.array([-np.sin(yaw_rad), np.cos(yaw_rad), 0.0], dtype=np.float32)
        lateral = body_left * float(self.recovery_side) * float(config.SAFETY_SHIELD_RECOVERY_LATERAL_ACTION)

        rel_pos = np.asarray(getattr(self.env, "current_rel_pos", np.zeros(3)), dtype=np.float32)
        goal_xy = rel_pos[:2].copy()
        goal_norm = float(np.linalg.norm(goal_xy))
        goal_action = np.zeros(3, dtype=np.float32)
        if goal_norm > 1e-6:
            goal_action[:2] = goal_xy / goal_norm * float(config.SAFETY_SHIELD_RECOVERY_GOAL_ACTION)

        front_clear = (
            diagnostics["front_min"] >= float(config.SAFETY_SHIELD_RECOVERY_FRONT_CLEAR_DIST)
            and diagnostics["depth_center_min"] >= float(config.SAFETY_SHIELD_RECOVERY_FRONT_CLEAR_DIST)
        )
        if front_clear:
            action = lateral + goal_action + 0.25 * brake
        else:
            back = -body_forward * float(config.SAFETY_SHIELD_RECOVERY_BACK_ACTION)
            action = back + lateral + 0.35 * brake
            action[2] = float(config.SAFETY_SHIELD_RECOVERY_CLIMB_ACTION)
        return action.astype(np.float32)

    def _apply_vertical_limits(self, action, diagnostics):
        action = np.asarray(action, dtype=np.float32).copy()
        if diagnostics["bottom_dist"] <= float(config.SAFETY_SHIELD_MIN_BOTTOM_DIST):
            # NED z is positive downward, so negative action requests climb.
            action[2] = min(float(action[2]), float(config.SAFETY_SHIELD_CLIMB_ACTION_Z))
        elif diagnostics["top_dist"] <= float(config.SAFETY_SHIELD_MIN_TOP_DIST):
            action[2] = max(float(action[2]), float(config.SAFETY_SHIELD_DESCEND_ACTION_Z))
        return action
