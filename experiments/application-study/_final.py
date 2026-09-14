"""Frozen study roles and paired identifiers, independent of generated values."""

from __future__ import annotations

from _study import keyed_rng

ROLE_KEYS = {"reg": 1, "stress": 2, "broad": 3, "constrained": 4}


def view_id(role: str, replicate: int, angle: float) -> str:
    return f"{role}-n{replicate:02d}-a{round(angle) + 180:03d}"


def role_specs(protocol: dict) -> dict:
    reg = protocol["registration"]
    acq = protocol["acquisition"]
    return {
        "reg": {
            "replicates": reg["noise_replicates_per_case_and_start"],
            "angles": sorted({a for angles in reg["view_sets_degrees"].values() for a in angles}),
            "beam": reg["summed_open_beam_per_pixel"] / 2,
        },
        "stress": {
            "replicates": reg["stress"]["noise_replicates_per_case"],
            "angles": reg["stress"]["views_degrees"],
            "beam": reg["stress"]["open_beam_per_pixel"],
        },
        "broad": {
            "replicates": acq["replicates_per_case"],
            "angles": [acq["base_view_degrees"], *acq["candidate_angles_degrees"]],
            "beam": acq["base_and_candidate_open_beam_per_pixel"],
        },
        "constrained": {
            "replicates": acq["constrained"]["replicates_per_case"],
            "angles": [acq["base_view_degrees"], *acq["constrained"]["candidate_angles_degrees"]],
            "beam": acq["base_and_candidate_open_beam_per_pixel"],
        },
    }


def registration_jobs(protocol: dict) -> list[dict]:
    result = []
    for replicate in range(role_specs(protocol)["reg"]["replicates"]):
        for start, offset in enumerate(
            protocol["registration"]["initial_offsets_local_se3_mm_rad"]
        ):
            for name, angles in protocol["registration"]["view_sets_degrees"].items():
                result.append(
                    {
                        "name": f"reg-{name}-n{replicate:02d}-s{start}",
                        "role": "reg",
                        "comparison": name,
                        "replicate": replicate,
                        "start": start,
                        "offset": offset,
                        "view_ids": [view_id("reg", replicate, a) for a in angles],
                    }
                )
    for replicate in range(role_specs(protocol)["stress"]["replicates"]):
        for start, offset in enumerate(
            protocol["registration"]["initial_offsets_local_se3_mm_rad"]
        ):
            result.append(
                {
                    "name": f"stress-n{replicate:02d}-s{start}",
                    "role": "stress",
                    "comparison": "stress",
                    "replicate": replicate,
                    "start": start,
                    "offset": offset,
                    "view_ids": [
                        view_id("stress", replicate, a)
                        for a in role_specs(protocol)["stress"]["angles"]
                    ],
                }
            )
    return result


def design_jobs(protocol: dict, case: int) -> list[dict]:
    result = []
    acq = protocol["acquisition"]
    for role in ("broad", "constrained"):
        angles = (
            acq["candidate_angles_degrees"]
            if role == "broad"
            else acq["constrained"]["candidate_angles_degrees"]
        )
        fixed = (
            acq["fixed_policy_angle_degrees"]
            if role == "broad"
            else acq["constrained"]["fixed_angle_degrees"]
        )
        for replicate in range(role_specs(protocol)[role]["replicates"]):
            random_angle = float(
                keyed_rng(
                    protocol["randomness"]["random_policy_root_seed"],
                    case,
                    ROLE_KEYS[role],
                    replicate,
                ).choice(angles)
            )
            result.append(
                {
                    "name": f"{role}-n{replicate:02d}",
                    "role": role,
                    "replicate": replicate,
                    "angles": angles,
                    "fixed_angle": fixed,
                    "random_angle": random_angle,
                    "base_id": view_id(role, replicate, acq["base_view_degrees"]),
                }
            )
    return result


def require_outcome_coverage(protocol: dict, case: int, rows: list[dict]) -> None:
    """Require each independently enumerated final outcome once, including failures."""
    expected = {job["name"]: job for job in registration_jobs(protocol)}
    for job in design_jobs(protocol, case):
        for policy in ("base", "selected", "fixed", "random"):
            name = job["name"] + "-" + policy
            expected[name] = {**job, "name": name, "policy": policy}
    names = [row["name"] for row in rows]
    if len(names) != len(expected) or set(names) != set(expected):
        raise ValueError("missing, duplicate or unexpected prespecified outcome")
    allowed = {"complete", "failed", "blocked_by_base_failure", "blocked_by_design_failure"}
    for row in rows:
        if row["status"] not in allowed:
            raise ValueError("unrecognised final outcome status")
        target = expected[row["name"]]
        for key in ("role", "replicate", "start", "comparison", "policy"):
            if key in target and row.get(key) != target[key]:
                raise ValueError("final outcome was assigned to a different paired role")
        if row["status"] == "blocked_by_base_failure" and row.get("policy") not in (
            "selected",
            "fixed",
            "random",
        ):
            raise ValueError("only downstream policies can be blocked by a base failure")
        if row["status"] == "blocked_by_design_failure" and row.get("policy") != "selected":
            raise ValueError("design failure must not omit fixed or random comparisons")
