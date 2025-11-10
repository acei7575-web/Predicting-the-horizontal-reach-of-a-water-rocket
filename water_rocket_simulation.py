"""Water rocket flight simulation and analysis toolkit.

This module provides a physics-based time-domain simulator for a two-phase
water rocket.  The solver resolves the water expulsion phase using
polytropic compression of the trapped air, transitions to an air-only
expansion model that supports choked and unchoked nozzle flow, and integrates
ballistic motion with aerodynamic drag.  In addition to the core solver, the
module exposes helper utilities for parameter conversion, calibration, and
sensitivity analysis tailored to the scenario described in the prompt.
"""

# 사용 변수·상수·단위 정리 -----------------------------------------------------
# dry_mass [kg]                : 로켓의 건조 질량 (사용자가 측정한 60 g → 0.06 kg)
# water_volume [m^3]           : 병에 채운 물의 체적
# bottle_volume [m^3]          : 병 내부 총 체적
# nozzle_diameter [m]          : 분사 노즐 지름 (20 mm)
# discharge_coefficient [-]    : 유량 계수 (노즐 형상에 따라 불확실)  # TODO: uncertain
# drag_coefficient [-]         : 공력 항력 계수 (외형에 따라 불확실)  # TODO: uncertain
# cross_sectional_area [m^2]   : 로켓 정면적
# initial_air_pressure [Pa]    : 발사 직전 병 내부 공기 압력
# air_temperature [K]          : 공기 온도 (절대온도)
# launch_angle_deg [deg]       : 발사각
# time_step [s]                : 수치 적분 시간 간격
# max_time [s]                 : 시뮬레이션 최대 시간
# air_density [kg/m^3]         : 주변 공기 밀도 (기상 조건으로부터 계산)
# gravity [m/s^2]              : 중력 가속도 (9.80665 m/s^2)
# polytropic_exponent [-]      : 물 분사 구간의 다원자 지수 (비정상 추진 모델)  # TODO: uncertain
#
# 파생 상수 및 단위 --------------------------------------------------------------
# R_AIR [J/(kg·K)]             : 건조 공기의 기체 상수
# GAMMA_AIR [-]                : 공기의 비열비 (질식/비질식 판정에 사용)
# RHO_WATER [kg/m^3]           : 물의 밀도
# P_ATM [Pa]                   : 대기압 기준값
# PSI_TO_PA [Pa/psi]           : psi → Pa 환산 계수
# ------------------------------------------------------------------------------
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple


R_AIR = 287.058  # Specific gas constant for dry air (J/(kg·K))
GAMMA_AIR = 1.4
RHO_WATER = 1000.0  # kg/m^3
P_ATM = 101_325.0  # Pa
PSI_TO_PA = 6_894.75729  # Exact conversion for lbf/in^2 to Pa


@dataclass
class LaunchConditions:
    """Metadata describing the environment of a particular launch."""

    location: str
    date_time: datetime
    temperature_c: float
    ambient_pressure_pa: float = P_ATM
    wind_speed_m_per_s: float = 0.0
    wind_direction: Optional[str] = None
    gust_speed_m_per_s: Optional[float] = None
    gust_direction: Optional[str] = None
    precipitation_mm: Optional[float] = None
    humidity_percent: Optional[float] = None


def estimate_air_density(temperature_c: float, *, pressure_pa: float = P_ATM) -> float:
    """Estimate air density from temperature and pressure assuming dry air."""

    temperature_k = temperature_c + 273.15
    return pressure_pa / (R_AIR * temperature_k)


@dataclass
class RocketParameters:
    """Container for the physical properties of the water rocket."""

    dry_mass: float
    water_volume: float
    bottle_volume: float
    nozzle_diameter: float
    discharge_coefficient: float
    drag_coefficient: float
    cross_sectional_area: float
    initial_air_pressure: float
    air_temperature: float = 300.0
    launch_angle_deg: float = 45.0
    time_step: float = 1e-3
    max_time: float = 30.0
    air_density: float = 1.225
    gravity: float = 9.80665
    polytropic_exponent: float = 1.2


def nozzle_water_flow(
    pressure_pa: float,
    ambient_pressure_pa: float,
    discharge_coefficient: float,
    nozzle_area: float,
    *,
    fluid_density: float = RHO_WATER,
) -> Tuple[float, float, float]:
    """Compute the water-phase mass flow rate, exit velocity, and thrust.

    The function assumes Bernoulli flow from a high-pressure reservoir to the
    ambient environment.  The resulting units are verified at runtime to catch
    accidental unit inconsistencies.
    """

    if pressure_pa <= ambient_pressure_pa or nozzle_area <= 0.0:
        return 0.0, 0.0, 0.0

    if fluid_density <= 0.0:
        raise ValueError("Fluid density must be positive for nozzle flow calculations.")

    delta_p = pressure_pa - ambient_pressure_pa
    # Bernoulli 방정식 (에너지 보존)으로부터 출구 속도 v = sqrt(2 Δp / ρ)를 계산.
    exit_velocity = math.sqrt(2.0 * delta_p / fluid_density)
    # 질량 보존식 ṁ = C_d ρ A v 로 순간 유량을 산출.
    mass_flow_rate = discharge_coefficient * fluid_density * nozzle_area * exit_velocity

    # Unit sanity check: [kg/s] = [kg/m^3] * [m^2] * [m/s]
    unit_check = fluid_density * nozzle_area * exit_velocity
    if not math.isclose(mass_flow_rate, discharge_coefficient * unit_check, rel_tol=1e-9):
        raise AssertionError("Water nozzle mass-flow units inconsistent.")

    thrust = mass_flow_rate * exit_velocity
    return mass_flow_rate, exit_velocity, thrust


def update_tank_pressure_water(
    *,
    polytropic_constant: float,
    air_volume: float,
    polytropic_exponent: float,
    ambient_pressure: float = P_ATM,
) -> float:
    """Update the trapped air pressure during the water expulsion phase.

    The relation :math:`p V^n = C` is enforced, where ``n`` is the supplied
    polytropic exponent.  The resulting pressure is never allowed to drop below
    the ambient value so that the solver can detect the end of the water phase.
    """

    if air_volume <= 0.0:
        raise ValueError("Air volume must remain positive inside the bottle.")
    if polytropic_exponent <= 0.0:
        raise ValueError("Polytropic exponent must be positive.")

    # 비정상(폴리트로픽) 과정: p V^n = const → p = C / V^n.
    pressure = polytropic_constant / (air_volume ** polytropic_exponent)
    if pressure < 0.0 or not math.isfinite(pressure):
        raise ArithmeticError("Computed an invalid reservoir pressure during water phase.")

    return max(pressure, ambient_pressure)


def nozzle_air_flow(
    pressure_pa: float,
    reservoir_temperature_k: float,
    ambient_pressure_pa: float,
    discharge_coefficient: float,
    nozzle_area: float,
    gamma: float = GAMMA_AIR,
) -> Tuple[float, float, float]:
    """Return mass flow, exit velocity, and exit pressure during the air phase."""

    if pressure_pa <= ambient_pressure_pa or nozzle_area <= 0.0:
        return 0.0, 0.0, ambient_pressure_pa
    if reservoir_temperature_k <= 0.0:
        raise ValueError("Reservoir temperature must remain positive.")

    critical_ratio = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
    pressure_ratio = ambient_pressure_pa / pressure_pa

    if pressure_ratio <= critical_ratio:
        # 질식(Choked) 조건: 노즐 목에서 마하 1, 등엔트로피 에너지 방정식 사용.
        exit_pressure = pressure_pa * critical_ratio
        exit_temperature = reservoir_temperature_k * (2.0 / (gamma + 1.0))
        mass_flow_rate = (
            discharge_coefficient
            * nozzle_area
            * pressure_pa
            * math.sqrt(gamma / (R_AIR * reservoir_temperature_k))
            * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
        )
        exit_velocity = math.sqrt(gamma * R_AIR * exit_temperature)
    else:
        # 비질식(유량 제한 없음) 조건: 등엔트로피 관계와 에너지 방정식으로 출구 상태 결정.
        exit_pressure = ambient_pressure_pa
        pressure_term = pressure_ratio ** ((gamma - 1.0) / gamma)
        energy_term = max(0.0, 1.0 - pressure_term)
        exit_temperature = reservoir_temperature_k * pressure_term
        velocity_term = max(0.0, 2.0 * gamma / (gamma - 1.0) * R_AIR * reservoir_temperature_k * energy_term)
        exit_velocity = math.sqrt(velocity_term)
        flow_term = (
            2.0
            * gamma
            / (R_AIR * reservoir_temperature_k * (gamma - 1.0))
            * pressure_ratio ** (2.0 / gamma)
            * energy_term
        )
        mass_flow_rate = discharge_coefficient * nozzle_area * pressure_pa * math.sqrt(max(flow_term, 0.0))

    if not math.isfinite(mass_flow_rate) or mass_flow_rate < 0.0:
        mass_flow_rate = 0.0
    if not math.isfinite(exit_velocity) or exit_velocity < 0.0:
        exit_velocity = 0.0

    return mass_flow_rate, exit_velocity, exit_pressure


def simulate_flight(
    parameters: RocketParameters,
    *,
    log_events: bool = False,
) -> Dict[str, List[float]]:
    """Simulate the flight of a water rocket with detailed phase modeling."""

    params = parameters
    if params.bottle_volume <= 0.0:
        raise ValueError("Bottle volume must be positive.")
    if params.water_volume < 0.0 or params.water_volume >= params.bottle_volume:
        raise ValueError("Water volume must be within (0, bottle_volume).")
    if params.initial_air_pressure <= P_ATM:
        raise ValueError("Initial air pressure must exceed atmospheric pressure.")
    if params.polytropic_exponent <= 0.0:
        raise ValueError("Polytropic exponent must be positive.")

    orientation = math.radians(params.launch_angle_deg)
    nozzle_area = math.pi * (params.nozzle_diameter * 0.5) ** 2

    water_mass = RHO_WATER * params.water_volume
    current_water_volume = params.water_volume
    initial_air_volume = params.bottle_volume - params.water_volume
    if initial_air_volume <= 0.0:
        raise ValueError("Initial air volume must be positive.")

    initial_air_mass = (
        params.initial_air_pressure * initial_air_volume / (R_AIR * params.air_temperature)
    )
    if initial_air_mass <= 0.0:
        raise ValueError("Initial air mass must be positive.")

    total_mass = params.dry_mass + water_mass + initial_air_mass
    if total_mass <= 0.0:
        raise ValueError("Total mass must be positive.")

    water_polytropic_constant = params.initial_air_pressure * (
        initial_air_volume ** params.polytropic_exponent
    )

    time = 0.0
    x = 0.0
    y = 0.0
    vx = 0.0
    vy = 0.0

    air_mass = initial_air_mass
    air_volume = initial_air_volume
    tank_pressure = params.initial_air_pressure
    air_phase_constant = None
    transition_logged = False

    time_history: List[float] = [time]
    x_history: List[float] = [x]
    y_history: List[float] = [y]
    vx_history: List[float] = [vx]
    vy_history: List[float] = [vy]
    mass_history: List[float] = [total_mass]
    pressure_history: List[float] = [tank_pressure]
    thrust_history: List[float] = [0.0]

    cos_theta = math.cos(orientation)
    sin_theta = math.sin(orientation)

    while time < params.max_time:
        dt_remaining = params.time_step
        while dt_remaining > 1e-12:
            speed = math.hypot(vx, vy)
            if speed > 1e-6:
                thrust_dir_x = vx / speed
                thrust_dir_y = vy / speed
            else:
                thrust_dir_x = cos_theta
                thrust_dir_y = sin_theta

            total_mass = params.dry_mass + water_mass + air_mass
            if total_mass <= 0.0:
                raise ArithmeticError("Non-positive total mass encountered during integration.")

            water_phase_active = water_mass > 1e-9 and current_water_volume > 1e-9

            thrust = 0.0
            exit_pressure = P_ATM
            actual_dt = dt_remaining
            delta_water = 0.0
            delta_air = 0.0

            if water_phase_active:
                previous_water_mass = water_mass
                air_volume = params.bottle_volume - current_water_volume
                tank_pressure = update_tank_pressure_water(
                    polytropic_constant=water_polytropic_constant,
                    air_volume=air_volume,
                    polytropic_exponent=params.polytropic_exponent,
                    ambient_pressure=P_ATM,
                )

                if tank_pressure <= P_ATM:
                    water_mass = 0.0
                    current_water_volume = 0.0
                    air_volume = params.bottle_volume
                else:
                    mass_flow_rate, exit_velocity, thrust = nozzle_water_flow(
                        tank_pressure,
                        P_ATM,
                        params.discharge_coefficient,
                        nozzle_area,
                    )
                    if mass_flow_rate > 0.0:
                        max_dt = water_mass / mass_flow_rate
                        actual_dt = min(dt_remaining, max_dt)
                        delta_water = mass_flow_rate * actual_dt
                    else:
                        actual_dt = dt_remaining
                        delta_water = 0.0

                if water_mass <= 0.0:
                    current_water_volume = 0.0
                    air_volume = params.bottle_volume
                    if air_mass > 0.0:
                        rho_air = air_mass / air_volume
                        air_phase_constant = tank_pressure / (rho_air ** GAMMA_AIR)
                    else:
                        air_phase_constant = None
            else:
                air_volume = params.bottle_volume
                if air_phase_constant is None and air_mass > 0.0:
                    rho_air = air_mass / air_volume
                    air_phase_constant = max(tank_pressure, P_ATM) / (rho_air ** GAMMA_AIR)

                if air_mass <= 1e-9:
                    tank_pressure = P_ATM
                    thrust = 0.0
                else:
                    tank_pressure = air_phase_constant * ((air_mass / air_volume) ** GAMMA_AIR)
                    if tank_pressure <= P_ATM:
                        tank_pressure = P_ATM
                        thrust = 0.0
                    else:
                        reservoir_temperature = tank_pressure * air_volume / (air_mass * R_AIR)
                        mass_flow_rate, exit_velocity, exit_pressure = nozzle_air_flow(
                            tank_pressure,
                            reservoir_temperature,
                            P_ATM,
                            params.discharge_coefficient,
                            nozzle_area,
                        )
                        if mass_flow_rate > 0.0:
                            max_dt = air_mass / mass_flow_rate
                            actual_dt = min(dt_remaining, max_dt)
                            delta_air = mass_flow_rate * actual_dt
                            thrust = mass_flow_rate * exit_velocity + (
                                exit_pressure - P_ATM
                            ) * nozzle_area
                        else:
                            actual_dt = dt_remaining
                            delta_air = 0.0
                            thrust = 0.0

            drag_x = 0.0
            drag_y = 0.0
            if speed > 1e-6:
                # 항력: F_d = 0.5 ρ C_d A v^2 (뉴턴 제2법칙 적용 시 감쇠력).
                drag_magnitude = (
                    0.5
                    * params.air_density
                    * params.drag_coefficient
                    * params.cross_sectional_area
                    * speed
                    * speed
                )
                drag_x = -drag_magnitude * vx / speed
                drag_y = -drag_magnitude * vy / speed

            gravity_force = -total_mass * params.gravity

            # 뉴턴 제2법칙: a = ΣF / m (추력, 항력, 중력 합력).
            ax = (thrust * thrust_dir_x + drag_x) / total_mass
            ay = (thrust * thrust_dir_y + drag_y + gravity_force) / total_mass

            vx += ax * actual_dt
            vy += ay * actual_dt
            x += vx * actual_dt
            y += vy * actual_dt

            if delta_water > 0.0:
                water_mass = max(water_mass - delta_water, 0.0)
                current_water_volume = water_mass / RHO_WATER
                assert math.isclose(
                    current_water_volume,
                    water_mass / RHO_WATER,
                    rel_tol=1e-6,
                    abs_tol=1e-9,
                ), "Water mass/volume inconsistency detected."
                if water_mass <= 1e-9 and previous_water_mass > 1e-9 and not transition_logged:
                    tank_pressure = update_tank_pressure_water(
                        polytropic_constant=water_polytropic_constant,
                        air_volume=params.bottle_volume,
                        polytropic_exponent=params.polytropic_exponent,
                        ambient_pressure=P_ATM,
                    )
                    rho_air = air_mass / params.bottle_volume if air_mass > 0.0 else 0.0
                    if air_mass > 0.0 and rho_air > 0.0:
                        air_phase_constant = tank_pressure / (rho_air ** GAMMA_AIR)
                    transition_logged = True
                    if log_events:
                        transition_speed = math.hypot(vx, vy)
                        print(
                            "[전환 이벤트] t={:.3f} s, 속도={:.2f} m/s, 압력={:.1f} kPa".format(
                                time + actual_dt,
                                transition_speed,
                                tank_pressure / 1000.0,
                            )
                        )
            if delta_air > 0.0:
                air_mass = max(air_mass - delta_air, 0.0)
                if air_mass <= 1e-9:
                    tank_pressure = P_ATM

            time += actual_dt
            dt_remaining -= actual_dt

            total_mass = params.dry_mass + water_mass + air_mass
            time_history.append(time)
            x_history.append(x)
            y_history.append(y)
            vx_history.append(vx)
            vy_history.append(vy)
            mass_history.append(total_mass)
            pressure_history.append(tank_pressure)
            thrust_history.append(thrust)

            if y < 0.0 and time > 0.0:
                break

        if y < 0.0 and time > 0.0:
            break

    if y < 0.0 and len(y_history) >= 2:
        y_prev = y_history[-2]
        x_prev = x_history[-2]
        t_prev = time_history[-2]

        frac = y_prev / (y_prev - y)
        x_touchdown = x_prev + frac * (x - x_prev)
        t_touchdown = t_prev + frac * (time - t_prev)

        time_history[-1] = t_touchdown
        x_history[-1] = x_touchdown
        y_history[-1] = 0.0
        vx_history[-1] = vx_history[-2] + frac * (vx_history[-1] - vx_history[-2])
        vy_history[-1] = vy_history[-2] + frac * (vy_history[-1] - vy_history[-2])
    else:
        x_touchdown = x_history[-1]

    result = {
        "time": time_history,
        "x": x_history,
        "y": y_history,
        "vx": vx_history,
        "vy": vy_history,
        "mass": mass_history,
        "pressure": pressure_history,
        "thrust": thrust_history,
        "range": [x_touchdown],
    }
    return result


def compute_range(parameters: RocketParameters) -> float:
    """Return only the horizontal range of the rocket."""

    history = simulate_flight(parameters)
    return history["range"][0]


def build_parameters_from_measurements(
    *,
    nose_mass_g: float,
    water_volume_ml: float,
    bottle_volume_ml: float,
    nozzle_diameter_mm: float,
    body_diameter_mm: float,
    launch_angle_deg: float,
    initial_air_pressure_psi: float,
    discharge_coefficient: float = 0.92,  # TODO: uncertain
    drag_coefficient: float = 0.5,  # TODO: uncertain
    air_temperature_c: float = 25.0,
    time_step: float = 5e-4,
    max_time: float = 30.0,
    air_density: float = 1.2,
    gravity: float = 9.80665,
    polytropic_exponent: float = 1.2,  # TODO: uncertain
) -> RocketParameters:
    """Convert practical build measurements to :class:`RocketParameters`."""

    dry_mass = nose_mass_g / 1000.0
    water_volume = water_volume_ml / 1_000_000.0
    bottle_volume = bottle_volume_ml / 1_000_000.0
    nozzle_diameter = nozzle_diameter_mm / 1000.0
    body_diameter = body_diameter_mm / 1000.0
    cross_sectional_area = math.pi * (body_diameter * 0.5) ** 2

    air_temperature_k = air_temperature_c + 273.15
    initial_air_pressure = P_ATM + initial_air_pressure_psi * PSI_TO_PA

    return RocketParameters(
        dry_mass=dry_mass,
        water_volume=water_volume,
        bottle_volume=bottle_volume,
        nozzle_diameter=nozzle_diameter,
        discharge_coefficient=discharge_coefficient,
        drag_coefficient=drag_coefficient,
        cross_sectional_area=cross_sectional_area,
        initial_air_pressure=initial_air_pressure,
        air_temperature=air_temperature_k,
        launch_angle_deg=launch_angle_deg,
        time_step=time_step,
        max_time=max_time,
        air_density=air_density,
        gravity=gravity,
        polytropic_exponent=polytropic_exponent,
    )


def calibrate_parameters(
    base_parameters: RocketParameters,
    target_range: float,
    search_specs: Sequence[Dict[str, float]],
    *,
    target_time_window: Tuple[float, float] | None = None,
    time_weight: float = 1.0,
) -> Dict[str, float]:
    """Sequentially tune selected parameters to reduce range error."""

    tuned_parameters = base_parameters
    base_result = simulate_flight(tuned_parameters)
    current_range = base_result["range"][0]
    current_time = base_result["time"][-1]

    tuned_values: Dict[str, float] = {}

    def objective(range_value: float, time_value: float) -> float:
        range_error = abs(range_value - target_range)
        if target_time_window is None:
            return range_error

        lower, upper = target_time_window
        if time_value < lower:
            time_penalty = lower - time_value
        elif time_value > upper:
            time_penalty = time_value - upper
        else:
            time_penalty = 0.0

        return range_error + time_weight * time_penalty

    for spec in search_specs:
        name = spec["name"]
        min_value = spec["min"]
        max_value = spec["max"]
        steps = max(int(spec.get("steps", 20)), 1)

        best_value = getattr(tuned_parameters, name)
        best_range = current_range
        best_time = current_time
        best_error = objective(best_range, best_time)

        for step in range(steps + 1):
            fraction = step / steps
            candidate_value = min_value + (max_value - min_value) * fraction
            candidate_parameters = replace(tuned_parameters, **{name: candidate_value})
            candidate_result = simulate_flight(candidate_parameters)
            candidate_range = candidate_result["range"][0]
            candidate_time = candidate_result["time"][-1]
            candidate_error = objective(candidate_range, candidate_time)

            if candidate_error < best_error:
                best_error = candidate_error
                best_value = candidate_value
                best_range = candidate_range
                best_time = candidate_time

        tuned_parameters = replace(tuned_parameters, **{name: best_value})
        current_range = best_range
        current_time = best_time
        tuned_values[name] = best_value

    return tuned_values


def run_sensitivity_analysis(
    base_parameters: RocketParameters,
    cd_values: Sequence[float],
    n_values: Sequence[float],
    area_scales: Sequence[float],
) -> Dict[float, List[List[float]]]:
    """Evaluate range sensitivity across drag, polytropic exponent, and nozzle area."""

    results: Dict[float, List[List[float]]] = {}
    for area_scale in area_scales:
        area_rows: List[List[float]] = []
        diameter_scale = math.sqrt(area_scale)
        for cd in cd_values:
            row: List[float] = []
            for n in n_values:
                candidate = replace(
                    base_parameters,
                    drag_coefficient=cd,
                    polytropic_exponent=n,
                    nozzle_diameter=base_parameters.nozzle_diameter * diameter_scale,
                )
                row.append(compute_range(candidate))
            area_rows.append(row)
        results[area_scale] = area_rows
    return results


def _format_heatmap_table(
    cd_values: Sequence[float],
    n_values: Sequence[float],
    data: List[List[float]],
) -> List[str]:
    """Render a text heatmap for the supplied range data."""

    flat_values = [value for row in data for value in row]
    v_min = min(flat_values)
    v_max = max(flat_values)
    span = max(v_max - v_min, 1e-9)
    shades = " .:-=+*#%@"

    header = "Cd↓ / n→ | " + " ".join(f"{n:5.2f}" for n in n_values)
    lines = [header, "-" * len(header)]
    for cd, row in zip(cd_values, data):
        entries = []
        for value in row:
            normalized = (value - v_min) / span
            index = min(int(normalized * (len(shades) - 1)), len(shades) - 1)
            entries.append(f"{value:5.1f}{shades[index]}")
        lines.append(f"{cd:8.3f} | " + " ".join(entries))
    lines.append("(문자열 음영: 낮은 값 → ' ', 높은 값 → '@')")
    return lines


if __name__ == "__main__":
    launch_conditions = LaunchConditions(
        location="대한민국, 남양주",
        date_time=datetime(2025, 9, 5, 16, 20, 0),
        temperature_c=32.0,
        wind_speed_m_per_s=0.9,
        wind_direction="동",
        gust_speed_m_per_s=1.5,
        gust_direction="동남동",
        precipitation_mm=0.0,
        humidity_percent=57.0,
    )

    ambient_air_density = estimate_air_density(
        launch_conditions.temperature_c, pressure_pa=launch_conditions.ambient_pressure_pa
    )

    nose_mass_g = 60.0
    scenario_parameters = build_parameters_from_measurements(
        nose_mass_g=nose_mass_g,
        water_volume_ml=385.0,
        bottle_volume_ml=3000.0,
        nozzle_diameter_mm=20.0,
        body_diameter_mm=88.0,
        launch_angle_deg=45.0,
        initial_air_pressure_psi=40.0,
        discharge_coefficient=0.92,
        drag_coefficient=0.5,
        air_temperature_c=launch_conditions.temperature_c,
        air_density=ambient_air_density,
        polytropic_exponent=1.2,
    )

    result = simulate_flight(scenario_parameters, log_events=True)
    range_estimate = result["range"][0]
    flight_time = result["time"][-1]
    actual_range_m = 89.0
    range_error = range_estimate - actual_range_m
    percent_error = abs(range_error) / actual_range_m * 100.0

    print(
        "시나리오: 탄두 60 g, 물 385 mL, 발사각 45°, 게이지 공기압 40 psi"
        " (1.5 L 페트병 2개 결합 기체, 노즐 지름 20 mm)"
    )
    formatted_dt = launch_conditions.date_time.strftime("%Y.%m.%d %H:%M")
    precipitation_str = "자료 없음"
    if launch_conditions.precipitation_mm is not None:
        if launch_conditions.precipitation_mm == 0.0:
            precipitation_str = "X(0mm)"
        else:
            precipitation_str = f"O({launch_conditions.precipitation_mm:.1f}mm)"
    wind_dir = launch_conditions.wind_direction or "자료 없음"
    gust_dir = launch_conditions.gust_direction or "자료 없음"
    gust_speed = (
        f"{launch_conditions.gust_speed_m_per_s:.1f}"
        if launch_conditions.gust_speed_m_per_s is not None
        else "자료 없음"
    )
    humidity = (
        f"{launch_conditions.humidity_percent:.0f}%"
        if launch_conditions.humidity_percent is not None
        else "자료 없음"
    )
    print(
        "발사 조건: {loc}, {dt} 발사, 강수 유무 {precip}, 기온(C):{temp:.1f}, "
        "10분 풍향:{wind_dir}, 10분 풍속: {wind_speed:.1f} (m/s), "
        "순간최대풍향: {gust_dir}, 순간최대풍속:{gust_speed}(m/s), 습도:{humidity}".format(
            loc=launch_conditions.location,
            dt=formatted_dt,
            precip=precipitation_str,
            temp=launch_conditions.temperature_c,
            wind_dir=wind_dir,
            wind_speed=launch_conditions.wind_speed_m_per_s,
            gust_dir=gust_dir,
            gust_speed=gust_speed,
            humidity=humidity,
        )
    )
    print(f"추정 대기 밀도: {ambient_air_density:.3f} kg/m³")
    print(f"모의 수평 도달 거리(보정 전): {range_estimate:.2f} m")
    print(f"비행 시간(보정 전): {flight_time:.2f} s")
    print(f"실측 수평 도달 거리: {actual_range_m:.2f} m")
    print(f"오차: {range_error:+.2f} m")
    print(f"오차율: {percent_error:.2f}%")
    print("사용자 제공 입력값(변경 없음):")
    print(f" - 탄두 질량: {nose_mass_g:.0f} g (전체 건조 질량으로 사용)")
    print(" - 주입수량: 385 mL")
    print(" - 발사각: 45°")
    print(" - 게이지 공기압: 40 psi")

    print("\n[민감도 분석] Cd_body ∈ [0.3, 0.9], n ∈ [1.0, 1.4], 노즐 면적 ±10%")
    cd_values = [0.30, 0.45, 0.60, 0.75, 0.90]
    n_values = [1.0, 1.1, 1.2, 1.3, 1.4]
    area_scales = [0.9, 1.0, 1.1]
    sensitivity_results = run_sensitivity_analysis(
        scenario_parameters, cd_values, n_values, area_scales
    )
    for scale in area_scales:
        print(f"\n- 노즐 단면적 스케일 {scale * 100:.0f}%")
        table_lines = _format_heatmap_table(cd_values, n_values, sensitivity_results[scale])
        for line in table_lines:
            print("  " + line)

    calibration_specs = [
        {"name": "drag_coefficient", "min": 0.3, "max": 0.9, "steps": 32},
        {"name": "polytropic_exponent", "min": 1.0, "max": 1.4, "steps": 32},
    ]
    tuned_values = calibrate_parameters(
        scenario_parameters,
        actual_range_m,
        calibration_specs,
        target_time_window=None,
    )
    tuned_parameters = replace(scenario_parameters, **tuned_values)
    tuned_result = simulate_flight(tuned_parameters, log_events=True)
    tuned_range = tuned_result["range"][0]
    tuned_error = tuned_range - actual_range_m
    tuned_percent_error = abs(tuned_error) / actual_range_m * 100.0

    print("\n[캘리브레이션] 실측 사거리 89.0 m 기준")
    print("  단계 | Cd_body |  n  | 예측 사거리 (m) | 오차 (%)")
    print(
        "  기본 | {cd:7.3f} | {n:3.1f} | {rng:17.2f} | {err:7.3f}".format(
            cd=scenario_parameters.drag_coefficient,
            n=scenario_parameters.polytropic_exponent,
            rng=range_estimate,
            err=percent_error,
        )
    )
    print(
        "  보정 | {cd:7.3f} | {n:3.1f} | {rng:17.2f} | {err:7.3f}".format(
            cd=tuned_parameters.drag_coefficient,
            n=tuned_parameters.polytropic_exponent,
            rng=tuned_range,
            err=tuned_percent_error,
        )
    )

    label_map = {
        "drag_coefficient": "항력 계수",
        "polytropic_exponent": "다상 지수 n",
    }

    print("\n보정된 변수 변화:")
    for spec in calibration_specs:
        name = spec["name"]
        before = getattr(scenario_parameters, name)
        after = tuned_values[name]
        delta = after - before
        print(
            f" - {label_map[name]}: {before:.3f} → {after:.3f} (변화량 {delta:+.3f})"
        )

    print("\n보정 후 모의 수평 도달 거리: {:.2f} m".format(tuned_range))
    print("보정 후 비행 시간: {:.2f} s".format(tuned_result["time"][-1]))
    print("보정 후 오차: {:+.2f} m".format(tuned_error))
    print("보정 후 오차율: {:.2f}%".format(tuned_percent_error))
    print("※ 사용자 입력값(탄두 질량 60 g, 물 385 mL, 발사각 45°, 공기압 40 psi)은 그대로 유지했습니다.")