"""Water rocket flight simulation.

This module provides a time-domain simulator for a two-dimensional water rocket.
The model resolves the thrust produced while water is expelled, includes the
mass variation due to the remaining water, and accounts for aerodynamic drag and
gravity.  The primary output is the horizontal range of the rocket, but the full
state history is also returned for further analysis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List


R_AIR = 287.058  # Specific gas constant for dry air (J/(kg·K))
GAMMA_AIR = 1.4
RHO_WATER = 1000.0  # kg/m^3
P_ATM = 101_325.0  # Pa


@dataclass
class RocketParameters:
    """Container for the physical properties of the water rocket.

    Attributes
    ----------
    dry_mass:
        Mass of the empty rocket (kg).
    water_volume:
        Initial volume of the water inside the bottle (m^3).
    bottle_volume:
        Total interior volume of the pressure vessel (m^3).
    nozzle_diameter:
        Diameter of the nozzle exit (m).
    discharge_coefficient:
        Empirical discharge coefficient (dimensionless, typically 0.8--1.0).
    drag_coefficient:
        Quadratic drag coefficient referenced to ``cross_sectional_area``.
    cross_sectional_area:
        Reference area used for the drag calculation (m^2).
    initial_air_pressure:
        Absolute pressure of the compressed air at launch (Pa).
    air_temperature:
        Absolute temperature of the compressed air (K).  Used to estimate the
        initial mass of the trapped air.
    launch_angle_deg:
        Launch rail elevation angle (degrees above the horizontal).
    time_step:
        Integration time step (s).
    max_time:
        Maximum simulation time before aborting (s).
    air_density:
        Ambient air density for aerodynamic drag (kg/m^3).
    gravity:
        Magnitude of the gravitational acceleration (m/s^2).
    """

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


def simulate_flight(parameters: RocketParameters) -> Dict[str, List[float]]:
    """Simulate the flight of a water rocket.

    Parameters
    ----------
    parameters:
        Fully-populated :class:`RocketParameters` describing the rocket.

    Returns
    -------
    dict
        Time history of the simulation with the following keys:

        ``"time"`` (s), ``"x"`` (m), ``"y"`` (m), ``"vx"`` (m/s), ``"vy"`` (m/s),
        ``"mass"`` (kg), ``"pressure"`` (Pa) and ``"thrust"`` (N).

        The final entry of the ``"x"`` series corresponds to the interpolated
        horizontal range when the rocket returns to ground level.

    Raises
    ------
    ValueError
        If the supplied parameters are physically inconsistent.
    """

    params = parameters
    if params.bottle_volume <= 0:
        raise ValueError("Bottle volume must be positive.")
    if params.water_volume < 0 or params.water_volume >= params.bottle_volume:
        raise ValueError("Water volume must be within (0, bottle_volume).")
    if params.initial_air_pressure <= P_ATM:
        raise ValueError("Initial air pressure must exceed atmospheric pressure.")

    orientation = math.radians(params.launch_angle_deg)
    nozzle_area = math.pi * (params.nozzle_diameter * 0.5) ** 2

    water_mass = RHO_WATER * params.water_volume
    initial_air_volume = params.bottle_volume - params.water_volume
    initial_air_mass = (
        params.initial_air_pressure
        * initial_air_volume
        / (R_AIR * params.air_temperature)
    )

    total_mass = params.dry_mass + water_mass + initial_air_mass
    if total_mass <= 0:
        raise ValueError("Total mass must be positive.")

    time = 0.0
    x = 0.0
    y = 0.0
    vx = 0.0
    vy = 0.0

    time_history: List[float] = [time]
    x_history: List[float] = [x]
    y_history: List[float] = [y]
    vx_history: List[float] = [vx]
    vy_history: List[float] = [vy]
    mass_history: List[float] = [total_mass]
    pressure_history: List[float] = [params.initial_air_pressure]
    thrust_history: List[float] = [0.0]

    current_water_volume = params.water_volume
    current_air_pressure = params.initial_air_pressure

    cos_theta = math.cos(orientation)
    sin_theta = math.sin(orientation)

    while time < params.max_time:
        dt = params.time_step

        # Update orientation: align thrust with current velocity when moving.
        speed = math.hypot(vx, vy)
        if speed > 1e-6:
            thrust_dir_x = vx / speed
            thrust_dir_y = vy / speed
        else:
            thrust_dir_x = cos_theta
            thrust_dir_y = sin_theta

        thrust = 0.0
        current_pressure = P_ATM

        if water_mass > 1e-9:
            current_air_volume = params.bottle_volume - current_water_volume
            current_air_volume = max(current_air_volume, 1e-9)
            current_pressure = params.initial_air_pressure * (
                initial_air_volume / current_air_volume
            ) ** GAMMA_AIR

            if current_pressure > P_ATM:
                exit_velocity = math.sqrt(2.0 * (current_pressure - P_ATM) / RHO_WATER)
                mass_flow_rate = (
                    params.discharge_coefficient
                    * RHO_WATER
                    * nozzle_area
                    * exit_velocity
                )

                max_mdot = water_mass / dt
                mass_flow_rate = min(mass_flow_rate, max_mdot)

                thrust = mass_flow_rate * exit_velocity
                delta_mass = mass_flow_rate * dt

                water_mass = max(water_mass - delta_mass, 0.0)
                current_water_volume = water_mass / RHO_WATER
                total_mass = params.dry_mass + water_mass + initial_air_mass
            else:
                current_pressure = P_ATM
                thrust = 0.0
        else:
            total_mass = params.dry_mass + initial_air_mass

        drag_x = 0.0
        drag_y = 0.0
        if speed > 1e-6:
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

        thrust_x = thrust * thrust_dir_x
        thrust_y = thrust * thrust_dir_y

        gravity_force = -total_mass * params.gravity

        ax = (thrust_x + drag_x) / total_mass
        ay = (thrust_y + drag_y + gravity_force) / total_mass

        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt

        time += dt

        time_history.append(time)
        x_history.append(x)
        y_history.append(y)
        vx_history.append(vx)
        vy_history.append(vy)
        mass_history.append(total_mass)
        pressure_history.append(current_pressure)
        thrust_history.append(thrust)

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


if __name__ == "__main__":
    example = RocketParameters(
        dry_mass=0.15,
        water_volume=0.0015,  # 1.5 L bottle filled to 1.5 L of water
        bottle_volume=0.002,
        nozzle_diameter=0.022,
        discharge_coefficient=0.95,
        drag_coefficient=0.35,
        cross_sectional_area=0.00785,
        initial_air_pressure=5.0 * P_ATM,
        launch_angle_deg=45.0,
    )

    result = simulate_flight(example)
    range_estimate = result["range"][0]
    print(f"Predicted horizontal range: {range_estimate:.2f} m")
    print(f"Flight time: {result['time'][-1]:.2f} s")
