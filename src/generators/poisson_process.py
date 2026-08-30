import numpy as np
import pandas as pd
from typing import Dict, Callable, Any


class InhomogeneousPoissonProcess:
    """
    Inhomogeneous Poisson Process (IPP) Traffic Generation Engine.

    This class models the arrival of traffic (vehicles or pedestrians) as a Poisson process
    with a time-varying intensity function lambda(t).

    Given a simulation time t in seconds, the probability of an arrival in an infinitesimal
    time step dt is:

        P(Arrival in [t, t+dt)) ≈ lambda(t) * dt

    Since our simulation uses discrete steps of dt = 1.0 second, the probability of
    generating a vehicle in a given second is approximated directly as lambda(t).
    """

    def __init__(self, data_frame: pd.DataFrame, cyclic: bool = True):
        """
        Initializes the Poisson Process Engine using preprocessed historical Excel data.

        :param data_frame: Pandas DataFrame cleaned and returned by TrafficDataReader.
        :param cyclic: If True, wraps simulation time around a 24-hour cycle (86400 seconds)
                       so multi-day simulations can run infinitely.
        """
        self.data = data_frame
        self.cyclic = cyclic
        self.seconds_per_bin = 900.0      # 15 minutes = 900 seconds
        self.seconds_per_day = 86400.0    # 24 hours = 86400 seconds
        self.total_bins = len(self.data)

        # Calculate rates (lambda) for each traffic volume column in vehicles/second
        # lambda_i = Volume_i / 900.0
        self.base_rates: Dict[str, np.ndarray] = {}
        for col in self.data.columns:
            if col != "start_time" and pd.api.types.is_numeric_dtype(self.data[col]):
                self.base_rates[col] = self.data[col].to_numpy() / self.seconds_per_bin

        # Pre-compute the time centers of each 15-minute bin for linear interpolation
        # Bin 0 center: 450s, Bin 1 center: 1350s, etc.
        self.bin_centers = np.arange(self.total_bins) * self.seconds_per_bin + (self.seconds_per_bin / 2.0)

        # Precompute full-day lookup table (86400 seconds) for fast indexed access
        # This eliminates per-step np.interp calls entirely.
        xp = np.concatenate([[-450.0], self.bin_centers, [self.seconds_per_day + 450.0]])
        t_all = np.arange(int(self.seconds_per_day), dtype=np.float32)

        self.rate_lookup_table: Dict[str, np.ndarray] = {}
        for col, rates in self.base_rates.items():
            yp = np.concatenate([[rates[-1]], rates, [rates[0]]])
            self.rate_lookup_table[col] = np.interp(t_all, xp, yp).astype(np.float32)

    def _get_cyclic_time(self, time_in_seconds: float) -> float:
        """Helper to wrap time around the 24-hour (86400 seconds) boundary if cyclic."""
        if self.cyclic:
            return time_in_seconds % self.seconds_per_day
        return min(time_in_seconds, (self.total_bins * self.seconds_per_bin) - 1.0)

    def get_rate_fast(self, second_idx: int, column_name: str) -> float:
        """
        Direct O(1) table lookup for a given second index.

        This method is the fastest way to query lambda(t) and should be used
        inside the simulation's per-second micro-step loop.

        :param second_idx: Integer simulation time in seconds.
        :param column_name: The column stream name (e.g., "north_thru").
        :return: Arrival probability for that second.
        """
        table = self.rate_lookup_table.get(column_name)
        if table is None:
            return 0.0
        idx = int(second_idx) % int(self.seconds_per_day)
        return float(table[idx])

    def get_rate_step(self, time_in_seconds: float, column_name: str) -> float:
        """
        Mode 1: Piecewise Constant (Step Function) Model.
        The rate lambda(t) remains constant throughout each 15-minute interval.

        lambda(t) = lambda_k  for t in [900k, 900(k+1))
        """
        t = self._get_cyclic_time(time_in_seconds)
        bin_idx = int(t // self.seconds_per_bin)

        # Safe bounds check
        if bin_idx >= self.total_bins:
            bin_idx = self.total_bins - 1

        rates = self.base_rates.get(column_name)
        if rates is None:
            return 0.0
        return float(rates[bin_idx])

    def get_rate_interpolated(self, time_in_seconds: float, column_name: str) -> float:
        """
        Mode 2: Smooth Cyclic Interpolation Model.

        This method is kept for backward compatibility.
        For performance-critical simulation loops, use get_rate_fast() instead.
        """
        return self.get_rate_fast(int(time_in_seconds), column_name)

    def get_rate_custom(self, time_in_seconds: float, column_name: str,
                        fitting_func: Callable[[float, str], float]) -> float:
        """
        Mode 3: Custom Mathematical Equation Model (Callback Interface).

        Directly evaluate any arbitrary intensity function lambda(t) mapped from
        R-code equations (e.g. Fourier polynomial regressions or non-linear curves).

        :param time_in_seconds: Current simulation time.
        :param column_name: The column stream name (e.g., "north_thru").
        :param fitting_func: A user-defined python function f(time, direction) that returns lambda.
        """
        t = self._get_cyclic_time(time_in_seconds)
        try:
            return max(0.0, fitting_func(t, column_name))
        except Exception as e:
            print(f"Error evaluating custom fitting function: {e}. Falling back to step rate.")
            return self.get_rate_step(time_in_seconds, column_name)