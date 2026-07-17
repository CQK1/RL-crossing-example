import numpy as np
import pandas as pd
from typing import Dict, Callable, Any

class InhomogeneousPoissonProcess:
    """
    Inhomogeneous Poisson Process (IPP) Traffic Generation Engine.
    
    This class models the arrival of traffic (vehicles or pedestrians) as a Poisson process
    with a time-varying intensity function $\lambda(t)$.
    
    Given a simulation time $t$ in seconds, the probability of an arrival in an infinitesimal 
    time step $dt$ is:
    
    $$P(\text{Arrival in } [t, t+dt)) \approx \lambda(t) \cdot dt$$
    
    Since our simulation uses discrete steps of $dt = 1.0$ second, the probability of 
    generating a vehicle in a given second is approximated directly as $\lambda(t)$.
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
        self.seconds_per_bin = 900.0  # 15 minutes = 900 seconds
        self.seconds_per_day = 86400.0  # 24 hours = 86400 seconds
        
        # Pre-extract values for fast lookup
        self.total_bins = len(self.data)
        
        # Calculate rates ($\lambda$) for each traffic volume column in vehicles/second
        # $\lambda_i = \text{Volume}_i / 900.0$
        self.base_rates: Dict[str, np.ndarray] = {}
        for col in self.data.columns:
            if col != "start_time" and pd.api.types.is_numeric_dtype(self.data[col]):
                self.base_rates[col] = self.data[col].to_numpy() / self.seconds_per_bin

        # Pre-compute the time centers of each 15-minute bin for linear interpolation
        # Bin 0 center: 450s, Bin 1 center: 1350s, etc.
        self.bin_centers = np.arange(self.total_bins) * self.seconds_per_bin + (self.seconds_per_bin / 2.0)

    def _get_cyclic_time(self, time_in_seconds: float) -> float:
        """Helper to wrap time around the 24-hour (86400 seconds) boundary if cyclic."""
        if self.cyclic:
            return time_in_seconds % self.seconds_per_day
        return min(time_in_seconds, (self.total_bins * self.seconds_per_bin) - 1.0)

    def get_rate_step(self, time_in_seconds: float, column_name: str) -> float:
        """
        Mode 1: Piecewise Constant (Step Function) Model.
        The rate $\lambda(t)$ remains constant throughout each 15-minute interval.
        
        $$\lambda(t) = \lambda_k \quad \text{for } t \in [900k, 900(k+1))$$
        """
        t = self._get_cyclic_time(time_in_seconds)
        bin_idx = int(t // self.seconds_per_bin)
        
        # Safe bounds check
        if bin_idx >= self.total_bins:
            bin_idx = self.total_bins - 1
            
        rates = self.base_rates.get(column_name)
        if rates is None:
            return 0.0
        return rates[bin_idx]

    def get_rate_interpolated(self, time_in_seconds: float, column_name: str) -> float:
        """
        Mode 2: Smooth Cyclic Interpolation Model.
        Linear interpolation between 15-minute bin centers to avoid sudden jumps 
        at the hour/interval boundaries, providing a smoother $\lambda(t)$ curve.
        """
        t = self._get_cyclic_time(time_in_seconds)
        rates = self.base_rates.get(column_name)
        if rates is None:
            return 0.0

        # Handing cyclic wrapping on boundary edges
        if self.cyclic:
            # Append cyclic boundaries to allow smooth wrap around
            # Wrap Bin -1 (previous day's end) to the left, and Bin 0 to the right end
            xp = np.concatenate([[-450.0], self.bin_centers, [self.seconds_per_day + 450.0]])
            yp = np.concatenate([[rates[-1]], rates, [rates[0]]])
            return float(np.interp(t, xp, yp))
        else:
            return float(np.interp(t, self.bin_centers, rates))

    def get_rate_custom(self, time_in_seconds: float, column_name: str, 
                        fitting_func: Callable[[float, str], float]) -> float:
        """
        Mode 3: Custom Mathematical Equation Model (Callback Interface).
        Directly evaluate any arbitrary intensity function $\lambda(t)$ mapped from R-code 
        equations (e.g. Fourier polynomial regressions or non-linear curves).
        
        :param time_in_seconds: Current simulation time.
        :param column_name: The column stream name (e.g., "north_thru").
        :param fitting_func: A user-defined python function `f(time, direction)` that returns $\lambda$.
        """
        t = self._get_cyclic_time(time_in_seconds)
        try:
            return max(0.0, fitting_func(t, column_name))
        except Exception as e:
            # Fallback to step function if the callback fails
            print(f"Error evaluating custom fitting function: {e}. Falling back to step rate.")
            return self.get_rate_step(time_in_seconds, column_name)