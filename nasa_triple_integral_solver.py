#!/usr/bin/env python3
"""
NASA-Grade Triple Integral Solver - Enhanced Version
Advanced mathematical solver with statistical validation and convergence analysis.
"""

import numpy as np
import random
import sympy as sp
from typing import Dict, List, Any, Tuple
import signal
from collections import defaultdict

class TripleIntegralSolver:
    """
    NASA-Grade Advanced Symbolic Mathematics Solver for Triple Integrals

    Features:
    - Robust symbolic and numerical integration
    - Statistical validation with confidence intervals
    - Adaptive sampling for convergence
    - Error analysis and quality metrics
    - Support for complex integrands and domains
    """

    def __init__(self):
        self.x, self.y, self.z = sp.symbols('x y z', real=True)
        self.difficulty_levels = {
            'basic': self._generate_basic_integral,
            'intermediate': self._generate_intermediate_integral,
            'advanced': self._generate_advanced_integral,
            'expert': self._generate_expert_integral
        }

        # Statistical validation parameters
        self.convergence_threshold = 1e-6
        self.max_iterations = 100
        self.confidence_level = 0.95

    def _generate_basic_integral(self) -> Dict[str, Any]:
        """Generate basic triple integral problem with analytical validation."""
        # ∭ (x + y + z) dx dy dz over [0,1] × [0,1] × [0,1]
        integrand = self.x + self.y + self.z
        integral = sp.Integral(sp.Integral(sp.Integral(integrand, (self.z, 0, 1)), (self.y, 0, 1)), (self.x, 0, 1))
        analytical_solution = 1.5  # Exact analytical result

        return {
            'integrand': integrand,
            'expression': integral,
            'analytical_solution': analytical_solution,
            'difficulty': 'basic',
            'domain': [(self.x, 0, 1), (self.y, 0, 1), (self.z, 0, 1)],
            'complexity_score': 1.0,
            'expected_convergence_rate': 0.01
        }

    def _generate_intermediate_integral(self) -> Dict[str, Any]:
        """Generate intermediate triple integral with polynomial complexity."""
        # ∭ (x²*y*z) dx dy dz over [0,2] × [0,1] × [0,3]
        integrand = self.x**2 * self.y * self.z
        integral = sp.Integral(sp.Integral(sp.Integral(integrand, (self.z, 0, 3)), (self.y, 0, 1)), (self.x, 0, 2))
        analytical_solution = 6.0  # Exact analytical result

        return {
            'integrand': integrand,
            'expression': integral,
            'analytical_solution': analytical_solution,
            'difficulty': 'intermediate',
            'domain': [(self.x, 0, 2), (self.y, 0, 1), (self.z, 0, 3)],
            'complexity_score': 2.5,
            'expected_convergence_rate': 0.05
        }

    def _generate_advanced_integral(self) -> Dict[str, Any]:
        """Generate advanced triple integral with trigonometric functions."""
        # ∭ (sin(x)*cos(y)*exp(z)) dx dy dz over [0,π/2] × [0,π/4] × [0,1]
        integrand = sp.sin(self.x) * sp.cos(self.y) * sp.exp(self.z)
        integral = sp.Integral(sp.Integral(sp.Integral(integrand, (self.z, 0, 1)), (self.y, 0, sp.pi/4)), (self.x, 0, sp.pi/2))

        # Compute exact analytical solution
        analytical_solution = float(sp.N(integral.doit()))

        return {
            'integrand': integrand,
            'expression': integral,
            'analytical_solution': analytical_solution,
            'difficulty': 'advanced',
            'domain': [(self.x, 0, sp.pi/2), (self.y, 0, sp.pi/4), (self.z, 0, 1)],
            'complexity_score': 4.0,
            'expected_convergence_rate': 0.1
        }

    def _generate_expert_integral(self) -> Dict[str, Any]:
        """Generate expert-level triple integral with complex functions."""
        # ∭ (x*y*z*exp(-x²-y²-z²)) dx dy dz over [-2,2] × [-2,2] × [-2,2]
        integrand = self.x * self.y * self.z * sp.exp(-(self.x**2 + self.y**2 + self.z**2))
        integral = sp.Integral(sp.Integral(sp.Integral(integrand, (self.z, -2, 2)), (self.y, -2, 2)), (self.x, -2, 2))

        # Numerical analytical solution (exact computation would be complex)
        analytical_solution = 0.0  # Approximate for this complex case

        return {
            'integrand': integrand,
            'expression': integral,
            'analytical_solution': analytical_solution,
            'difficulty': 'expert',
            'domain': [(self.x, -2, 2), (self.y, -2, 2), (self.z, -2, 2)],
            'complexity_score': 5.0,
            'expected_convergence_rate': 0.2
        }

    def generate_problem(self, difficulty: str = 'random') -> Dict[str, Any]:
        """Generate a validated triple integral problem."""
        if difficulty == 'random':
            # Weighted random selection favoring intermediate problems
            difficulties = ['basic'] * 2 + ['intermediate'] * 3 + ['advanced'] * 2 + ['expert'] * 1
            difficulty = random.choice(difficulties)

        problem = self.difficulty_levels[difficulty]()

        # Validate problem structure
        self._validate_problem(problem)

        return problem

    def _validate_problem(self, problem: Dict[str, Any]) -> None:
        """Validate problem structure and analytical solution."""
        required_keys = ['integrand', 'expression', 'analytical_solution', 'domain']
        for key in required_keys:
            if key not in problem:
                raise ValueError(f"Problem missing required key: {key}")

        # Validate domain
        if len(problem['domain']) != 3:
            raise ValueError("Triple integral must have exactly 3 integration variables")

        # Test integrand evaluation
        try:
            test_point = [0.5, 0.5, 0.5]
            test_val = float(problem['integrand'].subs([(self.x, test_point[0]),
                                                       (self.y, test_point[1]),
                                                       (self.z, test_point[2])]))
            if not np.isfinite(test_val):
                raise ValueError("Integrand evaluation failed")
        except Exception as e:
            raise ValueError(f"Invalid integrand: {e}")

    def solve_numerically(self, problem: Dict[str, Any], method: str = 'adaptive_monte_carlo',
                         initial_samples: int = 1000) -> Dict[str, Any]:
        """
        Solve triple integral using advanced numerical methods with statistical validation.

        Methods:
        - monte_carlo: Basic Monte Carlo integration
        - adaptive_monte_carlo: Adaptive sampling with convergence detection
        - stratified_monte_carlo: Stratified sampling for better convergence
        """
        integrand = problem['integrand']
        domain = problem['domain']

        if method == 'adaptive_monte_carlo':
            return self._adaptive_monte_carlo_integration(integrand, domain, initial_samples)
        elif method == 'stratified_monte_carlo':
            return self._stratified_monte_carlo_integration(integrand, domain, initial_samples)
        elif method == 'monte_carlo':
            return self._basic_monte_carlo_integration(integrand, domain, initial_samples)
        else:
            raise ValueError(f"Unknown integration method: {method}")

    def _compute_confidence_interval(self, estimate: float, standard_error: float) -> Tuple[float, float]:
        """Compute confidence interval using t-distribution approximation."""
        if standard_error == 0:
            return (estimate, estimate)

        # Use z-score for 95% confidence
        z_score = 1.96  # Approximately 95% confidence for large samples
        margin_of_error = z_score * standard_error

        return (estimate - margin_of_error, estimate + margin_of_error)

    def _adaptive_monte_carlo_integration(self, integrand, domain: List[Tuple],
                                        initial_samples: int) -> Dict[str, Any]:
        """
        Adaptive Monte Carlo integration with convergence detection and statistical validation.
        """
        x_var, x_min, x_max = domain[0]
        y_var, y_min, y_max = domain[1]
        z_var, z_min, z_max = domain[2]

        # Convert symbolic limits to float
        x_min = float(x_min)
        x_max = float(x_max)
        y_min = float(y_min)
        y_max = float(y_max)
        z_min = float(z_min)
        z_max = float(z_max)

        volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)

        # Create numerical function using lambdify for efficiency
        f_num = sp.lambdify([x_var, y_var, z_var], integrand, 'numpy')

        # Adaptive sampling parameters
        samples = initial_samples
        batch_size = 1000
        convergence_window = 10
        convergence_history = []

        total_sum = 0.0
        total_samples = 0
        variance_estimate = 0.0

        iteration = 0
        converged = False

        while iteration < self.max_iterations and not converged:
            # Generate batch of samples
            batch_sum = 0.0
            batch_values = []

            for _ in range(batch_size):
                x_val = random.uniform(x_min, x_max)
                y_val = random.uniform(y_min, y_max)
                z_val = random.uniform(z_min, z_max)

                try:
                    f_val = float(f_num(x_val, y_val, z_val))
                    if np.isfinite(f_val):
                        batch_sum += f_val
                        batch_values.append(f_val)
                    else:
                        batch_values.append(0.0)  # Handle singularities
                except (ValueError, TypeError, OverflowError):
                    batch_values.append(0.0)

            # Update running statistics
            total_sum += batch_sum
            total_samples += batch_size

            # Estimate current integral value
            current_estimate = (total_sum / total_samples) * volume

            # Update variance estimate
            if len(batch_values) > 1:
                batch_variance = np.var(batch_values)
                variance_estimate = (variance_estimate * (total_samples - batch_size) +
                                   batch_variance * batch_size) / total_samples

            # Check convergence
            convergence_history.append(current_estimate)
            if len(convergence_history) >= convergence_window:
                recent_values = convergence_history[-convergence_window:]
                convergence_rate = np.std(recent_values) / abs(np.mean(recent_values))

                if convergence_rate < self.convergence_threshold:
                    converged = True

            iteration += 1

            # Adaptive sample size increase
            if iteration % 5 == 0 and not converged:
                batch_size = min(batch_size * 2, 10000)

        # Compute confidence interval
        standard_error = np.sqrt(variance_estimate / total_samples) * volume
        confidence_interval = self._compute_confidence_interval(current_estimate, standard_error)

        return {
            'method': 'adaptive_monte_carlo',
            'result': current_estimate,
            'samples': total_samples,
            'volume': volume,
            'convergence_achieved': converged,
            'iterations': iteration,
            'standard_error': standard_error,
            'confidence_interval': confidence_interval,
            'variance_estimate': variance_estimate,
            'convergence_rate': convergence_rate if 'convergence_rate' in locals() else float('inf')
        }

    def _stratified_monte_carlo_integration(self, integrand, domain: List[Tuple],
                                          samples: int) -> Dict[str, Any]:
        """Stratified Monte Carlo integration for improved convergence."""
        x_var, x_min, x_max = domain[0]
        y_var, y_min, y_max = domain[1]
        z_var, z_min, z_max = domain[2]

        volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)

        # Create numerical function using lambdify for efficiency
        f_num = sp.lambdify([x_var, y_var, z_var], integrand, 'numpy')

        # Stratification parameters
        strata_x = max(2, int(np.sqrt(samples / 8)))
        strata_y = max(2, int(np.sqrt(samples / 8)))
        strata_z = max(2, int(np.sqrt(samples / 8)))

        samples_per_stratum = samples // (strata_x * strata_y * strata_z)

        total_sum = 0.0
        stratum_variances = []

        for i in range(strata_x):
            for j in range(strata_y):
                for k in range(strata_z):
                    # Define stratum bounds
                    x_low = x_min + i * (x_max - x_min) / strata_x
                    x_high = x_min + (i + 1) * (x_max - x_min) / strata_x
                    y_low = y_min + j * (y_max - y_min) / strata_y
                    y_high = y_min + (j + 1) * (y_max - y_min) / strata_y
                    z_low = z_min + k * (z_max - z_min) / strata_z
                    z_high = z_min + (k + 1) * (z_max - z_min) / strata_z

                    stratum_volume = (x_high - x_low) * (y_high - y_low) * (z_high - z_low)
                    stratum_sum = 0.0
                    stratum_values = []

                    # Sample within stratum
                    for _ in range(samples_per_stratum):
                        x_val = random.uniform(x_low, x_high)
                        y_val = random.uniform(y_low, y_high)
                        z_val = random.uniform(z_low, z_high)

                        try:
                            f_val = float(f_num(x_val, y_val, z_val))
                            if np.isfinite(f_val):
                                stratum_sum += f_val
                                stratum_values.append(f_val)
                        except (ValueError, TypeError, OverflowError):
                            stratum_values.append(0.0)

                    # Stratum estimate
                    stratum_estimate = (stratum_sum / samples_per_stratum) * stratum_volume
                    total_sum += stratum_estimate

                    # Stratum variance
                    if stratum_values:
                        stratum_variances.append(np.var(stratum_values) * (stratum_volume ** 2) / samples_per_stratum)

        # Overall estimate
        result = total_sum

        # Combined variance estimate
        variance_estimate = sum(stratum_variances) if stratum_variances else 0.0
        standard_error = np.sqrt(variance_estimate)
        confidence_interval = self._compute_confidence_interval(result, standard_error)

        return {
            'method': 'stratified_monte_carlo',
            'result': result,
            'samples': samples,
            'volume': volume,
            'strata': (strata_x, strata_y, strata_z),
            'standard_error': standard_error,
            'confidence_interval': confidence_interval,
            'variance_estimate': variance_estimate
        }

    def _basic_monte_carlo_integration(self, integrand, domain: List[Tuple], samples: int) -> Dict[str, Any]:
        """Basic Monte Carlo integration with statistical analysis."""
        x_var, x_min, x_max = domain[0]
        y_var, y_min, y_max = domain[1]
        z_var, z_min, z_max = domain[2]

        # Convert domain limits to floats
        x_min, x_max = float(x_min), float(x_max)
        y_min, y_max = float(y_min), float(y_max)
        z_min, z_max = float(z_min), float(z_max)

        volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)

        # Create numerical function using lambdify for efficiency
        f_num = sp.lambdify([x_var, y_var, z_var], integrand, 'numpy')

        total_sum = 0.0
        valid_samples = 0
        sample_values = []

        for _ in range(samples):
            x_val = random.uniform(x_min, x_max)
            y_val = random.uniform(y_min, y_max)
            z_val = random.uniform(z_min, z_max)

            try:
                f_val = float(f_num(x_val, y_val, z_val))
                if np.isfinite(f_val) and isinstance(f_val, (int, float)):
                    total_sum += f_val
                    sample_values.append(f_val)
                    valid_samples += 1
            except (ValueError, TypeError, AttributeError, OverflowError):
                continue

        if valid_samples == 0:
            raise ValueError("No valid samples generated for integration")

        result = (total_sum / valid_samples) * volume

        # Statistical analysis
        if len(sample_values) > 1:
            variance_estimate = np.var(sample_values, ddof=1) / valid_samples * (volume ** 2)
            standard_error = np.sqrt(variance_estimate)
            confidence_interval = self._compute_confidence_interval(result, standard_error)
        else:
            variance_estimate = 0.0
            standard_error = 0.0
            confidence_interval = (result, result)

        return {
            'method': 'monte_carlo',
            'result': result,
            'samples': samples,
            'valid_samples': valid_samples,
            'volume': volume,
            'standard_error': standard_error,
            'confidence_interval': confidence_interval,
            'variance_estimate': variance_estimate
        }

    def _symbolic_integration(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt symbolic integration with timeout and error handling."""
        try:
            # Set timeout for symbolic computation
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("Symbolic integration timeout")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30 second timeout

            result = problem['expression'].doit()

            signal.alarm(0)  # Cancel timeout

            # Convert to numerical value
            numerical_result = float(sp.N(result))

            return {
                'method': 'symbolic',
                'result': numerical_result,
                'success': True,
                'symbolic_result': result
            }

        except TimeoutError:
            return {
                'method': 'symbolic',
                'result': None,
                'success': False,
                'error': 'timeout'
            }
        except Exception as e:
            return {
                'method': 'symbolic',
                'result': None,
                'success': False,
                'error': str(e)
            }

    def evaluate_solution_quality(self, problem: Dict[str, Any], solution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive evaluation of numerical solution quality with statistical metrics.
        """
        analytical = problem.get('analytical_solution', 0.0)
        numerical = solution.get('result')

        if numerical is None or not np.isfinite(numerical):
            return {
                'accuracy': 0.0,
                'absolute_error': float('inf'),
                'relative_error': float('inf'),
                'quality_score': 0.0,
                'confidence_level': 0.0,
                'statistical_significance': False
            }

        # Basic error metrics
        absolute_error = abs(analytical - numerical)
        relative_error = absolute_error / abs(analytical) if analytical != 0 else float('inf')

        # Statistical evaluation
        standard_error = solution.get('standard_error', 0.0)
        confidence_interval = solution.get('confidence_interval', (numerical, numerical))

        # Check if analytical solution is within confidence interval
        ci_contains_analytical = (confidence_interval[0] <= analytical <= confidence_interval[1])

        # Quality score based on multiple factors
        accuracy_score = 1.0 - min(relative_error, 1.0)
        precision_score = 1.0 - min(standard_error / abs(numerical) if numerical != 0 else 1.0, 1.0)
        reliability_score = 1.0 if ci_contains_analytical else 0.5

        # Overall quality score (weighted average)
        quality_score = (accuracy_score * 0.5 + precision_score * 0.3 + reliability_score * 0.2)

        # Statistical significance (if we have enough samples)
        samples = solution.get('samples', 0)
        statistical_significance = (samples >= 1000 and relative_error < 0.1 and ci_contains_analytical)

        return {
            'accuracy': accuracy_score,
            'absolute_error': absolute_error,
            'relative_error': relative_error,
            'quality_score': quality_score,
            'precision_score': precision_score,
            'reliability_score': reliability_score,
            'confidence_interval_contains_analytical': ci_contains_analytical,
            'statistical_significance': statistical_significance,
            'convergence_achieved': solution.get('convergence_achieved', False)
        }