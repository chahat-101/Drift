"""Tests and diagnostics for physics_noise.py."""

import math
from pathlib import Path
import unittest

import matplotlib
import numpy as np
import stim

import decoder
from circuits import make_circuit
from physics_noise import (
    CHOSEN_SI1000_P,
    PHYS_REGIMES,
    NoiseModel,
    build_noisy_circuit,
    make_noise_state,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def noiseless_d5():
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=5, rounds=5
    ).flattened()


def split_layers(circuit):
    layers, layer = [], []
    for inst in circuit:
        if inst.name == "TICK":
            layers.append(layer)
            layer = []
        else:
            layer.append(inst)
    layers.append(layer)
    return layers


class PhysicsNoiseTests(unittest.TestCase):
    def test_t1_uniform_detector_error_model_matches_stim(self):
        reference = make_circuit(5, 5, 0.003)
        built = build_noisy_circuit(noiseless_d5(), NoiseModel("uniform", 0.003))
        actual_dem = built.detector_error_model()
        reference_dem = reference.detector_error_model()
        self.assertTrue(actual_dem.approx_equals(reference_dem, atol=1e-12))
        self.assertEqual(actual_dem.num_detectors, reference_dem.num_detectors)
        self.assertEqual(actual_dem.num_observables, reference_dem.num_observables)

    def test_t2_si1000_noise_applications_per_layer(self):
        p = 0.002
        circuit = build_noisy_circuit(noiseless_d5(), NoiseModel("si1000", p))
        layers = split_layers(circuit)

        def actual_noise(layer):
            return [
                (inst.name, inst.gate_args_copy()[0], len(inst.targets_copy()))
                for inst in layer
                if inst.name in ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR")
            ]

        self.assertEqual(actual_noise(layers[0]), [("X_ERROR", 2 * p, 49)])
        for index in (1, 6):
            self.assertEqual(
                actual_noise(layers[index]),
                [("DEPOLARIZE1", p / 10, 49)],
            )
        for index in (2, 3, 4, 5):
            self.assertEqual(
                actual_noise(layers[index]),
                [("DEPOLARIZE2", p, 40), ("DEPOLARIZE1", p / 10, 9)],
            )
        self.assertEqual(
            actual_noise(layers[7]),
            [
                ("X_ERROR", 5 * p, 24),
                ("X_ERROR", 2 * p, 24),
                ("DEPOLARIZE1", p / 10, 25),
                ("DEPOLARIZE1", 2 * p, 25),
            ],
        )
        self.assertEqual(
            actual_noise(layers[-1]),
            [("X_ERROR", 5 * p, 24), ("X_ERROR", 2 * p, 24), ("X_ERROR", 5 * p, 25)],
        )

    def test_t3_hand_checked_substep(self):
        base = stim.Circuit("""
            QUBIT_COORDS(0, 0) 0
            QUBIT_COORDS(2, 0) 1
            QUBIT_COORDS(4, 0) 2
            R 0 1 2
            TICK
            CX 0 1
            TICK
            CX 1 2
            TICK
            M 0 1 2
        """)
        for seed in range(100):
            state = make_noise_state(
                base, "phys_mixed", device_seed=0, stream_seed=seed, n_substeps=1500
            )
            if len(state.impact_steps):
                break
        self.assertTrue(len(state.impact_steps))
        t = int(state.impact_steps[0])

        for q in range(3):
            expected_m = state.heterogeneity[q] * math.exp(state.slow_x[t, q])
            for k in range(2):
                if state.tls_on[t, q, k]:
                    expected_m *= state.tls_strengths[q, k]
            expected_burst = 0.0
            for impact, point in zip(state.impact_steps, state.impact_points):
                if impact > t:
                    continue
                age = t - impact
                amplitude = 0.2 * math.exp(-age / 25)
                if amplitude < 1e-4:
                    continue
                sigma = 1 + 3 * min(age / 10, 1)
                r2 = sum(
                    ((state.coords[q, axis] - point[axis]) / state.data_spacing) ** 2
                    for axis in range(2)
                )
                expected_burst += amplitude * math.exp(-r2 / (2 * sigma * sigma))
            expected_burst = min(0.4, expected_burst)
            np.testing.assert_allclose(state.m[t, q], expected_m, rtol=1e-13)
            np.testing.assert_allclose(state.burst[t, q], expected_burst, rtol=1e-13)

    def test_t4_seed_determinism(self):
        base = noiseless_d5()
        first = make_noise_state(base, "phys_mixed", device_seed=0, stream_seed=123)
        second = make_noise_state(base, "phys_mixed", device_seed=0, stream_seed=123)
        different = make_noise_state(base, "phys_mixed", device_seed=0, stream_seed=124)
        for name in ("m", "burst", "coupler_on", "impact_steps", "impact_points"):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
        self.assertFalse(np.array_equal(first.m, different.m))
        np.testing.assert_array_equal(first.heterogeneity, different.heterogeneity)
        for regime in PHYS_REGIMES:
            for stream_seed in (0, 999):
                state = make_noise_state(base, regime, device_seed=0, stream_seed=stream_seed)
                np.testing.assert_array_equal(state.heterogeneity, first.heterogeneity)
        other_device = make_noise_state(base, "phys_mixed", device_seed=1, stream_seed=123)
        self.assertFalse(np.array_equal(first.heterogeneity, other_device.heterogeneity))

        first_circuit = build_noisy_circuit(base, first.model(CHOSEN_SI1000_P, 0))
        second_circuit = build_noisy_circuit(base, second.model(CHOSEN_SI1000_P, 0))
        different_circuit = build_noisy_circuit(base, different.model(CHOSEN_SI1000_P, 0))
        self.assertEqual(first_circuit, second_circuit)
        self.assertNotEqual(first_circuit, different_circuit)

    def test_t5_tls_spectrum(self):
        n_substeps = 32768
        average_power = None
        n_spectra = 0
        for seed in range(16):
            state = make_noise_state(
                noiseless_d5(), "phys_tls", device_seed=0, stream_seed=seed, n_substeps=n_substeps
            )
            spectra = []
            for q in range(len(state.qubits)):
                factor = np.ones(n_substeps)
                for k in range(2):
                    factor *= np.where(
                        state.tls_on[:, q, k], state.tls_strengths[q, k], 1.0
                    )
                centered = factor - factor.mean()
                spectra.append(np.abs(np.fft.rfft(centered)) ** 2 / n_substeps)
            seed_power = np.mean(spectra, axis=0)
            average_power = seed_power if average_power is None else average_power + seed_power
            n_spectra += len(spectra)
        average_power /= 16
        frequencies = np.fft.rfftfreq(n_substeps, d=1)
        fit_range = (frequencies >= 1 / 10000) & (frequencies <= 1 / 10)
        slope, intercept = np.polyfit(
            np.log10(frequencies[fit_range]),
            np.log10(average_power[fit_range]),
            1,
        )
        self.assertEqual(n_spectra, 16 * 49)
        self.assertTrue(np.isfinite(slope))

        path = Path("results/exp10/tls_spectrum.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(frequencies[1:], average_power[1:], lw=0.7, label="Average periodogram")
        fit_power = 10 ** intercept * frequencies[fit_range] ** slope
        ax.loglog(
            frequencies[fit_range], fit_power, "k--", label=f"Fit slope = {slope:.3f}"
        )
        ax.set(xlabel="Frequency (per sub-step)", ylabel="Power")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        self.assertTrue(path.is_file())
        print(f"\nT5: TLS spectrum slope = {slope:.6f}; plot = {path}")

    def test_t6_process_example_plot(self):
        base = noiseless_d5()
        mixed = make_noise_state(base, "phys_mixed", device_seed=0, stream_seed=11)
        cosmic = make_noise_state(base, "phys_cosmic", device_seed=0, stream_seed=11)
        coupler = make_noise_state(base, "phys_coupler", device_seed=0, stream_seed=11)
        time = np.arange(len(mixed.m)) / 10

        path = Path("results/exp10/processes_example.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        for q in range(3):
            axes[0].plot(time, mixed.m[:, q], label=f"Qubit {mixed.qubits[q]}")
        axes[0].set(ylabel="m (mixed)")
        axes[0].legend()
        axes[1].plot(time, cosmic.burst.max(axis=1))
        axes[1].set(ylabel="Maximum burst")
        for i, pair in enumerate(coupler.coupler_pairs):
            axes[2].step(time, coupler.coupler_on[:, i] + 1.2 * i, where="post", label=str(pair))
        axes[2].set(xlabel="2000-shot steps", ylabel="Coupler on/off", yticks=[])
        axes[2].legend()
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        self.assertTrue(path.is_file())
        print(f"\nT6: process plot = {path}")

    def test_t7_mean_detector_firing_rates(self):
        base = noiseless_d5()
        static_state = make_noise_state(base, "phys_static", device_seed=0, stream_seed=0)
        burst_state = make_noise_state(base, "phys_cosmic", device_seed=0, stream_seed=0)
        self.assertTrue(len(burst_state.impact_steps))
        burst_time = int(burst_state.impact_steps[0])
        circuits = {
            "uniform p=0.003": make_circuit(5, 5, 0.003),
            f"si1000 static p={CHOSEN_SI1000_P}": build_noisy_circuit(
                base, static_state.model(CHOSEN_SI1000_P, 0)
            ),
            f"si1000 active burst p={CHOSEN_SI1000_P}": build_noisy_circuit(
                base, burst_state.model(CHOSEN_SI1000_P, burst_time)
            ),
        }
        rates = {}
        for label, circuit in circuits.items():
            sampler = circuit.compile_detector_sampler(seed=0)
            detections = sampler.sample(shots=100_000)
            rates[label] = float(detections.mean())
        print("\nT7: " + "; ".join(f"{key}: {value:.6f}" for key, value in rates.items()))

    def test_t8_detector_grid_and_observable_layout(self):
        base = noiseless_d5()
        state = make_noise_state(base, "phys_static", device_seed=0, stream_seed=0)
        physical = build_noisy_circuit(base, state.model(0.0015, 0))
        reference = make_circuit(5, 5, 0.003)

        self.assertEqual(physical.num_detectors, reference.num_detectors)
        self.assertEqual(physical.num_observables, reference.num_observables)
        self.assertEqual(physical.get_detector_coordinates(), reference.get_detector_coordinates())

        def observables(circuit):
            return [
                (inst.gate_args_copy(), inst.targets_copy())
                for inst in circuit.flattened()
                if inst.name == "OBSERVABLE_INCLUDE"
            ]

        self.assertEqual(observables(physical), observables(reference))
        physical_grid = decoder.DetectorGrid(physical)
        reference_grid = decoder.DetectorGrid(reference)
        self.assertEqual(physical_grid.shape, reference_grid.shape)
        self.assertEqual(len(physical_grid.idx), len(reference_grid.idx))
        for physical_idx, reference_idx in zip(physical_grid.idx, reference_grid.idx):
            np.testing.assert_array_equal(physical_idx, reference_idx)
        np.testing.assert_array_equal(physical_grid.flat, reference_grid.flat)


if __name__ == "__main__":
    unittest.main(verbosity=2)
