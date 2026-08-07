#ifndef SUBWAVELENGTHRESONATORS_WORKFLOWS_LINE_DEFECT_BANDS_H
#define SUBWAVELENGTHRESONATORS_WORKFLOWS_LINE_DEFECT_BANDS_H

namespace workflows {

/* line-defect band diagram via the multipole defect operator M^eps
reference method from Habib Ammari, Erik Orvehed Hiltunen, and Sanghyeon Yu. 
“Subwavelength Guided Modes for Acoustic Waves in Bubbly Crystals with a Line Defect”
https://ems.press/journals/jems/articles/2227767

Works on a square-lattice crystal of high-contrast resonators with one defect row and period 1 in x.

Higher-frequency line-defect bands from the multipole operator M^eps, searched around the defect
resonator's interior Neumann resonances omega_0 = v_b * j'_{m,1}/R_def (dipole, quadrupole, ...),
for direct comparison with run_defect_capacitance_bands at the same geometry. Coarse sigma_min(M)
scan + golden refine within +/- window_factor*delta of each omega_0 is used to find the bands,
excluding the exterior Dirichlet flat-band frequencies.

It can also search the subwavelength (static or 0th monopole) band over [sub_lo, sub_hi] (labelled m=-1),
so one run gives the subwavelength + higher (dipole/quadrupole) bands at the same geometry; set
sub_lo >= sub_hi to skip it.

Writes "defect_neumann_bands.csv": m, alpha_x, omega, sigma_min(M), in_gap, near_dirichlet
(m = -1 subwavelength, 0 monopole, 1 dipole, 2 quadrupole, ...).
n_multipole = 0 selects a frequency-appropriate cutoff (avoids high-order J_n(kR) contamination).
v is the background wave speed, v_b the crystal resonators' interior speed, and v_bd the
defect resonators' interior speed; the searches recenter on omega_0 = v_bd * j'_{m,1}/R_def.*/
void run_line_defect_bands_neumann(double radius = 0.35,
                                   double defect_radius = 0.455,
                                   double delta = 0.001,
                                   int max_m = 2,
                                   double window_factor = 20.0,
                                   int num_alpha = 31,
                                   int n_gauss = 20,
                                   int n_multipole = 0,
                                   double sub_lo = 0.5,
                                   double sub_hi = 3.6,
                                   double v = 1.0,
                                   double v_b = 1.0,
                                   double v_bd = 1.0);

}  // namespace workflows

#endif  // SUBWAVELENGTHRESONATORS_WORKFLOWS_LINE_DEFECT_BANDS_H
