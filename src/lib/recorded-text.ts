/** Reviewed reading copy for retained records. Source records keep their original bytes. */
const readingCopy = new Map<string, string>([
  [
    "Assigned CT-derived material phantom with simulated independent Poisson channels; finite-budget reconstruction, not measured patient composition.",
    "Assigned CT-derived material phantom with simulated independent Poisson channels: finite-budget reconstruction, not measured patient composition.",
  ],
  [
    "Native pixel-zero-centre coordinates; the extent spans sampled-cell spacing, not the physical pixel aperture",
    "Native pixel-zero-centre coordinates. The extent spans sampled-cell spacing, not the physical pixel aperture",
  ],
  [
    "Acquired cumulative Total/High observations; displayed Low is their difference. The fitted effective response fails a reserved calibration criterion. The shown frame was used for fitting.",
    "Acquired cumulative Total/High observations, with displayed Low given by their difference. The fitted effective response fails a reserved calibration criterion. The shown frame was used for fitting.",
  ],
  [
    "Assigned material phantom; fractions are not measured patient composition.",
    "Assigned material phantom. Fractions are not measured patient composition.",
  ],
  [
    "1,000-update budget reached; stationarity and noise-limited performance are not established.",
    "1,000-update budget reached. Stationarity and noise-limited performance are not established.",
  ],
  [
    "Vertebral region includes cropped ribs; source-header laterality is not independently established.",
    "Vertebral region includes cropped ribs. Source-header laterality is not independently established.",
  ],
  [
    "Reference and recovered arrays share the inverse grid; surfaces remain unsmoothed and open at cropped boundaries.",
    "Reference and recovered arrays share the inverse grid. Surfaces remain unsmoothed and open at cropped boundaries.",
  ],
  [
    "Rister et al. CT-ORG (2019), case 2 bone labels; cropped, assigned and volume averaged",
    "Rister et al. CT-ORG (2019), case 2 bone labels, cropped, assigned and volume averaged",
  ],
  [
    "Hubbell and Seltzer, NIST SRD126; water coefficient source used for simulation",
    "Hubbell and Seltzer, NIST SRD126, the water coefficient source used for simulation",
  ],
  [
    "Response calibration v1 fails its frozen gate; absolute material quantification is unaccepted.",
    "Response calibration v1 fails its frozen gate. Absolute material quantification is unaccepted.",
  ],
  [
    "Equivalent coefficients are not HAP concentrations; the rod carrier composition is undocumented.",
    "Equivalent coefficients are not HAP concentrations. The rod carrier composition is undocumented.",
  ],
  [
    "342 retained updates, 344 executed including two lost before checkpoint recovery; time budget reached without convergence.",
    "342 retained updates, 344 executed including two lost before checkpoint recovery. Time budget reached without convergence.",
  ],
  [
    "Only the central z = -4 to +4 mm slab is shown; surfaces remain open and do not recover physical rod ends.",
    "Only the central z = -4 to +4 mm slab is shown. Surfaces remain open and do not recover physical rod ends.",
  ],
  [
    "Weak separation and speckle remain in the recorded fields; display levels do not classify material.",
    "Weak separation and speckle remain in the recorded fields. Display levels do not classify material.",
  ],
  [
    "Recorded mean cumulative Total/High intensities; frozen air normalisation",
    "Recorded mean cumulative Total/High intensities with frozen air normalisation",
  ],
  [
    "Zhou et al. (2025), calibration phantom measured data; fixed masks, native sample selection and project reconstruction",
    "Zhou et al. (2025), calibration phantom measured data, with fixed masks, native sample selection and project reconstruction",
  ],
  [
    "Lower thorax, abdomen and pelvis; not a full chest volume.",
    "Lower thorax, abdomen and pelvis, rather than a full chest volume.",
  ],
  [
    "Abdomen and pelvis, including hips and lower spine; native inferior crop retained.",
    "Abdomen and pelvis, including hips and lower spine, with the native inferior crop retained.",
  ],
  [
    "Coronal plane, y=5.81543 mm · −300 to 1,300 HU. x increases right; z increases up.",
    "Coronal plane, y=5.81543 mm · −300 to 1,300 HU. x increases right, and z increases up.",
  ],
]);

export function recordedText(text: string): string {
  return readingCopy.get(text) ?? text;
}
