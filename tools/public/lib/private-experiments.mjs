/** Root experiment records and their generated source copies stay private. */
export function isPrivateExperimentPath(relative) {
  return /^(?:experiments(?:\/|$)|public\/generated\/source-files\/experiments(?:\/|$)|public\/generated\/historical-sources\/[^/]+\/experiments(?:\/|$)|public\/generated\/worked-examples\/inputs\/reconstruction\/physics\/(?:prepare_open_physics|provision_inputs)\.py$)/.test(
    relative,
  );
}
