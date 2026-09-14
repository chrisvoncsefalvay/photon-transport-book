"""Shared host/device identifiers for transport derivative and random domains."""

# Nonnegative values select a material-density score in the replay kernel.
SOURCE_AMPLITUDE = -1
LOG_SOURCE_AMPLITUDE = -2

# Collision events occupy the lower half; source draws and angular rejection
# reserve this bit in the event and domain words, respectively.
RANDOM_NAMESPACE_BIT = 1 << 31
