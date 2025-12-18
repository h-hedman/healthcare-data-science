"""
Rule-Based Copper Control Policy
Author: Hayden Hedman
Revised: 2025-12-10

A simple heuristic policy that adjusts copper exposure based on
observed changes in antibiotic resistance (MIC_chloro + MIC_polyB).

Biological intuition:
- Rising MIC → reduce copper to avoid co-selection
- Stable MIC → maintain copper
- Falling MIC → continue decreasing copper (promote recovery)
- Special case: If copper is already very low but resistance rises,
  a small copper increase is allowed as an "emergency adjustment."

Actions:
0 = decrease copper
1 = maintain copper
2 = increase copper
"""


class RuleBasedPolicy:
    """Deterministic copper control strategy based on MIC trends."""

    def __init__(self):
        # Track previous MIC sum to detect directional changes
        self.prev_mic_sum = None

        # Tunable biological thresholds (kept simple for interpretability)
        self.rise_threshold = 1.0     # MIC increase considered meaningful
        self.stable_range = 0.5       # MIC considered "stable" within this range

    # --------------------------------------------------------------
    # RESET
    # --------------------------------------------------------------
    def reset(self):
        """Clear stored MIC history at the start of each episode."""
        self.prev_mic_sum = None

    # --------------------------------------------------------------
    # ACTION SELECTION
    # --------------------------------------------------------------
    def act(self, state):
        """
        Decide copper adjustment based on resistance trends.

        State structure:
            [ copper, MIC_chloro, MIC_polyB, cycle, growth_inhibition ]
        """
        copper, mic_c, mic_p, cycle, _ = state

        # Current total resistance burden
        mic_sum = mic_c + mic_p

        # First cycle: no trend available → maintain copper
        if self.prev_mic_sum is None:
            self.prev_mic_sum = mic_sum
            return 1  # maintain

        # Compute MIC change since last cycle
        delta_mic = mic_sum - self.prev_mic_sum
        self.prev_mic_sum = mic_sum

        # ----------------------------------------------------------
        # RULE 1 — MIC rising
        # ----------------------------------------------------------
        if delta_mic > self.rise_threshold:
            if copper > 5:
                return 0  # decrease copper
            else:
                return 2  # emergency increase if copper is already very low

        # ----------------------------------------------------------
        # RULE 2 — MIC stable
        # ----------------------------------------------------------
        if abs(delta_mic) <= self.stable_range:
            return 1  # maintain copper

        # ----------------------------------------------------------
        # RULE 3 — MIC decreasing
        # ----------------------------------------------------------
        return 0  # decrease copper further
