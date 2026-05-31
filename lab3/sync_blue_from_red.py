"""Однократная синхронизация testClientBlue.py из testClientRed.py."""
from pathlib import Path

root = Path(__file__).resolve().parent
text = (root / "testClientRed.py").read_text(encoding="utf-8")
text = text.replace('TEAM = "red"', 'TEAM = "blue"')
text = text.replace(
    "PACE_CATCH_UP_GAP = 120.0   # отставание от синих",
    "PACE_CATCH_UP_GAP = 120.0   # отставание от врага",
)
text = text.replace(
    "# м — как у Blue, чуть раньше pickup",
    "# м — ранний pickup",
)
text = text.replace(
    "(главный рычаг против Blue «ближайшая»)",
    "(приоритет ценности над «ближайшей»)",
)
text = text.replace("blue_agents", "enemy_agents")
text = text.replace("nearest_blue", "nearest_enemy")
text = text.replace(
    "# Гонка с синими: ближайший враг без груза",
    "# Гонка с врагом: ближайший противник без груза",
)
header = "# Зеркало testClientRed.py v7 — та же логика, команда BLUE\n\n"
if not text.startswith("# Зеркало"):
    text = header + text
out = root / "testClientBlue.py"
out.write_text(text, encoding="utf-8")
print("Wrote", out, "lines", len(text.splitlines()))
