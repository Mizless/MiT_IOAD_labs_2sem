"""Собрать testClientRed.py (v7) из testClientBlue.py (v8) + логика пользователя."""
from pathlib import Path

root = Path(__file__).resolve().parent
blue = (root / "testClientBlue.py").read_text(encoding="utf-8")

# --- шапка и команда ---
text = blue.replace("# BLUE v8 — роли: 2×COLLECTOR, 1×THIEF, 2×DISRUPTOR (все механики)",
                    "# RED v7 — единая стратегия (твой скрипт), без ролей")
text = text.replace('TEAM = "blue"', 'TEAM = "red"')

# --- убрать константы ролей ---
for block in [
    "THIEF_CHASE_DIST    = 10.0   # м — вор гонится за грузом врага\n",
    "RELAY_DROP_MIN      = 9.0    # м до базы — сброс для релея (тяжёлый груз)\n",
    "RELAY_DROP_MAX      = 17.0   # м — дальше сам несёт на базу\n",
    "RELAY_PICKUP_BONUS  = 25.0   # множитель приоритета relay-монет\n",
    "RELAY_SPOT_TTL      = 35.0   # сек — помним точку релея\n",
    "DISRUPT_CHARGE_MAX  = 85     # % — пока заряд steal, мешаем / байтим\n",
    "\n# Роли по порядку подключения агентов (0..4)\n"
    'ROLE_LAYOUT = ["COLLECTOR", "COLLECTOR", "THIEF", "DISRUPTOR", "DISRUPTOR"]\n',
]:
    text = text.replace(block, "")

text = text.replace(
    '    "agent_roles":     {},    # {agent_id: "COLLECTOR"|"THIEF"|"DISRUPTOR"}\n'
    '    "relay_spots":     [],    # [{pos, until}] — сбросы вора для подбора\n',
    "",
)

# --- relay в score ---
relay_block = """
    # Relay: монета рядом с точкой сброса вора
    now = time.time()
    for spot in team_memory.get("relay_spots", []):
        if spot["until"] < now:
            continue
        rp = spot["pos"]
        if math.hypot(treasure["pos"]["x"] - rp["x"], treasure["pos"]["z"] - rp["z"]) < 18.0:
            s *= RELAY_PICKUP_BONUS
            break

    return s"""
text = text.replace(relay_block, "\n    return s")

# --- хелперы ролей (между relay-блоком в score и EXPLORE) ---
start = text.find("\ndef role_for_agent_index")
end = text.find("\nEXPLORE_POINTS = [")
if start != -1 and end != -1:
    # plan_team_assignments остаётся — она перед EXPLORE в Blue
    plan_start = text.rfind("\ndef plan_team_assignments", 0, start)
    if plan_start != -1 and plan_start < start:
        text = text[:start] + text[plan_start:end] + text[end:]
    else:
        text = text[:start] + text[end:]

# --- Agent: убрать role в __init__ ---
text = text.replace(
    """    def __init__(self, agent_id, explore_index=0, role=None):
        self.id             = str(agent_id)
        self.role           = role or team_memory["agent_roles"].get(self.id, "COLLECTOR")
        team_memory["agent_roles"][self.id] = self.role
        self.target_id      = None""",
    """    def __init__(self, agent_id, explore_index=0):
        self.id             = str(agent_id)
        self.target_id      = None""",
)

# --- убрать методы ролей до decide ---
start = text.find("\n    def _tick_role(self, tactic):")
end = text.find("\n    def decide(self, me, treasures, bases, game_state, golems, all_agents):")
if start != -1 and end != -1:
    text = text[:start] + text[end:]

# --- decide: заменить ролевой хвост на v7 ---
old_tail = """        # ================================================================
        # 5. С ГРУЗОМ — по роли
        # ================================================================
        steal_ready = me.get("stealAbilityReady", False)
        steal_pct   = me.get("stealChargePercentage", 0) or 0

        if has_treasure:
            self._release_assignment()
            if self.role == "THIEF":
                return self._thief_deliver(pos, base_pos, my_weight)
            return self._deliver_cargo(pos, base_pos, all_agents)

        # ================================================================
        # 6. БЕЗ ГРУЗА — по роли
        # ================================================================
        if self.role == "THIEF":
            return self._thief_hunt(
                pos, base_pos, all_agents, golems, treasures, pace_mode,
                steal_ready, steal_pct,
            )
        if self.role == "DISRUPTOR":
            return self._disruptor_act(
                pos, base_pos, all_agents, golems, treasures, steal_pct,
            )

        return self._collector_gather(pos, base_pos, treasures, golems, all_agents, pace_mode)"""

new_tail = """        # ================================================================
        # 5. ДОСТАВКА СОКРОВИЩА НА БАЗУ
        # ================================================================
        if has_treasure:
            self._release_assignment()
            d_base = self.dist(pos, base_pos)

            for enemy in all_agents:
                if str(enemy.get("team", "")).lower() == TEAM.lower():
                    continue
                if enemy.get("isStunned"):
                    continue
                e_steal_ready = enemy.get("stealAbilityReady", False)
                e_steal_pct   = enemy.get("stealChargePercentage", 0)
                if not (e_steal_ready or e_steal_pct >= 90):
                    continue
                ed = self.dist(pos, enemy["pos"])

                if ed < ENEMY_STEAL_DROP:
                    team_memory["stats"]["drops_defensive"] += 1
                    return self._cmd("drop")

                if ed < ENEMY_STEAL_PANIC:
                    to_e_x = enemy["pos"]["x"] - pos["x"]
                    to_e_z = enemy["pos"]["z"] - pos["z"]
                    to_e_l = math.hypot(to_e_x, to_e_z) or 1.0
                    perp_x = -to_e_z / to_e_l
                    perp_z =  to_e_x / to_e_l
                    to_b_x = base_pos["x"] - pos["x"]
                    to_b_z = base_pos["z"] - pos["z"]
                    to_b_l = math.hypot(to_b_x, to_b_z) or 1.0
                    evade = {
                        "x": pos["x"] + (to_b_x / to_b_l) * 10 + perp_x * 6,
                        "y": pos["y"],
                        "z": pos["z"] + (to_b_z / to_b_l) * 10 + perp_z * 6,
                    }
                    return self._cmd("position", evade)

            if d_base < BASE_DROP_RADIUS:
                return self._cmd("drop")
            return self._cmd("position", base_pos)

        # ================================================================
        # 6. КРАЖА
        # ================================================================
        steal_ready = me.get("stealAbilityReady", False)
        steal_pct   = me.get("stealChargePercentage", 0)

        if steal_ready:
            best_enemy = None
            best_sc    = -1.0
            for enemy in all_agents:
                if str(enemy.get("team", "")).lower() == TEAM.lower():
                    continue
                if not enemy.get("hasTreasure") or enemy.get("isStunned"):
                    continue
                ed            = self.dist(pos, enemy["pos"])
                enemy_weight  = enemy.get("weight", 0)

                danger_bonus = 1.0
                if pace_mode == "catch_up" and score_diff < -100:
                    danger_bonus = 1.8
                elif is_winning:
                    danger_bonus = 0.3

                sc = danger_bonus * (1.0 + enemy_weight * 0.015) / (ed + 1.0)
                if sc > best_sc:
                    best_sc    = sc
                    best_enemy = enemy

            if best_enemy:
                ed = self.dist(pos, best_enemy["pos"])
                if ed < 1.0:
                    team_memory["stats"]["steal_attempts"] += 1
                    return self._cmd("steal")

                chase_dist = 0.0
                if pace_mode == "catch_up" and best_enemy.get("weight", 0) >= 50:
                    chase_dist = 5.0
                elif pace_mode == "normal" and score_diff < -80:
                    chase_dist = 3.0

                if ed < chase_dist:
                    return self._cmd("position", best_enemy["pos"])

        # ================================================================
        # 7. СБОР СОКРОВИЩ
        # ================================================================
        available = [
            t for t in treasures
            if not t.get("isPicked") and t.get("holderAgentId") is None
        ]

        if not available:
            ep = EXPLORE_POINTS[self.explore_index % len(EXPLORE_POINTS)]
            if self.dist(pos, ep) < 3.0:
                self.explore_index += 1
            return self._cmd("position", ep)

        enemy_agents = [
            a for a in all_agents
            if str(a.get("team", "")).lower() != TEAM.lower()
        ]

        def base_score(t):
            s = score_treasure_for_agent(
                pos, t, base_pos, golems, pace_mode, enemy_agents, self.id
            )
            dens = sum(1 for o in available if self.dist(t["pos"], o["pos"]) < 20.0)
            s *= 1.0 + 0.08 * dens
            return s

        best = max(available, key=base_score)

        if self.target_id is not None and self.target_id != best["id"]:
            current = next((t for t in available if t["id"] == self.target_id), None)
            if current and self.dist(pos, current["pos"]) < 15.0:
                if base_score(best) < base_score(current) * 1.5:
                    best = current

        if self.target_id and self.target_id != best["id"]:
            if team_memory["assignments"].get(self.target_id) == self.id:
                del team_memory["assignments"][self.target_id]

        self.target_id = best["id"]
        if self.target_id not in team_memory["assignments"]:
            team_memory["assignments"][self.target_id] = self.id

        d_best = self.dist(pos, best["pos"])

        if d_best < PICKUP_RADIUS:
            return self._cmd("pickup", best["pos"])

        return self._cmd("position", best["pos"])"""

if old_tail in text:
    text = text.replace(old_tail, new_tail)
else:
    raise SystemExit("decide tail block not found")

# --- print_summary v7 ---
text = text.replace(
    """def print_summary():
    s = team_memory["stats"]
    roles = s.get("role_ticks", {})
    top_roles = sorted(roles.items(), key=lambda x: -x[1])[:4]
    roles_str = " ".join(f"{k}:{v}" for k, v in top_roles) if top_roles else "-"
    print(
        f"  [SUMMARY] dlv={team_memory['delivery_count']} score={int(team_memory['prev_points'])} "
        f"stuns={s['stuns']} stucks={s['stucks']} steal_try={s['steal_attempts']} "
        f"steal_ok={s['steal_success']} relay={s.get('relay_drops', 0)} pace={team_memory['pace_mode']}"
    )
    print(f"  [ROLES] {roles_str}")""",
    '''def print_summary():
    s = team_memory["stats"]
    print(
        f"  [SUMMARY] dlv={team_memory['delivery_count']} score={int(team_memory['prev_points'])} "
        f"stuns={s['stuns']} stucks={s['stucks']} steal_try={s['steal_attempts']} "
        f"steal_ok={s['steal_success']} pace={team_memory['pace_mode']}"
    )''',
)

# --- старт игры ---
text = text.replace(
    'print(f"\\n{\'=\'*40}\\n  СТАРТ — {TEAM.upper()} (v8 ROLES)\\n")',
    'print(f"\\n{\'=\'*40}\\n  СТАРТ — {TEAM.upper()} (v7)\\n{\'=\'*40}\\n")',
)
text = text.replace('print(f"  {ROLE_LAYOUT}\\n{\'=\'*40}\\n")', "")
text = text.replace("team_memory[\"agent_roles\"].clear()\n                        ", "")
text = text.replace("team_memory[\"relay_spots\"]    = []\n                        ", "")
text = text.replace('"relay_drops": 0,\n                            "role_ticks": {},\n                            ', "")

# --- назначения: все пустые агенты, не только коллекторы ---
text = text.replace(
    """                for a in my_agents:
                    aid = str(a["agentId"])
                    if aid not in team_memory["agent_roles"]:
                        idx = len(team_memory["agent_roles"])
                        team_memory["agent_roles"][aid] = role_for_agent_index(idx)

                if game_state == STATE_INGAME and base_pos:""",
    "                if game_state == STATE_INGAME and base_pos:",
)
text = text.replace(
    """                    empty_collectors = [
                        a for a in my_agents
                        if not a.get("hasTreasure") and not a.get("isStunned")
                        and team_memory["agent_roles"].get(str(a["agentId"])) == "COLLECTOR"
                    ]
                    team_memory["assignments"] = plan_team_assignments(
                        empty_collectors, treasures, base_pos, golems, enemy_agents
                    )""",
    """                    empty_agents = [
                        a for a in my_agents
                        if not a.get("hasTreasure") and not a.get("isStunned")
                    ]
                    team_memory["assignments"] = plan_team_assignments(
                        empty_agents, treasures, base_pos, golems, enemy_agents
                    )""",
)
text = text.replace(
    """                    if aid not in agents_ai:
                        role = role_for_agent_index(agent_index)
                        agents_ai[aid] = Agent(
                            aid, explore_index=agent_index, role=role,
                        )
                        agent_index += 1
                        print(f"  [ROLE] агент {aid} → {role}")""",
    """                    if aid not in agents_ai:
                        agents_ai[aid] = Agent(aid, explore_index=agent_index)
                        agent_index += 1""",
)

# enemy_base в decide — оставить prune_relay вызов убрать
text = text.replace("        prune_relay_spots()\n\n        # ---- Адаптация", "        # ---- Адаптация")
text = text.replace(
    """        if team_memory["enemy_base_pos"] is None:
            for b in bases:
                if str(b.get("team", "")).lower() != TEAM.lower():
                    team_memory["enemy_base_pos"] = b["pos"]
                    break
        now = time.time()
        prune_relay_spots()

        # ---- Адаптация""",
    "        now = time.time()\n\n        # ---- Адаптация",
)

out = root / "testClientRed.py"
out.write_text(text, encoding="utf-8")
print("Wrote", out, "lines", len(text.splitlines()))
