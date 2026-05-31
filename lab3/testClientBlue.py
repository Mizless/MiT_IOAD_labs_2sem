import socket
import json
import math
import time
import random

HOST = "127.0.0.1"
PORT = 8080
TEAM = "blue"
STATE_INGAME = "InGame"

# ====== КОНСТАНТЫ ======
BASE_DROP_RADIUS    = 1.9    # м — радиус сдачи на базу
GOLEM_CHASE_SPEED   = 3.5    # м/с
GOLEM_VISION_DIST   = 10.0   # м — FOV дальность
GOLEM_LOSE_DIST     = 15.0   # м — дистанция потери цели
GOLEM_FOV_COS       = 0.766  # cos(40°) — запас к 45° конусу
PICKUP_RADIUS       = 1.4    # м — слать pickup начиная отсюда
DROP_WEIGHT_THRESH  = 3.6    # м/с — скорость ниже которой дропаем при Chase
ENEMY_STEAL_DROP    = 1.8    # м — вор вплотную → упреждающий дроп
ENEMY_STEAL_PANIC   = 6.0    # м — вор рядом → объезд
STEAL_CHASE_DIST    = 6.0    # м — гонимся за врагом только если он уже рядом
ESCORT_MIN_WEIGHT   = 80     # кг — порог для эскорта
STEAL_BONUS_TIME    = 5.0    # сек игнора веса после кражи

# ====== ОБЩАЯ ПАМЯТЬ КОМАНДЫ ======
team_memory = {
    "base_pos":       None,
    "enemy_base_pos": None,
    "assignments":    {},    # {treasure_id: agent_id}
    "golem_prev_pos": {},
    "golem_dir":      {},    # EMA-вектор движения голема (нормализованный)
    "escort_map":     {},    # {carrier_agent_id: escort_agent_id}
    "prev_points":    0,
    "enemy_points":   0,
    "delivery_count": 0,
    "stats": {
        "steal_attempts": 0,
        "steal_success": 0,
        "drops_defensive": 0,
        "stuns": 0,
        "stucks": 0,
        "last_summary": 0.0,
    },
    "agent_prev_stun": {},
}

EXPLORE_POINTS = [
    {"x":  80, "y": 0, "z":  80},
    {"x": -80, "y": 0, "z":  80},
    {"x":  80, "y": 0, "z": -80},
    {"x": -80, "y": 0, "z": -80},
    {"x":  80, "y": 0, "z":   0},
    {"x": -80, "y": 0, "z":   0},
    {"x":   0, "y": 0, "z":  80},
    {"x":   0, "y": 0, "z": -80},
    {"x":  40, "y": 0, "z":  40},
    {"x": -40, "y": 0, "z":  40},
    {"x":  40, "y": 0, "z": -40},
    {"x": -40, "y": 0, "z": -40},
    {"x":   0, "y": 0, "z":   0},
]


class Agent:
    def __init__(self, agent_id, explore_index=0):
        self.id             = str(agent_id)
        self.target_id      = None
        self.last_cmd_time  = 0.0
        self.cmd_interval   = 0.08
        self.last_sent_pos      = None
        self.last_sent_pos_time = 0.0
        self.pos_hist       = []
        self.unstuck_target = None
        self.unstuck_until  = 0.0
        self.explore_index  = explore_index
        self.steal_bonus_until = 0.0
        self.last_steal_pct = 0.0

    @staticmethod
    def dist(a, b):
        if not a or not b:
            return 1e9
        return math.hypot(a["x"] - b["x"], a["z"] - b["z"])

    @staticmethod
    def speed_from_weight(w):
        return max(5.0 * (1.0 - 0.005 * w), 0.5)

    def _cmd(self, action, target=None):
        now = time.time()
        if action == "position" and target is not None:
            if (self.last_sent_pos is not None
                    and self.dist(self.last_sent_pos, target) < 0.5
                    and now - self.last_sent_pos_time < 0.4):
                return None
        is_one_shot = action in ("pickup", "drop", "steal", "ready")
        if not is_one_shot and now - self.last_cmd_time < self.cmd_interval:
            return None
        self.last_cmd_time = now
        if action == "position" and target is not None:
            self.last_sent_pos      = target
            self.last_sent_pos_time = now
        cmd = {"Id": self.id, "action": action}
        if target is not None:
            cmd["target"] = target
        return cmd

    def _release_assignment(self):
        if self.target_id and team_memory["assignments"].get(self.target_id) == self.id:
            del team_memory["assignments"][self.target_id]
        self.target_id = None

    def _track_steal_bonus(self, me):
        cur_pct = me.get("stealChargePercentage", 0) or 0
        if (self.last_steal_pct >= 80 and cur_pct < 30
                and me.get("hasTreasure", False)):
            self.steal_bonus_until = time.time() + STEAL_BONUS_TIME
            team_memory["stats"]["steal_success"] += 1
            print(f"  [STEAL OK] Агент {self.id} получил 5с ускорения!")
        self.last_steal_pct = cur_pct

    def _has_steal_bonus(self):
        return time.time() < self.steal_bonus_until

    def _assess_golem_threat(self, pos, golems, eff_vision_dist):
        best_golem  = None
        best_threat = 0
        chasing_us  = False
        for g in golems:
            gd       = self.dist(pos, g["pos"])
            g_state  = g.get("state", "Patrol")
            g_target = g.get("targetAgentId")
            threat   = 0
            our_flag = False
            if g_state == "Chase":
                if g_target == self.id or str(g_target) == self.id:
                    if gd < GOLEM_LOSE_DIST:
                        threat   = 3
                        our_flag = True
                else:
                    if gd < 5.0:
                        threat = 2
            else:
                g_dir   = team_memory["golem_dir"].get(g["id"], {"x": 0, "z": 0})
                dir_len = math.hypot(g_dir["x"], g_dir["z"])
                if dir_len > 0.01 and gd < eff_vision_dist:
                    to_us_x = pos["x"] - g["pos"]["x"]
                    to_us_z = pos["z"] - g["pos"]["z"]
                    to_us_l = math.hypot(to_us_x, to_us_z) or 1.0
                    cos_a = ((g_dir["x"] / dir_len) * (to_us_x / to_us_l)
                           + (g_dir["z"] / dir_len) * (to_us_z / to_us_l))
                    if cos_a > GOLEM_FOV_COS:
                        threat   = 1
                        our_flag = True
            if threat > best_threat:
                best_threat = threat
                best_golem  = g
                chasing_us  = our_flag
        return best_golem, best_threat, chasing_us

    def decide(self, me, treasures, bases, game_state, golems, all_agents):
        pos = me.get("pos")
        if not pos or game_state != STATE_INGAME:
            return None

        self._track_steal_bonus(me)

        if me.get("isStunned"):
            self._release_assignment()
            team_memory["escort_map"] = {
                k: v for k, v in team_memory["escort_map"].items() if v != self.id
            }
            self.pos_hist.clear()
            self.last_sent_pos = None
            self.steal_bonus_until = 0.0
            return None

        has_treasure = me.get("hasTreasure", False)
        my_weight    = me.get("weight", 0)

        if team_memory["base_pos"] is None:
            for b in bases:
                if str(b.get("team", "")).lower() == TEAM.lower():
                    team_memory["base_pos"] = b["pos"]
                    break
        base_pos = team_memory["base_pos"] or {"x": 0, "y": 0, "z": 0}
        now = time.time()

        score_diff = team_memory["prev_points"] - team_memory["enemy_points"]
        is_losing = score_diff < -200

        if now < self.unstuck_until and self.unstuck_target:
            return self._cmd("position", self.unstuck_target)

        near_base   = self.dist(pos, base_pos) < 3.0
        near_pickup = not has_treasure and any(
            self.dist(pos, t["pos"]) < 2.0
            for t in treasures if not t.get("isPicked")
        )

        self.pos_hist.append({"x": pos["x"], "z": pos["z"]})
        if len(self.pos_hist) > 25:
            self.pos_hist.pop(0)
            xs = [p["x"] for p in self.pos_hist]
            zs = [p["z"] for p in self.pos_hist]
            dx = max(xs) - min(xs)
            dz = max(zs) - min(zs)
            if dx < 0.5 and dz < 0.5:
                self.pos_hist.clear()
                if near_base:
                    away_x = pos["x"] - base_pos["x"]
                    away_z = pos["z"] - base_pos["z"]
                    ln = math.hypot(away_x, away_z) or 1.0
                    self.unstuck_target = {
                        "x": pos["x"] + (away_x / ln) * 6,
                        "y": pos["y"],
                        "z": pos["z"] + (away_z / ln) * 6,
                    }
                else:
                    ang = random.uniform(0, 2 * math.pi)
                    self.unstuck_target = {
                        "x": pos["x"] + math.cos(ang) * 8,
                        "y": pos["y"],
                        "z": pos["z"] + math.sin(ang) * 8,
                    }
                self.unstuck_until = now + 1.5
                team_memory["stats"]["stucks"] += 1
                print(f"  [STUCK] Агент {self.id} застрял. Рывок.")
                return self._cmd("position", self.unstuck_target)

        eff_vision_dist = GOLEM_VISION_DIST if not is_losing else 7.0
        danger_golem, threat_level, chasing_us = self._assess_golem_threat(pos, golems, eff_vision_dist)

        if danger_golem and threat_level >= 2:
            gpos = danger_golem["pos"]
            gd   = self.dist(pos, gpos)
            if threat_level == 3 and chasing_us:
                if has_treasure:
                    my_spd = self.speed_from_weight(my_weight)
                    if not self._has_steal_bonus() and my_spd < DROP_WEIGHT_THRESH:
                        self._release_assignment()
                        team_memory["stats"]["drops_defensive"] += 1
                        return self._cmd("drop")
                    to_base_x = base_pos["x"] - pos["x"]
                    to_base_z = base_pos["z"] - pos["z"]
                    to_gol_x  = gpos["x"] - pos["x"]
                    to_gol_z  = gpos["z"] - pos["z"]
                    dot = to_base_x * to_gol_x + to_base_z * to_gol_z
                    if dot > 0 and gd < 9.0:
                        tb_len = math.hypot(to_base_x, to_base_z) or 1.0
                        perp_x = -to_base_z / tb_len
                        perp_z =  to_base_x / tb_len
                        escape = {
                            "x": pos["x"] + perp_x * 14,
                            "y": pos["y"],
                            "z": pos["z"] + perp_z * 14,
                        }
                    else:
                        escape = base_pos
                    return self._cmd("position", escape)
                else:
                    dx = pos["x"] - gpos["x"]
                    dz = pos["z"] - gpos["z"]
                    ln = math.hypot(dx, dz) or 1.0
                    escape = {
                        "x": pos["x"] + (dx / ln) * 18,
                        "y": pos["y"],
                        "z": pos["z"] + (dz / ln) * 18,
                    }
                    return self._cmd("position", escape)

        if danger_golem and threat_level == 1 and chasing_us:
            gpos = danger_golem["pos"]
            if has_treasure:
                to_base_x = base_pos["x"] - pos["x"]
                to_base_z = base_pos["z"] - pos["z"]
                tb_len    = math.hypot(to_base_x, to_base_z) or 1.0
                perp_x    = -to_base_z / tb_len
                perp_z    =  to_base_x / tb_len
                dodge = {
                    "x": pos["x"] + (to_base_x / tb_len) * 10 + perp_x * 6,
                    "y": pos["y"],
                    "z": pos["z"] + (to_base_z / tb_len) * 10 + perp_z * 6,
                }
                return self._cmd("position", dodge)
            else:
                nearest_t = min(
                    (t for t in treasures if not t.get("isPicked")),
                    key=lambda t: self.dist(pos, t["pos"]),
                    default=None,
                )
                if nearest_t is None or self.dist(pos, nearest_t["pos"]) > 5.0:
                    dx = pos["x"] - gpos["x"]
                    dz = pos["z"] - gpos["z"]
                    ln = math.hypot(dx, dz) or 1.0
                    escape = {
                        "x": pos["x"] + (dx / ln) * 12,
                        "y": pos["y"],
                        "z": pos["z"] + (dz / ln) * 12,
                    }
                    return self._cmd("position", escape)

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
                e_steal_ready = enemy.get("stealAbilityReady", False)
                e_steal_pct   = enemy.get("stealChargePercentage", 0)
                danger_bonus = 1.5 if (e_steal_ready or e_steal_pct >= 80) else 1.0
                if score_diff < -200:
                    danger_bonus *= 2.0
                elif score_diff > 200:
                    danger_bonus *= 0.5
                sc = danger_bonus * (1.0 + enemy_weight * 0.015) / (ed + 1.0)
                if sc > best_sc:
                    best_sc    = sc
                    best_enemy = enemy
            if best_enemy:
                ed = self.dist(pos, best_enemy["pos"])
                if ed < 1.0:
                    team_memory["stats"]["steal_attempts"] += 1
                    return self._cmd("steal")
                chase_dist = STEAL_CHASE_DIST
                if score_diff < -200:
                    chase_dist = 10.0
                elif score_diff > 200:
                    chase_dist = 3.0
                if ed < chase_dist:
                    return self._cmd("position", best_enemy["pos"])

        if steal_pct < 80:
            my_agents_list = [
                a for a in all_agents
                if str(a.get("team", "")).lower() == TEAM.lower()
            ]
            escort_needed = None
            escort_pos    = None
            min_ally_dist = 1e9
            for ally in my_agents_list:
                if str(ally["agentId"]) == self.id:
                    continue
                if not ally.get("hasTreasure"):
                    continue
                if ally.get("weight", 0) < ESCORT_MIN_WEIGHT:
                    continue
                ally_pos = ally.get("pos")
                if not ally_pos:
                    continue
                our_d = self.dist(pos, ally_pos)
                if our_d > 25.0:
                    continue
                for g in golems:
                    if (g.get("state") == "Chase"
                            and str(g.get("targetAgentId")) == str(ally["agentId"])):
                        if our_d < min_ally_dist:
                            min_ally_dist = our_d
                            escort_needed = ally
                            gpos   = g["pos"]
                            ag_x   = ally_pos["x"] - gpos["x"]
                            ag_z   = ally_pos["z"] - gpos["z"]
                            ag_len = math.hypot(ag_x, ag_z) or 1.0
                            escort_pos = {
                                "x": ally_pos["x"] - (ag_x / ag_len) * 3.0,
                                "y": ally_pos.get("y", 0),
                                "z": ally_pos["z"] - (ag_z / ag_len) * 3.0,
                            }
                            break
            if escort_needed:
                aid_esc = str(escort_needed["agentId"])
                existing = team_memory["escort_map"].get(aid_esc)
                if existing is None or existing == self.id:
                    team_memory["escort_map"][aid_esc] = self.id
                    if escort_pos and self.dist(pos, escort_pos) > 1.5:
                        return self._cmd("position", escort_pos)
            team_memory["escort_map"] = {
                k: v for k, v in team_memory["escort_map"].items()
                if v != self.id or escort_needed
            }

        available = [
            t for t in treasures
            if not t.get("isPicked") and t.get("holderAgentId") is None
        ]

        if not available:
            ep = EXPLORE_POINTS[self.explore_index % len(EXPLORE_POINTS)]
            if self.dist(pos, ep) < 3.0:
                self.explore_index += 1
            return self._cmd("position", ep)

        available_ids = {t["id"] for t in available}
        for tid in list(team_memory["assignments"].keys()):
            if tid not in available_ids:
                del team_memory["assignments"][tid]

        density_map = {}
        for t in available:
            count = sum(1 for other in available if self.dist(t["pos"], other["pos"]) < 20.0)
            density_map[t["id"]] = count

        def base_score(t):
            d_to   = self.dist(pos, t["pos"])
            d_home = self.dist(t["pos"], base_pos)
            w = t.get("weight", 0)
            v = t.get("value", 0) or 10
            spd          = self.speed_from_weight(w)
            speed_factor = spd / 5.0
            total_cost   = max((d_to + d_home) / speed_factor, 0.1)
            s = v / total_cost
            dens = density_map.get(t["id"], 1)
            s *= (1.0 + 0.1 * dens)
            assigned_to = team_memory["assignments"].get(t["id"])
            if assigned_to and assigned_to != self.id:
                s *= 0.002
            if w >= 60:
                for g in golems:
                    if self.dist(t["pos"], g["pos"]) < 12.0:
                        s *= 0.25
                        break
            if d_to < 2.0:
                s *= 80.0
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
        team_memory["assignments"][self.target_id] = self.id

        d_best = self.dist(pos, best["pos"])
        if d_best < PICKUP_RADIUS:
            return self._cmd("pickup", best["pos"])
        return self._cmd("position", best["pos"])


def send_msg(sock, obj):
    sock.sendall((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))


def track_stuns(all_agents):
    for a in all_agents:
        if str(a.get("team", "")).lower() != TEAM.lower():
            continue
        aid = str(a["agentId"])
        cur = a.get("isStunned", False)
        prev = team_memory["agent_prev_stun"].get(aid, False)
        if cur and not prev:
            team_memory["stats"]["stuns"] += 1
        team_memory["agent_prev_stun"][aid] = cur


def print_summary():
    s = team_memory["stats"]
    print(
        f"  [SUMMARY] dlv={team_memory['delivery_count']} score={int(team_memory['prev_points'])} "
        f"stuns={s['stuns']} stucks={s['stucks']} steal_try={s['steal_attempts']} steal_ok={s['steal_success']}"
    )


def start_client():
    sock        = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(None)
    agents_ai   = {}
    agent_index = 0

    try:
        sock.connect((HOST, PORT))
        print(f"[SYSTEM] Подключились: {TEAM.upper()}")
        send_msg(sock, {"team": TEAM})

        buffer    = ""
        player_id = None

        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            buffer += chunk.decode("utf-8", errors="ignore")

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type")

                if msg_type == "joinAccepted":
                    player_id = msg.get("playerId")
                    print(f"[SYSTEM] Вошли как {player_id}")
                    send_msg(sock, {"actions": [{"action": "ready"}]})
                    continue

                if msg_type == "joinRejected":
                    print(f"[SYSTEM] Отказ: {msg.get('reason')}")
                    return

                if msg_type == "gameEvent":
                    ev = msg.get("eventType")
                    if ev == "start":
                        print(f"\n{'='*40}\n  СТАРТ — {TEAM.upper()} (АДАПТИВНАЯ ВЕРСИЯ)\n{'='*40}\n")
                        team_memory["base_pos"]       = None
                        team_memory["enemy_base_pos"] = None
                        team_memory["assignments"].clear()
                        team_memory["golem_prev_pos"].clear()
                        team_memory["golem_dir"].clear()
                        team_memory["escort_map"].clear()
                        team_memory["prev_points"]    = 0
                        team_memory["enemy_points"]   = 0
                        team_memory["delivery_count"] = 0
                        team_memory["stats"] = {
                            "steal_attempts": 0,
                            "steal_success": 0,
                            "drops_defensive": 0,
                            "stuns": 0,
                            "stucks": 0,
                            "last_summary": time.time(),
                        }
                        team_memory["agent_prev_stun"].clear()
                        agents_ai.clear()
                        agent_index = 0
                    elif ev == "result":
                        print_summary()
                        print(f"\n[ИТОГ] {msg}")
                    elif ev == "end":
                        print("[SYSTEM] Игра завершена")
                    continue

                if not player_id or "agents" not in msg:
                    continue

                all_agents = msg["agents"]
                treasures  = msg.get("treasures", [])
                bases      = msg.get("bases", [])
                golems     = msg.get("golems", [])
                game_state = msg.get("gameState", "Lobby")

                for b in bases:
                    if b["team"].lower() == TEAM.lower():
                        pts = b["points"]
                        if pts > team_memory["prev_points"]:
                            team_memory["delivery_count"] += 1
                            print(
                                f"  [+] ДОСТАВЛЕНО! Счёт: {pts} "
                                f"(доставок: {team_memory['delivery_count']})"
                            )
                        team_memory["prev_points"] = pts
                    else:
                        team_memory["enemy_points"] = b["points"]

                track_stuns(all_agents)

                now = time.time()
                if (game_state == STATE_INGAME
                        and now - team_memory["stats"]["last_summary"] >= 30.0):
                    team_memory["stats"]["last_summary"] = now
                    print_summary()

                for g in golems:
                    gid  = g["id"]
                    prev = team_memory["golem_prev_pos"].get(gid)
                    if prev:
                        dx = g["pos"]["x"] - prev["x"]
                        dz = g["pos"]["z"] - prev["z"]
                        ln = math.hypot(dx, dz)
                        if ln > 0.01:
                            nx  = dx / ln
                            nz  = dz / ln
                            old = team_memory["golem_dir"].get(gid, {"x": nx, "z": nz})
                            a   = 0.4
                            team_memory["golem_dir"][gid] = {
                                "x": old["x"] * (1 - a) + nx * a,
                                "z": old["z"] * (1 - a) + nz * a,
                            }
                    team_memory["golem_prev_pos"][gid] = g["pos"]

                my_agents = [
                    a for a in all_agents
                    if str(a.get("team", "")).lower() == TEAM.lower()
                ]
                my_agents.sort(key=lambda a: 0 if a.get("hasTreasure") else 1)

                actions = []
                for a in my_agents:
                    aid = str(a["agentId"])
                    if aid not in agents_ai:
                        agents_ai[aid] = Agent(aid, explore_index=agent_index)
                        agent_index += 1
                    cmd = agents_ai[aid].decide(
                        a, treasures, bases, game_state, golems, all_agents
                    )
                    if cmd:
                        actions.append(cmd)

                if actions:
                    send_msg(sock, {"actions": actions})

    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()
    finally:
        sock.close()
        print("[SYSTEM] Отключились")


if __name__ == "__main__":
    start_client()
