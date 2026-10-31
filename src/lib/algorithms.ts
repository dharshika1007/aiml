import { Vector2D, Obstacle, SimulationConfig } from '../types';
import { normalize, dist } from './mosqeva';

const CANVAS_SIZE = 500;
const SPEED_LIMIT = 3;

export type AlgorithmKey =
  | 'astar'
  | 'dstar'
  | 'rrt'
  | 'apf'
  | 'ga'
  | 'pso'
  | 'dqn'
  | 'ppo'
  | 'mapf'
  | 'bt'
  | 'mosqeva';

export interface AlgorithmMeta {
  label: string;
  equation: string;        // LaTeX-like string rendered on canvas
  equationLine2?: string;  // optional second line
  color: string;           // robot body colour
}

export const ALGORITHM_META: Record<AlgorithmKey, AlgorithmMeta> = {
  astar: {
    label: 'A*',
    equation: 'f(n) = g(n) + h(n)',
    equationLine2: 'h(n) = Euclidean heuristic to goal',
    color: '#7c3aed',
  },
  dstar: {
    label: 'D* Lite',
    equation: 'key(s) = [min(g(s), rhs(s)) + h(s,sgoal);  min(g(s), rhs(s))]',
    equationLine2: 'rhs(s) = min_{s\'∈Succ(s)} (c(s,s\') + g(s\'))',
    color: '#0891b2',
  },
  rrt: {
    label: 'RRT / RRT*',
    equation: 'x_new = Steer(x_nearest, x_rand, η)',
    equationLine2: 'RRT*: rewire if cost(x_new via x_near) < cost(x_near)',
    color: '#059669',
  },
  apf: {
    label: 'Artificial Potential Field (APF)',
    equation: 'F_total = F_att + F_rep',
    equationLine2: 'F_att = −∇U_att,  F_rep = −∇U_rep',
    color: '#d97706',
  },
  ga: {
    label: 'Genetic Algorithm (GA)',
    equation: 'fitness(p) = 1 / (d(p,goal) + λ·collisions)',
    equationLine2: 'p\' = crossover(p₁,p₂) ⊕ mutate(ε)',
    color: '#be185d',
  },
  pso: {
    label: 'Particle Swarm Optimization (PSO)',
    equation: 'v_{i+1} = ωv_i + c₁r₁(p_best−x) + c₂r₂(g_best−x)',
    equationLine2: 'x_{i+1} = x_i + v_{i+1}',
    color: '#ea580c',
  },
  dqn: {
    label: 'Deep Q-Network (DQN)',
    equation: 'Q(s,a) ← r + γ·max_{a\'} Q(s\',a\')',
    equationLine2: 'L = E[(r + γ·max Q_target − Q(s,a))²]',
    color: '#4f46e5',
  },
  ppo: {
    label: 'Proximal Policy Optimization (PPO)',
    equation: 'L_CLIP = E[min(r_t·Â, clip(r_t, 1−ε, 1+ε)·Â)]',
    equationLine2: 'r_t(θ) = π_θ(a|s) / π_θ_old(a|s)',
    color: '#0f766e',
  },
  mapf: {
    label: 'Multi-Agent Path Finding (MAPF)',
    equation: 'min Σ_i cost(π_i)  s.t.  π_i ∩ π_j = ∅  ∀i≠j',
    equationLine2: 'CBS: conflict-based search on constraint tree',
    color: '#9333ea',
  },
  bt: {
    label: 'Behavior Tree Agent (Agentic AI)',
    equation: 'tick() → {Selector → [Sequence → [Condition, Action]]}',
    equationLine2: 'Fallback: Avoid → Navigate → Explore',
    color: '#b45309',
  },
  mosqeva: {
    label: 'MosqEVA',
    equation: 'v(t+1) = αA + βR + γD + δP',
    equationLine2: 'A=attraction  R=repulsion  D=disturbance  P=pheromone',
    color: '#3b82f6',
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

function repulsionVec(pos: Vector2D, obstacles: Obstacle[], safeBase = 35): Vector2D {
  let rep = { x: 0, y: 0 };
  obstacles.forEach(obs => {
    const d = dist(pos, obs);
    const safe = obs.radius + (obs.isDynamic ? safeBase + 10 : safeBase);
    if (d < safe) {
      const diff = normalize({ x: pos.x - obs.x, y: pos.y - obs.y });
      const str = Math.pow((safe - d) / safe, 2) * 10;
      rep.x += diff.x * str;
      rep.y += diff.y * str;
    }
  });
  return rep;
}

function clampVel(v: Vector2D, limit = SPEED_LIMIT): Vector2D {
  return {
    x: Math.max(-limit, Math.min(limit, v.x)),
    y: Math.max(-limit, Math.min(limit, v.y)),
  };
}

function isColliding(pos: Vector2D, obstacles: Obstacle[], margin = 2): boolean {
  return obstacles.some(o => dist(pos, o) < o.radius + margin);
}

// ─────────────────────────────────────────────────────────────────────────────
// A*  — grid-based, replan when dynamic obstacles shift significantly
// ─────────────────────────────────────────────────────────────────────────────
export class AStarAlgorithm {
  private CELL = 20; // px per cell
  private cols: number;
  private rows: number;
  private waypoints: Vector2D[] = [];
  private replanCounter = 0;

  constructor() {
    this.cols = Math.ceil(CANVAS_SIZE / this.CELL);
    this.rows = Math.ceil(CANVAS_SIZE / this.CELL);
  }

  private heuristic(a: { r: number; c: number }, b: { r: number; c: number }) {
    return Math.sqrt((a.r - b.r) ** 2 + (a.c - b.c) ** 2);
  }

  private buildGrid(obstacles: Obstacle[]): boolean[][] {
    const grid: boolean[][] = Array.from({ length: this.rows }, () =>
      Array(this.cols).fill(false)
    );
    obstacles.forEach(obs => {
      const margin = obs.radius + 12;
      const minC = Math.max(0, Math.floor((obs.x - margin) / this.CELL));
      const maxC = Math.min(this.cols - 1, Math.ceil((obs.x + margin) / this.CELL));
      const minR = Math.max(0, Math.floor((obs.y - margin) / this.CELL));
      const maxR = Math.min(this.rows - 1, Math.ceil((obs.y + margin) / this.CELL));
      for (let r = minR; r <= maxR; r++)
        for (let c = minC; c <= maxC; c++)
          grid[r][c] = true;
    });
    return grid;
  }

  private planPath(
    start: Vector2D,
    goal: Vector2D,
    obstacles: Obstacle[]
  ): Vector2D[] {
    const grid = this.buildGrid(obstacles);
    const toCell = (p: Vector2D) => ({
      r: Math.min(this.rows - 1, Math.max(0, Math.round(p.y / this.CELL))),
      c: Math.min(this.cols - 1, Math.max(0, Math.round(p.x / this.CELL))),
    });
    const toWorld = (r: number, c: number): Vector2D => ({
      x: c * this.CELL + this.CELL / 2,
      y: r * this.CELL + this.CELL / 2,
    });

    const sc = toCell(start);
    const gc = toCell(goal);

    type Node = { r: number; c: number; g: number; f: number; parent: Node | null };
    const open: Node[] = [{ r: sc.r, c: sc.c, g: 0, f: this.heuristic(sc, gc), parent: null }];
    const closed = new Set<string>();
    const key = (r: number, c: number) => `${r},${c}`;

    const dirs = [
      [0,1],[1,0],[0,-1],[-1,0],
      [1,1],[1,-1],[-1,1],[-1,-1],
    ];

    let found: Node | null = null;
    let iter = 0;
    while (open.length && iter++ < 5000) {
      open.sort((a, b) => a.f - b.f);
      const cur = open.shift()!;
      if (closed.has(key(cur.r, cur.c))) continue;
      closed.add(key(cur.r, cur.c));
      if (cur.r === gc.r && cur.c === gc.c) { found = cur; break; }

      for (const [dr, dc] of dirs) {
        const nr = cur.r + dr, nc = cur.c + dc;
        if (nr < 0 || nr >= this.rows || nc < 0 || nc >= this.cols) continue;
        if (grid[nr][nc] || closed.has(key(nr, nc))) continue;
        const g = cur.g + Math.sqrt(dr * dr + dc * dc);
        open.push({ r: nr, c: nc, g, f: g + this.heuristic({ r: nr, c: nc }, gc), parent: cur });
      }
    }

    if (!found) return [goal];
    const path: Vector2D[] = [];
    let cur: Node | null = found;
    while (cur) { path.unshift(toWorld(cur.r, cur.c)); cur = cur.parent; }
    return path.length > 1 ? path.slice(1) : [goal];
  }

  calculateNextVelocity(
    pos: Vector2D,
    goal: Vector2D,
    obstacles: Obstacle[]
  ): Vector2D {
    this.replanCounter++;
    // Replan every 40 frames or when waypoints exhausted
    if (this.waypoints.length === 0 || this.replanCounter % 40 === 0) {
      this.waypoints = this.planPath(pos, goal, obstacles);
    }

    // Advance waypoints
    while (this.waypoints.length > 1 && dist(pos, this.waypoints[0]) < 18) {
      this.waypoints.shift();
    }

    const target = this.waypoints[0] ?? goal;
    const toTarget = normalize({ x: target.x - pos.x, y: target.y - pos.y });
    const rep = repulsionVec(pos, obstacles);
    return clampVel({ x: toTarget.x * 2.5 + rep.x * 0.8, y: toTarget.y * 2.5 + rep.y * 0.8 });
  }

  reset() { this.waypoints = []; this.replanCounter = 0; }
}

// ─────────────────────────────────────────────────────────────────────────────
// D* Lite — incremental replanning (simplified continuous version)
// ─────────────────────────────────────────────────────────────────────────────
export class DStarLiteAlgorithm {
  private waypoints: Vector2D[] = [];
  private replanAt = 0;
  private astar = new AStarAlgorithm();

  calculateNextVelocity(pos: Vector2D, goal: Vector2D, obstacles: Obstacle[]): Vector2D {
    // D* Lite's key insight: replan incrementally when environment changes.
    // We approximate with frequent A* replanning triggered by proximity to obstacles.
    const nearDynamic = obstacles.some(o => o.isDynamic && dist(pos, o) < o.radius + 60);
    if (nearDynamic || this.replanAt <= 0) {
      this.waypoints = (this.astar as any).planPath(pos, goal, obstacles);
      this.replanAt = nearDynamic ? 15 : 50;
    }
    this.replanAt--;

    while (this.waypoints.length > 1 && dist(pos, this.waypoints[0]) < 18) {
      this.waypoints.shift();
    }

    const target = this.waypoints[0] ?? goal;
    const toTarget = normalize({ x: target.x - pos.x, y: target.y - pos.y });
    const rep = repulsionVec(pos, obstacles, 40);
    // D* is more reactive — add stronger dynamic-obstacle avoidance
    let dynRep = { x: 0, y: 0 };
    obstacles.filter(o => o.isDynamic).forEach(o => {
      const d = dist(pos, o);
      if (d < o.radius + 70 && o.velocity) {
        const away = normalize({ x: pos.x - o.x, y: pos.y - o.y });
        const str = ((o.radius + 70 - d) / 70) * 6;
        dynRep.x += away.x * str;
        dynRep.y += away.y * str;
      }
    });
    return clampVel({
      x: toTarget.x * 2.5 + rep.x * 0.7 + dynRep.x,
      y: toTarget.y * 2.5 + rep.y * 0.7 + dynRep.y,
    });
  }

  reset() { this.waypoints = []; this.replanAt = 0; (this.astar as any).reset(); }
}

// ─────────────────────────────────────────────────────────────────────────────
// RRT / RRT* — sampling-based
// ─────────────────────────────────────────────────────────────────────────────
export class RRTAlgorithm {
  private tree: { pos: Vector2D; parent: number | null }[] = [];
  private waypoints: Vector2D[] = [];
  private built = false;
  private eta = 25; // step size px

  private sample(goal: Vector2D): Vector2D {
    if (Math.random() < 0.1) return { ...goal };
    return { x: Math.random() * CANVAS_SIZE, y: Math.random() * CANVAS_SIZE };
  }

  private nearest(p: Vector2D): number {
    let best = 0, bd = Infinity;
    this.tree.forEach((n, i) => { const d = dist(n.pos, p); if (d < bd) { bd = d; best = i; } });
    return best;
  }

  private steer(from: Vector2D, to: Vector2D): Vector2D {
    const d = dist(from, to);
    if (d <= this.eta) return { ...to };
    const dir = normalize({ x: to.x - from.x, y: to.y - from.y });
    return { x: from.x + dir.x * this.eta, y: from.y + dir.y * this.eta };
  }

  private segmentClear(a: Vector2D, b: Vector2D, obstacles: Obstacle[]): boolean {
    const steps = Math.ceil(dist(a, b) / 8);
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const p = { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
      if (isColliding(p, obstacles, 8)) return false;
    }
    return true;
  }

  buildTree(start: Vector2D, goal: Vector2D, obstacles: Obstacle[]) {
    this.tree = [{ pos: { ...start }, parent: null }];
    for (let i = 0; i < 1500; i++) {
      const rand = this.sample(goal);
      const ni = this.nearest(rand);
      const newPos = this.steer(this.tree[ni].pos, rand);
      if (isColliding(newPos, obstacles)) continue;
      if (!this.segmentClear(this.tree[ni].pos, newPos, obstacles)) continue;

      // RRT* rewiring
      const newIdx = this.tree.length;
      let bestParent = ni;
      let bestCost = (this.tree[ni] as any).cost ?? 0 + dist(this.tree[ni].pos, newPos);
      const radius = 50;
      this.tree.forEach((n, idx) => {
        if (dist(n.pos, newPos) < radius && this.segmentClear(n.pos, newPos, obstacles)) {
          const cost = ((n as any).cost ?? 0) + dist(n.pos, newPos);
          if (cost < bestCost) { bestCost = cost; bestParent = idx; }
        }
      });
      this.tree.push({ pos: newPos, parent: bestParent, cost: bestCost } as any);

      if (dist(newPos, goal) < this.eta * 1.5) {
        this.tree.push({ pos: { ...goal }, parent: newIdx } as any);
        break;
      }
    }

    // Trace path
    let idx = this.tree.length - 1;
    const path: Vector2D[] = [];
    while (idx !== null && idx !== undefined) {
      path.unshift(this.tree[idx].pos);
      idx = this.tree[idx].parent as any;
      if (idx === null) break;
    }
    this.waypoints = path.slice(1);
    this.built = true;
  }

  calculateNextVelocity(pos: Vector2D, goal: Vector2D, obstacles: Obstacle[]): Vector2D {
    if (!this.built || this.waypoints.length === 0) {
      this.buildTree(pos, goal, obstacles);
    }

    // Replan if upcoming waypoint is now blocked
    if (this.waypoints.length > 0 && isColliding(this.waypoints[0], obstacles, 15)) {
      this.built = false;
      this.waypoints = [];
      return clampVel(repulsionVec(pos, obstacles, 45));
    }

    while (this.waypoints.length > 1 && dist(pos, this.waypoints[0]) < 20) {
      this.waypoints.shift();
    }

    const target = this.waypoints[0] ?? goal;
    const toTarget = normalize({ x: target.x - pos.x, y: target.y - pos.y });
    const rep = repulsionVec(pos, obstacles, 30);
    return clampVel({ x: toTarget.x * 2.8 + rep.x * 0.5, y: toTarget.y * 2.8 + rep.y * 0.5 });
  }

  reset() { this.tree = []; this.waypoints = []; this.built = false; }
}

// ─────────────────────────────────────────────────────────────────────────────
// APF — Artificial Potential Field
// ─────────────────────────────────────────────────────────────────────────────
export class APFAlgorithm {
  private escapeTimer = 0;
  private escapeDir = { x: 0, y: 0 };

  calculateNextVelocity(pos: Vector2D, goal: Vector2D, obstacles: Obstacle[]): Vector2D {
    // Attractive force (linear)
    const dGoal = dist(pos, goal);
    const att = dGoal < 1 ? { x: 0, y: 0 } : normalize({ x: goal.x - pos.x, y: goal.y - pos.y });

    // Repulsive force (inverse-square)
    let rep = { x: 0, y: 0 };
    obstacles.forEach(obs => {
      const d = Math.max(0.1, dist(pos, obs) - obs.radius);
      const d0 = obs.isDynamic ? 60 : 45;
      if (d < d0) {
        const str = (1 / d - 1 / d0) / (d * d) * 800;
        const away = normalize({ x: pos.x - obs.x, y: pos.y - obs.y });
        rep.x += away.x * str;
        rep.y += away.y * str;
      }
    });

    // Local-minima escape: if stuck, add tangential jitter
    const vel = { x: att.x * 2.5 + rep.x, y: att.y * 2.5 + rep.y };
    const speed = Math.sqrt(vel.x ** 2 + vel.y ** 2);
    if (speed < 0.3 && dGoal > 30) {
      if (this.escapeTimer <= 0) {
        this.escapeTimer = 25;
        const tang = normalize({ x: -(goal.y - pos.y), y: goal.x - pos.x });
        this.escapeDir = { x: tang.x * 2, y: tang.y * 2 };
      }
    }
    if (this.escapeTimer > 0) {
      this.escapeTimer--;
      return clampVel({ x: vel.x + this.escapeDir.x, y: vel.y + this.escapeDir.y });
    }
    return clampVel(vel);
  }

  reset() { this.escapeTimer = 0; }
}

// ─────────────────────────────────────────────────────────────────────────────
// GA — Genetic Algorithm (waypoint chromosome)
// ─────────────────────────────────────────────────────────────────────────────
export class GAAlgorithm {
  private waypoints: Vector2D[] = [];
  private generation = 0;
  private replanEvery = 80;

  private fitness(chromosome: Vector2D[], start: Vector2D, goal: Vector2D, obstacles: Obstacle[]): number {
    const full = [start, ...chromosome, goal];
    let pathLen = 0;
    let collisions = 0;
    for (let i = 1; i < full.length; i++) {
      pathLen += dist(full[i - 1], full[i]);
      if (isColliding(full[i], obstacles, 10)) collisions++;
    }
    return 1 / (pathLen / CANVAS_SIZE + collisions * 5 + 0.01);
  }

  private evolve(pop: Vector2D[][], start: Vector2D, goal: Vector2D, obstacles: Obstacle[]): Vector2D[][] {
    const scored = pop.map(c => ({ c, f: this.fitness(c, start, goal, obstacles) }));
    scored.sort((a, b) => b.f - a.f);
    const elites = scored.slice(0, 4).map(s => s.c);
    const next: Vector2D[][] = [...elites];
    while (next.length < 12) {
      const p1 = elites[Math.floor(Math.random() * elites.length)];
      const p2 = elites[Math.floor(Math.random() * elites.length)];
      const cut = Math.floor(Math.random() * p1.length);
      const child = [...p1.slice(0, cut), ...p2.slice(cut)].map(wp => ({
        x: Math.random() < 0.2 ? Math.random() * CANVAS_SIZE : wp.x + (Math.random() - 0.5) * 30,
        y: Math.random() < 0.2 ? Math.random() * CANVAS_SIZE : wp.y + (Math.random() - 0.5) * 30,
      }));
      next.push(child);
    }
    return next;
  }

  plan(start: Vector2D, goal: Vector2D, obstacles: Obstacle[]) {
    const N = 5; // waypoints per chromosome
    let pop: Vector2D[][] = Array.from({ length: 12 }, () =>
      Array.from({ length: N }, () => ({
        x: start.x + (goal.x - start.x) * (Math.random()) + (Math.random() - 0.5) * 100,
        y: start.y + (goal.y - start.y) * (Math.random()) + (Math.random() - 0.5) * 100,
      }))
    );
    for (let g = 0; g < 30; g++) pop = this.evolve(pop, start, goal, obstacles);
    const best = pop.reduce((a, b) =>
      this.fitness(a, start, goal, obstacles) > this.fitness(b, start, goal, obstacles) ? a : b
    );
    this.waypoints = [...best, goal];
  }

  calculateNextVelocity(pos: Vector2D, goal: Vector2D, obstacles: Obstacle[]): Vector2D {
    this.generation++;
    if (this.waypoints.length === 0 || this.generation % this.replanEvery === 0) {
      this.plan(pos, goal, obstacles);
    }
    while (this.waypoints.length > 1 && dist(pos, this.waypoints[0]) < 22) this.waypoints.shift();
    const target = this.waypoints[0] ?? goal;
    const toTarget = normalize({ x: target.x - pos.x, y: target.y - pos.y });
    const rep = repulsionVec(pos, obstacles);
    return clampVel({ x: toTarget.x * 2.5 + rep.x, y: toTarget.y * 2.5 + rep.y });
  }

  reset() { this.waypoints = []; this.generation = 0; }
}

// ─────────────────────────────────────────────────────────────────────────────
// PSO — Particle Swarm Optimization
// ─────────────────────────────────────────────────────────────────────────────
export class PSOAlgorithm {
  private particles: { pos: Vector2D; vel: Vector2D; best: Vector2D }[] = [];
  private gBest: Vector2D | null = null;
  private waypoints: Vector2D[] = [];
  private replanEvery = 60;
  private frame = 0;

  private costFn(p: Vector2D, goal: Vector2D, obstacles: Obstacle[]): number {
    const dGoal = dist(p, goal);
    const penalty = obstacles.reduce((acc, o) => {
      const d = dist(p, o) - o.radius;
      return acc + (d < 20 ? (20 - d) * 5 : 0);
    }, 0);
    return dGoal + penalty;
  }

  plan(start: Vector2D, goal: Vector2D, obstacles: Obstacle[]) {
    const N = 20;
    this.particles = Array.from({ length: N }, () => {
      const pos = {
        x: start.x + (Math.random() - 0.5) * 200,
        y: start.y + (Math.random() - 0.5) * 200,
      };
      return { pos, vel: { x: (Math.random() - 0.5) * 4, y: (Math.random() - 0.5) * 4 }, best: { ...pos } };
    });
    this.gBest = { ...start };

    for (let iter = 0; iter < 60; iter++) {
      this.particles.forEach(p => {
        const gB = this.gBest!;
        p.vel.x = 0.7 * p.vel.x + 1.5 * Math.random() * (p.best.x - p.pos.x) + 1.5 * Math.random() * (gB.x - p.pos.x);
        p.vel.y = 0.7 * p.vel.y + 1.5 * Math.random() * (p.best.y - p.pos.y) + 1.5 * Math.random() * (gB.y - p.pos.y);
        p.pos.x += p.vel.x;
        p.pos.y += p.vel.y;
        if (this.costFn(p.pos, goal, obstacles) < this.costFn(p.best, goal, obstacles)) p.best = { ...p.pos };
        if (this.costFn(p.best, goal, obstacles) < this.costFn(this.gBest!, goal, obstacles)) this.gBest = { ...p.best };
      });
    }
    // Trace waypoints from start toward gBest toward goal
    this.waypoints = [];
    if (this.gBest) {
      const steps = 5;
      for (let i = 1; i <= steps; i++) {
        const t = i / steps;
        this.waypoints.push({
          x: start.x + (this.gBest.x - start.x) * t,
          y: start.y + (this.gBest.y - start.y) * t,
        });
      }
    }
    this.waypoints.push(goal);
  }

  calculateNextVelocity(pos: Vector2D, goal: Vector2D, obstacles: Obstacle[]): Vector2D {
    this.frame++;
    if (this.waypoints.length === 0 || this.frame % this.replanEvery === 0) {
      this.plan(pos, goal, obstacles);
    }
    while (this.waypoints.length > 1 && dist(pos, this.waypoints[0]) < 22) this.waypoints.shift();
    const target = this.waypoints[0] ?? goal;
    const toTarget = normalize({ x: target.x - pos.x, y: target.y - pos.y });
    const rep = repulsionVec(pos, obstacles);
    return clampVel({ x: toTarget.x * 2.6 + rep.x * 0.9, y: toTarget.y * 2.6 + rep.y * 0.9 });
  }

  reset() { this.particles = []; this.gBest = null; this.waypoints = []; this.frame = 0; }
}

// ─────────────────────────────────────────────────────────────────────────────
// DQN — Deep Q-Network (tabular Q-table approximation for simulation)
// ─────────────────────────────────────────────────────────────────────────────
export class DQNAlgorithm {
  // 8 discrete actions: N NE E SE S SW W NW
  private ACTIONS = [
    { x: 0, y: -1 }, { x: 1, y: -1 }, { x: 1, y: 0 }, { x: 1, y: 1 },
    { x: 0, y: 1 },  { x: -1, y: 1 }, { x: -1, y: 0 }, { x: -1, y: -1 },
  ];
  private qTable: Map<string, number[]> = new Map();
  private epsilon = 0.3;
  private gamma = 0.95;
  private alpha = 0.1;
  private lastState = '';
  private lastAction = 0;

  private stateKey(pos: Vector2D, goal: Vector2D, obstacles: Obstacle[]): string {
    const gx = Math.round((goal.x - pos.x) / 50);
    const gy = Math.round((goal.y - pos.y) / 50);
    const nearObs = obstacles
      .map(o => dist(pos, o) - o.radius)
      .filter(d => d < 60)
      .length;
    return `${gx},${gy},${nearObs}`;
  }

  private getQ(state: string): number[] {
    if (!this.qTable.has(state)) this.qTable.set(state, this.ACTIONS.map(() => Math.random() * 0.1));
    return this.qTable.get(state)!;
  }

  private reward(pos: Vector2D, newPos: Vector2D, goal: Vector2D, obstacles: Obstacle[]): number {
    if (dist(newPos, goal) < 20) return 100;
    if (isColliding(newPos, obstacles)) return -20;
    return dist(pos, goal) - dist(newPos, goal); // positive if closer
  }

  calculateNextVelocity(pos: Vector2D, goal: Vector2D, obstacles: Obstacle[]): Vector2D {
    const state = this.stateKey(pos, goal, obstacles);
    const q = this.getQ(state);

    // Update Q for last step
    if (this.lastState) {
      const r = this.reward({ x: pos.x - this.ACTIONS[this.lastAction].x * 3, y: pos.y - this.ACTIONS[this.lastAction].y * 3 }, pos, goal, obstacles);
      const maxQ = Math.max(...q);
      const lastQ = this.getQ(this.lastState);
      lastQ[this.lastAction] += this.alpha * (r + this.gamma * maxQ - lastQ[this.lastAction]);
    }

    // ε-greedy
    let action: number;
    if (Math.random() < this.epsilon) {
      action = Math.floor(Math.random() * this.ACTIONS.length);
    } else {
      action = q.reduce((best, v, i) => (v > q[best] ? i : best), 0);
    }

    this.lastState = state;
    this.lastAction = action;
    this.epsilon = Math.max(0.05, this.epsilon * 0.9999);

    const dir = this.ACTIONS[action];
    const rep = repulsionVec(pos, obstacles);
    return clampVel({ x: normalize(dir).x * 2.5 + rep.x * 0.8, y: normalize(dir).y * 2.5 + rep.y * 0.8 });
  }

  reset() { this.qTable.clear(); this.epsilon = 0.3; this.lastState = ''; }
}

// ─────────────────────────────────────────────────────────────────────────────
// PPO — Proximal Policy Optimization (policy table approximation)
// ─────────────────────────────────────────────────────────────────────────────
export class PPOAlgorithm {
  private ACTIONS = [
    { x: 0, y: -1 }, { x: 1, y: -1 }, { x: 1, y: 0 }, { x: 1, y: 1 },
    { x: 0, y: 1 },  { x: -1, y: 1 }, { x: -1, y: 0 }, { x: -1, y: -1 },
  ];
  private policy: Map<string, number[]> = new Map(); // logits
  private oldPolicy: Map<string, number[]> = new Map();
  private clip = 0.2;
  private trajectory: { state: string; action: number; reward: number }[] = [];
  private updateEvery = 30;
  private frame = 0;

  private stateKey(pos: Vector2D, goal: Vector2D, obstacles: Obstacle[]): string {
    const gx = Math.round((goal.x - pos.x) / 40);
    const gy = Math.round((goal.y - pos.y) / 40);
    const near = obstacles.filter(o => dist(pos, o) - o.radius < 55).length;
    return `${gx},${gy},${Math.min(near, 3)}`;
  }

  private softmax(logits: number[]): number[] {
    const max = Math.max(...logits);
    const exp = logits.map(l => Math.exp(l - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(e => e / sum);
  }

  private getLogits(map: Map<string, number[]>, state: string): number[] {
    if (!map.has(state)) map.set(state, this.ACTIONS.map(() => 0.1 + Math.random() * 0.1));
    return map.get(state)!;
  }

  private sample(probs: number[]): number {
    let r = Math.random(), cum = 0;
    for (let i = 0; i < probs.length; i++) { cum += probs[i]; if (r < cum) return i; }
    return probs.length - 1;
  }

  calculateNextVelocity(pos: Vector2D, goal: Vector2D, obstacles: Obstacle[]): Vector2D {
    this.frame++;
    const state = this.stateKey(pos, goal, obstacles);
    const logits = this.getLogits(this.policy, state);
    const probs = this.softmax(logits);
    const action = this.sample(probs);

    const r = dist(pos, goal) < 25 ? 50 : (isColliding(pos, obstacles) ? -15 : (CANVAS_SIZE - dist(pos, goal)) / CANVAS_SIZE);
    this.trajectory.push({ state, action, reward: r });

    // PPO update: clip ratio and update logits
    if (this.frame % this.updateEvery === 0 && this.trajectory.length > 5) {
      this.trajectory.forEach(({ state: s, action: a, reward }) => {
        const logNew = this.getLogits(this.policy, s);
        const logOld = this.getLogits(this.oldPolicy, s);
        const probNew = this.softmax(logNew)[a];
        const probOld = this.softmax(logOld)[a] || 0.01;
        const ratio = probNew / probOld;
        const clipped = Math.max(1 - this.clip, Math.min(1 + this.clip, ratio));
        const adv = reward;
        const grad = Math.min(ratio * adv, clipped * adv) * 0.05;
        logNew[a] += grad;
        this.getLogits(this.oldPolicy, s); // ensure exists
        logOld[a] = logNew[a];
      });
      this.trajectory = [];
    }

    const dir = this.ACTIONS[action];
    const rep = repulsionVec(pos, obstacles);
    return clampVel({ x: normalize(dir).x * 2.5 + rep.x * 0.8, y: normalize(dir).y * 2.5 + rep.y * 0.8 });
  }

  reset() { this.policy.clear(); this.oldPolicy.clear(); this.trajectory = []; this.frame = 0; }
}

// ─────────────────────────────────────────────────────────────────────────────
// MAPF — Multi-Agent Path Finding (CBS-inspired single-agent with phantom agents)
// ─────────────────────────────────────────────────────────────────────────────
export class MAPFAlgorithm {
  // Simulates conflict-based search by treating dynamic obstacles as other agents
  // and routing around their predicted future positions
  private astar = new AStarAlgorithm();
  private waypoints: Vector2D[] = [];
  private replanAt = 0;

  calculateNextVelocity(pos: Vector2D, goal: Vector2D, obstacles: Obstacle[]): Vector2D {
    this.replanAt--;
    // Build augmented obstacle list including predicted agent positions
    const phantom: Obstacle[] = [];
    obstacles.filter(o => o.isDynamic && o.velocity).forEach(o => {
      for (let t = 1; t <= 8; t++) {
        phantom.push({
          ...o,
          id: `${o.id}_t${t}`,
          x: o.x + (o.velocity!.x * t * 3),
          y: o.y + (o.velocity!.y * t * 3),
          radius: o.radius + 4,
        });
      }
    });

    const augmented = [...obstacles, ...phantom];

    if (this.waypoints.length === 0 || this.replanAt <= 0) {
      this.waypoints = (this.astar as any).planPath(pos, goal, augmented);
      this.replanAt = 25;
    }

    while (this.waypoints.length > 1 && dist(pos, this.waypoints[0]) < 18) this.waypoints.shift();

    const target = this.waypoints[0] ?? goal;
    const toTarget = normalize({ x: target.x - pos.x, y: target.y - pos.y });
    const rep = repulsionVec(pos, obstacles, 50);
    return clampVel({ x: toTarget.x * 2.5 + rep.x, y: toTarget.y * 2.5 + rep.y });
  }

  reset() { this.waypoints = []; this.replanAt = 0; (this.astar as any).reset(); }
}

// ─────────────────────────────────────────────────────────────────────────────
// Behavior Tree Agent (Agentic AI)
// ─────────────────────────────────────────────────────────────────────────────
export class BehaviorTreeAlgorithm {
  // BT nodes: Selector → [Sequence(Emergency), Sequence(Avoid), Navigate]
  private exploreDir = { x: 1, y: 0 };
  private exploreTimer = 0;

  private conditionTooClose(pos: Vector2D, obstacles: Obstacle[]): Obstacle | null {
    return obstacles.find(o => dist(pos, o) - o.radius < 22) ?? null;
  }

  private conditionModeratelyClose(pos: Vector2D, obstacles: Obstacle[]): Obstacle | null {
    return obstacles.find(o => dist(pos, o) - o.radius < 55) ?? null;
  }

  // Emergency evasion — hard turn away
  private actionEmergencyEvade(pos: Vector2D, obs: Obstacle): Vector2D {
    const away = normalize({ x: pos.x - obs.x, y: pos.y - obs.y });
    const tang = { x: -away.y, y: away.x };
    return clampVel({ x: away.x * 3 + tang.x, y: away.y * 3 + tang.y });
  }

  // Smooth avoidance with tangential steering
  private actionAvoid(pos: Vector2D, goal: Vector2D, obs: Obstacle): Vector2D {
    const away = normalize({ x: pos.x - obs.x, y: pos.y - obs.y });
    const toGoal = normalize({ x: goal.x - pos.x, y: goal.y - pos.y });
    // Tangent that keeps heading toward goal side
    const tang1 = { x: -away.y, y: away.x };
    const tang2 = { x: away.y, y: -away.x };
    const t = (tang1.x * toGoal.x + tang1.y * toGoal.y) >= 0 ? tang1 : tang2;
    return clampVel({ x: away.x * 1.5 + t.x * 1.5 + toGoal.x * 0.5, y: away.y * 1.5 + t.y * 1.5 + toGoal.y * 0.5 });
  }

  // Navigate directly toward goal
  private actionNavigate(pos: Vector2D, goal: Vector2D): Vector2D {
    const toGoal = normalize({ x: goal.x - pos.x, y: goal.y - pos.y });
    return clampVel({ x: toGoal.x * 3, y: toGoal.y * 3 });
  }

  calculateNextVelocity(pos: Vector2D, goal: Vector2D, obstacles: Obstacle[]): Vector2D {
    // Tick the behavior tree
    const emergency = this.conditionTooClose(pos, obstacles);
    if (emergency) return this.actionEmergencyEvade(pos, emergency); // Sequence 1

    const nearby = this.conditionModeratelyClose(pos, obstacles);
    if (nearby) return this.actionAvoid(pos, goal, nearby); // Sequence 2

    return this.actionNavigate(pos, goal); // Fallback: Navigate
  }

  reset() { this.exploreTimer = 0; }
}