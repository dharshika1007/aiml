import { Vector2D, Obstacle, SimulationConfig } from './types';

export const normalize = (v: Vector2D): Vector2D => {
  const mag = Math.sqrt(v.x * v.x + v.y * v.y);
  return mag === 0 ? { x: 0, y: 0 } : { x: v.x / mag, y: v.y / mag };
};

export const dist = (v1: Vector2D, v2: Vector2D): number => {
  return Math.sqrt(Math.pow(v1.x - v2.x, 2) + Math.pow(v1.y - v2.y, 2));
};

export class MosqEvaAlgorithm {
  private pheromoneGrid: number[][];
  private gridSize: number;

  constructor(gridSize: number) {
    this.gridSize = gridSize;
    this.pheromoneGrid = Array(gridSize).fill(0).map(() => Array(gridSize).fill(0.1));
  }

  calculateNextVelocity(
    pos: Vector2D,
    goal: Vector2D,
    obstacles: Obstacle[],
    config: SimulationConfig,
    isConfused: boolean
  ): Vector2D {
    // 1. Attraction Vector (Ai) - Goal Attraction Mechanism (Section 3.2.4)
    const toGoal = { x: goal.x - pos.x, y: goal.y - pos.y };
    const attraction = normalize(toGoal);

    // 2. Repulsion Vector (Ri) - Repulsion Mechanism (Section 3.2.4)
    let repulsion = { x: 0, y: 0 };
    let minObstacleDist = Infinity;

    obstacles.forEach(obs => {
      const d = dist(pos, obs);
      minObstacleDist = Math.min(minObstacleDist, d - obs.radius);
      
      const safeDist = obs.isDynamic ? obs.radius + 45 : obs.radius + 30;
      
      if (d < safeDist) {
        const diff = { x: pos.x - obs.x, y: pos.y - obs.y };
        const normDiff = normalize(diff);
        
        // Force magnitude increases as distance decreases (Section 3.2.4)
        const strength = Math.pow((safeDist - d) / safeDist, 2) * 12;
        
        if (obs.isDynamic && obs.velocity) {
          const relVel = { x: obs.velocity.x, y: obs.velocity.y };
          const dot = relVel.x * -normDiff.x + relVel.y * -normDiff.y;
          const predictiveFactor = dot > 0 ? (1 + dot * 2) : 1;
          repulsion.x += normDiff.x * strength * predictiveFactor;
          repulsion.y += normDiff.y * strength * predictiveFactor;
        } else {
          repulsion.x += normDiff.x * strength;
          repulsion.y += normDiff.y * strength;
        }
      }
    });

    // 3. Disturbance Vector (Di) - Disturbance Response (Section 3.2.4)
    // Models environmental variations or sensor noise
    const disturbance = {
      x: (Math.random() - 0.5) * 0.4,
      y: (Math.random() - 0.5) * 0.4
    };

    // 4. Pheromone Vector (Pi) - Pheromone Reinforcement (Section 3.4.4)
    const gx = Math.floor(pos.x / (500 / this.gridSize));
    const gy = Math.floor(pos.y / (500 / this.gridSize));
    let pheromoneDir = { x: 0, y: 0 };
    
    for (let i = -1; i <= 1; i++) {
      for (let j = -1; j <= 1; j++) {
        const nx = gx + i;
        const ny = gy + j;
        if (nx >= 0 && nx < this.gridSize && ny >= 0 && ny < this.gridSize) {
          const val = this.pheromoneGrid[ny][nx];
          pheromoneDir.x += i * val;
          pheromoneDir.y += j * val;
        }
      }
    }
    const pheromone = normalize(pheromoneDir);

    // 5. Confusion Strategy (Section 3.3.5)
    // Randomize movement direction to avoid predictable motion patterns
    let confusionVector = { x: 0, y: 0 };
    if (isConfused) {
      confusionVector = {
        x: (Math.random() - 0.5) * 5,
        y: (Math.random() - 0.5) * 5
      };
    }

    // Adaptive weights: prioritize safety when very close
    let adaptiveAlpha = config.alpha;
    if (minObstacleDist < 20) {
      adaptiveAlpha *= (minObstacleDist / 20);
    }

    // Final Velocity: v = αA + βR + γD + δP + C (Confusion)
    return {
      x: adaptiveAlpha * attraction.x + config.beta * repulsion.x + config.gamma * disturbance.x + config.delta * pheromone.x + confusionVector.x,
      y: adaptiveAlpha * attraction.y + config.beta * repulsion.y + config.gamma * disturbance.y + config.delta * pheromone.y + confusionVector.y
    };
  }

  reinforcePath(path: Vector2D[]) {
    path.forEach(p => {
      const gx = Math.floor(p.x / (500 / this.gridSize));
      const gy = Math.floor(p.y / (500 / this.gridSize));
      if (gx >= 0 && gx < this.gridSize && gy >= 0 && gy < this.gridSize) {
        this.pheromoneGrid[gy][gx] += 0.5;
      }
    });
    // Evaporation
    for (let y = 0; y < this.gridSize; y++) {
      for (let x = 0; x < this.gridSize; x++) {
        this.pheromoneGrid[y][x] *= 0.95;
      }
    }
  }
}
