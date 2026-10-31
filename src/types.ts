// src/types.ts
export interface Vector2D {
  x: number;
  y: number;
}

export interface Obstacle {
  id: string;
  x: number;
  y: number;
  radius: number;
  isDynamic: boolean;
  velocity?: Vector2D;
}

export interface ThreatZone {
  id: string;
  x: number;
  y: number;
  radius: number;
  strength: number;     // 0.0 - 1.0
  pulsePhase: number;
}

export type EnvironmentMode = 'open' | 'urban' | 'indoor' | 'battlefield';

export interface SimulationConfig {
  staticObstacleCount: number;
  dynamicObstacleCount: number;
  alpha: number;
  beta: number;
  gamma: number;
  delta: number;
  gridSize: number;
}

export interface RobotState {
  position: Vector2D;
  velocity: Vector2D;
  path: Vector2D[];
  collisionCount: number;
  goalReached: boolean;
  startTime: number;
  endTime?: number;
  isConfused: boolean;
  confusionTimer: number;
  threatExposure: number;        // ← Added
}

export interface LiveRunStats {
  collisions: number;
  totalFrames: number;
  totalReplanMs: number;
  pathLengthPx: number;
  staticObstacles: number;
  dynamicObstacles: number;
  goalReached: boolean;
  energyConsumption: number;
  threatExposureTime: number;
  stealthScore: number;
  replans: number;
}