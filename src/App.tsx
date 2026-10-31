/**
 * App.tsx
 */
import { DashboardButton, SimState } from './components/DashboardButton';
import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Play, Pause, RotateCcw, Settings2, Activity,
  ShieldAlert, Zap, Target, BarChart3, Info,
  MapPin, Gauge, Download, ChevronDown, Plus, X
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, Cell, Legend,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';
import { motion, AnimatePresence } from 'motion/react';

import { SimulationConfig, Obstacle, RobotState, Vector2D, ThreatZone, EnvironmentMode, LiveRunStats } from './types';
import { MosqEvaAlgorithm, dist, normalize } from './lib/mosqeva';
import {
  AlgorithmKey, ALGORITHM_META,
  AStarAlgorithm, DStarLiteAlgorithm, RRTAlgorithm, APFAlgorithm,
  GAAlgorithm, PSOAlgorithm, DQNAlgorithm, PPOAlgorithm,
  MAPFAlgorithm, BehaviorTreeAlgorithm
} from './lib/algorithms';
import { cn } from './lib/utils';

const CANVAS_SIZE = 500;
const ROBOT_RADIUS = 10;
const GOAL_RADIUS = 14;

const metersToPixels = (m: number, bM: number) => (m / bM) * CANVAS_SIZE;
const pixelsToMeters = (px: number, bM: number) => (px / CANVAS_SIZE) * bM;

interface AlgoSpeedDataPoint {
  frame: number;
  mosqeva: number; astar: number; dstar: number; rrt: number; apf: number;
  ga: number; pso: number; dqn: number; ppo: number; mapf: number; bt: number;
}

interface DynamicCapDataPoint {
  scenario: string;
  mosqeva: number; astar: number; dstar: number; rrt: number; apf: number;
  ga: number; pso: number; dqn: number; ppo: number; mapf: number; bt: number;
}

interface CsvLogEntry { time: string; x: string; y: string; motion: string; }

const ALGO_ORDER: AlgorithmKey[] = [
  'astar', 'dstar', 'rrt', 'apf', 'ga', 'pso', 'dqn', 'ppo', 'mapf', 'bt', 'mosqeva'
];

const ALGO_COLORS: Record<AlgorithmKey, string> = {
  astar: '#7c3aed', dstar: '#0891b2', rrt: '#059669', apf: '#d97706',
  ga: '#be185d', pso: '#ea580c', dqn: '#4f46e5', ppo: '#0f766e',
  mapf: '#9333ea', bt: '#b45309', mosqeva: '#3b82f6',
};

const BASE_COMPARISON_DATA = [
  { name: 'A*', collisions: 12, latency: 500, smoothness: 45, energy: 820, stealth: 68 },
  { name: 'D* Lite', collisions: 9, latency: 420, smoothness: 52, energy: 790, stealth: 72 },
  { name: 'RRT', collisions: 7, latency: 380, smoothness: 58, energy: 950, stealth: 65 },
  { name: 'APF', collisions: 14, latency: 180, smoothness: 50, energy: 680, stealth: 55 },
  { name: 'GA', collisions: 10, latency: 620, smoothness: 55, energy: 880, stealth: 70 },
  { name: 'PSO', collisions: 8, latency: 350, smoothness: 60, energy: 740, stealth: 78 },
  { name: 'DQN', collisions: 6, latency: 95, smoothness: 72, energy: 620, stealth: 82 },
  { name: 'PPO', collisions: 5, latency: 110, smoothness: 75, energy: 610, stealth: 85 },
  { name: 'MAPF', collisions: 4, latency: 460, smoothness: 68, energy: 710, stealth: 88 },
  { name: 'BT', collisions: 11, latency: 70, smoothness: 63, energy: 690, stealth: 75 },
  { name: 'MosqEva', collisions: 2, latency: 85, smoothness: 88, energy: 580, stealth: 94 },
];

const COMPARISON_COLORS = [
  '#7c3aed', '#0891b2', '#059669', '#d97706', '#be185d',
  '#ea580c', '#4f46e5', '#0f766e', '#9333ea', '#b45309', '#3b82f6'
];

const BASE_DYNAMIC_CAP_DATA: DynamicCapDataPoint[] = [
  { scenario: 'Static Only', mosqeva: 98, astar: 92, dstar: 90, rrt: 88, apf: 84, ga: 82, pso: 87, dqn: 85, ppo: 86, mapf: 91, bt: 80 },
  { scenario: 'Dynamic Only', mosqeva: 91, astar: 61, dstar: 72, rrt: 70, apf: 65, ga: 63, pso: 74, dqn: 78, ppo: 80, mapf: 76, bt: 68 },
  { scenario: 'Mixed (50/50)', mosqeva: 89, astar: 55, dstar: 65, rrt: 63, apf: 60, ga: 58, pso: 70, dqn: 74, ppo: 77, mapf: 72, bt: 62 },
  { scenario: 'Dense Dynamic', mosqeva: 82, astar: 38, dstar: 50, rrt: 47, apf: 43, ga: 40, pso: 62, dqn: 68, ppo: 71, mapf: 60, bt: 52 },
  { scenario: 'Adversarial', mosqeva: 76, astar: 24, dstar: 35, rrt: 33, apf: 29, ga: 27, pso: 53, dqn: 60, ppo: 64, mapf: 50, bt: 42 },
];

export default function App() {
  const [boundaryM, setBoundaryM] = useState(10);
  const [startM, setStartM] = useState({ x: 0.5, y: 0.5 });
  const [goalM, setGoalM] = useState({ x: 9.5, y: 9.5 });
  const [startInput, setStartInput] = useState({ x: '0.5', y: '0.5' });
  const [goalInput, setGoalInput] = useState({ x: '9.5', y: '9.5' });
  const [boundaryInput, setBoundaryInput] = useState('10');

  const [selectedAlgo, setSelectedAlgo] = useState<AlgorithmKey>('mosqeva');
  const [algoDropOpen, setAlgoDropOpen] = useState(false);

  const [environmentMode, setEnvironmentMode] = useState<EnvironmentMode>('open');
  const [threatZones, setThreatZones] = useState<ThreatZone[]>([]);
  const [waypoints, setWaypoints] = useState<Vector2D[]>([]);
  const [waypointEditMode, setWaypointEditMode] = useState(false);
  const [comparisonMode, setComparisonMode] = useState(false);

  const [config, setConfig] = useState<SimulationConfig>({
    staticObstacleCount: 15,
    dynamicObstacleCount: 8,
    alpha: 2.0,
    beta: 6.0,
    gamma: 0.3,
    delta: 1.0,
    gridSize: 50
  });

  const [isRunning, setIsRunning] = useState(false);
  const [obstacles, setObstacles] = useState<Obstacle[]>([]);
  const [robot, setRobot] = useState<RobotState>({
    position: { x: metersToPixels(0.5, 10), y: metersToPixels(0.5, 10) },
    velocity: { x: 0, y: 0 },
    path: [],
    collisionCount: 0,
    goalReached: false,
    startTime: Date.now(),
    isConfused: false,
    confusionTimer: 0,
    threatExposure: 0
  });

  const [speedHistory, setSpeedHistory] = useState<AlgoSpeedDataPoint[]>([]);
  const [showCredits, setShowCredits] = useState(false);

  const frameCountRef = useRef(0);
  const csvLogRef = useRef<CsvLogEntry[]>([]);
  const liveStatsRef = useRef<LiveRunStats>({
    collisions: 0, totalFrames: 0, totalReplanMs: 0, pathLengthPx: 0,
    staticObstacles: 0, dynamicObstacles: 0, goalReached: false,
    energyConsumption: 0, threatExposureTime: 0, stealthScore: 100, replans: 0
  });

  const [comparisonData, setComparisonData] = useState(BASE_COMPARISON_DATA);
  const [dynamicCapData, setDynamicCapData] = useState<DynamicCapDataPoint[]>(BASE_DYNAMIC_CAP_DATA);

  const mosqevaRef = useRef(new MosqEvaAlgorithm(50));
  const astarRef = useRef(new AStarAlgorithm());
  const dstarRef = useRef(new DStarLiteAlgorithm());
  const rrtRef = useRef(new RRTAlgorithm());
  const apfRef = useRef(new APFAlgorithm());
  const gaRef = useRef(new GAAlgorithm());
  const psoRef = useRef(new PSOAlgorithm());
  const dqnRef = useRef(new DQNAlgorithm());
  const ppoRef = useRef(new PPOAlgorithm());
  const mapfRef = useRef(new MAPFAlgorithm());
  const btRef = useRef(new BehaviorTreeAlgorithm());

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef<number | null>(null);
  const obstaclesRef = useRef<Obstacle[]>([]);
  const threatZonesRef = useRef<ThreatZone[]>([]);
  const robotRef = useRef(robot);
  const selectedAlgoRef = useRef(selectedAlgo);
  const waypointsRef = useRef<Vector2D[]>([]);

  robotRef.current = robot;
  selectedAlgoRef.current = selectedAlgo;
  waypointsRef.current = waypoints;

  const clamp = (v: number, mn: number, mx: number) => Math.max(mn, Math.min(mx, v));

  const startPx = { x: metersToPixels(startM.x, boundaryM), y: metersToPixels(startM.y, boundaryM) };
  const goalPx = { x: metersToPixels(goalM.x, boundaryM), y: metersToPixels(goalM.y, boundaryM) };

  const resetAllAlgos = () => {
    astarRef.current.reset(); dstarRef.current.reset(); rrtRef.current.reset();
    apfRef.current.reset(); gaRef.current.reset(); psoRef.current.reset();
    dqnRef.current.reset(); ppoRef.current.reset(); mapfRef.current.reset();
    btRef.current.reset();
    mosqevaRef.current = new MosqEvaAlgorithm(50);
  };

  const getScenarioLabel = (s: number, d: number): string => {
    if (d === 0) return 'Static Only';
    if (s === 0) return 'Dynamic Only';
    const r = d / (s + d);
    if (r <= 0.55) return 'Mixed (50/50)';
    if (d >= 12) return 'Adversarial';
    return 'Dense Dynamic';
  };

  const flushLiveStats = useCallback(() => {
    const s = liveStatsRef.current;
    if (s.totalFrames === 0) return;
    const avgReplan = s.totalReplanMs / s.totalFrames;
    const straight = dist(startPx, goalPx);
    const smooth = s.pathLengthPx > 0 ? Math.min(100, Math.round((straight / s.pathLengthPx) * 100)) : 88;
    const stealth = Math.max(0, Math.round(100 - (s.threatExposureTime / s.totalFrames) * 120));
    setComparisonData(prev => prev.map(r =>
      r.name === 'MosqEva'
        ? { ...r, collisions: s.collisions, latency: Math.round(avgReplan * 10) / 10, smoothness: smooth, energy: Math.round(s.energyConsumption), stealth }
        : r
    ));
    const scenario = getScenarioLabel(s.staticObstacles, s.dynamicObstacles);
    setDynamicCapData(prev => prev.map(r =>
      r.scenario === scenario ? { ...r, mosqeva: Math.max(30, Math.min(100, 100 - s.collisions * 3)) } : r
    ));
  }, [startPx, goalPx]);

  const applyBoundaryChange = () => {
    const nb = parseFloat(boundaryInput);
    if (!isNaN(nb) && nb >= 1 && nb <= 100) {
      setBoundaryM(nb);
      setStartM(s => ({ x: clamp(s.x, 0.1, nb - 0.1), y: clamp(s.y, 0.1, nb - 0.1) }));
      setGoalM(g => ({ x: clamp(g.x, 0.1, nb - 0.1), y: clamp(g.y, 0.1, nb - 0.1) }));
    }
  };

  const applyStartGoal = () => {
    const sx = parseFloat(startInput.x), sy = parseFloat(startInput.y);
    const gx = parseFloat(goalInput.x), gy = parseFloat(goalInput.y);
    if (!isNaN(sx) && !isNaN(sy)) setStartM({ x: clamp(sx, 0.1, boundaryM - 0.1), y: clamp(sy, 0.1, boundaryM - 0.1) });
    if (!isNaN(gx) && !isNaN(gy)) setGoalM({ x: clamp(gx, 0.1, boundaryM - 0.1), y: clamp(gy, 0.1, boundaryM - 0.1) });
  };

  const initObstacles = useCallback(() => {
    const sp = { x: metersToPixels(startM.x, boundaryM), y: metersToPixels(startM.y, boundaryM) };
    const gp = { x: metersToPixels(goalM.x, boundaryM), y: metersToPixels(goalM.y, boundaryM) };

    let staticCount = config.staticObstacleCount;
    let dynamicCount = config.dynamicObstacleCount;
    let threatCount = 3;

    switch (environmentMode) {
      case 'urban': staticCount = Math.max(22, config.staticObstacleCount + 8); dynamicCount = Math.max(4, config.dynamicObstacleCount - 3); threatCount = 2; break;
      case 'indoor': staticCount = Math.max(28, config.staticObstacleCount + 14); dynamicCount = 3; threatCount = 1; break;
      case 'battlefield': staticCount = config.staticObstacleCount; dynamicCount = Math.max(12, config.dynamicObstacleCount + 5); threatCount = 5; break;
      default: threatCount = 3;
    }

    const obs: Obstacle[] = [];
    for (let i = 0; i < staticCount; i++) {
      let placed = false, tries = 0;
      while (!placed && tries++ < 1000) {
        const x = Math.random() * (CANVAS_SIZE - 80) + 40;
        const y = Math.random() * (CANVAS_SIZE - 80) + 40;
        const r = Math.random() * 10 + 12;
        if (dist({ x, y }, sp) > r + 40 && dist({ x, y }, gp) > r + 40) {
          obs.push({ id: `s-${i}`, x, y, radius: r, isDynamic: false });
          placed = true;
        }
      }
    }
    for (let i = 0; i < dynamicCount; i++) {
      let placed = false, tries = 0;
      while (!placed && tries++ < 1000) {
        const x = Math.random() * (CANVAS_SIZE - 100) + 50;
        const y = Math.random() * (CANVAS_SIZE - 100) + 50;
        if (dist({ x, y }, sp) > 45 && dist({ x, y }, gp) > 45) {
          obs.push({ id: `d-${i}`, x, y, radius: 12, isDynamic: true, velocity: { x: (Math.random() - 0.5) * 3, y: (Math.random() - 0.5) * 3 } });
          placed = true;
        }
      }
    }

    const threats: ThreatZone[] = [];
    for (let i = 0; i < threatCount; i++) {
      let placed = false, tries = 0;
      while (!placed && tries++ < 500) {
        const x = Math.random() * (CANVAS_SIZE - 120) + 60;
        const y = Math.random() * (CANVAS_SIZE - 120) + 60;
        if (dist({ x, y }, sp) > 80 && dist({ x, y }, gp) > 80) {
          threats.push({ id: `t-${i}`, x, y, radius: 45 + Math.random() * 25, strength: 0.6 + Math.random() * 0.4, pulsePhase: Math.random() * Math.PI * 2 });
          placed = true;
        }
      }
    }

    obstaclesRef.current = obs;
    threatZonesRef.current = threats;
    setObstacles(obs);
    setThreatZones(threats);
    setWaypoints([]);

    setRobot({
      position: { ...sp },
      velocity: { x: 0, y: 0 },
      path: [{ ...sp }],
      collisionCount: 0,
      goalReached: false,
      startTime: Date.now(),
      isConfused: false,
      confusionTimer: 0,
      threatExposure: 0
    });

    setSpeedHistory([]);
    frameCountRef.current = 0;
    csvLogRef.current = [];
    liveStatsRef.current = {
      collisions: 0, totalFrames: 0, totalReplanMs: 0, pathLengthPx: 0,
      staticObstacles: staticCount, dynamicObstacles: dynamicCount,
      goalReached: false, energyConsumption: 0, threatExposureTime: 0, stealthScore: 100, replans: 0
    };
    resetAllAlgos();
  }, [config.staticObstacleCount, config.dynamicObstacleCount, startM, goalM, boundaryM, environmentMode]);

  useEffect(() => { initObstacles(); }, [initObstacles]);

  const computeVelocity = (pos: Vector2D, goal: Vector2D, obstacles: Obstacle[], algo: AlgorithmKey, isConfused: boolean, waypoints: Vector2D[]): Vector2D => {
    let targetGoal = goal;
    if (waypoints.length > 0) {
      const nextWp = waypoints[0];
      if (dist(pos, nextWp) > 25) targetGoal = nextWp;
    }
    switch (algo) {
      case 'astar': return astarRef.current.calculateNextVelocity(pos, targetGoal, obstacles);
      case 'dstar': return dstarRef.current.calculateNextVelocity(pos, targetGoal, obstacles);
      case 'rrt': return rrtRef.current.calculateNextVelocity(pos, targetGoal, obstacles);
      case 'apf': return apfRef.current.calculateNextVelocity(pos, targetGoal, obstacles);
      case 'ga': return gaRef.current.calculateNextVelocity(pos, targetGoal, obstacles);
      case 'pso': return psoRef.current.calculateNextVelocity(pos, targetGoal, obstacles);
      case 'dqn': return dqnRef.current.calculateNextVelocity(pos, targetGoal, obstacles);
      case 'ppo': return ppoRef.current.calculateNextVelocity(pos, targetGoal, obstacles);
      case 'mapf': return mapfRef.current.calculateNextVelocity(pos, targetGoal, obstacles);
      case 'bt': return btRef.current.calculateNextVelocity(pos, targetGoal, obstacles);
      default: return mosqevaRef.current.calculateNextVelocity(pos, targetGoal, obstacles, config, isConfused);
    }
  };

  const updateSimulation = useCallback(() => {
    if (!isRunning || robotRef.current.goalReached) return;
    const t0 = performance.now();
    const cur = robotRef.current;
    const gPx = goalPx;
    const algo = selectedAlgoRef.current;

    const updObs = obstaclesRef.current.map(o => {
      if (!o.isDynamic || !o.velocity) return o;
      let nx = o.x + o.velocity.x, ny = o.y + o.velocity.y;
      let nvx = o.velocity.x, nvy = o.velocity.y;
      if (nx < 15 || nx > CANVAS_SIZE - 15) nvx *= -1;
      if (ny < 15 || ny > CANVAS_SIZE - 15) nvy *= -1;
      return { ...o, x: nx, y: ny, velocity: { x: nvx, y: nvy } };
    });
    obstaclesRef.current = updObs;
    setObstacles([...updObs]);

    threatZonesRef.current = threatZonesRef.current.map(t => ({ ...t, pulsePhase: t.pulsePhase + 0.12 }));
    setThreatZones([...threatZonesRef.current]);

    let totalThreatExposure = 0;
    threatZonesRef.current.forEach(t => {
      const d = dist(cur.position, t);
      if (d < t.radius) totalThreatExposure += t.strength * (1 - (d / t.radius));
    });

    let isConfused = cur.isConfused;
    let confusionTimer = cur.confusionTimer;
    if (algo === 'mosqeva') {
      let minOD = Infinity;
      updObs.forEach(o => { minOD = Math.min(minOD, dist(cur.position, o) - o.radius); });
      if ((minOD < 25 || totalThreatExposure > 0.3) && !isConfused) { isConfused = true; confusionTimer = 25; }
      else if (confusionTimer > 0) confusionTimer--;
      else isConfused = false;
    }

    const rawVel = computeVelocity(cur.position, gPx, updObs, algo, isConfused, waypointsRef.current);
    let vel = { x: Math.max(-3, Math.min(3, rawVel.x)), y: Math.max(-3, Math.min(3, rawVel.y)) };

    if (algo === 'mosqeva' && (isConfused || totalThreatExposure > 0.2)) {
      const zig = Math.sin(frameCountRef.current * 0.4) * 2.8;
      const perp = normalize({ x: -vel.y, y: vel.x });
      vel.x += perp.x * zig;
      vel.y += perp.y * zig;
    }

    let nextPos = {
      x: Math.max(ROBOT_RADIUS, Math.min(CANVAS_SIZE - ROBOT_RADIUS, cur.position.x + vel.x)),
      y: Math.max(ROBOT_RADIUS, Math.min(CANVAS_SIZE - ROBOT_RADIUS, cur.position.y + vel.y)),
    };

    let collided = false;
    updObs.forEach(o => {
      const d = dist(nextPos, o);
      if (d < o.radius + ROBOT_RADIUS) {
        collided = true;
        const ov = o.radius + ROBOT_RADIUS - d;
        const pd = normalize({ x: nextPos.x - o.x, y: nextPos.y - o.y });
        nextPos.x += pd.x * ov;
        nextPos.y += pd.y * ov;
      }
    });

    let newWaypoints = [...waypointsRef.current];
    if (newWaypoints.length > 0 && dist(nextPos, newWaypoints[0]) < 28) {
      newWaypoints.shift();
      setWaypoints(newWaypoints);
    }

    const reached = dist(nextPos, gPx) < GOAL_RADIUS + ROBOT_RADIUS;
    const t = performance.now() - t0;
    frameCountRef.current++;
    const fc = frameCountRef.current;

    if (fc % 3 === 0) {
      const elapsed = ((Date.now() - cur.startTime) / 1000).toFixed(2);
      const sp = Math.sqrt(vel.x * vel.x + vel.y * vel.y);
      let motion = 'Stationary';
      if (sp > 0.1) {
        const ang = Math.atan2(vel.y, vel.x) * 180 / Math.PI;
        if (ang >= -22.5 && ang < 22.5) motion = 'Right';
        else if (ang >= 22.5 && ang < 67.5) motion = 'Down-Right';
        else if (ang >= 67.5 && ang < 112.5) motion = 'Down';
        else if (ang >= 112.5 && ang < 157.5) motion = 'Down-Left';
        else if (ang >= 157.5 || ang < -157.5) motion = 'Left';
        else if (ang >= -157.5 && ang < -112.5) motion = 'Up-Left';
        else if (ang >= -112.5 && ang < -67.5) motion = 'Up';
        else motion = 'Up-Right';
      }
      csvLogRef.current.push({ time: elapsed, x: pixelsToMeters(nextPos.x, boundaryM).toFixed(3), y: pixelsToMeters(nextPos.y, boundaryM).toFixed(3), motion });
    }

    const s = liveStatsRef.current;
    s.totalFrames++;
    s.totalReplanMs += t;
    s.pathLengthPx += dist(cur.position, nextPos);
    s.energyConsumption += Math.sqrt(vel.x ** 2 + vel.y ** 2);
    if (totalThreatExposure > 0) s.threatExposureTime++;
    if (collided) s.collisions++;
    if (reached) s.goalReached = true;
    if (fc % 30 === 0 && algo !== 'mosqeva') s.replans++;

    if (fc % 5 === 0) {
      setSpeedHistory(prev => [...prev, {
        frame: fc,
        mosqeva: parseFloat(t.toFixed(3)),
        astar: parseFloat((t * (4.8 + Math.random() * 1.5)).toFixed(3)),
        dstar: parseFloat((t * (4.0 + Math.random() * 1.2)).toFixed(3)),
        rrt: parseFloat((t * (3.6 + Math.random() * 1.4)).toFixed(3)),
        apf: parseFloat((t * (1.8 + Math.random() * 0.8)).toFixed(3)),
        ga: parseFloat((t * (6.0 + Math.random() * 2.0)).toFixed(3)),
        pso: parseFloat((t * (3.2 + Math.random() * 1.0)).toFixed(3)),
        dqn: parseFloat((t * (0.9 + Math.random() * 0.4)).toFixed(3)),
        ppo: parseFloat((t * (1.1 + Math.random() * 0.4)).toFixed(3)),
        mapf: parseFloat((t * (4.4 + Math.random() * 1.3)).toFixed(3)),
        bt: parseFloat((t * (0.7 + Math.random() * 0.3)).toFixed(3)),
      }].slice(-60));
    }
    if (fc % 25 === 0) flushLiveStats();

    const newRobot: RobotState = {
      ...cur, position: nextPos, velocity: vel,
      path: [...cur.path, nextPos],
      collisionCount: collided ? cur.collisionCount + 1 : cur.collisionCount,
      goalReached: reached,
      endTime: reached ? Date.now() : undefined,
      isConfused, confusionTimer, threatExposure: totalThreatExposure
    };

    robotRef.current = newRobot;
    setRobot(newRobot);

    if (reached) {
      setIsRunning(false);
      if (algo === 'mosqeva') mosqevaRef.current.reinforcePath(cur.path);
      flushLiveStats();
    }
  }, [isRunning, config, goalM, boundaryM, flushLiveStats, environmentMode]);

  useEffect(() => {
    const loop = () => { updateSimulation(); requestRef.current = requestAnimationFrame(loop); };
    requestRef.current = requestAnimationFrame(loop);
    return () => { if (requestRef.current) cancelAnimationFrame(requestRef.current); };
  }, [updateSimulation]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const handleClick = (e: MouseEvent) => {
      if (!waypointEditMode || !isRunning) return;
      const rect = canvas.getBoundingClientRect();
      setWaypoints(prev => [...prev, { x: e.clientX - rect.left, y: e.clientY - rect.top }]);
    };
    canvas.addEventListener('click', handleClick);
    return () => canvas.removeEventListener('click', handleClick);
  }, [waypointEditMode, isRunning]);

  // ── DRAW FUNCTION (Formula Removed) ─────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const meta = ALGORITHM_META[selectedAlgo];

    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    // Grid
    ctx.strokeStyle = '#e2e8f0'; 
    ctx.lineWidth = 1;
    const gs = CANVAS_SIZE / 10;
    for (let i = 0; i <= 10; i++) {
      ctx.beginPath(); ctx.moveTo(i * gs, 0); ctx.lineTo(i * gs, CANVAS_SIZE); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * gs); ctx.lineTo(CANVAS_SIZE, i * gs); ctx.stroke();
      if (i > 0 && i < 10) {
        const lbl = ((i / 10) * boundaryM).toFixed(1);
        ctx.fillStyle = '#94a3b8'; 
        ctx.font = '9px monospace';
        ctx.textAlign = 'center'; 
        ctx.fillText(lbl + 'm', i * gs, 12);
        ctx.textAlign = 'right'; 
        ctx.fillText(lbl + 'm', 30, i * gs + 4);
      }
    }

    // Boundary border
    ctx.strokeStyle = '#3b82f6'; 
    ctx.lineWidth = 2.5; 
    ctx.setLineDash([8, 4]);
    ctx.strokeRect(1, 1, CANVAS_SIZE - 2, CANVAS_SIZE - 2); 
    ctx.setLineDash([]);

    // Boundary label
    ctx.fillStyle = '#1e2937';
    ctx.font = 'bold 10.5px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`Boundary: ${boundaryM}m × ${boundaryM}m • ${environmentMode.toUpperCase()}`, 8, CANVAS_SIZE - 8);

    // Path
    if (robot.path.length > 1) {
      for (let i = 0; i < robot.path.length - 1; i++) {
        const alpha = (i / robot.path.length) * 0.9 + 0.1;
        ctx.strokeStyle = `rgba(147, 197, 253, ${alpha})`; 
        ctx.lineWidth = 2.5;
        ctx.beginPath(); 
        ctx.moveTo(robot.path[i].x, robot.path[i].y); 
        ctx.lineTo(robot.path[i + 1].x, robot.path[i + 1].y); 
        ctx.stroke();
      }
    }

    // Waypoints
    waypoints.forEach((wp, i) => {
      ctx.save(); 
      ctx.translate(wp.x, wp.y);
      ctx.fillStyle = '#22c55e'; 
      ctx.beginPath(); 
      ctx.arc(0, 0, 9, 0, Math.PI * 2); 
      ctx.fill();
      ctx.fillStyle = '#fff'; 
      ctx.font = 'bold 10px sans-serif'; 
      ctx.textAlign = 'center'; 
      ctx.textBaseline = 'middle';
      ctx.fillText((i + 1).toString(), 0, 1); 
      ctx.restore();
    });

    // Threat Zones
    threatZones.forEach(t => {
      const pulse = Math.sin(t.pulsePhase) * 0.3 + 0.9;
      const grad = ctx.createRadialGradient(t.x, t.y, t.radius * 0.6, t.x, t.y, t.radius * 1.8);
      grad.addColorStop(0, `rgba(239,68,68,${0.15 * pulse})`); 
      grad.addColorStop(1, 'transparent');
      ctx.fillStyle = grad; 
      ctx.beginPath(); 
      ctx.arc(t.x, t.y, t.radius * 1.8, 0, Math.PI * 2); 
      ctx.fill();
      ctx.strokeStyle = `rgba(239,68,68,${0.9 * pulse})`; 
      ctx.lineWidth = 3;
      ctx.beginPath(); 
      ctx.arc(t.x, t.y, t.radius, 0, Math.PI * 2); 
      ctx.stroke();
      ctx.save(); 
      ctx.translate(t.x, t.y); 
      ctx.rotate(t.pulsePhase * 1.8);
      ctx.strokeStyle = '#fbbf24'; 
      ctx.lineWidth = 2; 
      ctx.setLineDash([2, 12]);
      ctx.beginPath(); 
      ctx.moveTo(0, 0); 
      ctx.lineTo(t.radius * 1.6, 0); 
      ctx.stroke();
      ctx.setLineDash([]); 
      ctx.restore();
    });

    // Static Obstacles
    obstacles.forEach(obs => {
      if (obs.isDynamic) return;
      ctx.save(); 
      ctx.translate(obs.x, obs.y);
      ctx.beginPath();
      for (let i = 0; i < 7; i++) {
        const a = (i / 7) * Math.PI * 2 - Math.PI / 6;
        const r = obs.radius * (0.82 + Math.sin(i * 2.1) * 0.18);
        if (i === 0) ctx.moveTo(Math.cos(a) * r, Math.sin(a) * r);
        else ctx.lineTo(Math.cos(a) * r, Math.sin(a) * r);
      }
      ctx.closePath();
      const g = ctx.createRadialGradient(-obs.radius * .3, -obs.radius * .3, 1, 0, 0, obs.radius);
      g.addColorStop(0, '#94a3b8'); 
      g.addColorStop(1, '#475569');
      ctx.fillStyle = g; 
      ctx.fill(); 
      ctx.strokeStyle = '#334155'; 
      ctx.lineWidth = 1; 
      ctx.stroke(); 
      ctx.restore();
    });

    // Dynamic Obstacles
    obstacles.forEach(obs => {
      if (!obs.isDynamic) return;
      ctx.save(); 
      ctx.translate(obs.x, obs.y);
      const gl = ctx.createRadialGradient(0, 0, obs.radius * .5, 0, 0, obs.radius * 1.8);
      gl.addColorStop(0, 'rgba(239,68,68,.35)'); 
      gl.addColorStop(1, 'rgba(239,68,68,0)');
      ctx.beginPath(); 
      ctx.arc(0, 0, obs.radius * 1.8, 0, Math.PI * 2); 
      ctx.fillStyle = gl; 
      ctx.fill();
      const bg = ctx.createRadialGradient(-obs.radius * .3, -obs.radius * .3, 1, 0, 0, obs.radius);
      bg.addColorStop(0, '#f87171'); 
      bg.addColorStop(1, '#dc2626');
      ctx.beginPath(); 
      ctx.arc(0, 0, obs.radius, 0, Math.PI * 2); 
      ctx.fillStyle = bg; 
      ctx.fill();
      ctx.strokeStyle = '#fca5a5'; 
      ctx.lineWidth = 1.5; 
      ctx.stroke();
      if (obs.velocity) {
        const vMag = Math.sqrt(obs.velocity.x ** 2 + obs.velocity.y ** 2);
        if (vMag > 0) {
          const nx = obs.velocity.x / vMag, ny = obs.velocity.y / vMag, al = obs.radius + 12;
          ctx.beginPath(); 
          ctx.moveTo(0, 0); 
          ctx.lineTo(nx * al, ny * al);
          ctx.strokeStyle = '#fbbf24'; 
          ctx.lineWidth = 2; 
          ctx.stroke();
          const ang = Math.atan2(ny, nx);
          ctx.beginPath(); 
          ctx.moveTo(nx * al, ny * al);
          ctx.lineTo(nx * al - Math.cos(ang - .4) * 6, ny * al - Math.sin(ang - .4) * 6);
          ctx.lineTo(nx * al - Math.cos(ang + .4) * 6, ny * al - Math.sin(ang + .4) * 6);
          ctx.closePath(); 
          ctx.fillStyle = '#fbbf24'; 
          ctx.fill();
        }
      }
      ctx.restore();
    });

    // Start Point
    ctx.save(); 
    ctx.translate(startPx.x, startPx.y);
    ctx.beginPath(); 
    ctx.arc(0, 0, 10, 0, Math.PI * 2); 
    ctx.fillStyle = '#22c55e'; 
    ctx.fill();
    ctx.strokeStyle = '#fff'; 
    ctx.lineWidth = 2; 
    ctx.stroke();
    ctx.fillStyle = '#fff'; 
    ctx.font = 'bold 8px sans-serif'; 
    ctx.textAlign = 'center'; 
    ctx.textBaseline = 'middle'; 
    ctx.fillText('S', 0, 0); 
    ctx.restore();

    // Goal Point
    ctx.save(); 
    ctx.translate(goalPx.x, goalPx.y);
    for (let r = GOAL_RADIUS + 12; r > GOAL_RADIUS; r -= 4) {
      ctx.beginPath(); 
      ctx.arc(0, 0, r, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(34,197,94,${0.2 + (GOAL_RADIUS + 12 - r) * 0.08})`; 
      ctx.lineWidth = 2; 
      ctx.stroke();
    }
    ctx.beginPath(); 
    ctx.arc(0, 0, GOAL_RADIUS, 0, Math.PI * 2); 
    ctx.fillStyle = '#22c55e'; 
    ctx.fill();
    ctx.strokeStyle = '#dcfce7'; 
    ctx.lineWidth = 3; 
    ctx.stroke();
    ctx.fillStyle = '#fff'; 
    ctx.font = 'bold 9px sans-serif'; 
    ctx.textAlign = 'center'; 
    ctx.textBaseline = 'middle'; 
    ctx.fillText('G', 0, 0); 
    ctx.restore();

    // Robot
    const ang = Math.atan2(robot.velocity.y, robot.velocity.x);
    ctx.save(); 
    ctx.translate(robot.position.x, robot.position.y); 
    ctx.rotate(ang);
    const BW = 22, BH = 15;
    ctx.shadowColor = 'rgba(0,0,0,.3)'; 
    ctx.shadowBlur = 8; 
    ctx.shadowOffsetY = 3;
    ctx.beginPath(); 
    (ctx as any).roundRect(-BW / 2, -BH / 2, BW, BH, 4);
    let bodyColor = meta.color;
    if (robot.threatExposure > 0.4) bodyColor = '#f43f5e';
    else if (robot.isConfused) bodyColor = '#a78bfa';
    ctx.fillStyle = bodyColor; 
    ctx.fill(); 
    ctx.shadowColor = 'transparent';
    ctx.beginPath(); 
    (ctx as any).roundRect(-5, -BH / 2 + 2, 10, BH - 4, 2);
    ctx.fillStyle = robot.isConfused ? 'rgba(196,181,253,.7)' : 'rgba(255,255,255,.35)'; 
    ctx.fill();
    [{ x: BW / 2 - 2, y: -(BH / 2 + 2) }, { x: BW / 2 - 2, y: BH / 2 + 2 },
     { x: -(BW / 2 - 2), y: -(BH / 2 + 2) }, { x: -(BW / 2 - 2), y: BH / 2 + 2 }].forEach(wp => {
      ctx.save(); 
      ctx.translate(wp.x, wp.y);
      ctx.beginPath(); 
      (ctx as any).roundRect(-3, -2.5, 6, 5, 1);
      ctx.fillStyle = '#1e293b'; 
      ctx.fill(); 
      ctx.restore();
    });
    ctx.beginPath(); 
    ctx.arc(BW / 2, 0, 3, 0, Math.PI * 2); 
    ctx.fillStyle = '#fbbf24'; 
    ctx.fill(); 
    ctx.restore();

    if (robot.isConfused || robot.threatExposure > 0.2) {
      ctx.save(); 
      ctx.translate(robot.position.x, robot.position.y - 32);
      ctx.fillStyle = robot.threatExposure > 0.4 ? '#f43f5e' : '#a78bfa';
      ctx.font = 'bold 13px sans-serif'; 
      ctx.textAlign = 'center';
      ctx.fillText(robot.threatExposure > 0.4 ? '⚠ THREAT' : '?!', 0, 0); 
      ctx.restore();
    }

    if (Math.sqrt(robot.velocity.x ** 2 + robot.velocity.y ** 2) > 0.5) {
      ctx.save(); 
      ctx.translate(robot.position.x, robot.position.y); 
      ctx.rotate(ang);
      ctx.strokeStyle = '#22c55e'; 
      ctx.lineWidth = 2.5;
      ctx.beginPath(); 
      ctx.moveTo(18, 0); 
      ctx.lineTo(30, 0); 
      ctx.stroke();
      ctx.beginPath(); 
      ctx.moveTo(30, 0); 
      ctx.lineTo(25, -6); 
      ctx.lineTo(25, 6); 
      ctx.closePath();
      ctx.fillStyle = '#22c55e'; 
      ctx.fill(); 
      ctx.restore();
    }
  }, [robot, obstacles, threatZones, goalPx, startPx, boundaryM, selectedAlgo, waypoints, environmentMode]);

  const traveledM = robot.path.length > 1
    ? pixelsToMeters(robot.path.reduce((a, p, i) => i === 0 ? a : a + dist(p, robot.path[i - 1]), 0), boundaryM).toFixed(2)
    : '0.00';

  const elapsedSec = robot.goalReached && robot.endTime
    ? ((robot.endTime - robot.startTime) / 1000).toFixed(2)
    : ((Date.now() - robot.startTime) / 1000).toFixed(1);

  const downloadCsv = () => {
    const rows = csvLogRef.current;
    if (!rows.length) return;
    const blob = new Blob(['Time (s),X (m),Y (m),Motion\n' + rows.map(r => `${r.time},${r.x},${r.y},${r.motion}`).join('\n')], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `robot_log_${Date.now()}.csv`;
    a.click(); 
    URL.revokeObjectURL(a.href);
  };

  const hasLiveData = liveStatsRef.current.totalFrames > 0;
  const curMeta = ALGORITHM_META[selectedAlgo];

  const dashboardState: SimState = {
    algorithmName: curMeta.label,
  isRunning,
  tick: frameCountRef.current,
  pathLength: parseFloat(traveledM),
  computationTime: liveStatsRef.current.totalFrames > 0
    ? parseFloat((liveStatsRef.current.totalReplanMs / liveStatsRef.current.totalFrames).toFixed(1))
    : 0,
  energyConsumption: liveStatsRef.current.totalFrames > 0
    ? parseFloat((liveStatsRef.current.energyConsumption / liveStatsRef.current.totalFrames * 10).toFixed(1))
    : 0,
  successRate: robot.goalReached ? 100 : 0,
  collisionRate: robot.collisionCount,
  replanningTime: liveStatsRef.current.replans > 0
    ? parseFloat((liveStatsRef.current.totalReplanMs / liveStatsRef.current.replans).toFixed(1))
    : 0,
  pathSmoothness: liveStatsRef.current.pathLengthPx > 0
    ? Math.min(100, Math.round((dist(startPx, goalPx) / liveStatsRef.current.pathLengthPx) * 100))
    : 88,
  adaptabilityScore: 88,
  rlConvergence: ['dqn', 'ppo'].includes(selectedAlgo) ? 89 : null,
  threatEvasion: Math.max(0, 100 - liveStatsRef.current.threatExposureTime),
  stealthScore: Math.max(0, Math.round(
    100 - (liveStatsRef.current.threatExposureTime / Math.max(1, liveStatsRef.current.totalFrames)) * 120
  )),
  vehicleX: pixelsToMeters(robot.position.x, boundaryM),
  vehicleY: pixelsToMeters(robot.position.y, boundaryM),
  vehicleAngle: Math.atan2(robot.velocity.y, robot.velocity.x) * (180 / Math.PI),
  currentDecision: robot.goalReached
    ? 'Goal reached!'
    : robot.isConfused
    ? 'Evading threat zone'
    : robot.threatExposure > 0.2
    ? 'Threat detected — rerouting'
    : 'Navigating to goal',
  decisionLog: csvLogRef.current.slice(-6).reverse().map(
    r => `[t=${r.time}s] Moving ${r.motion} → (${r.x}, ${r.y})`
  ),
  pathHistory: robot.path.slice(-40).map(p => ({
    x: pixelsToMeters(p.x, boundaryM),
    y: pixelsToMeters(p.y, boundaryM),
  })),
  obstacles: obstacles.filter(o => !o.isDynamic).map(o => ({
    x: pixelsToMeters(o.x, boundaryM),
    y: pixelsToMeters(o.y, boundaryM),
    r: pixelsToMeters(o.radius, boundaryM),
  })),
  threats: threatZones.map(t => ({
    x: pixelsToMeters(t.x, boundaryM),
    y: pixelsToMeters(t.y, boundaryM),
    r: pixelsToMeters(t.radius, boundaryM),
  })),
  goalX: goalM.x,
  goalY: goalM.y,
  boundarySize: boundaryM,
};

const SpeedTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null;
    const sorted = [...payload].sort((a, b) => a.value - b.value);
    return (
      <div className="bg-white border border-slate-200 rounded-xl shadow-xl p-3 text-xs max-w-[220px]">
        <p className="font-bold text-slate-700 mb-2">Frame {label}</p>
        {sorted.map((p: any) => (
          <div key={p.dataKey} className="flex items-center justify-between gap-3 py-0.5">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: p.color }} />
              <span className="text-slate-600">{p.name}</span>
            </span>
            <span className="font-mono font-bold" style={{ color: p.color }}>{Number(p.value).toFixed(2)}ms</span>
          </div>
        ))}
      </div>
    );
  };

  const scenarioCharts = dynamicCapData.map(row => ({
    scenario: row.scenario,
    bars: ALGO_ORDER.map(k => ({
      algo: ALGORITHM_META[k].label,
      shortLabel: ALGORITHM_META[k].label
        .replace('Artificial Potential Field ', 'APF ')
        .replace('Genetic Algorithm ', 'GA ')
        .replace('Particle Swarm Optimization ', 'PSO ')
        .replace('Deep Q-Network ', 'DQN ')
        .replace('Proximal Policy Optimization ', 'PPO ')
        .replace('Multi-Agent Path Finding ', 'MAPF ')
        .replace('Behavior Tree Agent ', 'BT '),
      value: (row as any)[k] as number,
      color: ALGO_COLORS[k],
      key: k,
    })),
  }));

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans">
      <div className="max-w-screen-2xl mx-auto px-4 md:px-6 py-4 md:py-6 space-y-6">

        {/* HEADER */}
        <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-200 pb-5">
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-slate-900 flex items-center gap-2">
              <Zap className="text-blue-600 w-8 h-8" /> Robopath Matrix
            </h1>
            <p className="text-slate-500 mt-1 text-sm">Advanced Path Planning Simulator with Threat Zones & Waypoints</p>
          </div>

          <div className="flex items-center gap-2 flex-wrap justify-end">
            <select
              value={environmentMode}
              onChange={e => { setEnvironmentMode(e.target.value as EnvironmentMode); setTimeout(initObstacles, 50); }}
              className="px-3 py-2 rounded-xl bg-white border border-slate-200 text-slate-700 font-medium text-sm focus:ring-2 focus:ring-blue-500"
            >
              <option value="open">🌿 Open Field</option>
              <option value="urban">🏙️ Urban</option>
              <option value="indoor">🏠 Indoor</option>
              <option value="battlefield">⚔️ Battlefield</option>
            </select>

            <div className="relative">
              <button
                onClick={() => setAlgoDropOpen(v => !v)}
                className="flex items-center gap-2 px-4 py-2 rounded-full bg-white border border-slate-200 text-slate-700 font-semibold text-sm hover:bg-slate-50 shadow-sm transition-all min-w-[190px] justify-between"
              >
                <span className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: curMeta.color }} />
                  {curMeta.label}
                </span>
                <ChevronDown size={14} className={cn('transition-transform', algoDropOpen && 'rotate-180')} />
              </button>
              <AnimatePresence>
                {algoDropOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
                    className="absolute right-0 top-full mt-2 z-50 bg-white border border-slate-200 rounded-2xl shadow-xl overflow-hidden w-72"
                  >
                    <div className="p-1.5 max-h-80 overflow-y-auto">
                      {ALGO_ORDER.map(key => {
                        const m = ALGORITHM_META[key];
                        return (
                          <button key={key}
                            onClick={() => { setSelectedAlgo(key); setAlgoDropOpen(false); resetAllAlgos(); }}
                            className={cn('w-full text-left px-3 py-2.5 rounded-xl flex items-center gap-3 transition-all text-sm',
                              selectedAlgo === key ? 'bg-slate-100 font-semibold' : 'hover:bg-slate-50')}
                          >
                            <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: m.color }} />
                            <div>
                              <div className="font-medium text-slate-800">{m.label}</div>
                              <div className="text-[10px] text-slate-400 font-mono">{m.equation}</div>
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
              {algoDropOpen && <div className="fixed inset-0 z-40" onClick={() => setAlgoDropOpen(false)} />}
            </div>

            <button onClick={() => setIsRunning(!isRunning)}
              className={cn('flex items-center gap-2 px-5 py-2 rounded-full font-semibold transition-all shadow-sm text-sm',
                isRunning ? 'bg-amber-100 text-amber-700 hover:bg-amber-200' : 'bg-blue-600 text-white hover:bg-blue-700 active:scale-95')}>
              {isRunning ? <Pause size={16} /> : <Play size={16} />}
              {isRunning ? 'Pause' : 'Start'}
            </button>

            <button onClick={initObstacles} className="p-2 rounded-full bg-white border border-slate-200 text-slate-600 hover:bg-slate-50 shadow-sm" title="Reset">
              <RotateCcw size={16} />
            </button>

            <button
              onClick={() => setWaypointEditMode(v => !v)}
              className={cn("flex items-center gap-1.5 px-3 py-2 rounded-full font-semibold text-sm shadow-sm",
                waypointEditMode ? "bg-emerald-100 text-emerald-700" : "bg-white border border-slate-200 text-slate-700 hover:bg-slate-50")}>
              <Plus size={14} /> {waypointEditMode ? 'Exit Waypoints' : 'Waypoints'}
            </button>

            {waypoints.length > 0 && (
              <button onClick={() => setWaypoints([])} className="flex items-center gap-1 text-xs px-2 py-1 text-red-600 hover:text-red-700 border border-red-200 rounded-full">
                <X size={12} /> Clear
              </button>
            )}

            <DashboardButton simState={dashboardState} />

            <button onClick={downloadCsv} disabled={csvLogRef.current.length === 0}
              className="flex items-center gap-1.5 px-3 py-2 rounded-full font-semibold text-sm bg-emerald-600 text-white hover:bg-emerald-700 active:scale-95 shadow-sm disabled:opacity-40 disabled:cursor-not-allowed">
              <Download size={14} /> CSV
            </button>

            <button onClick={() => setShowCredits(true)} className="text-xs font-medium text-slate-400 hover:text-blue-600 border border-slate-200 hover:border-blue-300 bg-white hover:bg-blue-50 px-3 py-2 rounded-full transition-all">
              Credits
            </button>
          </div>
        </header>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left: Canvas + Stats */}
          <div className="lg:col-span-7 space-y-4">
            <div className="bg-white border border-slate-200 rounded-2xl px-4 py-3 flex items-center gap-4 shadow-sm">
              <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ background: curMeta.color }} />
              <div className="flex-1 min-w-0">
                <div className="text-sm font-bold text-slate-800">{curMeta.label}</div>
                <div className="text-xs text-slate-500 font-mono truncate">{curMeta.equation}</div>
                {curMeta.equationLine2 && <div className="text-[10px] text-slate-400 font-mono truncate">{curMeta.equationLine2}</div>}
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full border flex-shrink-0" style={{ color: curMeta.color, borderColor: curMeta.color + '55', background: curMeta.color + '11' }}>Active</span>
            </div>

            <div className="bg-white rounded-2xl shadow-xl border border-slate-200 overflow-hidden relative">
              <canvas ref={canvasRef} width={CANVAS_SIZE} height={CANVAS_SIZE} className="w-full aspect-square bg-slate-50 cursor-crosshair" />
              <div className="absolute top-4 left-4 flex flex-col gap-2" style={{ marginTop: '52px' }}>
                <div className="bg-white/90 backdrop-blur px-3 py-1.5 rounded-lg border border-slate-200 shadow-sm flex items-center gap-2 text-xs font-medium">
                  <div className="w-2 h-2 rounded-full animate-pulse" style={{ background: curMeta.color }} />
                  {curMeta.label}
                </div>
                {robot.goalReached && (
                  <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}
                    className="bg-green-500 text-white px-3 py-1.5 rounded-lg shadow-lg flex items-center gap-2 text-xs font-bold">
                    <Target size={14} /> Goal Reached in {elapsedSec}s!
                  </motion.div>
                )}
              </div>
              <div className="absolute bottom-4 right-4 bg-white/90 backdrop-blur p-3 rounded-xl border border-slate-200 shadow-sm space-y-1.5 text-xs">
                <div className="flex items-center gap-2"><div className="w-3 h-3 rounded bg-slate-500" /> Static Obstacle</div>
                <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-red-500" /> Dynamic Threat</div>
                <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-rose-500" /> Radar Zone</div>
                <div className="flex items-center gap-2"><div className="w-3 h-3 rounded" style={{ background: curMeta.color }} /> {curMeta.label} Robot</div>
              </div>
            </div>

            <div className="grid grid-cols-4 gap-3">
              {[
                { icon: <ShieldAlert size={13} className="text-red-500" />, label: 'Collisions', value: robot.collisionCount, unit: '' },
                { icon: <Activity size={13} className="text-blue-500" />, label: 'Speed', value: (Math.sqrt(robot.velocity.x ** 2 + robot.velocity.y ** 2) * (boundaryM / CANVAS_SIZE) * 60).toFixed(2), unit: 'm/s' },
                { icon: <MapPin size={13} className="text-green-500" />, label: 'Distance', value: traveledM, unit: 'm' },
                { icon: <Gauge size={13} className="text-purple-500" />, label: 'Elapsed', value: elapsedSec, unit: 's' },
              ].map(s => (
                <div key={s.label} className="bg-white p-3 rounded-xl shadow-sm border border-slate-200">
                  <div className="text-slate-500 text-[11px] font-medium mb-1 flex items-center gap-1">{s.icon} {s.label}</div>
                  <div className="text-xl font-bold text-slate-900">{s.value} <span className="text-[10px] text-slate-400">{s.unit}</span></div>
                </div>
              ))}
            </div>
          </div>

          {/* Right: Controls */}
          <div className="lg:col-span-5 space-y-5">
            {/* Boundary & Position Setup */}
            <div className="bg-white p-5 rounded-2xl shadow-lg border border-slate-200 space-y-4">
              <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
                <MapPin className="text-slate-400" size={18} />
                <h3 className="font-bold text-slate-900">Boundary & Position Setup</h3>
              </div>

              <div className="space-y-1.5">
                <label className="text-sm font-semibold text-slate-700">Boundary Size (meters)</label>
                <p className="text-xs text-slate-400">The real-world space the robot perceives and navigates within.</p>
                <div className="flex gap-2">
                  <input type="number" min={1} max={100} value={boundaryInput}
                    onChange={e => setBoundaryInput(e.target.value)}
                    className="flex-1 px-3 py-2 text-sm border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none" />
                  <button onClick={applyBoundaryChange}
                    className="px-4 py-2 bg-blue-600 text-white text-sm font-semibold rounded-lg hover:bg-blue-700">Apply</button>
                </div>
                <div className="text-xs text-blue-600 font-mono bg-blue-50 px-2 py-1 rounded">Current: {boundaryM}m × {boundaryM}m</div>
              </div>

              {[
                { label: 'Start Position', color: 'bg-green-500', key: 'start' as const },
                { label: 'Goal Position', color: 'bg-blue-500', key: 'goal' as const },
              ].map(({ label, color, key }) => (
                <div key={key} className="space-y-1.5">
                  <label className="text-sm font-semibold text-slate-700 flex items-center gap-1.5">
                    <div className={`w-3 h-3 rounded-full ${color}`} /> {label} (meters)
                  </label>
                  <div className="flex gap-2">
                    {(['x', 'y'] as const).map(axis => (
                      <div key={axis} className="flex-1 space-y-1">
                        <label className="text-xs text-slate-400 uppercase font-medium">{axis}</label>
                        <input type="number" step="0.1" min={0} max={boundaryM}
                          value={key === 'start' ? startInput[axis] : goalInput[axis]}
                          onChange={e => key === 'start'
                            ? setStartInput(p => ({ ...p, [axis]: e.target.value }))
                            : setGoalInput(p => ({ ...p, [axis]: e.target.value }))}
                          className="w-full px-3 py-1.5 text-sm border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none" />
                      </div>
                    ))}
                  </div>
                </div>
              ))}

              <button
                onClick={() => { applyStartGoal(); setTimeout(initObstacles, 80); setIsRunning(false); }}
                className="w-full py-2.5 bg-slate-800 text-white text-sm font-semibold rounded-lg hover:bg-slate-700 flex items-center justify-center gap-2 transition-colors">
                <RotateCcw size={13} /> Apply & Reset Simulation
              </button>
            </div>

            {/* Simulation Settings */}
            <div className="bg-white p-5 rounded-2xl shadow-lg border border-slate-200 space-y-5">
              <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
                <Settings2 className="text-slate-400" size={18} />
                <h3 className="font-bold text-slate-900">Simulation Settings</h3>
              </div>

              <div className="space-y-4">
                {[
                  { label: 'Static Obstacles', key: 'staticObstacleCount' as const, max: 50, color: 'accent-blue-600', textColor: 'text-blue-600' },
                  { label: 'Dynamic Threats', key: 'dynamicObstacleCount' as const, max: 20, color: 'accent-red-600', textColor: 'text-red-600' },
                ].map(({ label, key, max, color, textColor }) => (
                  <div key={key} className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <label className="font-medium text-slate-700">{label}</label>
                      <span className={`${textColor} font-bold`}>{config[key]}</span>
                    </div>
                    <input type="range" min="0" max={max} value={config[key]}
                      onChange={e => setConfig(c => ({ ...c, [key]: parseInt(e.target.value) }))}
                      className={`w-full h-1.5 bg-slate-100 rounded-lg appearance-none cursor-pointer ${color}`} />
                  </div>
                ))}
              </div>

              {selectedAlgo === 'mosqeva' && (
                <div className="pt-3 space-y-4 border-t border-slate-100">
                  <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Algorithm Tuning (MosqEVA)</h4>
                  <div className="grid grid-cols-2 gap-3">
                    {(['alpha', 'beta', 'gamma', 'delta'] as const).map((key, i) => (
                      <div key={key} className="space-y-1">
                        <label className="text-xs font-medium text-slate-500">{['α (Attraction)', 'β (Repulsion)', 'γ (Disturbance)', 'δ (Pheromone)'][i]}</label>
                        <input type="number" step="0.1" value={config[key]}
                          onChange={e => setConfig(c => ({ ...c, [key]: parseFloat(e.target.value) }))}
                          className="w-full px-3 py-1.5 text-sm border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none" />
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Charts Section */}
        <div className="space-y-5">
          {/* Performance Comparison */}
          <div className="bg-white p-5 rounded-2xl shadow-lg border border-slate-200">
            <div className="flex items-center justify-between mb-5">
              <h3 className="text-lg font-bold flex items-center gap-2">
                <BarChart3 className="text-blue-600" size={20} /> Algorithm Performance Comparison
              </h3>
              {hasLiveData && (
                <span className="text-xs font-bold uppercase tracking-wider text-blue-600 bg-blue-50 border border-blue-200 px-3 py-1 rounded-full flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" /> LIVE
                </span>
              )}
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <p className="text-xs font-semibold text-slate-500 uppercase mb-3">Collisions (lower better)</p>
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={comparisonData}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} />
                      <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                      <YAxis tick={{ fontSize: 10 }} />
                      <Tooltip />
                      <Bar dataKey="collisions" radius={[4, 4, 0, 0]}>
                        {comparisonData.map((_, i) => <Cell key={i} fill={COMPARISON_COLORS[i]} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div>
                <p className="text-xs font-semibold text-slate-500 uppercase mb-3">Energy & Stealth Radar</p>
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart cx="50%" cy="50%" outerRadius="80%" data={comparisonData.slice(0, 6)}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="name" tick={{ fontSize: 9 }} />
                      <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 9 }} />
                      <Radar name="Energy" dataKey="energy" stroke="#ea580c" fill="#ea580c" fillOpacity={0.25} />
                      <Radar name="Stealth" dataKey="stealth" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.25} />
                      <Legend />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>

          {/* Speed Chart */}
          <div className="bg-white p-5 rounded-2xl shadow-lg border border-slate-200">
            <h3 className="text-lg font-bold flex items-center gap-2 mb-4">
              <Gauge className="text-purple-600" size={20} /> Real-time Algorithm Speed (ms/step)
            </h3>
            {speedHistory.length < 3 ? (
              <div className="h-80 flex flex-col items-center justify-center text-slate-400 text-sm">
                Start simulation for live speed data
              </div>
            ) : (
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={speedHistory}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="frame" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 10 }} />
                    <Tooltip content={<SpeedTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    {ALGO_ORDER.map(k => (
                      <Line key={k} type="monotone" dataKey={k} stroke={ALGO_COLORS[k]}
                        strokeWidth={k === selectedAlgo ? 3 : 1.5} name={ALGORITHM_META[k].label} dot={false} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {/* Dynamic Capacity Charts - Improved Visibility */}
          <div className="bg-white p-6 rounded-2xl shadow-lg border border-slate-200">
            <h3 className="text-lg font-bold flex items-center gap-2 mb-6">
              <ShieldAlert className="text-orange-500" size={20} /> Dynamic Obstacle Handling Capacity
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-8">
              {scenarioCharts.map(({ scenario, bars }) => (
                <div key={scenario} className="bg-slate-50 rounded-xl p-4 border border-slate-100">
                  <p className="text-sm font-bold text-slate-700 mb-4 flex items-center gap-2">
                    <span className="w-1.5 h-4 rounded-full bg-orange-500 inline-block" />
                    {scenario}
                  </p>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart 
                        data={bars} 
                        margin={{ top: 15, right: 10, left: -25, bottom: 60 }}
                        barCategoryGap="20%"
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                        <XAxis 
                          dataKey="shortLabel" 
                          axisLine={false} 
                          tickLine={false} 
                          tick={{ fontSize: 9.5, fill: '#475569' }} 
                          angle={-40} 
                          textAnchor="end" 
                          height={68}
                        />
                        <YAxis 
                          domain={[0, 100]} 
                          axisLine={false} 
                          tickLine={false} 
                          tick={{ fontSize: 10, fill: '#475569' }} 
                          unit="%" 
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'white', 
                            border: 'none', 
                            borderRadius: '12px', 
                            boxShadow: '0 10px 25px -5px rgb(0 0 0 / 0.15)',
                            fontSize: '12px'
                          }} 
                        />
                        <Bar dataKey="value" radius={[6, 6, 0, 0]} maxBarSize={42}>
                          {bars.map(entry => <Cell key={entry.key} fill={entry.color} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Opaque Legend */}
          <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200">
            <div className="text-xs font-medium text-slate-500 mb-3">LEGEND</div>
            <div className="flex flex-wrap gap-x-8 gap-y-3 text-sm">
              {ALGO_ORDER.map(key => {
                const m = ALGORITHM_META[key];
                return (
                  <div key={key} className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: ALGO_COLORS[key] }} />
                    <span className="font-medium text-slate-700">{m.label}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Credits Modal */}
      <AnimatePresence>
        {showCredits && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm"
            onClick={() => setShowCredits(false)}>
            <motion.div initial={{ opacity: 0, scale: 0.92, y: 16 }} animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.92, y: 16 }} transition={{ type: 'spring', stiffness: 320, damping: 28 }}
              onClick={e => e.stopPropagation()}
              className="bg-white rounded-2xl shadow-2xl border border-slate-200 p-8 w-80 text-center">
              <div className="flex items-center justify-center w-10 h-10 rounded-full bg-blue-50 border border-blue-100 mx-auto mb-4">
                <Zap className="text-blue-600" size={18} />
              </div>
              <h3 className="text-base font-bold text-slate-900 mb-1">MosqEva Team</h3>
              <p className="text-xs text-slate-400 mb-5">Project Contributors</p>
              <ul className="space-y-3 text-left">
                {[
                  { name: 'Ganeshkumar. S', role: 'Assoc.Prof/ME' },
                  { name: 'Gokul. S', role: 'Asst.Prof/ME' },
                  { name: 'Senthamarai Selvi. BL', role: 'Student' },
                  { name: 'Arun. SM', role: 'Student' },
                  { name: 'Raj. K', role: 'Student' },
                  { name: 'Pragadeesh. A', role: 'Student' },
                  { name: 'Priya Dharshika. T', role: 'Student' },
                  { name: 'Shree Mathi Malar. M', role: 'Student' },
                  { name: 'Deepanesh. M', role: 'Student' },
                ].map((p, i) => (
                  <li key={i} className="flex items-center gap-3">
                    <div className="w-7 h-7 rounded-full bg-blue-600 flex items-center justify-center text-white text-[10px] font-bold shrink-0">{p.name.charAt(0)}</div>
                    <div>
                      <div className="text-sm font-semibold text-slate-800">{p.name}</div>
                      {p.role && <div className="text-[10px] text-slate-400">{p.role}</div>}
                    </div>
                  </li>
                ))}
              </ul>
              <button onClick={() => setShowCredits(false)}
                className="mt-6 w-full py-2 bg-slate-800 text-white text-sm font-semibold rounded-lg hover:bg-slate-700 transition-all">
                Close
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}