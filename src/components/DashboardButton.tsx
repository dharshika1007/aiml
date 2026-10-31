// DashboardButton.tsx

import { useState, useEffect, useRef, useCallback } from 'react';

export interface SimState {
  algorithmName?: string;
  isRunning?: boolean;
  tick?: number;
  pathLength?: number;
  computationTime?: number;
  energyConsumption?: number;
  successRate?: number;
  collisionRate?: number;
  replanningTime?: number;
  pathSmoothness?: number;
  adaptabilityScore?: number;
  rlConvergence?: number | null;
  threatEvasion?: number;
  stealthScore?: number;
  vehicleX?: number;
  vehicleY?: number;
  vehicleAngle?: number;
  currentDecision?: string;
  decisionLog?: string[];
  pathHistory?: Array<{ x: number; y: number }>;
  obstacles?: Array<{ x: number; y: number; r: number }>;
  threats?: Array<{ x: number; y: number; r: number }>;
  goalX?: number;
  goalY?: number;
  boundarySize?: number;
}

interface Props {
  simState?: SimState;
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function r1(v: number) { return v.toFixed(1); }
function r0(v: number) { return v.toFixed(0); }

type StatusLevel = 'good' | 'warn' | 'bad';
function statusOf(val: number, goodAbove: number, warnAbove: number): StatusLevel {
  return val >= goodAbove ? 'good' : val >= warnAbove ? 'warn' : 'bad';
}
function statusLow(val: number, goodBelow: number, warnBelow: number): StatusLevel {
  return val <= goodBelow ? 'good' : val <= warnBelow ? 'warn' : 'bad';
}

const STATUS_COLORS: Record<StatusLevel, string> = {
  good: '#22c55e',
  warn: '#f59e0b',
  bad:  '#ef4444',
};
const STATUS_BG: Record<StatusLevel, string> = {
  good: 'rgba(34,197,94,0.12)',
  warn: 'rgba(245,158,11,0.12)',
  bad:  'rgba(239,68,68,0.12)',
};
const STATUS_LABEL: Record<StatusLevel, string> = {
  good: 'Good',
  warn: 'Avg',
  bad:  'Poor',
};

// ── Demo data (used when no real simState is provided) ─────────────────────
function makeDemoState(tick: number): SimState {
  const t = tick / 60;
  return {
    algorithmName:    'MosqEVA',
    isRunning:        true,
    tick,
    pathLength:       +(82 + Math.sin(t) * 6).toFixed(1),
    computationTime:  +(11 + Math.cos(t * 1.3) * 2).toFixed(1),
    energyConsumption:+(44 + Math.sin(t * 0.7) * 5).toFixed(1),
    successRate:      +(93 + Math.sin(t * 0.4) * 2).toFixed(1),
    collisionRate:    +(1.2 + Math.abs(Math.sin(t * 2)) * 0.8).toFixed(1),
    replanningTime:   +(10 + Math.cos(t) * 3).toFixed(1),
    pathSmoothness:   +(80 + Math.sin(t * 0.5) * 6).toFixed(1),
    adaptabilityScore:+(88 + Math.sin(t * 0.3) * 4).toFixed(1),
    rlConvergence:    +(89 + Math.sin(t * 0.2) * 3).toFixed(1),
    threatEvasion:    +(85 + Math.cos(t * 0.6) * 5).toFixed(1),
    stealthScore:     +(74 + Math.sin(t * 0.8) * 6).toFixed(1),
    vehicleX:          0.5 + (t % 9),
    vehicleY:          0.5 + Math.sin(t * 0.8) * 4 + 4,
    vehicleAngle:      t * 60,
    currentDecision:  ['Navigating to goal', 'Avoiding threat zone', 'Recalculating path',
                       'Pheromone attraction', 'Evading obstacle'][Math.floor(t * 2) % 5],
    decisionLog:      Array.from({ length: 6 }, (_, i) =>
      `[t=${tick - i * 3}] ${['Path recalculated','Threat detected','Waypoint reached',
        'Energy optimised','Stealth mode ON','Replan triggered'][i % 6]}`),
    pathHistory:       Array.from({ length: Math.min(tick, 40) }, (_, i) => ({
      x: 0.5 + i * 0.22,
      y: 0.5 + Math.sin(i * 0.3) * 3 + 4,
    })),
    obstacles: [
      { x: 2, y: 2, r: 0.3 }, { x: 4, y: 5, r: 0.3 }, { x: 6, y: 3, r: 0.3 },
      { x: 7, y: 7, r: 0.3 }, { x: 3, y: 7, r: 0.3 }, { x: 8, y: 4, r: 0.3 },
    ],
    threats: [
      { x: 3, y: 3, r: 1.2 }, { x: 6.5, y: 3.5, r: 1.4 },
    ],
    goalX: 9.5, goalY: 9.5, boundarySize: 10,
  };
}

// ── Sub-components ────────────────────────────────────────────────────────────

// Mini metric pill
function MetricPill({ label, value, unit, status }: {
  label: string; value: string; unit: string; status: StatusLevel;
}) {
  return (
    <div style={{
      background: STATUS_BG[status],
      border: `1px solid ${STATUS_COLORS[status]}33`,
      borderRadius: 8,
      padding: '7px 10px',
      display: 'flex', flexDirection: 'column', gap: 2,
      minWidth: 90,
    }}>
      <span style={{ fontSize: 10, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</span>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 3 }}>
        <span style={{ fontSize: 18, fontWeight: 600, color: '#f1f5f9', fontVariantNumeric: 'tabular-nums' }}>{value}</span>
        <span style={{ fontSize: 10, color: '#64748b' }}>{unit}</span>
      </div>
      <span style={{ fontSize: 10, color: STATUS_COLORS[status], fontWeight: 500 }}>{STATUS_LABEL[status]}</span>
    </div>
  );
}

// Bar row for table
function BarRow({ label, value, max = 100, color, unit = '' }: {
  label: string; value: number; max?: number; color: string; unit?: string;
}) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <span style={{ fontSize: 11, color: '#94a3b8', width: 130, flexShrink: 0 }}>{label}</span>
      <div style={{ flex: 1, height: 5, background: 'rgba(255,255,255,0.06)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 3, transition: 'width 0.4s' }} />
      </div>
      <span style={{ fontSize: 11, color: '#e2e8f0', fontVariantNumeric: 'tabular-nums', width: 48, textAlign: 'right' }}>
        {r1(value)}{unit}
      </span>
    </div>
  );
}

// Canvas-based nav visualization
function NavCanvas({ s }: { s: SimState }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef   = useRef<number>(0);
  const tickRef   = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;

    function draw() {
      const W = canvas!.width, H = canvas!.height;
      const bound = s.boundarySize || 10;
      const scale = Math.min(W, H) / bound;

      ctx.clearRect(0, 0, W, H);

      // Background
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, W, H);

      // Grid
      ctx.strokeStyle = 'rgba(148,163,184,0.07)';
      ctx.lineWidth = 0.5;
      for (let i = 0; i <= bound; i++) {
        ctx.beginPath(); ctx.moveTo(i * scale, 0); ctx.lineTo(i * scale, H); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, i * scale); ctx.lineTo(W, i * scale); ctx.stroke();
      }

      // Threats
      (s.threats || []).forEach(t => {
        const tx = t.x * scale, ty = t.y * scale;
        const pulse = 0.85 + 0.15 * Math.sin(tickRef.current * 0.05);
        const grad = ctx.createRadialGradient(tx, ty, 0, tx, ty, t.r * scale * pulse);
        grad.addColorStop(0, 'rgba(239,68,68,0.35)');
        grad.addColorStop(1, 'rgba(239,68,68,0)');
        ctx.beginPath(); ctx.arc(tx, ty, t.r * scale * pulse, 0, Math.PI * 2);
        ctx.fillStyle = grad; ctx.fill();
        ctx.beginPath(); ctx.arc(tx, ty, t.r * scale, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(239,68,68,0.5)'; ctx.lineWidth = 1; ctx.stroke();
        ctx.beginPath(); ctx.arc(tx, ty, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#ef4444'; ctx.fill();
      });

      // Obstacles
      (s.obstacles || []).forEach(o => {
        ctx.save();
        ctx.translate(o.x * scale, o.y * scale);
        ctx.rotate(Math.PI / 5);
        ctx.fillStyle = '#475569';
        ctx.strokeStyle = '#64748b';
        ctx.lineWidth = 1;
        // Pentagon
        ctx.beginPath();
        for (let i = 0; i < 5; i++) {
          const angle = (i * 2 * Math.PI / 5) - Math.PI / 2;
          const r = o.r * scale * 2.5;
          i === 0 ? ctx.moveTo(r * Math.cos(angle), r * Math.sin(angle))
                  : ctx.lineTo(r * Math.cos(angle), r * Math.sin(angle));
        }
        ctx.closePath(); ctx.fill(); ctx.stroke();
        ctx.restore();
      });

      // Path history trail
      const trail = s.pathHistory || [];
      if (trail.length > 1) {
        ctx.beginPath();
        ctx.moveTo(trail[0].x * scale, trail[0].y * scale);
        trail.forEach(p => ctx.lineTo(p.x * scale, p.y * scale));
        ctx.strokeStyle = 'rgba(56,189,248,0.4)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Goal
      const gx = (s.goalX || 9.5) * scale, gy = (s.goalY || 9.5) * scale;
      const gPulse = 0.8 + 0.2 * Math.sin(tickRef.current * 0.08);
      ctx.beginPath(); ctx.arc(gx, gy, 10 * gPulse, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(34,197,94,0.2)'; ctx.fill();
      ctx.beginPath(); ctx.arc(gx, gy, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#22c55e'; ctx.fill();

      // Vehicle
      const vx = (s.vehicleX || 0.5) * scale;
      const vy = (s.vehicleY || 0.5) * scale;
      const angle = ((s.vehicleAngle || 0) * Math.PI) / 180;

      // Pheromone trail dots
      for (let i = 0; i < 3; i++) {
        const dx = -Math.cos(angle) * (i + 1) * 8;
        const dy = -Math.sin(angle) * (i + 1) * 8;
        ctx.beginPath(); ctx.arc(vx + dx, vy + dy, 2 - i * 0.5, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(250,204,21,${0.6 - i * 0.15})`; ctx.fill();
      }

      // Vehicle body
      ctx.save();
      ctx.translate(vx, vy);
      ctx.rotate(angle);
      // Body
      ctx.fillStyle = '#38bdf8';
      ctx.strokeStyle = '#7dd3fc';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(12, 0); ctx.lineTo(-8, -7); ctx.lineTo(-5, 0); ctx.lineTo(-8, 7);
      ctx.closePath(); ctx.fill(); ctx.stroke();
      // Direction indicator
      ctx.fillStyle = '#fff';
      ctx.beginPath(); ctx.arc(6, 0, 2.5, 0, Math.PI * 2); ctx.fill();
      ctx.restore();

      // State label
      ctx.fillStyle = 'rgba(15,23,42,0.75)';
      ctx.fillRect(4, 4, 170, 18);
      ctx.fillStyle = '#94a3b8';
      ctx.font = '10px monospace';
      ctx.fillText(`▶ ${s.currentDecision || '—'}`, 8, 16);

      tickRef.current++;
      animRef.current = requestAnimationFrame(draw);
    }

    draw();
    return () => cancelAnimationFrame(animRef.current);
  }, [s]);

  return (
    <canvas
      ref={canvasRef}
      width={440}
      height={320}
      style={{ width: '100%', height: 320, borderRadius: 10, display: 'block' }}
    />
  );
}

// ── Main Dashboard Modal ──────────────────────────────────────────────────────
function DashboardModal({ simState, onClose }: { simState: SimState; onClose: () => void }) {
  const s = simState;
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = 0;
  }, [s.decisionLog]);

  // Close on Escape
  useEffect(() => {
    const fn = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', fn);
    return () => window.removeEventListener('keydown', fn);
  }, [onClose]);

  const pl  = s.pathLength        ?? 0;
  const ct  = s.computationTime   ?? 0;
  const en  = s.energyConsumption ?? 0;
  const sr  = s.successRate       ?? 0;
  const cr  = s.collisionRate     ?? 0;
  const rt  = s.replanningTime    ?? 0;
  const ps  = s.pathSmoothness    ?? 0;
  const as_ = s.adaptabilityScore ?? 0;
  const rl  = s.rlConvergence     ?? null;
  const te  = s.threatEvasion     ?? 0;
  const st  = s.stealthScore      ?? 0;

  return (
    <div
      onClick={e => { if (e.target === e.currentTarget) onClose(); }}
      style={{
        position: 'fixed', inset: 0, zIndex: 9999,
        background: 'rgba(2,6,23,0.82)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: 16,
        backdropFilter: 'blur(4px)',
      }}
    >
      <div style={{
        background: '#0f172a',
        border: '1px solid rgba(148,163,184,0.12)',
        borderRadius: 16,
        width: '100%', maxWidth: 960,
        maxHeight: '90vh',
        overflowY: 'auto',
        boxShadow: '0 24px 80px rgba(0,0,0,0.7)',
        scrollbarWidth: 'thin',
        scrollbarColor: '#334155 transparent',
      }}>

        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '14px 20px',
          borderBottom: '1px solid rgba(148,163,184,0.08)',
          position: 'sticky', top: 0, background: '#0f172a', zIndex: 10,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{
              width: 32, height: 32, borderRadius: 8,
              background: 'rgba(56,189,248,0.15)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 16,
            }}>📊</div>
            <div>
              <div style={{ fontSize: 15, fontWeight: 600, color: '#f1f5f9' }}>
                Performance Dashboard
              </div>
              <div style={{ fontSize: 11, color: '#64748b' }}>
                {s.algorithmName || 'Algorithm'} · tick {s.tick ?? 0} ·&nbsp;
                <span style={{ color: s.isRunning ? '#22c55e' : '#f59e0b' }}>
                  {s.isRunning ? '● Running' : '○ Paused'}
                </span>
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'rgba(148,163,184,0.1)', border: '1px solid rgba(148,163,184,0.15)',
              borderRadius: 8, color: '#94a3b8', width: 32, height: 32,
              cursor: 'pointer', fontSize: 16, display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}
          >✕</button>
        </div>

        <div style={{ padding: '16px 20px', display: 'flex', flexDirection: 'column', gap: 16 }}>

          {/* Top row: Canvas + Core metrics */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>

            {/* Nav visualization */}
            <div style={{
              background: '#020817', border: '1px solid rgba(148,163,184,0.08)',
              borderRadius: 12, overflow: 'hidden',
            }}>
              <div style={{ padding: '10px 14px', borderBottom: '1px solid rgba(148,163,184,0.06)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 11, fontWeight: 500, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Live Navigation</span>
                <div style={{ display: 'flex', gap: 6, fontSize: 10 }}>
                  <span style={{ color: '#ef4444' }}>● Threats</span>
                  <span style={{ color: '#475569' }}>◆ Obstacles</span>
                  <span style={{ color: '#38bdf8' }}>▶ Vehicle</span>
                  <span style={{ color: '#22c55e' }}>● Goal</span>
                </div>
              </div>
              <NavCanvas s={s} />
            </div>

            {/* Core metrics grid */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <span style={{ fontSize: 11, fontWeight: 500, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Core Metrics</span>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                <MetricPill label="Path length"   value={r0(pl)} unit="units" status={statusLow(pl, 85, 105)} />
                <MetricPill label="Comp. time"    value={r1(ct)} unit="ms"    status={statusLow(ct, 12, 22)} />
                <MetricPill label="Success rate"  value={r1(sr)} unit="%"     status={statusOf(sr, 92, 85)} />
                <MetricPill label="Collision rate" value={r1(cr)} unit="hits" status={statusLow(cr, 1, 3)} />
                <MetricPill label="Threat evasion" value={r1(te)} unit="%"    status={statusOf(te, 80, 65)} />
                <MetricPill label="Stealth score" value={r1(st)} unit="/100"  status={statusOf(st, 70, 55)} />
              </div>
            </div>
          </div>

          {/* Performance bars */}
          <div style={{
            background: '#020817', border: '1px solid rgba(148,163,184,0.08)',
            borderRadius: 12, padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: 10,
          }}>
            <span style={{ fontSize: 11, fontWeight: 500, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Performance Overview</span>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px 24px' }}>
              <BarRow label="Energy consumption" value={en}  color="#f59e0b" unit="%" />
              <BarRow label="Path smoothness"    value={ps}  color="#38bdf8" />
              <BarRow label="Adaptability score" value={as_} color="#a78bfa" />
              <BarRow label="Threat evasion"     value={te}  color="#f87171" />
              <BarRow label="Stealth score"      value={st}  color="#34d399" />
              {rl !== null && <BarRow label="RL convergence" value={rl} color="#fb923c" unit="%" />}
            </div>
          </div>

          {/* Bottom row: full table + vehicle state + decision log */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>

            {/* Full metrics table */}
            <div style={{
              background: '#020817', border: '1px solid rgba(148,163,184,0.08)',
              borderRadius: 12, padding: '12px 16px',
            }}>
              <span style={{ fontSize: 11, fontWeight: 500, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'block', marginBottom: 10 }}>All Metrics</span>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr>
                    {['Metric','Value','Unit','Status'].map(h => (
                      <th key={h} style={{ textAlign: 'left', color: '#475569', fontWeight: 500, padding: '4px 8px', borderBottom: '1px solid rgba(148,163,184,0.07)', fontSize: 10, textTransform: 'uppercase' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[
                    { m: 'Path length',       v: r0(pl),              u: 'units', s: statusLow(pl, 85, 105) },
                    { m: 'Computation time',  v: r1(ct),              u: 'ms',    s: statusLow(ct, 12, 22) },
                    { m: 'Energy',            v: r1(en),              u: '%',     s: statusLow(en, 45, 60) },
                    { m: 'Success rate',      v: r1(sr),              u: '%',     s: statusOf(sr, 92, 85) },
                    { m: 'Collision rate',    v: r1(cr),              u: 'hits',  s: statusLow(cr, 1, 3) },
                    { m: 'Replanning time',   v: r1(rt),              u: 'ms',    s: statusLow(rt, 12, 22) },
                    { m: 'Path smoothness',   v: r1(ps),              u: '/100',  s: statusOf(ps, 80, 65) },
                    { m: 'Adaptability',      v: r1(as_),             u: '/100',  s: statusOf(as_, 85, 70) },
                    { m: 'RL convergence',    v: rl !== null ? r1(rl) : 'N/A', u: rl !== null ? '%' : '—', s: (rl !== null ? statusOf(rl, 88, 75) : 'warn') as StatusLevel },
                    { m: 'Threat evasion',    v: r1(te),              u: '%',     s: statusOf(te, 80, 65) },
                    { m: 'Stealth score',     v: r1(st),              u: '/100',  s: statusOf(st, 70, 55) },
                  ].map(row => (
                    <tr key={row.m} style={{ borderBottom: '1px solid rgba(148,163,184,0.04)' }}>
                      <td style={{ padding: '5px 8px', color: '#94a3b8' }}>{row.m}</td>
                      <td style={{ padding: '5px 8px', color: '#e2e8f0', fontWeight: 500, fontVariantNumeric: 'tabular-nums' }}>{row.v}</td>
                      <td style={{ padding: '5px 8px', color: '#475569' }}>{row.u}</td>
                      <td style={{ padding: '5px 8px' }}>
                        <span style={{
                          background: STATUS_BG[row.s], color: STATUS_COLORS[row.s],
                          fontSize: 10, padding: '1px 6px', borderRadius: 4, fontWeight: 500,
                        }}>{STATUS_LABEL[row.s]}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Vehicle state + decision log */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>

              {/* Vehicle state card */}
              <div style={{
                background: '#020817', border: '1px solid rgba(148,163,184,0.08)',
                borderRadius: 12, padding: '12px 16px',
              }}>
                <span style={{ fontSize: 11, fontWeight: 500, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'block', marginBottom: 10 }}>Vehicle State</span>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  {[
                    { label: 'Position X',  value: r1(s.vehicleX  ?? 0), unit: 'm' },
                    { label: 'Position Y',  value: r1(s.vehicleY  ?? 0), unit: 'm' },
                    { label: 'Heading',     value: r0(s.vehicleAngle ?? 0), unit: '°' },
                    { label: 'Status',      value: s.isRunning ? 'Active' : 'Idle', unit: '' },
                  ].map(item => (
                    <div key={item.label} style={{
                      background: 'rgba(148,163,184,0.04)', borderRadius: 8, padding: '8px 10px',
                    }}>
                      <div style={{ fontSize: 10, color: '#475569', marginBottom: 2 }}>{item.label}</div>
                      <div style={{ fontSize: 15, fontWeight: 600, color: '#38bdf8', fontVariantNumeric: 'tabular-nums' }}>
                        {item.value}<span style={{ fontSize: 10, color: '#64748b', marginLeft: 2 }}>{item.unit}</span>
                      </div>
                    </div>
                  ))}
                </div>
                <div style={{
                  marginTop: 10, padding: '8px 10px',
                  background: 'rgba(56,189,248,0.06)',
                  border: '1px solid rgba(56,189,248,0.15)',
                  borderRadius: 8,
                }}>
                  <div style={{ fontSize: 10, color: '#475569', marginBottom: 2 }}>Current decision</div>
                  <div style={{ fontSize: 12, color: '#7dd3fc', fontWeight: 500 }}>{s.currentDecision || '—'}</div>
                </div>
              </div>

              {/* Decision log */}
              <div style={{
                background: '#020817', border: '1px solid rgba(148,163,184,0.08)',
                borderRadius: 12, padding: '12px 16px', flex: 1,
              }}>
                <span style={{ fontSize: 11, fontWeight: 500, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'block', marginBottom: 8 }}>Decision Log</span>
                <div ref={logRef} style={{
                  fontFamily: 'monospace', fontSize: 11, color: '#475569',
                  display: 'flex', flexDirection: 'column', gap: 4,
                  maxHeight: 160, overflowY: 'auto',
                }}>
                  {(s.decisionLog || []).map((line, i) => (
                    <div key={i} style={{
                      color: i === 0 ? '#7dd3fc' : '#475569',
                      padding: '2px 0',
                      borderBottom: '1px solid rgba(148,163,184,0.04)',
                    }}>{line}</div>
                  ))}
                </div>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}

// ── The actual exported button ────────────────────────────────────────────────
export function DashboardButton({ simState }: Props) {
  const [open, setOpen]         = useState(false);
  const [demoTick, setDemoTick] = useState(0);
  const intervalRef             = useRef<ReturnType<typeof setInterval> | null>(null);

  // Run demo ticker only when modal is open and no real state is running
  useEffect(() => {
    if (open && !simState?.isRunning) {
      intervalRef.current = setInterval(() => setDemoTick(t => t + 1), 200);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [open, simState?.isRunning]);

  // Merge real state with demo fallback
  const effectiveState: SimState = simState?.isRunning
    ? simState
    : makeDemoState(demoTick);

  const handleClose = useCallback(() => setOpen(false), []);

  return (
    <>
      {/* ── Navbar button ── */}
      <button
        onClick={() => setOpen(true)}
        title="Open Performance Dashboard"
        style={{
          display: 'inline-flex', alignItems: 'center', gap: 6,
          padding: '6px 12px',
          background: open ? 'rgba(56,189,248,0.15)' : 'transparent',
          border: `1px solid ${open ? 'rgba(56,189,248,0.4)' : 'rgba(148,163,184,0.25)'}`,
          borderRadius: 8,
          color: open ? '#38bdf8' : '#64748b',
          fontSize: 13, fontWeight: 500, cursor: 'pointer',
          transition: 'all 0.15s',
          whiteSpace: 'nowrap',
        }}
        onMouseEnter={e => {
          if (!open) {
            (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(56,189,248,0.4)';
            (e.currentTarget as HTMLButtonElement).style.color = '#38bdf8';
          }
        }}
        onMouseLeave={e => {
          if (!open) {
            (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(148,163,184,0.25)';
            (e.currentTarget as HTMLButtonElement).style.color = '#64748b';
          }
        }}
      >
        {/* Bar chart icon */}
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <rect x="1" y="8"  width="2.5" height="5" rx="0.5" fill="currentColor" opacity="0.7"/>
          <rect x="5" y="4"  width="2.5" height="9" rx="0.5" fill="currentColor"/>
          <rect x="9" y="6"  width="2.5" height="7" rx="0.5" fill="currentColor" opacity="0.85"/>
          <rect x="0.5" y="13" width="13" height="0.5" rx="0.25" fill="currentColor" opacity="0.3"/>
        </svg>
        Dashboard
      </button>

      {/* ── Modal portal ── */}
      {open && (
        <DashboardModal simState={effectiveState} onClose={handleClose} />
      )}
    </>
  );
}

export default DashboardButton;