// roblox-up-and-coming-forecast-01 forecaster — Monte-Carlo rank diffusion.
// Reads /input/task.json + history.ndjson, writes /output/forecast.json.
// Node 22, stdlib only. Fully deterministic: fixed-seed PRNG, no wall clock,
// no network, fixed iteration order. Env overrides (MAKEFUN_INPUT_DIR /
// MAKEFUN_OUTPUT_DIR) per the kit; FF_VARIANT selects the parameter set
// (default "v6"; an unknown value throws — it never silently falls back).
// The shipped default is the chosen variant; see LANE-B report.
//
// Model (per game i, rank r_i in the Up-and-Coming chart):
//   r_i(h+1) = r_i(h) + drift(band) + vel_i + N(0, sigma_i),
//     h = 1..STEPS steps (STEPS = task.window.slots, each step
//     task.window.slot_minutes long; 72 x 60min on the practice task).
//   drift: ranks <=50 drift down slowly (+0.05 ranks/h), ranks 51+ drift up
//     (-0.07/h) — measured means of per-step rank changes in the practice
//     history (34 reads, 2026-10-30..2026-11-07).
//   vel_i: shrunk OLS slope of the game's rank vs time over the last 48h,
//     capped to [-0.35, +0.35] ranks/h (rank velocity autocorrelates only
//     ~0.20 step-to-step, so it is heavily shrunk; v3 ablates it to 0).
//   sigma_i: rank-step volatility per sqrt-hour, strongly rank-dependent
//     (measured: 0.5 for ranks 1-5, 1.1 for 6-10, 1.7 for 11-20, 2.9 for
//     21-50, 3.6 for 51+), blended 60/40 with the game's own measured step sd.
//   Exit hazard per hour, rank-dependent (measured per ~6.4h step: 0.6% top 10,
//     0% 11-20, 1.9% 21-50, 9.2% 51+), ramped up as the game nears the 28-day
//     age-out (details.created when present, else first-seen-in-history as a
//     lower bound on age). Exited games leave the simulation.
//   Games with hazard forced to 1 (age >= 28d at close) get zero probability
//     at every place: the MC kills them before hour 1 and the kernel blend is
//     zeroed for them too — they cannot appear anywhere in the window.
//   New games are NOT simulated (they enter at the bottom, ranks 80-99, and
//     reach the top 10 within 72h only ~11% of top-10 slots in walk-forward
//     cuts); their mass is the unnamed reserve U left at every place.
//   Variant choice (walk-forward, RULES log score; CORRECTED in the
//     2026-11-12 revision — the original "best 6-cut mean" paragraph mixed
//     denominators and the "best agreement" claim was false; see the
//     REVISION section of the LANE-B report):
//     TRUE 6-cut means (cuts 20/22/24/26/27/30): v6 -35.71 (SHIPPED),
//     v2 -35.87, v7 -36.10, v4 -36.46, v3 -36.80, v10 -37.12.
//     Common-4-cut means (20/24/27/30): v8 -34.89, v6 -34.97, v2 -35.02,
//     v9 -35.43, v7 -35.47, v4 -35.54, v3 -35.61, v5 -36.01, v10 -36.29,
//     starter -42.60 (v8's 0.08 edge over v6 on 4 cuts is noise, and v8
//     has no runs on cuts 22/26, the hardest cuts). v6 vs v2 is within
//     noise (3-3 cuts each); no variant is significantly best, and v6
//     does NOT have the best agreement (common-4: v3 49.90, v6 48.58).
//     Practice-window scores (10 public answer reads): v1 -33.67,
//     v3 -33.96, v6 -34.19, v8 -34.23, v2 -34.32, v10 -35.57. No variant
//     wins both proxies; v6 is kept as default as best true 6-cut WF mean
//     and 3rd on the practice window. Absolute WF scores do NOT predict
//     the final score (final history adds hourly house-recorder reads + details).
// Place probabilities are the MC top-10 occupancy averaged over all STEPS
// steps (exactly the scored quantity: the mean over the window's hourly
// reads), blended with a broad Gaussian kernel around the current rank,
// capped, floored, and renormalised so each place sums to 1 - U.
// Playing counts are deliberately NOT extrapolated: in the practice history
// the game's rank is not its playing level (#8 had the highest playing) and
// per-step log-playing growth barely predicts the next rank change
// (fraction improving 0.38-0.49 across growth buckets). Rank is the signal.

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// ---- parameter sets (variants) ----
type Params = {
  sims: number; seed: number;
  velWeight: number; sigmaScale: number;
  unnamed: number; pMax: number; pMin: number;
  blend: number; kernelBw: number;
  // occBlend: weight on the empirical per-place historical occupancy
  // frequency (fraction of history reads the game held that place). Only
  // v10 sets it; 0/absent leaves the other variants byte-identical.
  occBlend?: number;
};
const VARIANTS: Record<string, Params> = {
  // v1: persistence kernel only (no MC) — baseline-beater sanity check.
  v1: { sims: 0, seed: 20261112, velWeight: 0, sigmaScale: 1.0, unnamed: 0.20, pMax: 0.30, pMin: 0.002, blend: 1.0, kernelBw: 3.5 },
  // v2: full MC diffusion + shrunk velocity (default guess).
  v2: { sims: 4000, seed: 20261112, velWeight: 1, sigmaScale: 1.0, unnamed: 0.20, pMax: 0.30, pMin: 0.002, blend: 0.25, kernelBw: 3.5 },
  // v3: MC, no velocity (pumpfun v6 lesson: slopes extrapolate badly).
  v3: { sims: 4000, seed: 20261112, velWeight: 0, sigmaScale: 1.0, unnamed: 0.20, pMax: 0.30, pMin: 0.002, blend: 0.25, kernelBw: 3.5 },
  // v4: MC, wider diffusion.
  v4: { sims: 4000, seed: 20261112, velWeight: 1, sigmaScale: 1.5, unnamed: 0.20, pMax: 0.30, pMin: 0.002, blend: 0.25, kernelBw: 3.5 },
  // v5: MC, large unnamed reserve + low cap (diffuse, pumpfun-style).
  v5: { sims: 4000, seed: 20261112, velWeight: 1, sigmaScale: 1.0, unnamed: 0.35, pMax: 0.22, pMin: 0.002, blend: 0.25, kernelBw: 3.5 },
  // v6: MC, tighter diffusion.
  v6: { sims: 4000, seed: 20261112, velWeight: 1, sigmaScale: 0.7, unnamed: 0.20, pMax: 0.30, pMin: 0.002, blend: 0.25, kernelBw: 3.5 },
  // v7: MC, no kernel blend (pure occupancy).
  v7: { sims: 4000, seed: 20261112, velWeight: 1, sigmaScale: 1.0, unnamed: 0.20, pMax: 0.30, pMin: 0.002, blend: 0.0, kernelBw: 3.5 },
  // v8: MC, small unnamed reserve (confident).
  v8: { sims: 4000, seed: 20261112, velWeight: 1, sigmaScale: 1.0, unnamed: 0.10, pMax: 0.30, pMin: 0.002, blend: 0.25, kernelBw: 3.5 },
  // v9: MC, heavier kernel blend + larger reserve.
  v9: { sims: 4000, seed: 20261112, velWeight: 1, sigmaScale: 1.0, unnamed: 0.28, pMax: 0.25, pMin: 0.002, blend: 0.40, kernelBw: 3.5 },
  // v10: v6 + 25% empirical place-occupancy blend (Attack 2 experiment:
  // blend MC output with per-place historical occupancy frequencies from
  // the INPUT history only — never from the answer reads).
  v10: { sims: 4000, seed: 20261112, velWeight: 1, sigmaScale: 0.7, unnamed: 0.20, pMax: 0.30, pMin: 0.002, blend: 0.25, kernelBw: 3.5, occBlend: 0.25 },
};
// Unknown FF_VARIANT throws: a typo'd A/B probe must never silently run v6.
const variantName = process.env.FF_VARIANT ?? 'v6';
if (!(variantName in VARIANTS)) {
  throw new Error(`Unknown FF_VARIANT ${JSON.stringify(variantName)} (known: ${Object.keys(VARIANTS).join(', ')})`);
}
const P = VARIANTS[variantName]!;

// ---- deterministic PRNG (mulberry32) + Box-Muller ----
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(P.seed);
let spare: number | null = null;
function gauss(): number {
  if (spare !== null) { const v = spare; spare = null; return v; }
  let u = 0, v = 0;
  while (u === 0) u = rand();
  v = rand();
  const m = Math.sqrt(-2 * Math.log(u));
  spare = m * Math.sin(2 * Math.PI * v);
  return m * Math.cos(2 * Math.PI * v);
}

// ---- measured calibration (practice history, see header) ----
function bandSigma(rank: number): number {
  if (rank <= 5) return 0.5;
  if (rank <= 10) return 1.1;
  if (rank <= 20) return 1.7;
  if (rank <= 50) return 2.9;
  return 3.6;
}
function bandDrift(rank: number): number {
  return rank <= 50 ? 0.05 : -0.07;
}
function bandHazard(rank: number): number {
  if (rank <= 10) return 0.001;
  if (rank <= 20) return 0.0008;
  if (rank <= 50) return 0.003;
  return 0.014;
}

// ---- input ----
interface ChartRow { rank: number; universe_id: number; name?: string; playing?: number | null; }
interface Snap {
  at: string;
  up_and_coming: ChartRow[];
  details?: Record<string, { created?: string | null } | null> | null;
}
const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as {
  places: number; output: string;
  window?: { slots?: number; slot_minutes?: number; from?: string; until?: string };
};
// Horizon from task.json (RULES: "Read the window from task.json rather than
// writing it into your code"). Defaults 72 x 60min match the practice task;
// on a 60-minute slot the per-step scaling below is exactly 1, so output on
// the practice task.json is byte-identical to the hardcoded-72 version.
const STEPS = task.window?.slots ?? 72;
const HOURS_PER_STEP = (task.window?.slot_minutes ?? 60) / 60;
const lines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8').split('\n').filter((l) => l !== '');
const snaps = lines.map((l) => JSON.parse(l) as Snap);
const t0 = Date.parse(snaps[0]!.at);
const tLast = Date.parse(snaps[snaps.length - 1]!.at);
const hours = (s: Snap): number => (Date.parse(s.at) - t0) / 3600000;

// Per-game observation series.
interface Obs { t: number; rank: number; }
const series = new Map<number, Obs[]>();
const firstSeen = new Map<number, number>();
for (const s of snaps) {
  const t = hours(s);
  for (const r of s.up_and_coming) {
    if (!series.has(r.universe_id)) { series.set(r.universe_id, []); firstSeen.set(r.universe_id, t); }
    series.get(r.universe_id)!.push({ t, rank: r.rank });
  }
}
// Latest known creation date per game (house-recorder details; null in the
// archive-only practice history).
const created = new Map<number, number>();
for (const s of snaps) {
  if (!s.details) continue;
  for (const [uid, d] of Object.entries(s.details)) {
    if (d && d.created) created.set(Number(uid), Date.parse(d.created));
  }
}

const lastChart = [...snaps[snaps.length - 1]!.up_and_coming].sort((a, b) => a.rank - b.rank);
const n = lastChart.length;
const ids = lastChart.map((r) => r.universe_id);
const r0 = lastChart.map((r) => r.rank);

// Per-game parameters.
const sigma = new Float64Array(n);
const drift = new Float64Array(n);
const hazard = new Float64Array(n);
for (let i = 0; i < n; i++) {
  const uid = ids[i]!;
  const obs = series.get(uid)!;
  const bSig = bandSigma(r0[i]!);
  // Own step volatility, shrunk toward the band value.
  let own = NaN;
  if (obs.length >= 4) {
    const steps: number[] = [];
    for (let j = 1; j < obs.length; j++) {
      const dt = obs[j]!.t - obs[j - 1]!.t;
      if (dt > 0) steps.push((obs[j]!.rank - obs[j - 1]!.rank) / Math.sqrt(dt));
    }
    if (steps.length >= 3) {
      const m = steps.reduce((a, b) => a + b, 0) / steps.length;
      const v = steps.reduce((a, b) => a + (b - m) * (b - m), 0) / steps.length;
      own = Math.sqrt(v);
    }
  }
  let sig = bSig;
  if (Number.isFinite(own)) {
    const clamped = Math.min(Math.max(own, 0.3 * bSig), 2.5 * bSig);
    sig = 0.6 * bSig + 0.4 * clamped;
  }
  sigma[i] = sig * P.sigmaScale;
  // Shrunk OLS rank velocity over the last 48h.
  let vel = 0;
  const recent = obs.filter((o) => (hours(snaps[snaps.length - 1]!) - o.t) <= 48);
  if (recent.length >= 3) {
    const mt = recent.reduce((a, o) => a + o.t, 0) / recent.length;
    const mr = recent.reduce((a, o) => a + o.rank, 0) / recent.length;
    let num = 0, den = 0;
    for (const o of recent) { num += (o.t - mt) * (o.rank - mr); den += (o.t - mt) * (o.t - mt); }
    if (den > 0) {
      const slope = num / den;
      const shrink = recent.length / (recent.length + 4);
      vel = Math.min(Math.max(slope * shrink, -0.35), 0.35);
    }
  }
  drift[i] = bandDrift(r0[i]!) + P.velWeight * vel;
  // Age-out hazard ramp: 28-day eligibility window.
  let hz = bandHazard(r0[i]!);
  const c = created.get(uid);
  if (c !== undefined) {
    const ageDays = (tLast - c) / 86400000;
    if (ageDays >= 28) hz = 1;
    else if (ageDays > 21) hz *= 1 + 20 * ((ageDays - 21) / 7);
  } else {
    const ageLbDays = (hours(snaps[snaps.length - 1]!) - (firstSeen.get(uid) ?? 0)) / 24;
    if (ageLbDays > 21) hz *= 1 + 8 * ((ageLbDays - 21) / 7);
  }
  hazard[i] = Math.min(hz, 1);
}

// ---- Monte-Carlo over STEPS steps (window from task.json) ----
const PLACES = task.places;
const counts = new Float64Array(n * PLACES); // counts[i*PLACES + (k-1)]
if (P.sims > 0) {
  const ranks = new Float64Array(n);
  const alive = new Uint8Array(n);
  const order = new Array<number>(n);
  for (let s = 0; s < P.sims; s++) {
    for (let i = 0; i < n; i++) { ranks[i] = r0[i]!; alive[i] = 1; order[i] = i; }
    for (let h = 0; h < STEPS; h++) {
      for (let i = 0; i < n; i++) {
        if (!alive[i]) continue;
        // Per-step hazard/drift/sigma scale with step length; at 60min the
        // factors are exactly 1, so the practice output is unchanged.
        const hzStep = HOURS_PER_STEP === 1 ? hazard[i]! : 1 - Math.pow(1 - hazard[i]!, HOURS_PER_STEP);
        if (rand() < hzStep) { alive[i] = 0; continue; }
        ranks[i] = Math.max(0.5, ranks[i]! + drift[i]! * HOURS_PER_STEP + gauss() * sigma[i]! * Math.sqrt(HOURS_PER_STEP));
      }
      // Top 10 of this simulated hour.
      const idx: number[] = [];
      for (let i = 0; i < n; i++) if (alive[i]) idx.push(i);
      idx.sort((a, b) => ranks[a]! - ranks[b]!);
      const top = Math.min(PLACES, idx.length);
      for (let k = 0; k < top; k++) counts[idx[k]! * PLACES + k] += 1;
    }
  }
}
const denom = P.sims > 0 ? P.sims * STEPS : 1;

// ---- Empirical per-place occupancy frequencies (v10 only) ----
// occ[i][k] = fraction of history reads in which game i held place k+1.
// Computed from the INPUT history only. Guarded so every other variant
// skips the computation entirely and keeps its exact code path.
const occBlend = P.occBlend ?? 0;
const occ = new Float64Array(n * PLACES);
if (occBlend > 0) {
  const idxOf = new Map<number, number>();
  ids.forEach((uid, i) => idxOf.set(uid, i));
  for (const s of snaps) {
    for (const r of s.up_and_coming) {
      if (r.rank < 1 || r.rank > PLACES) continue;
      const i = idxOf.get(r.universe_id);
      if (i !== undefined) occ[i * PLACES + (r.rank - 1)] += 1 / snaps.length;
    }
  }
}

// ---- Blend with a Gaussian kernel around the current rank ----
const probs: Array<{ universe_id: number; place: number; p: number }> = [];
const kernelDen = new Float64Array(PLACES);
const kernelW = new Float64Array(n * PLACES);
for (let k = 0; k < PLACES; k++) {
  for (let i = 0; i < n; i++) {
    const w = Math.exp(-0.5 * Math.pow((r0[i]! - (k + 1)) / P.kernelBw, 2));
    kernelW[i * PLACES + k] = w;
    kernelDen[k] += w;
  }
}
const placeScore = new Float64Array(n); // for top10: sum_k p*(PLACES+1-k)
for (let k = 0; k < PLACES; k++) {
  const raw: number[] = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const mc = P.sims > 0 ? counts[i * PLACES + k] / denom : 0;
    const ker = kernelDen[k]! > 0 ? kernelW[i * PLACES + k]! / kernelDen[k]! : 0;
    let base = (1 - P.blend) * mc + P.blend * ker;
    // v10: blend in the empirical occupancy frequency for this place.
    if (occBlend > 0) base = (1 - occBlend) * base + occBlend * occ[i * PLACES + k]!;
    // Age-dead games (hazard == 1, i.e. age >= 28d at close) cannot appear in
    // the window: zero the whole row, kernel included (the MC already gives
    // them 0 — they exit before hour 1). Fixes the 0.1722 phantom mass.
    raw[i] = hazard[i]! >= 1 ? 0 : Math.min(base, P.pMax);
  }
  const sum = raw.reduce((a, b) => a + b, 0);
  const target = 1 - P.unnamed;
  const scale = sum > target ? target / sum : 1;
  for (let i = 0; i < n; i++) {
    const p = raw[i]! * scale;
    if (p >= P.pMin) {
      probs.push({ universe_id: ids[i]!, place: k + 1, p: Math.round(p * 1e6) / 1e6 });
      placeScore[i] += p * (PLACES + 1 - (k + 1));
    }
  }
}
// Deterministic row order: by place, then p desc, then universe_id.
probs.sort((a, b) => a.place - b.place || b.p - a.p || a.universe_id - b.universe_id);

// Age-dead games are excluded from top10 (their placeScore is 0 anyway).
const topIdx = [...Array(n).keys()].filter((i) => hazard[i]! < 1).sort((a, b) => placeScore[b]! - placeScore[a]! || r0[a]! - r0[b]!);
const top10 = topIdx.slice(0, PLACES).map((i) => ids[i]!);
// Fallback (should not happen): fill from the last chart order.
for (const uid of ids) {
  if (top10.length >= PLACES) break;
  if (!top10.includes(uid)) top10.push(uid);
}

mkdirSync(OUTPUT_DIR, { recursive: true });
writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities: probs, top10 })}\n`);
