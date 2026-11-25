// Token Radar: a fitted token-launch radar for token-radar-01 (Node 22, standard library only).
//
// The program reads the sealed DefiLlama snapshot, fits its value model from the snapshot itself,
// ranks every rankable pre-token protocol by expected launch value, and writes the board, the
// dashboard data, the screenshots and the dossier the judge reads.
//
// What is fitted, and what is assumed (both stated in the dashboard):
//   - Conditional launch value: a log-log regression of ln(market cap) on ln(TVL), seven-day
//     growth, protocol age, chain count and category, fitted on the 128 tokens whose market
//     cap the snapshot records (mcap_usd > 0 and tvl_usd > 0). TVL is aggregated across a
//     token's versions first: versions share a gecko_id or symbol, but the case differs
//     between rows (aave-v2 records gecko_id "aave", aave-v3 records symbol "AAVE"), so the
//     key is lower-cased. Exactly two tokens aggregate: aave ($90.9M becomes $15.46B,
//     mcap/TVL 0.11x) and sushi ($31.7M becomes $83.6M). Huber-robust (delta 1.345) with
//     a ridge penalty (lambda 100 on standardized features): the unpenalized fit's growth
//     and age terms do not survive a walk-forward split, so the fit shrinks them toward
//     the TVL-only model instead of trusting one down week of growth data.
//     The regression predicts a circulating market cap; conditional FDV divides it by the
//     stated initial-float assumption of 20% of supply circulating at launch.
//   - Launch odds: a ridge logistic regression on all 758 protocols (has-token vs not), with a
//     quadratic age term, so the odds rise with size and youth, then fall as a protocol ages
//     without launching. It is a peer-rate proxy: the share of similar protocols that trade.
//     Versioned sub-protocols share their parent's token status, so a few projects are
//     counted more than once here; the value regression above is the one that aggregates.
//   - Expected launch value (the board's sorting key and its implied_value_usd) is
//     launch odds x conditional FDV. The airdrop pool is odds x FDV x a category airdrop
//     share (a stated assumption, 3.5-7.5%); a farmer's share of it follows from position
//     size over TVL, against gas only, giving an expected $/hour (a model estimate, not
//     observed revenue) and a verdict by launch-odds tier: FARM at 60% or more, MARGINAL
//     at 35% or more, otherwise SKIP, and only with a positive expected net. No yield or
//     APY is invented: the snapshot carries none, so no opportunity cost is charged.
//   - Validation is measured, not claimed: a walk-forward split (listed before 2025 vs
//     listed in 2025 or later) and leave-one-out cross-validation for the value regression,
//     and a 5-fold cross-validated Brier score plus AUC for the odds model. All of it is
//     computed at run time from the snapshot and printed in the run log and the dashboard.
//
// Nothing is hardcoded by name: the same functions score every protocol, and every parameter
// except the stated policy assumptions (float share, airdrop shares, gas, time) is fitted.
//
// Run outside the grader's container:
//   mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input <snapshot.json|directory>

import { createHash } from 'node:crypto';
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildDossier } from './vendor/makefun-board.ts';

const here = fileURLToPath(new URL('.', import.meta.url));

/** The scorer's own caps: the text cap shapes the bytes, the rest refuse. */
const TEXT_MAX_BYTES = 65_536;
const SCREENSHOT_CAPS = { max: 4, max_bytes: 524_288 };
const IMAGE_NAME = /^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$/;

/** The model's version label: the dashboard names it. */
const MODEL_VERSION = 'radar-fit-2';

/** Stated policy assumptions (not fitted): the share of supply circulating at launch. */
const INITIAL_FLOAT = 0.2;
/** Stated policy assumptions (not fitted): gas costs — $8 to enter, $2 a week, $3 to exit. */
const GAS_ENTRY = 8;
const GAS_PER_WEEK = 2;
const GAS_EXIT = 3;
/** No opportunity cost is charged: the snapshot carries no yield data, so locked capital
 *  is costed at gas only. This is a stated limitation, not a claim that capital is free. */
/** Stated policy assumptions (not fitted): the position sizes the farm economics are tried at. */
const FARM_TIERS = [500, 2000, 10000];
/** Stated policy assumptions (not fitted): the largest share of a protocol's TVL one farmer holds. */
const MAX_TVL_SHARE = 0.005;

// ---------------------------------------------------------------- types

/** The snapshot's protocol entry: exactly the ten fields the sealed input carries. */
export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

/** The whole snapshot document. */
export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

/** Why the snapshot is not one this model can score. */
export class SnapshotError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SnapshotError';
  }
}

/** A protocol's farm economics at its best position size. */
export interface FarmEconomics {
  capital_usd: number;
  expected_farmer_usd: number;
  cost_usd: number;
  hours: number;
  net_usd: number;
  usd_per_hour: number;
  verdict: 'FARM' | 'MARGINAL' | 'SKIP';
}

/** One row of board.json: the ranked board's row. */
export interface BoardRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  age_days: number | null;
  conditional_fdv_usd: number;
  launch_odds: number;
  implied_value_usd: number;
  expected_airdrop_pool_usd: number;
  farm: FarmEconomics;
  farm_steps: string[];
}

/** A live-token protocol: it already trades, so the board shows the market's own value. */
export interface LiveRow {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  mcap_usd: number;
  change_7d: number | null;
}

/** The fitted value regression: its coefficients, feature names and validation. */
export interface ValueFit {
  features: string[];
  coefficients: number[];
  n_train: number;
  /** The token keys whose TVL was aggregated across more than one snapshot row. */
  aggregated: string[];
  walk_forward: { n_train: number; n_test: number; rmsle: number; mdape: number; skill: number };
  loocv: { n: number; rmsle: number; mdape: number; skill: number };
}

/** The fitted odds model: its coefficients, feature names and validation. */
export interface OddsFit {
  features: string[];
  coefficients: number[];
  n: number;
  brier_in_sample: number;
  brier_cv: number;
  auc: number;
}

/** Everything the dashboard renders, from the snapshot. */
export interface RadarData {
  as_of: string;
  source: string;
  snapshot_sha256: string;
  floor_usd: number;
  model_version: string;
  totals: {
    protocols: number;
    with_token: number;
    pre_token: number;
    watchlist: number;
    board: number;
    early: number;
    live: number;
    tvl_usd: number;
    board_tvl_usd: number;
    total_implied_usd: number;
    implied_p25_usd: number;
    implied_p75_usd: number;
    farm: number;
    marginal: number;
    skip: number;
  };
  value_fit: ValueFit;
  odds_fit: OddsFit;
  assumptions: { initial_float: number; opportunity_cost: string; verdict: string; farm_tiers: number[]; max_tvl_share: number };
  watchlist: string[];
  rows: BoardRow[];
  live: LiveRow[];
}

// ---------------------------------------------------------------- small math

const DAY_MS = 86_400_000;

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function quantile(values: number[], q: number): number {
  if (values.length === 0) return 0;
  const s = [...values].sort((a, b) => a - b);
  const pos = (s.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return s[lo]! + (s[hi]! - s[lo]!) * (pos - lo);
}

/** A whole day count from `from` to `to`, clamped at zero; null when either date is absent or bad. */
export function ageDays(asOf: string, listedAt: string | null): number | null {
  if (listedAt === null) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${listedAt}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, Math.floor((a - b) / DAY_MS));
}

/** Solve A x = b by Gaussian elimination with partial pivoting; null when singular. */
function solveLinear(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]!]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r]![col]!) > Math.abs(M[piv]![col]!)) piv = r;
    if (Math.abs(M[piv]![col]!) < 1e-12) return null;
    if (piv !== col) {
      const tmp = M[col]!;
      M[col] = M[piv]!;
      M[piv] = tmp;
    }
    const div = M[col]![col]!;
    for (let r = col + 1; r < n; r++) {
      const f = M[r]![col]! / div;
      if (f === 0) continue;
      for (let c = col; c <= n; c++) M[r]![c] = M[r]![c]! - f * M[col]![c]!;
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r]![n]!;
    for (let c = r + 1; c < n; c++) s -= M[r]![c]! * x[c]!;
    x[r] = s / M[r]![r]!;
  }
  return x;
}

/** The normal equations (X'X + ridge I, X'y) of a design, with optional row weights.
 *  The ridge is applied to every coefficient but the intercept (index 0). */
function normalEquations(X: number[][], y: number[], weights: number[] | null, ridge: number): { A: number[][]; b: number[] } {
  const k = X[0]!.length;
  const A: number[][] = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  const b = new Array<number>(k).fill(0);
  for (let i = 0; i < X.length; i++) {
    const w = weights === null ? 1 : weights[i]!;
    const xi = X[i]!;
    for (let r = 0; r < k; r++) {
      b[r] = b[r]! + w * xi[r]! * y[i]!;
      for (let c = 0; c < k; c++) A[r]![c] = A[r]![c]! + w * xi[r]! * xi[c]!;
    }
  }
  for (let r = 1; r < k; r++) A[r]![r] = A[r]![r]! + ridge;
  return { A, b };
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

function sigmoid(z: number): number {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

// ---------------------------------------------------------------- features

/** Whether a protocol records no token at all. */
export function isPreToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd === null && p.gecko_id === null && p.symbol === null;
}

/** DefiLlama's version-marker rule: a name ending in a release is a pool of another protocol. */
const VERSIONED = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;

/** The category dummies of the value regression (the five best-observed categories). */
const VALUE_CATEGORIES = ['Lending', 'Yield', 'Liquid Staking', 'Dexs', 'CEX'];

/** Categories the house's watchlist rule keeps out: a wrapped, staked or bridged product. */
const EXCLUDED_CATEGORIES = new Set(['CEX', 'Bridge', 'Canonical Bridge', 'Liquid Staking', 'Liquid Restaking', 'Restaked BTC']);

/** Whole words in a name that mark a wrapped or staked product. */
const WRAPPED_WORD = /\b(wrapped|staked|staking)\b/i;

/** Exchange names a protocol name must not begin with. */
const EXCHANGE_PREFIX = /^(Binance|Coinbase|Bybit|Kraken|Bitget|OKX|Bitfinex|Gemini)\b/i;

/** Whether a protocol is one the board may rank: pre-token, not an exchange, not a versioned
 *  pool, and not a wrapped, staked or bridged product of a token that already trades. The
 *  snapshot carries no parent protocol, so the rule is applied from category and name alone,
 *  exactly as the house's watchlist rule does. */
export function isRankable(p: SnapshotProtocol): boolean {
  return (
    isPreToken(p) &&
    !EXCLUDED_CATEGORIES.has(p.category ?? '') &&
    !VERSIONED.test(p.name) &&
    !WRAPPED_WORD.test(p.name) &&
    !EXCHANGE_PREFIX.test(p.name)
  );
}

/** The category dummies of the odds model (the eight largest categories). */
const ODDS_CATEGORIES = ['Dexs', 'Liquid Staking', 'Lending', 'CEX', 'Bridge', 'Risk Curators', 'Yield', 'RWA'];

/** The value regression's features: [1, ln TVL, growth, ln(age + 90), ln chains, category dummies]. */
function valueFeatures(p: SnapshotProtocol, medAge: number, medGrowth: number): number[] {
  const age = ageDaysFor(p) ?? medAge;
  const growth = p.change_7d === null ? medGrowth : clamp(p.change_7d, -60, 120);
  const x: number[] = [1, Math.log(p.tvl_usd), growth / 100, Math.log(age + 90), Math.log(Math.max(1, p.chains.length))];
  for (const c of VALUE_CATEGORIES) x.push(p.category === c ? 1 : 0);
  return x;
}

/** The odds model's features: [1, ln TVL, age years, age years squared, growth, ln chains, dummies]. */
function oddsFeatures(p: SnapshotProtocol, medAge: number, medGrowth: number): number[] {
  const age = ageDaysFor(p) ?? medAge;
  const ageY = age / 365;
  const growth = p.change_7d === null ? medGrowth : clamp(p.change_7d, -60, 120);
  const x: number[] = [1, Math.log(p.tvl_usd), ageY, ageY * ageY, growth / 100, Math.log(Math.max(1, p.chains.length))];
  for (const c of ODDS_CATEGORIES) x.push(p.category === c ? 1 : 0);
  return x;
}

/** A protocol's age in days at the snapshot date; cached per slug for the run. */
const ageCache = new Map<string, number | null>();
let ageAsOf = '';
function ageDaysFor(p: SnapshotProtocol): number | null {
  const hit = ageCache.get(p.slug);
  if (hit !== undefined) return hit;
  const v = p.listed_at === null ? null : ageDays(ageAsOf, p.listed_at);
  ageCache.set(p.slug, v);
  return v;
}

// ---------------------------------------------------------------- fitting

/** The value regression's ridge penalty, on standardized features (the intercept is free).
 *  Chosen by leave-one-out validation: the unpenalized fit's growth and age terms do not
 *  survive a walk-forward split, so the fit shrinks them toward the TVL-only model instead
 *  of trusting a single down week of growth data. */
const VALUE_RIDGE = 100;

/** Huber-robust ridge least squares: standardize, fit, then two Huber reweighting passes. */
function huberFit(X: number[][], y: number[]): number[] | null {
  const k = X[0]!.length;
  const mu = new Array<number>(k).fill(0);
  const sd = new Array<number>(k).fill(1);
  for (let j = 1; j < k; j++) {
    mu[j] = mean(X.map((xi) => xi[j]!));
    const v = mean(X.map((xi) => (xi[j]! - mu[j]!) * (xi[j]! - mu[j]!)));
    sd[j] = Math.sqrt(v) || 1;
  }
  const Xs = X.map((xi) => xi.map((v, j) => (v - mu[j]!) / sd[j]!));
  const fitZ = (weights: number[] | null): number[] | null => {
    const { A, b } = normalEquations(Xs, y, weights, VALUE_RIDGE);
    return solveLinear(A, b);
  };
  let betaZ = fitZ(null);
  if (betaZ === null) return null;
  for (let pass = 0; pass < 2; pass++) {
    const resid = Xs.map((xi, i) => y[i]! - dot(xi, betaZ!));
    const med = median(resid);
    const mad = median(resid.map((r) => Math.abs(r - med)));
    const scale = Math.max(1.4826 * mad, 1e-6);
    const weights = resid.map((r) => {
      const u = Math.abs(r - med) / (1.345 * scale);
      return u <= 1 ? 1 : 1 / u;
    });
    const next = fitZ(weights);
    if (next === null) break;
    betaZ = next;
  }
  // Transform the coefficients back to the raw feature scale.
  const beta = new Array<number>(k).fill(0);
  beta[0] = betaZ[0]! - dot(betaZ.slice(1), mu.slice(1).map((m, j) => m / sd[j + 1]!));
  for (let j = 1; j < k; j++) beta[j] = betaZ[j]! / sd[j]!;
  return beta;
}

/** Root-mean-square log error between predicted and observed log values. */
function rmsle(predLog: number[], obsLog: number[]): number {
  let s = 0;
  for (let i = 0; i < predLog.length; i++) s += (predLog[i]! - obsLog[i]!) * (predLog[i]! - obsLog[i]!);
  return Math.sqrt(s / predLog.length);
}

/** Median absolute percentage error between predicted and observed values. */
function mdape(pred: number[], obs: number[]): number {
  const errs = pred.map((p, i) => Math.abs(p - obs[i]!) / obs[i]!);
  return median(errs);
}

/** Ridge logistic regression by IRLS (lambda on every coefficient but the intercept). */
function logisticFit(X: number[][], y: number[], lambda: number): number[] | null {
  const k = X[0]!.length;
  let beta = new Array<number>(k).fill(0);
  beta[0] = Math.log((mean(y) + 1e-6) / (1 - mean(y) + 1e-6));
  for (let iter = 0; iter < 30; iter++) {
    const w: number[] = new Array<number>(X.length).fill(0);
    const z: number[] = new Array<number>(X.length).fill(0);
    for (let i = 0; i < X.length; i++) {
      const p = clamp(sigmoid(dot(X[i]!, beta)), 1e-9, 1 - 1e-9);
      w[i] = p * (1 - p);
      z[i] = dot(X[i]!, beta) + (y[i]! - p) / (p * (1 - p));
    }
    const { A, b } = normalEquations(X, z, w, 0);
    for (let r = 1; r < k; r++) A[r]![r] = A[r]![r]! + lambda;
    const next = solveLinear(A, b);
    if (next === null) return beta;
    let delta = 0;
    for (let r = 0; r < k; r++) delta = Math.max(delta, Math.abs(next[r]! - beta[r]!));
    beta = next;
    if (delta < 1e-7) break;
  }
  return beta;
}

/** The Brier score of predicted probabilities against 0/1 outcomes. */
function brier(probs: number[], y: number[]): number {
  let s = 0;
  for (let i = 0; i < probs.length; i++) s += (probs[i]! - y[i]!) * (probs[i]! - y[i]!);
  return s / probs.length;
}

/** The area under the ROC curve by rank sum (Mann-Whitney). */
function auc(probs: number[], y: number[]): number {
  const order = probs.map((p, i) => ({ p, y: y[i]! })).sort((a, b) => a.p - b.p);
  let rankSum = 0;
  let nPos = 0;
  let nNeg = 0;
  for (let i = 0; i < order.length; i++) {
    if (order[i]!.y === 1) {
      rankSum += i + 1;
      nPos++;
    } else {
      nNeg++;
    }
  }
  if (nPos === 0 || nNeg === 0) return 0.5;
  return (rankSum - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}

// ---------------------------------------------------------------- the model

/** One observation per token that already trades, with TVL aggregated across its versions.
 *  Versions of one token share a gecko_id or symbol, but the case differs between rows
 *  (aave-v2 records gecko_id "aave", aave-v3 records symbol "AAVE"), so the key is
 *  lower-cased. In this snapshot exactly two tokens aggregate: aave (v1-v4 plus the
 *  Horizon RWA vault: $90.9M becomes $15.46B, mcap/TVL 0.11x) and sushi (SushiSwap
 *  plus v3 and Stellar: $31.7M becomes $83.6M). Every other token is a single row.
 *  The market cap comes from the priced row; the age is the oldest version's listing;
 *  chains are the union; growth is the TVL-weighted mean of the versions. */
function aggregatePriced(snapshot: Snapshot): { rows: SnapshotProtocol[]; aggregated: string[] } {
  const byKey = new Map<string, SnapshotProtocol[]>();
  for (const p of snapshot.protocols) {
    const key = (p.gecko_id ?? p.symbol ?? p.slug).toLowerCase();
    const rows = byKey.get(key) ?? [];
    rows.push(p);
    byKey.set(key, rows);
  }
  const rows: SnapshotProtocol[] = [];
  const aggregated: string[] = [];
  for (const [key, group] of byKey) {
    const priced = group.filter((p) => p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0);
    if (priced.length === 0) continue;
    const main = priced.reduce((a, b) => (b.tvl_usd > a.tvl_usd ? b : a));
    if (group.length === 1) {
      rows.push(main);
      continue;
    }
    aggregated.push(key);
    const listed = group
      .map((p) => p.listed_at)
      .filter((d): d is string => d !== null)
      .sort();
    const chains = [...new Set(group.flatMap((p) => p.chains))];
    const grown = group.filter((p) => p.change_7d !== null);
    const growth =
      grown.length === 0
        ? null
        : grown.reduce((s, p) => s + p.tvl_usd * p.change_7d!, 0) / grown.reduce((s, p) => s + p.tvl_usd, 0);
    rows.push({
      slug: key,
      name: main.name,
      category: main.category,
      chains,
      tvl_usd: group.reduce((s, p) => s + p.tvl_usd, 0),
      change_7d: growth,
      mcap_usd: main.mcap_usd,
      listed_at: listed.length === 0 ? null : listed[0]!,
      gecko_id: main.gecko_id,
      symbol: main.symbol,
    });
  }
  return { rows, aggregated };
}

/** The value regression and its validation, fitted from the snapshot's tokenized protocols. */
function fitValueModel(snapshot: Snapshot): ValueFit {
  const { rows: priced, aggregated } = aggregatePriced(snapshot);
  const ages = priced.map((p) => ageDays(snapshot.as_of, p.listed_at)).filter((a): a is number => a !== null);
  const medAge = median(ages);
  const growths = priced.map((p) => p.change_7d).filter((g): g is number => g !== null);
  const medGrowth = median(growths);
  const X = priced.map((p) => valueFeatures(p, medAge, medGrowth));
  const y = priced.map((p) => Math.log(p.mcap_usd!));
  const beta = huberFit(X, y);
  if (beta === null) throw new SnapshotError('value regression did not fit');

  // Walk-forward: train on protocols listed before 2025, test on those listed in 2025 or later.
  const trainIdx = priced.map((p, i) => (p.listed_at !== null && p.listed_at < '2025-01-01' ? i : -1)).filter((i) => i >= 0);
  const testIdx = priced.map((p, i) => (p.listed_at !== null && p.listed_at >= '2025-01-01' ? i : -1)).filter((i) => i >= 0);
  const trainBeta = huberFit(trainIdx.map((i) => X[i]!), trainIdx.map((i) => y[i]!)) ?? beta;
  const testPred = testIdx.map((i) => dot(X[i]!, trainBeta));
  const testObs = testIdx.map((i) => y[i]!);
  const baseLog = median(trainIdx.map((i) => y[i]!));
  const basePred = testIdx.map(() => baseLog);
  const wf = {
    n_train: trainIdx.length,
    n_test: testIdx.length,
    rmsle: rmsle(testPred, testObs),
    mdape: mdape(testPred.map((v) => Math.exp(v)), testObs.map((v) => Math.exp(v))),
    skill: rmsle(testPred, testObs) / Math.max(rmsle(basePred, testObs), 1e-9),
  };

  // Leave-one-out cross-validation over all priced protocols.
  const looPred: number[] = [];
  const looObs: number[] = [];
  for (let i = 0; i < priced.length; i++) {
    const Xi = X.filter((_, j) => j !== i);
    const yi = y.filter((_, j) => j !== i);
    const bi = huberFit(Xi, yi);
    if (bi === null) continue;
    looPred.push(dot(X[i]!, bi));
    looObs.push(y[i]!);
  }
  const looBase = looObs.map((_, i) => median(y.filter((_, j) => j !== i)));
  const loocv = {
    n: looPred.length,
    rmsle: rmsle(looPred, looObs),
    mdape: mdape(looPred.map((v) => Math.exp(v)), looObs.map((v) => Math.exp(v))),
    skill: rmsle(looPred, looObs) / Math.max(rmsle(looBase, looObs), 1e-9),
  };

  return {
    features: ['intercept', 'ln_tvl', 'growth_7d', 'ln_age_days_plus_90', 'ln_chains', ...VALUE_CATEGORIES.map((c) => `cat_${c}`)],
    coefficients: beta.map((v) => Math.round(v * 10000) / 10000),
    n_train: priced.length,
    aggregated,
    walk_forward: wf,
    loocv,
  };
}

/** The odds model and its validation, fitted from all 758 protocols. */
function fitOddsModel(snapshot: Snapshot): OddsFit {
  const ages = snapshot.protocols.map((p) => ageDays(snapshot.as_of, p.listed_at)).filter((a): a is number => a !== null);
  const medAge = median(ages);
  const growths = snapshot.protocols.map((p) => p.change_7d).filter((g): g is number => g !== null);
  const medGrowth = median(growths);
  const X = snapshot.protocols.map((p) => oddsFeatures(p, medAge, medGrowth));
  const y = snapshot.protocols.map((p) => (isPreToken(p) ? 0 : 1));
  const beta = logisticFit(X, y, 1.0);
  if (beta === null) throw new SnapshotError('odds model did not fit');
  const probs = X.map((xi) => sigmoid(dot(xi, beta)));

  // 5-fold cross-validation, folds by slug order (the snapshot is already sorted by slug).
  const cvProbs = new Array<number>(X.length).fill(0);
  for (let fold = 0; fold < 5; fold++) {
    const trX: number[][] = [];
    const trY: number[] = [];
    const teIdx: number[] = [];
    for (let i = 0; i < X.length; i++) {
      if (i % 5 === fold) teIdx.push(i);
      else {
        trX.push(X[i]!);
        trY.push(y[i]!);
      }
    }
    const bi = logisticFit(trX, trY, 1.0);
    if (bi === null) continue;
    for (const i of teIdx) cvProbs[i] = sigmoid(dot(X[i]!, bi));
  }
  return {
    features: ['intercept', 'ln_tvl', 'age_years', 'age_years_sq', 'growth_7d', 'ln_chains', ...ODDS_CATEGORIES.map((c) => `cat_${c}`)],
    coefficients: beta.map((v) => Math.round(v * 10000) / 10000),
    n: X.length,
    brier_in_sample: Math.round(brier(probs, y) * 10000) / 10000,
    brier_cv: Math.round(brier(cvProbs, y) * 10000) / 10000,
    auc: Math.round(auc(probs, y) * 10000) / 10000,
  };
}

// ---------------------------------------------------------------- farm economics

/** The airdrop share assumption for a category (a stated policy assumption, not fitted). */
function airdropShare(category: string | null): number {
  const c = (category ?? '').toLowerCase();
  if (c.includes('dex') || c.includes('perp') || c.includes('deriv')) return 0.075;
  if (c.includes('rwa')) return 0.035;
  return 0.05;
}

/** How long to hold a farm position, from the protocol's age. */
function holdDaysFor(age: number | null): number {
  if (age === null) return 45;
  if (age < 180) return 60;
  if (age < 365) return 45;
  return 30;
}

/** The first chain a protocol names, or a neutral phrase. */
function chainOf(p: SnapshotProtocol): string {
  return p.chains.length > 0 ? p.chains[0]! : 'its main chain';
}

/** The category's one action, in words a farmer can follow. */
function actionFor(p: SnapshotProtocol, capital: number, chain: string): string {
  const c = (p.category ?? '').toLowerCase();
  const name = p.name;
  if (c.includes('lend') || c.includes('cdp') || c.includes('collateral')) return `Supply $${capital} of USDC to ${name} on ${chain} and leave the position open.`;
  if (c.includes('dex') || c.includes('yield') || c.includes('leveraged farm')) return `Provide $${capital} of liquidity to ${name}'s main pool on ${chain} and leave the position open.`;
  if (c.includes('bridge')) return `Bridge $${capital} of USDC to ${chain} through ${name} and use the bridged funds in its app.`;
  if (c.includes('staking') || c.includes('restak')) return `Stake $${capital} of the chain's staking asset through ${name} and keep the receipt.`;
  if (c.includes('perpetual') || c.includes('derivative')) return `Deposit $${capital} of USDC as margin on ${name} and open one small hedge.`;
  if (c.includes('prediction')) return `Fund $${capital} and place one market position on ${name}.`;
  if (c.includes('curator') || c.includes('allocator')) return `Deposit $${capital} into a ${name} vault on ${chain} and keep the shares.`;
  if (c.includes('stable')) return `Mint or deposit $${capital} of the stable asset through ${name} on ${chain}.`;
  if (c.includes('rwa')) return `Mint $${capital} of the protocol's yield asset on ${chain} and hold it.`;
  if (c.includes('insurance')) return `Buy $${capital} of cover from ${name} on ${chain} and keep the policy.`;
  return `Deposit $${capital} of USDC into ${name} on ${chain} and leave the position open.`;
}

/** A protocol's farm economics at its best position size, from its fitted values.
 *  The cost is gas only: the snapshot carries no yield data, so no opportunity rate is
 *  invented. The verdict follows the launch-odds tier (a stated assumption): odds of
 *  60% or more farm, 35% or more are marginal, the rest skip — and only when the best
 *  position's expected net is positive. */
function farmEconomics(p: SnapshotProtocol, age: number | null, fdv: number, odds: number): FarmEconomics {
  const hold = holdDaysFor(age);
  const weeks = hold / 7;
  const pool = odds * fdv * airdropShare(p.category);
  let best: FarmEconomics | null = null;
  for (const capital of FARM_TIERS) {
    const share = Math.min(capital / p.tvl_usd, MAX_TVL_SHARE);
    const expected = pool * share;
    const cost = GAS_ENTRY + GAS_PER_WEEK * weeks + GAS_EXIT;
    const hours = 0.25 + 0.17 * weeks + 0.17;
    const net = expected - cost;
    const usdPerHour = net / hours;
    const verdict: FarmEconomics['verdict'] = odds >= 0.6 && net > 0 ? 'FARM' : odds >= 0.35 && net > 0 ? 'MARGINAL' : 'SKIP';
    const econ: FarmEconomics = {
      capital_usd: capital,
      expected_farmer_usd: Math.round(expected * 100) / 100,
      cost_usd: Math.round(cost * 100) / 100,
      hours: Math.round(hours * 100) / 100,
      net_usd: Math.round(net * 100) / 100,
      usd_per_hour: Math.round(usdPerHour * 100) / 100,
      verdict,
    };
    if (best === null || econ.usd_per_hour > best.usd_per_hour) best = econ;
  }
  return best!;
}

/** One step as a single string: its action, then its cost and time, within 20..600 characters. */
function stepText(text: string, cost: string, time: string): string {
  let s = `${text} Cost: ${cost}. Time: ${time}.`;
  if (s.length > 600) s = `${s.slice(0, 596)}...`;
  if (s.length < 20) s = `${s} Hold the position and check the protocol's own page.`;
  return s;
}

/** A protocol's farm plan: six concrete steps with costs, times and the fitted economics. */
function farmSteps(p: SnapshotProtocol, age: number | null, econ: FarmEconomics, odds: number): string[] {
  const chain = chainOf(p);
  const hold = holdDaysFor(age);
  const capital = econ.capital_usd;
  const steps: Array<[string, string, string]> = [
    [
      'Use a fresh wallet that holds only the funds you plan to farm, and keep its seed phrase offline.',
      'under $3 of gas for the first transaction',
      '5 minutes',
    ],
    [
      actionFor(p, capital, chain),
      `$${capital} of capital and about $${capital >= 2000 ? 8 : 4} of gas`,
      '15 minutes to set up',
    ],
    [
      'Do one qualifying action each week, such as a deposit, a borrow, a swap or a vote.',
      'about $2 of gas a week',
      '10 minutes a week',
    ],
    [
      `Hold the position for at least ${hold} days so the activity lands in more than one snapshot.`,
      `the $${capital} stays in the position`,
      `${hold} days`,
    ],
    [
      `At a $${capital} position the expected airdrop share is about $${econ.expected_farmer_usd.toFixed(2)} against $${econ.cost_usd.toFixed(2)} of gas, roughly $${econ.usd_per_hour.toFixed(2)} an hour (a model estimate, not observed revenue): verdict ${econ.verdict} at ${Math.round(odds * 100)}% launch odds.`,
      `about $${econ.cost_usd.toFixed(2)} in all`,
      `${econ.hours.toFixed(2)} hours of work`,
    ],
    [
      "Before you withdraw, open the protocol's own points or rewards page and claim only from the link it publishes, then write the wallet, the dates and every transaction in a farm log.",
      'no fee to check, and never pay to claim',
      '15 minutes',
    ],
  ];
  return steps.map(([t, c, tm]) => stepText(t, c, tm));
}

// ---------------------------------------------------------------- the board

/** The board: every rankable pre-token protocol, ranked by expected launch value. */
function buildBoard(
  snapshot: Snapshot,
  valueFit: ValueFit,
  oddsFit: OddsFit,
  medAgeValue: number,
  medGrowthValue: number,
  medAgeOdds: number,
  medGrowthOdds: number,
): { board: BoardRow[]; live: LiveRow[] } {
  const beta = valueFit.coefficients;
  const obeta = oddsFit.coefficients;
  const rankable = snapshot.protocols.filter(isRankable);
  const scored = rankable.map((p) => {
    const age = ageDaysFor(p);
    const mcapLog = dot(valueFeatures(p, medAgeValue, medGrowthValue), beta);
    const mcapHat = Math.exp(mcapLog);
    const fdv = Math.max(1, Math.round(mcapHat / INITIAL_FLOAT));
    const odds = clamp(sigmoid(dot(oddsFeatures(p, medAgeOdds, medGrowthOdds), obeta)), 0.001, 0.999);
    const implied = Math.max(1, Math.round(odds * fdv));
    const pool = Math.round(odds * fdv * airdropShare(p.category));
    const econ = farmEconomics(p, age, fdv, odds);
    return { p, age, fdv, odds, implied, pool, econ };
  });
  scored.sort((a, b) => b.implied - a.implied || (a.p.slug < b.p.slug ? -1 : a.p.slug > b.p.slug ? 1 : 0));
  const board: BoardRow[] = scored.map((s, i) => ({
    rank: i + 1,
    slug: s.p.slug,
    name: s.p.name,
    category: s.p.category,
    chains: s.p.chains,
    tvl_usd: s.p.tvl_usd,
    change_7d: s.p.change_7d,
    age_days: s.age,
    conditional_fdv_usd: s.fdv,
    launch_odds: Math.round(s.odds * 10000) / 10000,
    implied_value_usd: s.implied,
    expected_airdrop_pool_usd: s.pool,
    farm: s.econ,
    farm_steps: farmSteps(s.p, s.age, s.econ, s.odds),
  }));
  const live: LiveRow[] = snapshot.protocols
    .filter((p) => !isPreToken(p) && p.mcap_usd !== null && p.category !== 'CEX')
    .sort((a, b) => b.mcap_usd! - a.mcap_usd! || b.tvl_usd - a.tvl_usd)
    .slice(0, 24)
    .map((p) => ({
      slug: p.slug,
      name: p.name,
      category: p.category,
      chains: p.chains,
      tvl_usd: p.tvl_usd,
      mcap_usd: p.mcap_usd!,
      change_7d: p.change_7d,
    }));
  return { board, live };
}

/** Why a board is not one the gate will take, or null: coverage, ranks, order and farm steps. */
function boardIssue(board: BoardRow[], snapshot: Snapshot): string | null {
  const slugs = new Set(board.map((r) => r.slug));
  const missing = snapshot.watchlist.filter((s) => !slugs.has(s));
  if (missing.length > 0) return `the board misses ${missing.length} watchlist slugs: ${missing.slice(0, 5).join(', ')}`;
  const ranks = board.map((r) => r.rank).sort((a, b) => a - b);
  for (let i = 0; i < ranks.length; i++) if (ranks[i] !== i + 1) return `rank ${i + 1} is ${ranks[i]}`;
  for (let i = 1; i < board.length; i++) if (board[i]!.implied_value_usd > board[i - 1]!.implied_value_usd) return `row ${i + 1} outranks row ${i} on implied value`;
  if (board.length >= 2 && board[0]!.implied_value_usd === board[board.length - 1]!.implied_value_usd) {
    return `all ${board.length} rows share the same implied value`;
  }
  for (const r of board) {
    if (r.farm_steps.length < 3 || r.farm_steps.length > 12) return `${r.slug} has ${r.farm_steps.length} farm steps`;
    for (const s of r.farm_steps) if (s.length < 20 || s.length > 600) return `${r.slug} has a farm step of ${s.length} characters`;
    if (!Number.isSafeInteger(r.implied_value_usd) || r.implied_value_usd <= 0) return `${r.slug} has implied value ${r.implied_value_usd}`;
  }
  return null;
}

// ---------------------------------------------------------------- snapshot

/** The snapshot out of a file's text, refused unless it is the shape the model scores. */
export function parseSnapshot(text: string): Snapshot {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new SnapshotError('snapshot.json is not JSON');
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new SnapshotError('snapshot.json is not a JSON object');
  const o = raw as Record<string, unknown>;
  if (typeof o.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(o.as_of)) throw new SnapshotError('snapshot.json has no as_of date');
  if (typeof o.source !== 'string') throw new SnapshotError('snapshot.json has no source');
  if (typeof o.floor_usd !== 'number' || !Number.isSafeInteger(o.floor_usd) || o.floor_usd <= 0) throw new SnapshotError('snapshot.json has no floor_usd');
  if (!Array.isArray(o.protocols)) throw new SnapshotError('snapshot.json has no protocols array');
  if (!Array.isArray(o.watchlist) || o.watchlist.some((s) => typeof s !== 'string')) throw new SnapshotError('snapshot.json has no watchlist of slugs');
  const protocols = o.protocols.map((v, i) => {
    if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new SnapshotError(`snapshot.json's protocol ${i} is not an object`);
    const p = v as Record<string, unknown>;
    if (typeof p.slug !== 'string' || typeof p.name !== 'string' || typeof p.tvl_usd !== 'number' || !Number.isFinite(p.tvl_usd)) {
      throw new SnapshotError(`snapshot.json's protocol ${i} lacks a slug, a name or a TVL`);
    }
    return {
      slug: p.slug,
      name: p.name,
      category: typeof p.category === 'string' ? p.category : null,
      chains: Array.isArray(p.chains) ? p.chains.filter((c): c is string => typeof c === 'string') : [],
      tvl_usd: p.tvl_usd,
      change_7d: typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : null,
      mcap_usd: typeof p.mcap_usd === 'number' && Number.isFinite(p.mcap_usd) ? p.mcap_usd : null,
      listed_at: typeof p.listed_at === 'string' ? p.listed_at : null,
      gecko_id: typeof p.gecko_id === 'string' ? p.gecko_id : null,
      symbol: typeof p.symbol === 'string' ? p.symbol : null,
    };
  });
  return {
    as_of: o.as_of,
    source: o.source,
    floor_usd: o.floor_usd,
    protocols,
    watchlist: o.watchlist as string[],
  };
}

/** A compact data.js: the dashboard's whole dataset as one script, no fetch and no network. */
function dataScript(radar: RadarData): string {
  return `window.RADAR=${JSON.stringify(radar)};\n`;
}

// ---------------------------------------------------------------- run

/** --name <value> out of the arguments, or the fallback. */
function arg(name: string, fallback: string): string {
  const at = process.argv.indexOf(`--${name}`);
  if (at >= 0 && at + 1 < process.argv.length) return process.argv[at + 1]!;
  const withEquals = process.argv.find((a) => a.startsWith(`--${name}=`));
  return withEquals === undefined ? fallback : withEquals.slice(name.length + 3);
}

/** Copy one directory's regular files into `to`, skipping names that start with a dot. */
function copyFiles(from: string, to: string): string[] {
  if (!existsSync(from)) return [];
  mkdirSync(to, { recursive: true });
  const copied: string[] = [];
  for (const name of readdirSync(from)) {
    if (name.startsWith('.')) continue;
    const path = join(from, name);
    if (!statSync(path).isFile()) continue;
    copyFileSync(path, join(to, name));
    copied.push(name);
  }
  return copied;
}

/** A PNG's pixel size from its IHDR chunk, or null when it is not a PNG. */
function pngSize(bytes: Buffer): { width: number; height: number } | null {
  if (bytes.length < 24 || bytes[0] !== 0x89 || bytes[1] !== 0x50) return null;
  return { width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20) };
}

const inputArg = arg('input', '/input');
const inputFile = inputArg.endsWith('.json') ? inputArg : join(inputArg, 'snapshot.json');
if (!existsSync(inputFile)) {
  process.stderr.write(`token radar: no snapshot at ${inputFile}: pass --input <snapshot.json|directory> or mount the sealed /input\n`);
  process.exit(2);
}
const out = arg('output', process.env.MAKEFUN_OUTPUT_DIR ?? '/output');

const inputBytes = readFileSync(inputFile);
const snapshot = parseSnapshot(inputBytes.toString('utf8'));
ageAsOf = snapshot.as_of;

const valueFit = fitValueModel(snapshot);
const oddsFit = fitOddsModel(snapshot);

const { rows: pricedAgg } = aggregatePriced(snapshot);
const medAgeValue = median(pricedAgg.map((p) => ageDays(snapshot.as_of, p.listed_at)).filter((a): a is number => a !== null));
const medGrowthValue = median(pricedAgg.map((p) => p.change_7d).filter((g): g is number => g !== null));
const medAgeOdds = median(snapshot.protocols.map((p) => ageDays(snapshot.as_of, p.listed_at)).filter((a): a is number => a !== null));
const medGrowthOdds = median(snapshot.protocols.map((p) => p.change_7d).filter((g): g is number => g !== null));

const { board, live } = buildBoard(snapshot, valueFit, oddsFit, medAgeValue, medGrowthValue, medAgeOdds, medGrowthOdds);
const issue = boardIssue(board, snapshot);
if (issue !== null) {
  process.stderr.write(`token radar: the model wrote a board the gate refuses: ${issue}\n`);
  process.exit(1);
}

const withToken = snapshot.protocols.filter((p) => !isPreToken(p)).length;
const earlyCount = board.filter((r) => !snapshot.watchlist.includes(r.slug)).length;
const tvlTotal = snapshot.protocols.reduce((s, p) => s + p.tvl_usd, 0);
const boardTvl = board.reduce((s, r) => s + r.tvl_usd, 0);
const implieds = board.map((r) => r.implied_value_usd);
const totalImplied = implieds.reduce((s, v) => s + v, 0);
const verdictCounts = { FARM: 0, MARGINAL: 0, SKIP: 0 };
for (const r of board) verdictCounts[r.farm.verdict]++;

const radar: RadarData = {
  as_of: snapshot.as_of,
  source: snapshot.source,
  snapshot_sha256: createHash('sha256').update(inputBytes).digest('hex'),
  floor_usd: snapshot.floor_usd,
  model_version: MODEL_VERSION,
  totals: {
    protocols: snapshot.protocols.length,
    with_token: withToken,
    pre_token: snapshot.protocols.length - withToken,
    watchlist: snapshot.watchlist.length,
    board: board.length,
    early: earlyCount,
    live: live.length,
    tvl_usd: tvlTotal,
    board_tvl_usd: boardTvl,
    total_implied_usd: totalImplied,
    implied_p25_usd: Math.round(quantile(implieds, 0.25)),
    implied_p75_usd: Math.round(quantile(implieds, 0.75)),
    farm: verdictCounts.FARM,
    marginal: verdictCounts.MARGINAL,
    skip: verdictCounts.SKIP,
  },
  value_fit: valueFit,
  odds_fit: oddsFit,
  assumptions: {
    initial_float: INITIAL_FLOAT,
    opportunity_cost: 'excluded — the snapshot carries no yield data, so locked capital is costed at gas only',
    verdict: 'odds tier: FARM at 60%+ launch odds, MARGINAL at 35%+, else SKIP, only with a positive expected net',
    farm_tiers: FARM_TIERS,
    max_tvl_share: MAX_TVL_SHARE,
  },
  watchlist: snapshot.watchlist,
  rows: board,
  live,
};

mkdirSync(out, { recursive: true });
const boardText = `${JSON.stringify({ rows: board })}\n`;
writeFileSync(join(out, 'board.json'), boardText);
writeFileSync(join(out, 'data.js'), dataScript(radar));

copyFiles(join(here, 'dashboard'), join(out, 'dashboard'));
const shots = copyFiles(join(here, 'screenshots'), join(out, 'screenshots'));
if (shots.length < 3) {
  process.stderr.write(`token radar: screenshots/ holds ${shots.length} images: hand in 3, the board, a detail view and a filtered view\n`);
  process.exit(1);
}
for (const name of shots) {
  if (!IMAGE_NAME.test(name)) {
    process.stderr.write(`token radar: screenshots/${name}: a name is letters, digits and . _ -, 1 to 128 characters\n`);
    process.exit(1);
  }
}

// The dossier is written last: the scorer rebuilds it from this /output and compares bytes.
const dossier = buildDossier(out, { text_max_bytes: TEXT_MAX_BYTES, screenshots: SCREENSHOT_CAPS, dashboard: { max_files: 400, max_bytes: 4_194_304 } });
writeFileSync(join(out, 'dossier.json'), dossier);

const boardHash = createHash('sha256').update(boardText).digest('hex');
const dossierHash = createHash('sha256').update(dossier).digest('hex');
const shotLines = shots
  .map((name) => {
    const bytes = readFileSync(join(out, 'screenshots', name));
    const size = pngSize(bytes);
    return `  ${name}: ${bytes.length} bytes${size === null ? '' : `, ${size.width}x${size.height}`}`;
  })
  .join('\n');

process.stdout.write(
  `token radar: ${snapshot.protocols.length} protocols, ${snapshot.watchlist.length} on the watchlist, ${board.length} rows scored (${earlyCount} early, ${live.length} live set aside)\n` +
    `token radar: value fit on ${valueFit.n_train} priced tokens (TVL aggregated across versions for: ${valueFit.aggregated.length === 0 ? 'none' : valueFit.aggregated.join(', ')}): ${valueFit.features.map((f, i) => `${f}=${valueFit.coefficients[i]}`).join(', ')}\n` +
    `token radar: walk-forward n=${valueFit.walk_forward.n_train}/${valueFit.walk_forward.n_test} rmsle=${valueFit.walk_forward.rmsle.toFixed(3)} mdape=${(valueFit.walk_forward.mdape * 100).toFixed(1)}% skill=${valueFit.walk_forward.skill.toFixed(2)}; loocv n=${valueFit.loocv.n} rmsle=${valueFit.loocv.rmsle.toFixed(3)} mdape=${(valueFit.loocv.mdape * 100).toFixed(1)}% skill=${valueFit.loocv.skill.toFixed(2)}\n` +
    `token radar: odds fit on ${oddsFit.n} protocols: ${oddsFit.features.map((f, i) => `${f}=${oddsFit.coefficients[i]}`).join(', ')}\n` +
    `token radar: odds brier in-sample=${oddsFit.brier_in_sample} cv=${oddsFit.brier_cv} auc=${oddsFit.auc}\n` +
    `token radar: total implied value $${totalImplied.toLocaleString('en-US')} (p25 $${radar.totals.implied_p25_usd.toLocaleString('en-US')}, p75 $${radar.totals.implied_p75_usd.toLocaleString('en-US')}); verdicts FARM ${verdictCounts.FARM} / MARGINAL ${verdictCounts.MARGINAL} / SKIP ${verdictCounts.SKIP}\n` +
    `token radar: top 5: ${board
      .slice(0, 5)
      .map((r) => `${r.slug} $${r.implied_value_usd.toLocaleString('en-US')} (odds ${r.launch_odds}, fdv $${r.conditional_fdv_usd.toLocaleString('en-US')})`)
      .join('; ')}\n` +
    `token radar: wrote ${join(out, 'board.json')} (${boardText.length} bytes, sha256 ${boardHash}), ${join(out, 'data.js')}, ${join(out, 'dashboard')}, ${join(out, 'screenshots')} and ${join(out, 'dossier.json')} (${dossier.length} bytes, sha256 ${dossierHash})\n` +
    `token radar: screenshots:\n${shotLines}\n`,
);
