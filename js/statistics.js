/* ============================================
   Statistics Engine v2
   ============================================ */
class StatisticsEngine {
    static descriptive(values) {
        const sorted = [...values].sort((a, b) => a - b), n = values.length, mean = ss.mean(values), std = ss.standardDeviation(values), sem = std / Math.sqrt(n);
        return { n, mean, std, variance: ss.variance(values), se: sem, median: ss.median(sorted), mode: ss.mode(values), min: sorted[0], max: sorted[n - 1], range: sorted[n - 1] - sorted[0], q1: ss.quantile(sorted, .25), q3: ss.quantile(sorted, .75), iqr: ss.quantile(sorted, .75) - ss.quantile(sorted, .25), skewness: ss.sampleSkewness(values), kurtosis: n > 3 ? ss.sampleKurtosis(values) : NaN, cv: mean !== 0 ? (std / Math.abs(mean)) * 100 : NaN, ci95: [mean - 1.96 * sem, mean + 1.96 * sem], sum: ss.sum(values) };
    }
    static oneSampleTTest(values, mu0, alt = 'two-sided') {
        const n = values.length, m = ss.mean(values), s = ss.standardDeviation(values), se = s / Math.sqrt(n), t = (m - mu0) / se, df = n - 1;
        let p; if (alt === 'two-sided') p = 2 * (1 - jStat.studentt.cdf(Math.abs(t), df)); else if (alt === 'less') p = jStat.studentt.cdf(t, df); else p = 1 - jStat.studentt.cdf(t, df);
        const tc = jStat.studentt.inv(.975, df);
        return { testName: '1-Sample t-Test', n, mean: m, std: s, se, t, df, pValue: p, mu0, ci95: [m - tc * se, m + tc * se], significant: p < .05, conclusion: p < .05 ? `Reject H₀ (p=${p.toFixed(4)})` : `Fail to reject H₀ (p=${p.toFixed(4)})` };
    }
    static twoSampleTTest(v1, v2, alt = 'two-sided') {
        const n1 = v1.length, n2 = v2.length, m1 = ss.mean(v1), m2 = ss.mean(v2), s1 = ss.standardDeviation(v1), s2 = ss.standardDeviation(v2);
        const se = Math.sqrt(s1 * s1 / n1 + s2 * s2 / n2), t = (m1 - m2) / se;
        const df = Math.floor(Math.pow(s1 * s1 / n1 + s2 * s2 / n2, 2) / (Math.pow(s1 * s1 / n1, 2) / (n1 - 1) + Math.pow(s2 * s2 / n2, 2) / (n2 - 1)));
        let p; if (alt === 'two-sided') p = 2 * (1 - jStat.studentt.cdf(Math.abs(t), df)); else if (alt === 'less') p = jStat.studentt.cdf(t, df); else p = 1 - jStat.studentt.cdf(t, df);
        return { testName: '2-Sample t-Test', n1, n2, mean1: m1, mean2: m2, std1: s1, std2: s2, se, t, df, pValue: p, meanDiff: m1 - m2, significant: p < .05, conclusion: p < .05 ? `Significant difference (p=${p.toFixed(4)})` : `No significant difference (p=${p.toFixed(4)})` };
    }
    static pairedTTest(v1, v2) { return this.oneSampleTTest(v1.map((v, i) => v - v2[i]), 0); }
    static chiSquareTest(observed, expected) {
        const chi2 = observed.reduce((s, o, i) => s + Math.pow(o - (expected ? expected[i] : ss.mean(observed)), 2) / (expected ? expected[i] : ss.mean(observed)), 0);
        const df = observed.length - 1; const p = 1 - jStat.chisquare.cdf(chi2, df);
        return { testName: 'Chi-Square Test', chi2, df, pValue: p, significant: p < .05, conclusion: p < .05 ? `Significant (p=${p.toFixed(4)})` : `Not significant (p=${p.toFixed(4)})` };
    }
    static normalityTest(values) {
        const n = values.length, sk = ss.sampleSkewness(values), ku = n > 3 ? ss.sampleKurtosis(values) : 0;
        const jb = n / 6 * (sk * sk + (ku - 3) * (ku - 3) / 4), p = 1 - jStat.chisquare.cdf(jb, 2);
        return { testName: 'Normality Test (Jarque-Bera)', n, skewness: sk, kurtosis: ku, statistic: jb, df: 2, pValue: p, significant: p < .05, conclusion: p > .05 ? `Normal distribution (p=${p.toFixed(4)})` : `Non-normal (p=${p.toFixed(4)})` };
    }
    static fTest(v1, v2) {
        const var1 = ss.variance(v1), var2 = ss.variance(v2), f = var1 / var2, df1 = v1.length - 1, df2 = v2.length - 1;
        const p = 2 * Math.min(jStat.centralF.cdf(f, df1, df2), 1 - jStat.centralF.cdf(f, df1, df2));
        return { testName: 'F-Test', var1, var2, f, df1, df2, pValue: p, significant: p < .05 };
    }
    static linearRegression(x, y) {
        const n = x.length, mx = ss.mean(x), my = ss.mean(y); let ssxx = 0, ssxy = 0, ssyy = 0;
        for (let i = 0; i < n; i++) { ssxx += (x[i] - mx) * (x[i] - mx); ssxy += (x[i] - mx) * (y[i] - my); ssyy += (y[i] - my) * (y[i] - my) }
        const b1 = ssxy / ssxx, b0 = my - b1 * mx, yHat = x.map(xi => b0 + b1 * xi), res = y.map((yi, i) => yi - yHat[i]);
        const sse = res.reduce((s, r) => s + r * r, 0), ssr = yHat.reduce((s, yh) => s + (yh - my) * (yh - my), 0), sst = ssyy;
        const r2 = 1 - sse / sst, adjR2 = 1 - (1 - r2) * (n - 1) / (n - 2), mse = sse / (n - 2);
        const seB1 = Math.sqrt(mse / ssxx), seB0 = Math.sqrt(mse * (1 / n + mx * mx / ssxx));
        const tB1 = b1 / seB1, tB0 = b0 / seB0, pB1 = 2 * (1 - jStat.studentt.cdf(Math.abs(tB1), n - 2)), pB0 = 2 * (1 - jStat.studentt.cdf(Math.abs(tB0), n - 2));
        const fStat = ssr / mse, pF = 1 - jStat.centralF.cdf(fStat, 1, n - 2);
        return { intercept: b0, slope: b1, rSquared: r2, adjR2, rmse: Math.sqrt(mse), mae: ss.mean(res.map(Math.abs)), n, coefficients: [{ term: 'Intercept', est: b0, se: seB0, t: tB0, p: pB0 }, { term: 'X', est: b1, se: seB1, t: tB1, p: pB1 }], anova: { ssr, sse, sst, fStat, pValue: pF }, yHat, residuals: res, equation: `Y = ${b0.toFixed(4)} + ${b1.toFixed(4)}·X` };
    }
    static polynomialRegression(x, y, degree) {
        const n = x.length, p = degree + 1;
        const X = x.map(xi => { const r = []; for (let d = 0; d <= degree; d++)r.push(Math.pow(xi, d)); return r });
        const XtX = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => X.reduce((s, r) => s + r[i] * r[j], 0)));
        const Xty = Array.from({ length: p }, (_, i) => X.reduce((s, r, k) => s + r[i] * y[k], 0));
        const coeffs = this.solveLinear(XtX, Xty);
        const yHat = x.map(xi => coeffs.reduce((s, c, d) => s + c * Math.pow(xi, d), 0));
        const res = y.map((yi, i) => yi - yHat[i]), my = ss.mean(y);
        const sst = y.reduce((s, yi) => s + (yi - my) * (yi - my), 0), sse = res.reduce((s, r) => s + r * r, 0);
        return { coefficients: coeffs, rSquared: 1 - sse / sst, rmse: Math.sqrt(sse / (n - p)), yHat, residuals: res, degree, equation: 'Y = ' + coeffs.map((c, d) => d === 0 ? c.toFixed(4) : `${c.toFixed(4)}·X^${d}`).join(' + ') };
    }
    static solveLinear(A, b) { const n = b.length, aug = A.map((r, i) => [...r, b[i]]); for (let i = 0; i < n; i++) { let mr = i; for (let k = i + 1; k < n; k++)if (Math.abs(aug[k][i]) > Math.abs(aug[mr][i])) mr = k;[aug[i], aug[mr]] = [aug[mr], aug[i]]; for (let k = i + 1; k < n; k++) { const f = aug[k][i] / aug[i][i]; for (let j = i; j <= n; j++)aug[k][j] -= f * aug[i][j] } } const x = new Array(n); for (let i = n - 1; i >= 0; i--) { x[i] = aug[i][n]; for (let j = i + 1; j < n; j++)x[i] -= aug[i][j] * x[j]; x[i] /= aug[i][i] } return x }
    static multipleRegression(X, y) {
        // X is array of arrays (each inner array is one predictor column)
        const n = y.length, p = X.length;
        // Use normal equations: β = (X'X)^-1 X'y with intercept
        const design = y.map((_, i) => [1, ...X.map(col => col[i])]);
        const pp = p + 1;
        const XtX = Array.from({ length: pp }, (_, i) => Array.from({ length: pp }, (_, j) => design.reduce((s, r) => s + r[i] * r[j], 0)));
        const Xty = Array.from({ length: pp }, (_, i) => design.reduce((s, r, k) => s + r[i] * y[k], 0));
        const coeffs = this.solveLinear(XtX, Xty);
        const yHat = design.map(r => coeffs.reduce((s, c, j) => s + c * r[j], 0));
        const res = y.map((yi, i) => yi - yHat[i]), my = ss.mean(y);
        const sst = y.reduce((s, yi) => s + (yi - my) * (yi - my), 0), sse = res.reduce((s, r) => s + r * r, 0);
        const r2 = 1 - sse / sst;
        return { coefficients: coeffs, rSquared: r2, adjR2: 1 - (1 - r2) * (n - 1) / (n - pp), rmse: Math.sqrt(sse / (n - pp)), yHat, residuals: res };
    }
    static oneWayAnova(groups) {
        const k = groups.length, N = groups.reduce((s, g) => s + g.length, 0), gm = ss.mean(groups.flat());
        const ssb = groups.reduce((s, g) => s + g.length * Math.pow(ss.mean(g) - gm, 2), 0);
        const ssw = groups.reduce((s, g) => s + g.reduce((s2, v) => s2 + Math.pow(v - ss.mean(g), 2), 0), 0);
        const dfb = k - 1, dfw = N - k, msb = ssb / dfb, msw = ssw / dfw, f = msb / msw, p = 1 - jStat.centralF.cdf(f, dfb, dfw);
        return { table: [{ source: 'Between', ss: ssb, df: dfb, ms: msb, f, p }, { source: 'Within', ss: ssw, df: dfw, ms: msw }, { source: 'Total', ss: ssb + ssw, df: N - 1 }], groupStats: groups.map((g, i) => ({ n: g.length, mean: ss.mean(g), std: ss.standardDeviation(g) })), significant: p < .05, conclusion: p < .05 ? `Significant (F=${f.toFixed(2)}, p=${p.toFixed(4)})` : `Not significant (F=${f.toFixed(2)}, p=${p.toFixed(4)})` };
    }
    static correlation(x, y) {
        const r = ss.sampleCorrelation(x, y), n = x.length, t = r * Math.sqrt((n - 2) / (1 - r * r)), p = 2 * (1 - jStat.studentt.cdf(Math.abs(t), n - 2));
        return { r, rSquared: r * r, t, df: n - 2, pValue: p, n, significant: p < .05, strength: Math.abs(r) > .7 ? 'Strong' : Math.abs(r) > .4 ? 'Moderate' : 'Weak', direction: r > 0 ? 'Positive' : r < 0 ? 'Negative' : 'None' };
    }
    static correlationMatrix(cols, dm) {
        const matrix = [];
        for (let i = 0; i < cols.length; i++) { matrix[i] = []; for (let j = 0; j < cols.length; j++) { if (i === j) { matrix[i][j] = 1; continue } const x = dm.getNumericValues(cols[i]), y = dm.getNumericValues(cols[j]), n = Math.min(x.length, y.length); matrix[i][j] = ss.sampleCorrelation(x.slice(0, n), y.slice(0, n)); } }
        return matrix;
    }
    static xbarRChart(values, n) {
        const sgs = []; for (let i = 0; i < values.length; i += n) { const sg = values.slice(i, i + n); if (sg.length === n) sgs.push(sg) }
        const means = sgs.map(sg => ss.mean(sg)), ranges = sgs.map(sg => Math.max(...sg) - Math.min(...sg));
        const xbb = ss.mean(means), rb = ss.mean(ranges);
        const c = { 2: { A2: 1.88, D3: 0, D4: 3.267 }, 3: { A2: 1.023, D3: 0, D4: 2.574 }, 4: { A2: .729, D3: 0, D4: 2.282 }, 5: { A2: .577, D3: 0, D4: 2.114 }, 6: { A2: .483, D3: 0, D4: 2.004 }, 7: { A2: .419, D3: .076, D4: 1.924 }, 8: { A2: .373, D3: .136, D4: 1.864 }, 9: { A2: .337, D3: .184, D4: 1.816 }, 10: { A2: .308, D3: .223, D4: 1.777 } };
        const cc = c[n] || c[5];
        return { xbar: { data: means, cl: xbb, ucl: xbb + cc.A2 * rb, lcl: xbb - cc.A2 * rb }, range: { data: ranges, cl: rb, ucl: cc.D4 * rb, lcl: cc.D3 * rb }, numSubgroups: sgs.length };
    }
    static iMRChart(values) {
        const mr = []; for (let i = 1; i < values.length; i++)mr.push(Math.abs(values[i] - values[i - 1]));
        const xb = ss.mean(values), mrb = ss.mean(mr), sig = mrb / 1.128;
        return { individuals: { data: values, cl: xb, ucl: xb + 3 * sig, lcl: xb - 3 * sig }, mr: { data: mr, cl: mrb, ucl: 3.267 * mrb, lcl: 0 } };
    }
    static capability(values, lsl, usl, target) {
        const m = ss.mean(values), s = ss.standardDeviation(values), cp = (usl - lsl) / (6 * s), cpu = (usl - m) / (3 * s), cpl = (m - lsl) / (3 * s), cpk = Math.min(cpu, cpl);
        const ppmA = 1e6 * (1 - jStat.normal.cdf(usl, m, s)), ppmB = 1e6 * jStat.normal.cdf(lsl, m, s);
        return { mean: m, std: s, cp, cpk, cpu, cpl, pp: cp, ppk: cpk, ppmAbove: Math.round(ppmA), ppmBelow: Math.round(ppmB), ppmTotal: Math.round(ppmA + ppmB), lsl, usl, target: target || (lsl + usl) / 2, sigmaLevel: U.round(cpk * 3, 1), rating: cpk >= 1.33 ? 'Capable' : cpk >= 1 ? 'Marginal' : 'Not Capable' };
    }
    static pareto(categories) {
        const counts = {}; categories.forEach(c => counts[c] = (counts[c] || 0) + 1);
        const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]), total = categories.length;
        let cum = 0; return sorted.map(([cat, cnt]) => { cum += cnt / total * 100; return { category: cat, count: cnt, pct: cnt / total * 100, cumulative: cum } });
    }
    static movingAverage(values, w) { return values.map((_, i) => i < w - 1 ? null : ss.mean(values.slice(i - w + 1, i + 1))); }
    static exponentialSmoothing(values, alpha, periods = 10) {
        const sm = [values[0]]; for (let i = 1; i < values.length; i++)sm.push(alpha * values[i] + (1 - alpha) * sm[i - 1]);
        const fc = Array(periods).fill(sm[sm.length - 1]); return { smoothed: sm, forecast: fc };
    }
    static autocorrelation(values, maxLag = 30) {
        const n = values.length, m = ss.mean(values), v = values.reduce((s, x) => s + (x - m) * (x - m), 0), acf = [], lim = Math.min(maxLag, Math.floor(n / 4));
        for (let lag = 0; lag <= lim; lag++) { let s = 0; for (let i = 0; i < n - lag; i++)s += (values[i] - m) * (values[i + lag] - m); acf.push(s / v) }
        return { acf, ci: 1.96 / Math.sqrt(n) };
    }
}
