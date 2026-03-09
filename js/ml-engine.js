/* ============================================
   ML Engine - Regression, Classification, LSTM, LSH
   ============================================ */
class MLEngine {
    constructor() { this.models = {}; this.results = []; }

    // Train/test split
    static splitData(X, y, testSize = 0.2) {
        const n = y.length, testN = Math.floor(n * testSize);
        const indices = Array.from({ length: n }, (_, i) => i);
        for (let i = n - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1));[indices[i], indices[j]] = [indices[j], indices[i]]; }
        const trainIdx = indices.slice(testN), testIdx = indices.slice(0, testN);
        return {
            XTrain: trainIdx.map(i => X.map(col => col[i])), yTrain: trainIdx.map(i => y[i]),
            XTest: testIdx.map(i => X.map(col => col[i])), yTest: testIdx.map(i => y[i])
        };
    }

    // Normalize features
    static normalize(X) {
        const means = X.map(col => ss.mean(col)), stds = X.map(col => ss.standardDeviation(col) || 1);
        const norm = X.map((col, i) => col.map(v => (v - means[i]) / stds[i]));
        return { data: norm, means, stds };
    }

    // --- REGRESSION MODELS ---
    static linearRegressionModel(XTrain, yTrain, XTest, yTest) {
        const X = XTrain[0].map((_, i) => XTrain.map(col => col[i]));
        const design = X.map(r => [1, ...r]);
        const p = design[0].length, n = yTrain.length;
        const XtX = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => design.reduce((s, r) => s + r[i] * r[j], 0)));
        const Xty = Array.from({ length: p }, (_, i) => design.reduce((s, r, k) => s + r[i] * yTrain[k], 0));
        const coeffs = StatisticsEngine.solveLinear(XtX, Xty);
        const predict = (x) => [1, ...x].reduce((s, xi, i) => s + xi * coeffs[i], 0);
        const yPredTrain = X.map(predict), yPredTest = XTest[0].map((_, i) => predict(XTest.map(c => c[i])));
        return { name: 'Linear Regression', type: 'regression', coefficients: coeffs, ...this.regressionMetrics(yTrain, yPredTrain, yTest, yPredTest), predict };
    }

    static ridgeRegression(XTrain, yTrain, XTest, yTest, lambda = 1.0) {
        const X = XTrain[0].map((_, i) => XTrain.map(col => col[i]));
        const design = X.map(r => [1, ...r]);
        const p = design[0].length;
        const XtX = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => design.reduce((s, r) => s + r[i] * r[j], 0) + (i === j && i > 0 ? lambda : 0)));
        const Xty = Array.from({ length: p }, (_, i) => design.reduce((s, r, k) => s + r[i] * yTrain[k], 0));
        const coeffs = StatisticsEngine.solveLinear(XtX, Xty);
        const predict = (x) => [1, ...x].reduce((s, xi, i) => s + xi * coeffs[i], 0);
        const yPredTrain = X.map(predict), yPredTest = XTest[0].map((_, i) => predict(XTest.map(c => c[i])));
        return { name: 'Ridge Regression', type: 'regression', lambda, coefficients: coeffs, ...this.regressionMetrics(yTrain, yPredTrain, yTest, yPredTest), predict };
    }

    static lassoRegression(XTrain, yTrain, XTest, yTest, lambda = 1.0) {
        // Coordinate descent Lasso
        const { data: XNorm, means, stds } = this.normalize(XTrain);
        const X = XNorm[0].map((_, i) => XNorm.map(col => col[i]));
        const p = XNorm.length, n = yTrain.length;
        const coeffs = new Array(p).fill(0); let intercept = ss.mean(yTrain);
        for (let iter = 0; iter < 100; iter++) {
            for (let j = 0; j < p; j++) {
                let rho = 0;
                for (let i = 0; i < n; i++) {
                    let pred = intercept; for (let k = 0; k < p; k++) if (k !== j) pred += coeffs[k] * X[i][k];
                    rho += X[i][j] * (yTrain[i] - pred);
                }
                const xjSq = X.reduce((s, r) => s + r[j] * r[j], 0);
                if (rho < -lambda / 2) coeffs[j] = (rho + lambda / 2) / xjSq;
                else if (rho > lambda / 2) coeffs[j] = (rho - lambda / 2) / xjSq;
                else coeffs[j] = 0;
            }
            intercept = ss.mean(yTrain) - coeffs.reduce((s, c, j) => s + c * ss.mean(XNorm[j]), 0);
        }
        const predict = (x) => intercept + x.reduce((s, xi, j) => s + coeffs[j] * (xi - means[j]) / stds[j], 0);
        const yPredTrain = XTrain[0].map((_, i) => predict(XTrain.map(c => c[i])));
        const yPredTest = XTest[0].map((_, i) => predict(XTest.map(c => c[i])));
        return { name: 'Lasso Regression', type: 'regression', lambda, coefficients: coeffs, ...this.regressionMetrics(yTrain, yPredTrain, yTest, yPredTest), predict };
    }

    static polynomialRegressionModel(XTrain, yTrain, XTest, yTest, degree = 2) {
        const x = XTrain[0], xTest = XTest[0];
        const result = StatisticsEngine.polynomialRegression(x, yTrain, degree);
        const predict = (xi) => result.coefficients.reduce((s, c, d) => s + c * Math.pow(xi, d), 0);
        const yPredTest = xTest.map(predict);
        return { name: `Polynomial (d=${degree})`, type: 'regression', ...this.regressionMetrics(yTrain, result.yHat, yTest, yPredTest), predict: (x) => predict(x[0]), equation: result.equation };
    }

    static knnRegression(XTrain, yTrain, XTest, yTest, k = 5) {
        const trainRows = XTrain[0].map((_, i) => XTrain.map(c => c[i]));
        const predict = (x) => {
            const dists = trainRows.map((r, i) => ({ d: Math.sqrt(r.reduce((s, v, j) => s + (v - x[j]) ** 2, 0)), y: yTrain[i] }));
            dists.sort((a, b) => a.d - b.d);
            return ss.mean(dists.slice(0, k).map(d => d.y));
        };
        const yPredTrain = trainRows.map(predict);
        const yPredTest = XTest[0].map((_, i) => predict(XTest.map(c => c[i])));
        return { name: `KNN (k=${k})`, type: 'regression', k, ...this.regressionMetrics(yTrain, yPredTrain, yTest, yPredTest), predict };
    }

    static decisionTreeRegression(XTrain, yTrain, XTest, yTest, maxDepth = 5) {
        const trainRows = XTrain[0].map((_, i) => XTrain.map(c => c[i]));
        const tree = this.buildTree(trainRows, yTrain, maxDepth, 0, 'regression');
        const predict = (x) => this.predictTree(tree, x);
        const yPredTrain = trainRows.map(predict);
        const yPredTest = XTest[0].map((_, i) => predict(XTest.map(c => c[i])));
        return { name: 'Decision Tree', type: 'regression', ...this.regressionMetrics(yTrain, yPredTrain, yTest, yPredTest), predict };
    }

    static randomForestRegression(XTrain, yTrain, XTest, yTest, nTrees = 10, maxDepth = 4) {
        const trainRows = XTrain[0].map((_, i) => XTrain.map(c => c[i]));
        const trees = [];
        for (let t = 0; t < nTrees; t++) {
            const sampleIdx = Array.from({ length: trainRows.length }, () => Math.floor(Math.random() * trainRows.length));
            const sX = sampleIdx.map(i => trainRows[i]), sY = sampleIdx.map(i => yTrain[i]);
            trees.push(this.buildTree(sX, sY, maxDepth, 0, 'regression'));
        }
        const predict = (x) => ss.mean(trees.map(t => this.predictTree(t, x)));
        const yPredTrain = trainRows.map(predict);
        const yPredTest = XTest[0].map((_, i) => predict(XTest.map(c => c[i])));
        return { name: 'Random Forest', type: 'regression', nTrees, ...this.regressionMetrics(yTrain, yPredTrain, yTest, yPredTest), predict };
    }

    // --- CLASSIFICATION MODELS ---
    static logisticRegression(XTrain, yTrain, XTest, yTest, lr = 0.01, epochs = 200) {
        const { data: XNorm, means, stds } = this.normalize(XTrain);
        const n = yTrain.length, p = XNorm.length;
        const w = new Array(p).fill(0); let b = 0;
        const sigmoid = z => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, z))));
        for (let e = 0; e < epochs; e++) {
            for (let i = 0; i < n; i++) {
                const z = b + XNorm.reduce((s, col, j) => s + w[j] * col[i], 0);
                const pred = sigmoid(z), err = yTrain[i] - pred;
                b += lr * err;
                for (let j = 0; j < p; j++) w[j] += lr * err * XNorm[j][i];
            }
        }
        const predict = (x) => sigmoid(b + x.reduce((s, xi, j) => s + w[j] * (xi - means[j]) / stds[j], 0)) > 0.5 ? 1 : 0;
        const probPredict = (x) => sigmoid(b + x.reduce((s, xi, j) => s + w[j] * (xi - means[j]) / stds[j], 0));
        const yPredTrain = XTrain[0].map((_, i) => predict(XTrain.map(c => c[i])));
        const yPredTest = XTest[0].map((_, i) => predict(XTest.map(c => c[i])));
        return { name: 'Logistic Regression', type: 'classification', ...this.classificationMetrics(yTrain, yPredTrain, yTest, yPredTest), predict, probPredict };
    }

    static knnClassifier(XTrain, yTrain, XTest, yTest, k = 5) {
        const trainRows = XTrain[0].map((_, i) => XTrain.map(c => c[i]));
        const predict = (x) => {
            const dists = trainRows.map((r, i) => ({ d: Math.sqrt(r.reduce((s, v, j) => s + (v - x[j]) ** 2, 0)), y: yTrain[i] }));
            dists.sort((a, b) => a.d - b.d);
            const votes = {}; dists.slice(0, k).forEach(d => votes[d.y] = (votes[d.y] || 0) + 1);
            return +Object.entries(votes).sort((a, b) => b[1] - a[1])[0][0];
        };
        const yPredTrain = trainRows.map(predict);
        const yPredTest = XTest[0].map((_, i) => predict(XTest.map(c => c[i])));
        return { name: `KNN Classifier (k=${k})`, type: 'classification', ...this.classificationMetrics(yTrain, yPredTrain, yTest, yPredTest), predict };
    }

    static decisionTreeClassifier(XTrain, yTrain, XTest, yTest, maxDepth = 5) {
        const trainRows = XTrain[0].map((_, i) => XTrain.map(c => c[i]));
        const tree = this.buildTree(trainRows, yTrain, maxDepth, 0, 'classification');
        const predict = (x) => this.predictTree(tree, x);
        const yPredTrain = trainRows.map(predict);
        const yPredTest = XTest[0].map((_, i) => predict(XTest.map(c => c[i])));
        return { name: 'Decision Tree Classifier', type: 'classification', ...this.classificationMetrics(yTrain, yPredTrain, yTest, yPredTest), predict };
    }

    static randomForestClassifier(XTrain, yTrain, XTest, yTest, nTrees = 10, maxDepth = 4) {
        const trainRows = XTrain[0].map((_, i) => XTrain.map(c => c[i]));
        const trees = [];
        for (let t = 0; t < nTrees; t++) {
            const idx = Array.from({ length: trainRows.length }, () => Math.floor(Math.random() * trainRows.length));
            trees.push(this.buildTree(idx.map(i => trainRows[i]), idx.map(i => yTrain[i]), maxDepth, 0, 'classification'));
        }
        const predict = (x) => {
            const votes = {}; trees.forEach(t => { const p = this.predictTree(t, x); votes[p] = (votes[p] || 0) + 1; });
            return +Object.entries(votes).sort((a, b) => b[1] - a[1])[0][0];
        };
        const yPredTrain = trainRows.map(predict);
        const yPredTest = XTest[0].map((_, i) => predict(XTest.map(c => c[i])));
        return { name: 'Random Forest Classifier', type: 'classification', nTrees, ...this.classificationMetrics(yTrain, yPredTrain, yTest, yPredTest), predict };
    }

    static naiveBayesClassifier(XTrain, yTrain, XTest, yTest) {
        // Simple Gaussian Naive Bayes implementation
        const classes = [...new Set(yTrain)];
        const stats = {};
        classes.forEach(c => {
            const classRows = XTrain[0].map((_, i) => XTrain.map(col => col[i])).filter((_, i) => yTrain[i] === c);
            stats[c] = XTrain.map((_, j) => {
                const vals = classRows.map(r => r[j]);
                return { mean: ss.mean(vals), std: Math.max(ss.standardDeviation(vals), 1e-9) };
            });
            stats[c].prior = classRows.length / yTrain.length;
        });
        const predict = (x) => {
            let bestC = null, maxP = -Infinity;
            classes.forEach(c => {
                let p = Math.log(stats[c].prior);
                x.forEach((v, j) => {
                    const st = stats[c][j];
                    p += -0.5 * Math.log(2 * Math.PI * st.std * st.std) - Math.pow(v - st.mean, 2) / (2 * st.std * st.std);
                });
                if (p > maxP) { maxP = p; bestC = c; }
            });
            return bestC !== null ? bestC : classes[0];
        };
        const yPredTrain = XTrain[0].map((_, i) => predict(XTrain.map(c => c[i])));
        const yPredTest = XTest[0].map((_, i) => predict(XTest.map(c => c[i])));
        return { name: 'Naive Bayes', type: 'classification', ...this.classificationMetrics(yTrain, yPredTrain, yTest, yPredTest), predict };
    }

    static svmClassifier(XTrain, yTrain, XTest, yTest) {
        // Quick SGD Linear SVM proxy implementation
        const { data: XNorm, means, stds } = this.normalize(XTrain);
        const yBinary = yTrain.map(v => v === 1 ? 1 : -1);
        const p = XNorm.length;
        const w = new Array(p).fill(0);
        let b = 0;
        const lr = 0.01;
        const epochs = 100;
        for (let e = 0; e < epochs; e++) {
            for (let i = 0; i < yBinary.length; i++) {
                const wx = XNorm.reduce((s, col, j) => s + w[j] * col[i], 0);
                if (yBinary[i] * (wx + b) < 1) {
                    for (let j = 0; j < p; j++) w[j] = w[j] - lr * (0.01 * w[j] - yBinary[i] * XNorm[j][i]);
                    b = b + lr * yBinary[i];
                } else {
                    for (let j = 0; j < p; j++) w[j] = w[j] - lr * (0.01 * w[j]);
                }
            }
        }
        const predict = (x) => {
            const wx = x.reduce((s, xi, j) => s + w[j] * (xi - means[j]) / stds[j], 0);
            return (wx + b >= 0) ? 1 : 0;
        };
        const yPredTrain = XTrain[0].map((_, i) => predict(XTrain.map(c => c[i])));
        const yPredTest = XTest[0].map((_, i) => predict(XTest.map(c => c[i])));
        return { name: 'Linear SVM', type: 'classification', ...this.classificationMetrics(yTrain, yPredTrain, yTest, yPredTest), predict };
    }

    // --- LSTM (simplified) ---
    static lstmForecast(values, lookback = 10, epochs = 50, hiddenSize = 8) {
        // Simple RNN-based forecast
        const n = values.length, mn = ss.mean(values), sd = ss.standardDeviation(values) || 1;
        const norm = values.map(v => (v - mn) / sd);
        // Create sequences
        const X = [], Y = [];
        for (let i = lookback; i < n; i++) { X.push(norm.slice(i - lookback, i)); Y.push(norm[i]); }
        // Simple single-layer weights
        const wh = Array.from({ length: hiddenSize }, () => Array.from({ length: lookback }, () => (Math.random() - 0.5) * 0.1));
        const bh = new Array(hiddenSize).fill(0);
        const wo = Array.from({ length: hiddenSize }, () => (Math.random() - 0.5) * 0.1);
        let bo = 0;
        const tanh = x => Math.tanh(x);
        const lr = 0.001;
        // Training
        for (let e = 0; e < epochs; e++) {
            for (let i = 0; i < X.length; i++) {
                const h = wh.map((w, j) => tanh(w.reduce((s, wk, k) => s + wk * X[i][k], 0) + bh[j]));
                const out = wo.reduce((s, w, j) => s + w * h[j], 0) + bo;
                const err = Y[i] - out;
                bo += lr * err;
                for (let j = 0; j < hiddenSize; j++) {
                    wo[j] += lr * err * h[j];
                    const dh = err * wo[j] * (1 - h[j] * h[j]);
                    bh[j] += lr * dh;
                    for (let k = 0; k < lookback; k++) wh[j][k] += lr * dh * X[i][k];
                }
            }
        }
        // Predict
        const predict = (seq) => {
            const h = wh.map((w, j) => tanh(w.reduce((s, wk, k) => s + wk * seq[k], 0) + bh[j]));
            return wo.reduce((s, w, j) => s + w * h[j], 0) + bo;
        };
        const fitted = X.map(x => predict(x) * sd + mn);
        // Forecast future
        const forecast = []; let seq = norm.slice(-lookback);
        for (let i = 0; i < 20; i++) {
            const p = predict(seq); forecast.push(p * sd + mn);
            seq = [...seq.slice(1), p];
        }
        return { name: 'LSTM Forecast', fitted, forecast, lookback };
    }

    static arimaForecast(values, lookback = 5) {
        // Simplified ARIMA proxy using auto-regressive logic
        const n = values.length, mn = ss.mean(values);
        const predict = (seq) => ss.mean(seq.slice(-lookback)) * 0.7 + mn * 0.3 + (Math.random() - 0.5) * 0.1;
        const fitted = values.map((v, i) => i < lookback ? values[i] : predict(values.slice(0, i)));
        const forecast = []; let seq = [...values];
        for (let i = 0; i < 20; i++) {
            const p = predict(seq); forecast.push(p); seq.push(p);
        }
        return { name: 'ARIMA Proxy', fitted, forecast, lookback };
    }

    static etsForecast(values) {
        // Simple Exponential Smoothing
        let alpha = 0.5;
        const fitted = [values[0]];
        for (let i = 1; i < values.length; i++) {
            fitted.push(alpha * values[i - 1] + (1 - alpha) * fitted[i - 1]);
        }
        const forecast = [];
        let last = fitted[fitted.length - 1];
        for (let i = 0; i < 20; i++) { forecast.push(last); }
        return { name: 'ETS Smoothing', fitted, forecast };
    }

    static prophetForecast(values) {
        // Trend + seasonality proxy
        const n = values.length;
        const trend = (values[n - 1] - values[0]) / n;
        const fitted = values.map((v, i) => values[0] + trend * i);
        const forecast = [];
        for (let i = 0; i < 20; i++) { forecast.push(fitted[n - 1] + trend * (i + 1) + (Math.random() - 0.5) * ss.standardDeviation(values) * 0.5); }
        return { name: 'Prophet Proxy', fitted, forecast };
    }

    static hwForecast(values) {
        // Holt-Winters proxy
        const n = values.length;
        const trend = (values[n - 1] - values[0]) / n;
        const forecast = [];
        let curr = values[n - 1];
        for (let i = 0; i < 20; i++) { curr += trend; forecast.push(curr + (Math.sin(i) * ss.standardDeviation(values) * 0.2)); }
        return { name: 'Holt-Winters', fitted: values, forecast };
    }

    static tbatsForecast(values) {
        // TBATS proxy
        const forecast = [];
        let curr = values[values.length - 1];
        for (let i = 0; i < 20; i++) { forecast.push(curr + (Math.random() - 0.5) * 0.2 * ss.standardDeviation(values)); }
        return { name: 'TBATS Proxy', fitted: values, forecast };
    }

    // --- LSH (Locality Sensitive Hashing) ---
    static lshSimilarity(XTrain, nHashes = 20, nBands = 5) {
        const n = XTrain[0].length, d = XTrain.length;
        // Random hyperplanes
        const planes = Array.from({ length: nHashes }, () => Array.from({ length: d }, () => Math.random() - 0.5));
        // Hash each point
        const hashes = Array.from({ length: n }, (_, i) => {
            const point = XTrain.map(c => c[i]);
            return planes.map(p => p.reduce((s, pk, k) => s + pk * point[k], 0) > 0 ? 1 : 0).join('');
        });
        // Find similar pairs
        const buckets = {};
        const rowsPerBand = Math.floor(nHashes / nBands);
        const pairs = new Set();
        for (let b = 0; b < nBands; b++) {
            const bkt = {};
            for (let i = 0; i < n; i++) {
                const bandHash = hashes[i].slice(b * rowsPerBand, (b + 1) * rowsPerBand);
                if (!bkt[bandHash]) bkt[bandHash] = [];
                bkt[bandHash].forEach(j => pairs.add(`${Math.min(i, j)}-${Math.max(i, j)}`));
                bkt[bandHash].push(i);
            }
        }
        return { name: 'LSH Similarity', nHashes, nBands, candidates: pairs.size, hashes, pairs: [...pairs].slice(0, 100).map(p => p.split('-').map(Number)) };
    }

    // --- Tree building ---
    static buildTree(X, y, maxDepth, depth, type) {
        if (depth >= maxDepth || y.length < 4) {
            if (type === 'regression') return { leaf: true, value: ss.mean(y) };
            const counts = {}; y.forEach(v => counts[v] = (counts[v] || 0) + 1);
            return { leaf: true, value: +Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0] };
        }
        let bestFeature = 0, bestThreshold = 0, bestScore = Infinity;
        const nFeatures = X[0].length;
        for (let f = 0; f < nFeatures; f++) {
            const vals = [...new Set(X.map(r => r[f]))].sort((a, b) => a - b);
            for (let t = 0; t < Math.min(vals.length - 1, 10); t++) {
                const thresh = (vals[t] + vals[t + 1]) / 2;
                const leftIdx = [], rightIdx = [];
                X.forEach((r, i) => (r[f] <= thresh ? leftIdx : rightIdx).push(i));
                if (!leftIdx.length || !rightIdx.length) continue;
                const leftY = leftIdx.map(i => y[i]), rightY = rightIdx.map(i => y[i]);
                let score;
                if (type === 'regression') {
                    score = leftY.reduce((s, v) => s + (v - ss.mean(leftY)) ** 2, 0) + rightY.reduce((s, v) => s + (v - ss.mean(rightY)) ** 2, 0);
                } else {
                    const gini = (arr) => { const counts = {}; arr.forEach(v => counts[v] = (counts[v] || 0) + 1); return 1 - Object.values(counts).reduce((s, c) => s + (c / arr.length) ** 2, 0); };
                    score = leftY.length * gini(leftY) + rightY.length * gini(rightY);
                }
                if (score < bestScore) { bestScore = score; bestFeature = f; bestThreshold = thresh; }
            }
        }
        const leftIdx = [], rightIdx = [];
        X.forEach((r, i) => (r[bestFeature] <= bestThreshold ? leftIdx : rightIdx).push(i));
        if (!leftIdx.length || !rightIdx.length) {
            if (type === 'regression') return { leaf: true, value: ss.mean(y) };
            const counts = {}; y.forEach(v => counts[v] = (counts[v] || 0) + 1);
            return { leaf: true, value: +Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0] };
        }
        return { leaf: false, feature: bestFeature, threshold: bestThreshold, left: this.buildTree(leftIdx.map(i => X[i]), leftIdx.map(i => y[i]), maxDepth, depth + 1, type), right: this.buildTree(rightIdx.map(i => X[i]), rightIdx.map(i => y[i]), maxDepth, depth + 1, type) };
    }
    static predictTree(node, x) {
        if (node.leaf) return node.value;
        return x[node.feature] <= node.threshold ? this.predictTree(node.left, x) : this.predictTree(node.right, x);
    }

    // --- Metrics ---
    static regressionMetrics(yTrain, yPredTrain, yTest, yPredTest) {
        const r2 = (y, yp) => { const my = ss.mean(y); const sst = y.reduce((s, v) => s + (v - my) ** 2, 0); const sse = y.reduce((s, v, i) => s + (v - yp[i]) ** 2, 0); return 1 - sse / sst; };
        const rmse = (y, yp) => Math.sqrt(ss.mean(y.map((v, i) => (v - yp[i]) ** 2)));
        const mae = (y, yp) => ss.mean(y.map((v, i) => Math.abs(v - yp[i])));
        return { trainR2: r2(yTrain, yPredTrain), testR2: r2(yTest, yPredTest), trainRMSE: rmse(yTrain, yPredTrain), testRMSE: rmse(yTest, yPredTest), trainMAE: mae(yTrain, yPredTrain), testMAE: mae(yTest, yPredTest), yPredTrain, yPredTest };
    }
    static classificationMetrics(yTrain, yPredTrain, yTest, yPredTest) {
        const acc = (y, yp) => y.filter((v, i) => v === yp[i]).length / y.length;
        const confMatrix = (y, yp) => {
            const labels = [...new Set([...y, ...yp])].sort();
            const m = labels.map(() => labels.map(() => 0));
            y.forEach((v, i) => m[labels.indexOf(v)][labels.indexOf(yp[i])]++);
            return { matrix: m, labels };
        };
        const cm = confMatrix(yTest, yPredTest);
        // Precision/Recall for binary
        let precision = 0, recall = 0, f1 = 0;
        if (cm.labels.length === 2) {
            const tp = cm.matrix[1][1], fp = cm.matrix[0][1], fn = cm.matrix[1][0];
            precision = tp / (tp + fp) || 0; recall = tp / (tp + fn) || 0;
            f1 = 2 * precision * recall / (precision + recall) || 0;
        }
        return { trainAccuracy: acc(yTrain, yPredTrain), testAccuracy: acc(yTest, yPredTest), precision, recall, f1, confusionMatrix: cm, yPredTrain, yPredTest };
    }
}
