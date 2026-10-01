// ABP optional native kernels (C++20, CPython C-API only; no NumPy headers).
//
// Why this file exists
// --------------------
// Profiling (docs/benchmarks.md) showed that the pure-Python Meuwissen & Luo
// (1992) inbreeding tracer needs ~90 s for a 100k-animal, 20-generation
// pedigree.  The algorithm is inherently sequential and pointer-chasing, so it
// is implemented here once, in portable C++20, and cross-checked in the test
// suite against the Python reference implementation in abp/core/pedigree.py
// and against the independent tabular method.  The Python path remains the
// scientific reference and the automatic fallback when this module is not
// compiled (see ADR 0002).
//
// Interface contract
// ------------------
// inbreeding_ml(sire: buffer[int64], dam: buffer[int64]) -> bytes
//   * sire/dam hold internal indices, -1 for an unknown parent;
//   * parents must precede offspring (index < own index) - checked;
//   * returns n little-endian float64 values (the inbreeding coefficients).
// symbolic_cholesky / takahashi: sparse selected inversion, see abp/solvers/selinv.py.
// mindegree_order / ldl_numeric / ldl_solve: sparse LDL', see abp/solvers/cholesky.py.
// ml_general(sire, dam, c, e, fext) -> bytes (2n float64: diag(A), then d)
//   * metafounder generalisation, see abp/core/metafounders.py.
// inbreeding_depth(sire, dam[, mode]) -> (bytes, depths, ml_pairs, colleau_columns, colleau_depths)
//   * same F as inbreeding_ml, computed depth by depth with the cheaper of Meuwissen-Luo
//     traces and Colleau columns per depth (see inbreeding_depth_kernel).
// colleau_times(sire, dam, d, x, k) -> bytes (n*k float64): A x for x of shape n x k
//   (row-major) by the two pedigree recursions of A = T D T' (Colleau 2002), see
//   abp/core/pedigree.py::Pedigree.a_times (the SuperLU triangular solves are the
//   reference).
// The GIL is released while computing.  Memory: O(n) doubles/ints plus a
// hash map with one entry per distinct (sire, dam) pair.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <functional>
#include <iterator>
#include <utility>
#include <cmath>
#include <cstdint>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// Returns an empty string on success, otherwise an error message.
std::string inbreeding_kernel(const int64_t* sire, const int64_t* dam, int64_t n,
                              double* F_out) {
    for (int64_t i = 0; i < n; ++i) {
        if (sire[i] >= i || dam[i] >= i || sire[i] < -1 || dam[i] < -1) {
            return "parents must precede offspring (animal index " + std::to_string(i) + ")";
        }
    }
    // F[n] = -1 encodes an unknown parent so one formula yields d_i.
    std::vector<double> F(static_cast<size_t>(n) + 1, 0.0);
    F[static_cast<size_t>(n)] = -1.0;
    std::vector<double> d(static_cast<size_t>(n), 0.0);
    std::vector<double> coef(static_cast<size_t>(n), 0.0);  // row of T being traced
    std::priority_queue<int64_t> heap;                      // max-heap: youngest first
    std::unordered_map<uint64_t, double> family;            // F of a (sire, dam) pair
    family.reserve(static_cast<size_t>(n / 4 + 16));

    for (int64_t i = 0; i < n; ++i) {
        const int64_t s = sire[i], m = dam[i];
        const double Fs = s < 0 ? -1.0 : F[static_cast<size_t>(s)];
        const double Fm = m < 0 ? -1.0 : F[static_cast<size_t>(m)];
        d[static_cast<size_t>(i)] = 0.5 - 0.25 * (Fs + Fm);
        if (s < 0 || m < 0) {  // at least one unknown parent: not inbred
            F[static_cast<size_t>(i)] = 0.0;
            continue;
        }
        const uint64_t lo = static_cast<uint64_t>(std::min(s, m));
        const uint64_t hi = static_cast<uint64_t>(std::max(s, m));
        const uint64_t key = lo * static_cast<uint64_t>(n) + hi;
        auto it = family.find(key);
        if (it != family.end()) {  // full sib of an animal already processed
            F[static_cast<size_t>(i)] = it->second;
            continue;
        }
        // Trace ancestors in decreasing index order; coef[j] = T[i, j].
        double acc = 0.0;
        coef[static_cast<size_t>(i)] = 1.0;
        heap.push(i);
        while (!heap.empty()) {
            const int64_t j = heap.top();
            heap.pop();
            const double lj = coef[static_cast<size_t>(j)];
            coef[static_cast<size_t>(j)] = 0.0;  // reset for the next animal
            acc += lj * lj * d[static_cast<size_t>(j)];
            const double half = 0.5 * lj;
            const int64_t par[2] = {sire[j], dam[j]};
            for (int64_t p : par) {
                if (p >= 0) {
                    // Coefficients are strictly positive once set, so 0 marks "not queued".
                    if (coef[static_cast<size_t>(p)] == 0.0) heap.push(p);
                    coef[static_cast<size_t>(p)] += half;
                }
            }
        }
        F[static_cast<size_t>(i)] = acc - 1.0;
        family.emplace(key, F[static_cast<size_t>(i)]);
    }
    std::copy(F.begin(), F.begin() + n, F_out);
    return std::string();
}

// A x = T D T' x for x (n x k, row-major):  v = T' x by the backward recursion
// v_s += v_i / 2, v_d += v_i / 2 (offspring before parents), w = D v, then the forward
// recursion x_i = w_i + (x_s + x_d) / 2 (parents before offspring).  The inner loops run
// over the k contiguous columns of one animal.
std::string colleau_kernel(const int64_t* sire, const int64_t* dam, const double* D,
                           const double* x, int64_t n, int64_t k, double* out) {
    for (int64_t i = 0; i < n; ++i) {
        if (sire[i] >= i || dam[i] >= i || sire[i] < -1 || dam[i] < -1) {
            return "parents must precede offspring (animal index " + std::to_string(i) + ")";
        }
    }
    std::copy(x, x + n * k, out);
    for (int64_t i = n - 1; i >= 0; --i) {
        const double* vi = out + i * k;
        if (sire[i] >= 0) {
            double* vs = out + sire[i] * k;
            for (int64_t j = 0; j < k; ++j) vs[j] += 0.5 * vi[j];
        }
        if (dam[i] >= 0) {
            double* vd = out + dam[i] * k;
            for (int64_t j = 0; j < k; ++j) vd[j] += 0.5 * vi[j];
        }
    }
    for (int64_t i = 0; i < n; ++i) {
        double* xi = out + i * k;
        const double di = D[i];
        for (int64_t j = 0; j < k; ++j) xi[j] *= di;
        if (sire[i] >= 0) {
            const double* xs = out + sire[i] * k;
            for (int64_t j = 0; j < k; ++j) xi[j] += 0.5 * xs[j];
        }
        if (dam[i] >= 0) {
            const double* xd = out + dam[i] * k;
            for (int64_t j = 0; j < k; ++j) xi[j] += 0.5 * xd[j];
        }
    }
    return "";
}

// Inbreeding by pedigree depth (round 9).  Animals are processed depth by depth
// (depth 0: both parents unknown; otherwise 1 + the larger parental depth), so when
// depth g is reached F and d are known for every animal of depth < g, and the
// relationships a(s, m) of the new (sire, dam) pairs of depth g are entries of A
// restricted to that prefix.  Per depth the kernel picks the cheaper of
//  * Meuwissen & Luo traces (one per new pair; cost ~ ancestors traced), and
//  * Colleau columns: y = A e_p over the prefix for every distinct parent p on the
//    smaller side of the new pairs (cost ~ 2 x prefix size per column, 16 columns at a
//    time), then F = y[other parent] / 2 (indirect approach of Colleau 2002).
// mode 0 chooses automatically, 1 forces Meuwissen & Luo, 2 forces Colleau columns (tests).
// The choice uses the traced-ancestor count of the first (up to 16) pairs, which are
// computed by Meuwissen & Luo in any case.  Both paths give F exactly (up to rounding).
// stats[0..3]: depths, pairs by Meuwissen & Luo, Colleau columns, depths using Colleau.
std::string inbreeding_depth_kernel(const int64_t* sire0, const int64_t* dam0, int64_t n,
                                    int64_t mode, double* F_out, int64_t* stats) {
    for (int64_t i = 0; i < n; ++i) {
        if (sire0[i] >= i || dam0[i] >= i || sire0[i] < -1 || dam0[i] < -1) {
            return "parents must precede offspring (animal index " + std::to_string(i) + ")";
        }
    }
    const size_t N = static_cast<size_t>(n);
    std::vector<int64_t> depth(N, 0);
    int64_t maxd = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t ds = sire0[i] < 0 ? -1 : depth[static_cast<size_t>(sire0[i])];
        const int64_t dd = dam0[i] < 0 ? -1 : depth[static_cast<size_t>(dam0[i])];
        depth[static_cast<size_t>(i)] = 1 + std::max(ds, dd);
        maxd = std::max(maxd, depth[static_cast<size_t>(i)]);
    }
    // stable counting sort by depth: perm[k] = original index of position k
    std::vector<int64_t> start(static_cast<size_t>(maxd) + 2, 0);
    for (int64_t i = 0; i < n; ++i) ++start[static_cast<size_t>(depth[static_cast<size_t>(i)]) + 1];
    for (int64_t g = 0; g <= maxd; ++g) start[static_cast<size_t>(g) + 1] += start[static_cast<size_t>(g)];
    std::vector<int64_t> perm(N), pos(N);
    {
        std::vector<int64_t> fill(start.begin(), start.end() - 1);
        for (int64_t i = 0; i < n; ++i) {
            const int64_t k = fill[static_cast<size_t>(depth[static_cast<size_t>(i)])]++;
            perm[static_cast<size_t>(k)] = i;
            pos[static_cast<size_t>(i)] = k;
        }
    }
    std::vector<int64_t> S(N), M(N);
    for (int64_t k = 0; k < n; ++k) {
        const int64_t i = perm[static_cast<size_t>(k)];
        S[static_cast<size_t>(k)] = sire0[i] < 0 ? -1 : pos[static_cast<size_t>(sire0[i])];
        M[static_cast<size_t>(k)] = dam0[i] < 0 ? -1 : pos[static_cast<size_t>(dam0[i])];
    }
    std::vector<double> F(N, 0.0), d(N, 0.0), coef(N, 0.0);
    std::priority_queue<int64_t> heap;
    std::unordered_map<uint64_t, double> family;
    family.reserve(N / 4 + 16);
    auto key_of = [n](int64_t s, int64_t m) {
        return static_cast<uint64_t>(std::min(s, m)) * static_cast<uint64_t>(n)
               + static_cast<uint64_t>(std::max(s, m));
    };
    // Meuwissen & Luo trace of animal i (d known for i and its ancestors); returns
    // the number of animals traced.
    auto trace = [&](int64_t i, double& Fi) {
        int64_t count = 0;
        double acc = 0.0;
        coef[static_cast<size_t>(i)] = 1.0;
        heap.push(i);
        while (!heap.empty()) {
            const int64_t j = heap.top();
            heap.pop();
            ++count;
            const double lj = coef[static_cast<size_t>(j)];
            coef[static_cast<size_t>(j)] = 0.0;
            acc += lj * lj * d[static_cast<size_t>(j)];
            const double half = 0.5 * lj;
            const int64_t par[2] = {S[static_cast<size_t>(j)], M[static_cast<size_t>(j)]};
            for (int64_t p : par) {
                if (p >= 0) {
                    if (coef[static_cast<size_t>(p)] == 0.0) heap.push(p);
                    coef[static_cast<size_t>(p)] += half;
                }
            }
        }
        Fi = acc - 1.0;
        return count;
    };
    constexpr int64_t K = 16;
    std::vector<double> X;
    stats[0] = maxd + 1;
    stats[1] = stats[2] = stats[3] = 0;
    for (int64_t g = 0; g <= maxd; ++g) {
        const int64_t a = start[static_cast<size_t>(g)], b = start[static_cast<size_t>(g) + 1];
        // Mendelian sampling variances of depth g (parents are in the prefix)
        for (int64_t i = a; i < b; ++i) {
            const int64_t s = S[static_cast<size_t>(i)], m = M[static_cast<size_t>(i)];
            const double Fs = s < 0 ? -1.0 : F[static_cast<size_t>(s)];
            const double Fm = m < 0 ? -1.0 : F[static_cast<size_t>(m)];
            d[static_cast<size_t>(i)] = 0.5 - 0.25 * (Fs + Fm);
        }
        // new pairs of this depth (first offspring of each)
        std::vector<int64_t> first;
        std::unordered_map<uint64_t, int64_t> seen;
        for (int64_t i = a; i < b; ++i) {
            const int64_t s = S[static_cast<size_t>(i)], m = M[static_cast<size_t>(i)];
            if (s < 0 || m < 0) continue;
            const uint64_t key = key_of(s, m);
            if (family.count(key) == 0 && seen.emplace(key, i).second) first.push_back(i);
        }
        if (first.empty()) {
            for (int64_t i = a; i < b; ++i) {
                const int64_t s = S[static_cast<size_t>(i)], m = M[static_cast<size_t>(i)];
                if (s >= 0 && m >= 0) F[static_cast<size_t>(i)] = family[key_of(s, m)];
            }
            continue;
        }
        const size_t n_sample = std::min<size_t>(first.size(), mode == 0 ? 16 : 0);
        double traced = 0.0;
        for (size_t f = 0; f < n_sample; ++f) {
            const int64_t i = first[f];
            double Fi = 0.0;
            traced += static_cast<double>(trace(i, Fi));
            family.emplace(key_of(S[static_cast<size_t>(i)], M[static_cast<size_t>(i)]), Fi);
        }
        stats[1] += static_cast<int64_t>(n_sample);
        const size_t rest = first.size() - n_sample;
        if (rest > 0) {
            std::vector<int64_t> sires, dams;
            for (size_t f = n_sample; f < first.size(); ++f) {
                sires.push_back(S[static_cast<size_t>(first[f])]);
                dams.push_back(M[static_cast<size_t>(first[f])]);
            }
            auto distinct = [](std::vector<int64_t> v) {
                std::sort(v.begin(), v.end());
                return static_cast<size_t>(std::unique(v.begin(), v.end()) - v.begin());
            };
            const size_t ns = distinct(sires), nd = distinct(dams);
            const bool by_sire = ns <= nd;
            const size_t ncol = by_sire ? ns : nd;
            const double avg = n_sample ? traced / static_cast<double>(n_sample) : 0.0;
            // a heap trace costs ~ log2 of the heap size per animal traced
            const double cost_ml = static_cast<double>(rest) * avg * (1.0 + std::log2(avg + 1.0));
            const double cost_col = 2.0 * static_cast<double>(a) * static_cast<double>(ncol)
                                    + static_cast<double>(a);
            if (mode == 2 || (mode == 0 && cost_col < cost_ml)) {
                ++stats[3];
                // columns: distinct parents p of the chosen side, each with its pairs
                std::unordered_map<int64_t, std::vector<std::pair<int64_t, uint64_t>>> need;
                for (size_t f = n_sample; f < first.size(); ++f) {
                    const int64_t s = S[static_cast<size_t>(first[f])];
                    const int64_t m = M[static_cast<size_t>(first[f])];
                    const int64_t p = by_sire ? s : m, o = by_sire ? m : s;
                    need[p].emplace_back(o, key_of(s, m));
                }
                std::vector<int64_t> cols;
                cols.reserve(need.size());
                for (const auto& kv : need) cols.push_back(kv.first);
                std::sort(cols.begin(), cols.end());
                stats[2] += static_cast<int64_t>(cols.size());
                X.assign(static_cast<size_t>(a) * K, 0.0);
                for (size_t c0 = 0; c0 < cols.size(); c0 += K) {
                    const int64_t kc = std::min<int64_t>(K, static_cast<int64_t>(cols.size() - c0));
                    std::fill(X.begin(), X.end(), 0.0);
                    for (int64_t c = 0; c < kc; ++c)
                        X[static_cast<size_t>(cols[c0 + static_cast<size_t>(c)] * K + c)] = 1.0;
                    const int64_t top = cols[c0 + static_cast<size_t>(kc) - 1];
                    for (int64_t i = top; i >= 0; --i) {       // v = T' e
                        const double* vi = X.data() + i * K;
                        const int64_t s = S[static_cast<size_t>(i)], m = M[static_cast<size_t>(i)];
                        if (s >= 0) {
                            double* vs = X.data() + s * K;
                            for (int64_t c = 0; c < K; ++c) vs[c] += 0.5 * vi[c];
                        }
                        if (m >= 0) {
                            double* vm = X.data() + m * K;
                            for (int64_t c = 0; c < K; ++c) vm[c] += 0.5 * vi[c];
                        }
                    }
                    for (int64_t i = 0; i < a; ++i) {          // y = T D v
                        double* xi = X.data() + i * K;
                        const double di = d[static_cast<size_t>(i)];
                        for (int64_t c = 0; c < K; ++c) xi[c] *= di;
                        const int64_t s = S[static_cast<size_t>(i)], m = M[static_cast<size_t>(i)];
                        if (s >= 0) {
                            const double* xs = X.data() + s * K;
                            for (int64_t c = 0; c < K; ++c) xi[c] += 0.5 * xs[c];
                        }
                        if (m >= 0) {
                            const double* xm = X.data() + m * K;
                            for (int64_t c = 0; c < K; ++c) xi[c] += 0.5 * xm[c];
                        }
                    }
                    for (int64_t c = 0; c < kc; ++c) {
                        for (const auto& ok : need[cols[c0 + static_cast<size_t>(c)]])
                            family.emplace(ok.second,
                                           0.5 * X[static_cast<size_t>(ok.first * K + c)]);
                    }
                }
            } else {
                for (size_t f = n_sample; f < first.size(); ++f) {
                    const int64_t i = first[f];
                    double Fi = 0.0;
                    trace(i, Fi);
                    family.emplace(key_of(S[static_cast<size_t>(i)], M[static_cast<size_t>(i)]), Fi);
                }
                stats[1] += static_cast<int64_t>(rest);
            }
        }
        for (int64_t i = a; i < b; ++i) {
            const int64_t s = S[static_cast<size_t>(i)], m = M[static_cast<size_t>(i)];
            if (s >= 0 && m >= 0) F[static_cast<size_t>(i)] = family[key_of(s, m)];
        }
    }
    for (int64_t k = 0; k < n; ++k) F_out[perm[static_cast<size_t>(k)]] = F[static_cast<size_t>(k)];
    return std::string();
}

PyObject* py_inbreeding_depth(PyObject*, PyObject* args) {
    Py_buffer sb, db;
    long long mode = 0;
    if (!PyArg_ParseTuple(args, "y*y*|L", &sb, &db, &mode)) return nullptr;
    PyObject* result = nullptr;
    if (sb.len != db.len || sb.len % 8 != 0) {
        PyErr_SetString(PyExc_ValueError, "sire and dam must be int64 buffers of equal length");
    } else {
        const int64_t n = static_cast<int64_t>(sb.len / 8);
        PyObject* fb = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(n * 8));
        if (fb != nullptr) {
            double* out = reinterpret_cast<double*>(PyBytes_AS_STRING(fb));
            int64_t st[4] = {0, 0, 0, 0};
            std::string err;
            Py_BEGIN_ALLOW_THREADS
            err = inbreeding_depth_kernel(static_cast<const int64_t*>(sb.buf),
                                          static_cast<const int64_t*>(db.buf), n,
                                          static_cast<int64_t>(mode), out, st);
            Py_END_ALLOW_THREADS
            if (!err.empty()) {
                Py_DECREF(fb);
                PyErr_SetString(PyExc_ValueError, err.c_str());
            } else {
                result = Py_BuildValue("(NLLLL)", fb, static_cast<long long>(st[0]),
                                       static_cast<long long>(st[1]),
                                       static_cast<long long>(st[2]),
                                       static_cast<long long>(st[3]));
            }
        }
    }
    PyBuffer_Release(&sb);
    PyBuffer_Release(&db);
    return result;
}

PyObject* py_colleau_times(PyObject*, PyObject* args) {
    Py_buffer sb, db, Db, xb;
    long long k = 0;
    if (!PyArg_ParseTuple(args, "y*y*y*y*L", &sb, &db, &Db, &xb, &k)) return nullptr;
    PyObject* result = nullptr;
    const int64_t n = static_cast<int64_t>(sb.len / 8);
    if (sb.len != db.len || sb.len % 8 != 0 || Db.len != sb.len || k < 1
        || xb.len != static_cast<Py_ssize_t>(n * k * 8)) {
        PyErr_SetString(PyExc_ValueError, "colleau_times: inconsistent buffer sizes");
    } else {
        result = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(n * k * 8));
        if (result != nullptr) {
            double* out = reinterpret_cast<double*>(PyBytes_AS_STRING(result));
            std::string err;
            Py_BEGIN_ALLOW_THREADS
            err = colleau_kernel(static_cast<const int64_t*>(sb.buf),
                                 static_cast<const int64_t*>(db.buf),
                                 static_cast<const double*>(Db.buf),
                                 static_cast<const double*>(xb.buf), n, k, out);
            Py_END_ALLOW_THREADS
            if (!err.empty()) {
                Py_DECREF(result);
                result = nullptr;
                PyErr_SetString(PyExc_ValueError, err.c_str());
            }
        }
    }
    PyBuffer_Release(&sb);
    PyBuffer_Release(&db);
    PyBuffer_Release(&Db);
    PyBuffer_Release(&xb);
    return result;
}

PyObject* py_inbreeding_ml(PyObject*, PyObject* args) {
    Py_buffer sb, db;
    if (!PyArg_ParseTuple(args, "y*y*", &sb, &db)) return nullptr;
    PyObject* result = nullptr;
    if (sb.len != db.len || sb.len % 8 != 0) {
        PyErr_SetString(PyExc_ValueError, "sire and dam must be int64 buffers of equal length");
    } else {
        const int64_t n = static_cast<int64_t>(sb.len / 8);
        result = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(n * 8));
        if (result != nullptr) {
            double* out = reinterpret_cast<double*>(PyBytes_AS_STRING(result));
            std::string err;
            Py_BEGIN_ALLOW_THREADS
            err = inbreeding_kernel(static_cast<const int64_t*>(sb.buf),
                                    static_cast<const int64_t*>(db.buf), n, out);
            Py_END_ALLOW_THREADS
            if (!err.empty()) {
                Py_DECREF(result);
                result = nullptr;
                PyErr_SetString(PyExc_ValueError, err.c_str());
            }
        }
    }
    PyBuffer_Release(&sb);
    PyBuffer_Release(&db);
    return result;
}


// ---------------------------------------------------------------------------
// Bayesian marker-regression sweep (see abp/solvers/bayes.py::sweep_python,
// which is the reference; the algorithm below is line-for-line the same).
// Wt is W transposed and C-contiguous: marker j occupies Wt[j*n .. j*n+n).
// method: 0 = single normal (BRR/BayesA), 1 = zero + normal (BayesB/C/Cpi),
//         2 = zero + K-1 normals (BayesR). Random numbers z (normal) and u
// (uniform) are pre-drawn by the caller. e, beta, delta are updated in place.
std::string bayes_kernel(const double* Wt, const double* wtw, double* e, double* beta,
                         int64_t* delta, const double* var_j, const double* log_pi,
                         const double* comp_var, int64_t K, double sigma_e2, int method,
                         const double* z, const double* u, int64_t n, int64_t m) {
    std::vector<double> logs(static_cast<size_t>(K > 0 ? K : 1)), lhs_k(logs.size());
    for (int64_t j = 0; j < m; ++j) {
        const double* w = Wt + j * n;
        double dot = 0.0;
        for (int64_t i = 0; i < n; ++i) dot += w[i] * e[i];
        const double bj = beta[j];
        const double rhs = (dot + wtw[j] * bj) / sigma_e2;
        double nb = 0.0;
        if (method == 0) {
            const double lhs = wtw[j] / sigma_e2 + 1.0 / var_j[j];
            nb = rhs / lhs + z[j] / std::sqrt(lhs);
            delta[j] = 1;
        } else if (method == 1) {
            const double v = var_j[j];
            const double lhs = wtw[j] / sigma_e2 + 1.0 / v;
            const double logd1 = -0.5 * (std::log(lhs) + std::log(v)) + 0.5 * rhs * rhs / lhs + log_pi[1];
            const double d = log_pi[0] - logd1;
            const double p1 = d < 700.0 ? 1.0 / (1.0 + std::exp(d)) : 0.0;
            if (u[j] < p1) {
                delta[j] = 1;
                nb = rhs / lhs + z[j] / std::sqrt(lhs);
            } else {
                delta[j] = 0;
                nb = 0.0;
            }
        } else if (method == 2) {
            logs[0] = log_pi[0];
            double mx = logs[0];
            for (int64_t k = 1; k < K; ++k) {
                const double v = comp_var[k];
                lhs_k[k] = wtw[j] / sigma_e2 + 1.0 / v;
                logs[k] = -0.5 * (std::log(lhs_k[k]) + std::log(v)) + 0.5 * rhs * rhs / lhs_k[k] + log_pi[k];
                if (logs[k] > mx) mx = logs[k];
            }
            double tot = 0.0;
            for (int64_t k = 0; k < K; ++k) { logs[k] = std::exp(logs[k] - mx); tot += logs[k]; }
            double cum = 0.0;
            int64_t pick = K - 1;
            for (int64_t k = 0; k < K; ++k) {
                cum += logs[k] / tot;
                if (cum > u[j]) { pick = k; break; }
            }
            delta[j] = pick;
            nb = pick == 0 ? 0.0 : rhs / lhs_k[pick] + z[j] / std::sqrt(lhs_k[pick]);
        } else {
            return "unknown method code";
        }
        if (nb != bj) {
            const double diff = nb - bj;
            for (int64_t i = 0; i < n; ++i) e[i] -= w[i] * diff;
            beta[j] = nb;
        }
    }
    return std::string();
}

PyObject* py_bayes_sweep(PyObject*, PyObject* args) {
    Py_buffer Wb, wtwb, eb, betab, deltab, varb, lpb, cvb, zb, ub;
    double sigma_e2;
    int method;
    if (!PyArg_ParseTuple(args, "y*y*w*w*w*y*y*y*diy*y*", &Wb, &wtwb, &eb, &betab, &deltab, &varb,
                          &lpb, &cvb, &sigma_e2, &method, &zb, &ub))
        return nullptr;
    const int64_t m = static_cast<int64_t>(betab.len / 8);
    const int64_t n = static_cast<int64_t>(eb.len / 8);
    std::string err;
    if (Wb.len != n * m * 8 || wtwb.len != m * 8 || deltab.len != m * 8 || varb.len != m * 8 ||
        zb.len != m * 8 || ub.len != m * 8 || lpb.len != cvb.len) {
        err = "bayes_sweep: buffer sizes do not match (n, m)";
    } else {
        const int64_t K = static_cast<int64_t>(cvb.len / 8);
        Py_BEGIN_ALLOW_THREADS
        err = bayes_kernel(static_cast<const double*>(Wb.buf), static_cast<const double*>(wtwb.buf),
                           static_cast<double*>(eb.buf), static_cast<double*>(betab.buf),
                           static_cast<int64_t*>(deltab.buf), static_cast<const double*>(varb.buf),
                           static_cast<const double*>(lpb.buf), static_cast<const double*>(cvb.buf),
                           K, sigma_e2, method, static_cast<const double*>(zb.buf),
                           static_cast<const double*>(ub.buf), n, m);
        Py_END_ALLOW_THREADS
    }
    for (Py_buffer* b : {&Wb, &wtwb, &eb, &betab, &deltab, &varb, &lpb, &cvb, &zb, &ub})
        PyBuffer_Release(b);
    if (!err.empty()) {
        PyErr_SetString(PyExc_ValueError, err.c_str());
        return nullptr;
    }
    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
// Generalised Meuwissen-Luo trace for metafounders (reference:
// abp/core/metafounders.py::ml_general_python; same algorithm).
//   d_i  = 1 - (A_s + A_d)/4 - c_i       (A_p = diag of an animal parent, else 0)
//   A_ii = 1 + fext_i                    (a parent is not an animal)
//   A_ii = sum_j T_ij^2 d_j + e_i        (both parents are animals)
// With c = e = fext = 0 this is the ordinary algorithm (A_ii = 1 + F_i).
// Output: 2n doubles, diag(A) followed by d.
std::string ml_general_kernel(const int64_t* sire, const int64_t* dam, const double* c,
                              const double* e, const double* fext, int64_t n, double* out) {
    for (int64_t i = 0; i < n; ++i) {
        if (sire[i] >= i || dam[i] >= i || sire[i] < -1 || dam[i] < -1) {
            return "parents must precede offspring (animal index " + std::to_string(i) + ")";
        }
    }
    double* adiag = out;
    double* d = out + n;
    std::vector<double> coef(static_cast<size_t>(n), 0.0);
    std::priority_queue<int64_t> heap;
    std::unordered_map<uint64_t, double> family;
    family.reserve(static_cast<size_t>(n / 4 + 16));
    for (int64_t i = 0; i < n; ++i) {
        const int64_t s = sire[i], m = dam[i];
        const double as = s < 0 ? 0.0 : adiag[s];
        const double am = m < 0 ? 0.0 : adiag[m];
        d[i] = 1.0 - 0.25 * (as + am) - c[i];
        if (s < 0 || m < 0) {
            adiag[i] = 1.0 + fext[i];
            continue;
        }
        const uint64_t lo = static_cast<uint64_t>(std::min(s, m));
        const uint64_t hi = static_cast<uint64_t>(std::max(s, m));
        const uint64_t key = lo * static_cast<uint64_t>(n) + hi;
        auto it = family.find(key);
        if (it != family.end()) {
            adiag[i] = it->second;
            continue;
        }
        double acc = 0.0;
        coef[static_cast<size_t>(i)] = 1.0;
        heap.push(i);
        while (!heap.empty()) {
            const int64_t j = heap.top();
            heap.pop();
            const double lj = coef[static_cast<size_t>(j)];
            coef[static_cast<size_t>(j)] = 0.0;
            acc += lj * lj * d[j];
            const double half = 0.5 * lj;
            const int64_t par[2] = {sire[j], dam[j]};
            for (int64_t p : par) {
                if (p >= 0) {
                    if (coef[static_cast<size_t>(p)] == 0.0) heap.push(p);
                    coef[static_cast<size_t>(p)] += half;
                }
            }
        }
        adiag[i] = acc + e[i];
        family.emplace(key, adiag[i]);
    }
    return std::string();
}

PyObject* py_ml_general(PyObject*, PyObject* args) {
    Py_buffer sb, db, cb, eb, fb;
    if (!PyArg_ParseTuple(args, "y*y*y*y*y*", &sb, &db, &cb, &eb, &fb)) return nullptr;
    PyObject* result = nullptr;
    const Py_ssize_t len = sb.len;
    if (db.len != len || cb.len != len || eb.len != len || fb.len != len || len % 8 != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "ml_general: sire, dam (int64) and c, e, fext (float64) must have equal length");
    } else {
        const int64_t n = static_cast<int64_t>(len / 8);
        result = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(2 * n * 8));
        if (result != nullptr) {
            double* out = reinterpret_cast<double*>(PyBytes_AS_STRING(result));
            std::string err;
            Py_BEGIN_ALLOW_THREADS
            err = ml_general_kernel(static_cast<const int64_t*>(sb.buf),
                                    static_cast<const int64_t*>(db.buf),
                                    static_cast<const double*>(cb.buf),
                                    static_cast<const double*>(eb.buf),
                                    static_cast<const double*>(fb.buf), n, out);
            Py_END_ALLOW_THREADS
            if (!err.empty()) {
                Py_DECREF(result);
                result = nullptr;
                PyErr_SetString(PyExc_ValueError, err.c_str());
            }
        }
    }
    for (Py_buffer* b : {&sb, &db, &cb, &eb, &fb}) PyBuffer_Release(b);
    return result;
}

// ---------------------------------------------------------------------------
// Sparse selected inversion (reference: abp/solvers/selinv.py, same algorithms).
// symbolic_cholesky: strictly lower pattern of the Cholesky factor of a
//   symmetric CSC matrix (union of the column's rows > j and the children's
//   patterns in the elimination tree).
// takahashi: Z = B^-1 on that pattern from B = L D L' (L unit lower).
std::string symbolic_kernel(const int64_t* indptr, const int64_t* indices, int64_t n,
                            std::vector<int64_t>& colptr, std::vector<int64_t>& rowidx) {
    std::vector<std::vector<int64_t>> pats(static_cast<size_t>(n));
    std::vector<std::vector<int64_t>> children(static_cast<size_t>(n));
    std::vector<int64_t> mark(static_cast<size_t>(n), -1);
    for (int64_t j = 0; j < n; ++j) {
        std::vector<int64_t>& pj = pats[static_cast<size_t>(j)];
        mark[static_cast<size_t>(j)] = j;
        for (int64_t p = indptr[j]; p < indptr[j + 1]; ++p) {
            const int64_t i = indices[p];
            if (i < 0 || i >= n) return "symbolic_cholesky: row index out of range";
            if (i > j && mark[static_cast<size_t>(i)] != j) {
                mark[static_cast<size_t>(i)] = j;
                pj.push_back(i);
            }
        }
        for (int64_t c : children[static_cast<size_t>(j)]) {
            for (int64_t i : pats[static_cast<size_t>(c)]) {
                if (i > j && mark[static_cast<size_t>(i)] != j) {
                    mark[static_cast<size_t>(i)] = j;
                    pj.push_back(i);
                }
            }
        }
        std::sort(pj.begin(), pj.end());
        if (!pj.empty()) children[static_cast<size_t>(pj.front())].push_back(j);
    }
    colptr.assign(static_cast<size_t>(n) + 1, 0);
    for (int64_t j = 0; j < n; ++j)
        colptr[static_cast<size_t>(j) + 1] = colptr[static_cast<size_t>(j)] +
                                             static_cast<int64_t>(pats[static_cast<size_t>(j)].size());
    rowidx.resize(static_cast<size_t>(colptr.back()));
    for (int64_t j = 0; j < n; ++j) {
        std::copy(pats[static_cast<size_t>(j)].begin(), pats[static_cast<size_t>(j)].end(),
                  rowidx.begin() + colptr[static_cast<size_t>(j)]);
        std::vector<int64_t>().swap(pats[static_cast<size_t>(j)]);
    }
    return std::string();
}

std::string takahashi_kernel(const int64_t* colptr, const int64_t* rowidx, const double* lval,
                             const double* d, int64_t n, double* zdiag, double* zval) {
    std::vector<double> acc(static_cast<size_t>(n), 0.0);
    for (int64_t j = n - 1; j >= 0; --j) {
        const int64_t lo = colptr[j], hi = colptr[j + 1];
        if (!(d[j] > 0.0)) return "takahashi: non-positive pivot";
        for (int64_t t = lo; t < hi; ++t) {
            const int64_t k = rowidx[t];
            const double lkj = lval[t];
            acc[static_cast<size_t>(k)] -= lkj * zdiag[k];
            // rows i of column j after k are a subset of column k's rows: merge walk
            int64_t q = colptr[k];
            const int64_t qend = colptr[k + 1];
            for (int64_t u = t + 1; u < hi; ++u) {
                const int64_t i = rowidx[u];
                while (q < qend && rowidx[q] < i) ++q;
                if (q == qend || rowidx[q] != i) return "takahashi: pattern not closed";
                const double z = zval[q];
                acc[static_cast<size_t>(i)] -= lkj * z;
                acc[static_cast<size_t>(k)] -= lval[u] * z;
            }
        }
        double s = 0.0;
        for (int64_t t = lo; t < hi; ++t) {
            const size_t i = static_cast<size_t>(rowidx[t]);
            zval[t] = acc[i];
            s += lval[t] * acc[i];
            acc[i] = 0.0;
        }
        zdiag[j] = 1.0 / d[j] - s;
    }
    return std::string();
}

PyObject* py_symbolic_cholesky(PyObject*, PyObject* args) {
    Py_buffer pb, ib;
    long long n_ll;
    if (!PyArg_ParseTuple(args, "y*y*L", &pb, &ib, &n_ll)) return nullptr;
    const int64_t n = static_cast<int64_t>(n_ll);
    std::vector<int64_t> colptr, rowidx;
    std::string err;
    if (pb.len != (n + 1) * 8 || ib.len % 8 != 0) {
        err = "symbolic_cholesky: indptr must hold n + 1 int64 values";
    } else {
        Py_BEGIN_ALLOW_THREADS
        err = symbolic_kernel(static_cast<const int64_t*>(pb.buf),
                              static_cast<const int64_t*>(ib.buf), n, colptr, rowidx);
        Py_END_ALLOW_THREADS
    }
    PyBuffer_Release(&pb);
    PyBuffer_Release(&ib);
    if (!err.empty()) {
        PyErr_SetString(PyExc_ValueError, err.c_str());
        return nullptr;
    }
    PyObject* a = PyBytes_FromStringAndSize(reinterpret_cast<const char*>(colptr.data()),
                                            static_cast<Py_ssize_t>(colptr.size() * 8));
    PyObject* b = PyBytes_FromStringAndSize(reinterpret_cast<const char*>(rowidx.data()),
                                            static_cast<Py_ssize_t>(rowidx.size() * 8));
    if (a == nullptr || b == nullptr) {
        Py_XDECREF(a);
        Py_XDECREF(b);
        return nullptr;
    }
    return Py_BuildValue("(NN)", a, b);
}

PyObject* py_takahashi(PyObject*, PyObject* args) {
    Py_buffer cb, rb, lb, db;
    if (!PyArg_ParseTuple(args, "y*y*y*y*", &cb, &rb, &lb, &db)) return nullptr;
    const int64_t n = static_cast<int64_t>(db.len / 8);
    const int64_t nnz = static_cast<int64_t>(rb.len / 8);
    PyObject* zd = nullptr;
    PyObject* zv = nullptr;
    std::string err;
    if (cb.len != (n + 1) * 8 || lb.len != rb.len) {
        err = "takahashi: buffer sizes do not match (colptr n+1, rowidx = lval)";
    } else {
        zd = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(n * 8));
        zv = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(nnz * 8));
        if (zd == nullptr || zv == nullptr) {
            err = "takahashi: out of memory";
        } else {
            const int64_t* colptr = static_cast<const int64_t*>(cb.buf);
            if (colptr[n] != nnz) {
                err = "takahashi: colptr[n] != number of entries";
            } else {
                double* zdp = reinterpret_cast<double*>(PyBytes_AS_STRING(zd));
                double* zvp = reinterpret_cast<double*>(PyBytes_AS_STRING(zv));
                Py_BEGIN_ALLOW_THREADS
                err = takahashi_kernel(colptr, static_cast<const int64_t*>(rb.buf),
                                       static_cast<const double*>(lb.buf),
                                       static_cast<const double*>(db.buf), n, zdp, zvp);
                Py_END_ALLOW_THREADS
            }
        }
    }
    for (Py_buffer* b : {&cb, &rb, &lb, &db}) PyBuffer_Release(b);
    if (!err.empty()) {
        Py_XDECREF(zd);
        Py_XDECREF(zv);
        PyErr_SetString(PyExc_ValueError, err.c_str());
        return nullptr;
    }
    return Py_BuildValue("(NN)", zd, zv);
}

// ---------------------------------------------------------------------------
// Sparse LDL' with minimum-degree ordering (reference: abp/solvers/cholesky.py,
// same algorithms).
// mindegree_order: eliminate a node of smallest current degree (ties: smallest
//   index), join its neighbours into a clique; lazy min-heap of (degree, node).
// ldl_numeric: up-looking LDL' of a full symmetric CSC matrix on a given
//   symbolic pattern (column k of the upper part scattered, elimination-tree
//   reach processed in increasing index order).
// ldl_solve: L z = b, z /= d, L' x = z for m right-hand sides stored row-wise.
std::string mindegree_kernel(const int64_t* indptr, const int64_t* indices, int64_t n,
                             std::vector<int64_t>& order) {
    std::vector<std::vector<int64_t>> adj(static_cast<size_t>(n));
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t p = indptr[j]; p < indptr[j + 1]; ++p) {
            const int64_t i = indices[p];
            if (i < 0 || i >= n) return "mindegree_order: index out of range";
            if (i != j) {
                adj[static_cast<size_t>(j)].push_back(i);
                adj[static_cast<size_t>(i)].push_back(j);
            }
        }
    }
    for (auto& a : adj) {
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
    }
    // Dense rows (degree > max(16, 10 sqrt(n)), e.g. an intercept linked to every recorded
    // animal) are removed from the graph and eliminated last, as in AMD; otherwise each
    // elimination would merge into their huge adjacency lists.
    const int64_t dense_limit = std::max<int64_t>(16, static_cast<int64_t>(10.0 * std::sqrt(static_cast<double>(n))));
    std::vector<char> dense(static_cast<size_t>(n), 0);
    for (int64_t v = 0; v < n; ++v)
        if (static_cast<int64_t>(adj[static_cast<size_t>(v)].size()) > dense_limit) dense[static_cast<size_t>(v)] = 1;
    for (int64_t v = 0; v < n; ++v) {
        auto& a = adj[static_cast<size_t>(v)];
        if (dense[static_cast<size_t>(v)]) { std::vector<int64_t>().swap(a); continue; }
        a.erase(std::remove_if(a.begin(), a.end(), [&](int64_t w) { return dense[static_cast<size_t>(w)] != 0; }), a.end());
    }
    using Item = std::pair<int64_t, int64_t>;   // (degree, node)
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
    for (int64_t v = 0; v < n; ++v)
        if (!dense[static_cast<size_t>(v)]) heap.emplace(static_cast<int64_t>(adj[static_cast<size_t>(v)].size()), v);
    std::vector<char> done(static_cast<size_t>(n), 0);
    order.clear();
    order.reserve(static_cast<size_t>(n));
    std::vector<int64_t> merged;
    while (!heap.empty()) {
        const auto [deg, v] = heap.top();
        heap.pop();
        auto& av = adj[static_cast<size_t>(v)];
        if (done[static_cast<size_t>(v)] || deg != static_cast<int64_t>(av.size())) continue;
        done[static_cast<size_t>(v)] = 1;
        order.push_back(v);
        const std::vector<int64_t> nb(av.begin(), av.end());
        for (int64_t u : nb) {
            auto& au = adj[static_cast<size_t>(u)];
            merged.clear();
            merged.reserve(au.size() + nb.size());
            std::set_union(au.begin(), au.end(), nb.begin(), nb.end(), std::back_inserter(merged));
            // drop v and u itself
            merged.erase(std::remove_if(merged.begin(), merged.end(),
                                        [&](int64_t w) { return w == v || w == u; }), merged.end());
            au.swap(merged);
            heap.emplace(static_cast<int64_t>(au.size()), u);
        }
        std::vector<int64_t>().swap(av);
    }
    for (int64_t v = 0; v < n; ++v)
        if (dense[static_cast<size_t>(v)]) order.push_back(v);
    return std::string();
}

std::string ldl_numeric_kernel(const int64_t* Bp, const int64_t* Bi, const double* Bx,
                               const int64_t* colptr, const int64_t* rowidx, int64_t n,
                               double* d, double* lval) {
    std::vector<int64_t> parent(static_cast<size_t>(n), -1), fill(static_cast<size_t>(n));
    for (int64_t j = 0; j < n; ++j) {
        if (colptr[j + 1] > colptr[j]) parent[static_cast<size_t>(j)] = rowidx[colptr[j]];
        fill[static_cast<size_t>(j)] = colptr[j];
    }
    std::vector<double> y(static_cast<size_t>(n), 0.0);
    std::vector<int64_t> mark(static_cast<size_t>(n), -1), reach;
    for (int64_t k = 0; k < n; ++k) {
        reach.clear();
        double dkk = 0.0;
        for (int64_t p = Bp[k]; p < Bp[k + 1]; ++p) {
            int64_t i = Bi[p];
            if (i == k) {
                dkk += Bx[p];
            } else if (i < k) {
                y[static_cast<size_t>(i)] += Bx[p];
                while (i != -1 && i < k && mark[static_cast<size_t>(i)] != k) {
                    mark[static_cast<size_t>(i)] = k;
                    reach.push_back(i);
                    i = parent[static_cast<size_t>(i)];
                }
            }
        }
        std::sort(reach.begin(), reach.end());
        for (int64_t j : reach) {
            const double yj = y[static_cast<size_t>(j)];
            y[static_cast<size_t>(j)] = 0.0;
            const int64_t end = fill[static_cast<size_t>(j)];
            for (int64_t p = colptr[j]; p < end; ++p) y[static_cast<size_t>(rowidx[p])] -= lval[p] * yj;
            const double lkj = yj / d[j];
            dkk -= lkj * yj;
            const int64_t p = fill[static_cast<size_t>(j)];
            if (p >= colptr[j + 1] || rowidx[p] != k) return "ldl_numeric: entry outside the symbolic pattern";
            lval[p] = lkj;
            fill[static_cast<size_t>(j)] = p + 1;
        }
        if (!(dkk > 0.0)) return "coefficient matrix is not positive definite (non-positive pivot "
                                 "at permuted index " + std::to_string(k) + ")";
        d[k] = dkk;
    }
    return std::string();
}

void ldl_solve_kernel(const int64_t* colptr, const int64_t* rowidx, const double* lval,
                      const double* d, int64_t n, int64_t m, double* x) {
    for (int64_t r = 0; r < m; ++r) {
        double* xr = x + r * n;
        for (int64_t j = 0; j < n; ++j) {
            const double xj = xr[j];
            if (xj != 0.0)
                for (int64_t p = colptr[j]; p < colptr[j + 1]; ++p) xr[rowidx[p]] -= lval[p] * xj;
        }
        for (int64_t j = 0; j < n; ++j) xr[j] /= d[j];
        for (int64_t j = n - 1; j >= 0; --j) {
            double s = xr[j];
            for (int64_t p = colptr[j]; p < colptr[j + 1]; ++p) s -= lval[p] * xr[rowidx[p]];
            xr[j] = s;
        }
    }
}

PyObject* py_mindegree_order(PyObject*, PyObject* args) {
    Py_buffer pb, ib;
    long long n_ll;
    if (!PyArg_ParseTuple(args, "y*y*L", &pb, &ib, &n_ll)) return nullptr;
    const int64_t n = static_cast<int64_t>(n_ll);
    std::vector<int64_t> order;
    std::string err;
    if (pb.len != (n + 1) * 8) {
        err = "mindegree_order: indptr must hold n + 1 int64 values";
    } else {
        Py_BEGIN_ALLOW_THREADS
        err = mindegree_kernel(static_cast<const int64_t*>(pb.buf),
                               static_cast<const int64_t*>(ib.buf), n, order);
        Py_END_ALLOW_THREADS
    }
    PyBuffer_Release(&pb);
    PyBuffer_Release(&ib);
    if (!err.empty()) {
        PyErr_SetString(PyExc_ValueError, err.c_str());
        return nullptr;
    }
    return PyBytes_FromStringAndSize(reinterpret_cast<const char*>(order.data()),
                                     static_cast<Py_ssize_t>(order.size() * 8));
}

PyObject* py_ldl_numeric(PyObject*, PyObject* args) {
    Py_buffer bp, bi, bx, cp, ri;
    if (!PyArg_ParseTuple(args, "y*y*y*y*y*", &bp, &bi, &bx, &cp, &ri)) return nullptr;
    const int64_t n = static_cast<int64_t>(bp.len / 8) - 1;
    const int64_t nnz = static_cast<int64_t>(ri.len / 8);
    PyObject* dd = nullptr;
    PyObject* lv = nullptr;
    std::string err;
    if (n < 0 || cp.len != bp.len || bi.len != bx.len) {
        err = "ldl_numeric: buffer sizes do not match";
    } else {
        dd = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(n * 8));
        lv = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(nnz * 8));
        if (dd == nullptr || lv == nullptr) {
            err = "ldl_numeric: out of memory";
        } else {
            double* dp = reinterpret_cast<double*>(PyBytes_AS_STRING(dd));
            double* lp = reinterpret_cast<double*>(PyBytes_AS_STRING(lv));
            Py_BEGIN_ALLOW_THREADS
            err = ldl_numeric_kernel(static_cast<const int64_t*>(bp.buf),
                                     static_cast<const int64_t*>(bi.buf),
                                     static_cast<const double*>(bx.buf),
                                     static_cast<const int64_t*>(cp.buf),
                                     static_cast<const int64_t*>(ri.buf), n, dp, lp);
            Py_END_ALLOW_THREADS
        }
    }
    for (Py_buffer* b : {&bp, &bi, &bx, &cp, &ri}) PyBuffer_Release(b);
    if (!err.empty()) {
        Py_XDECREF(dd);
        Py_XDECREF(lv);
        PyErr_SetString(PyExc_ValueError, err.c_str());
        return nullptr;
    }
    return Py_BuildValue("(NN)", dd, lv);
}

PyObject* py_ldl_solve(PyObject*, PyObject* args) {
    Py_buffer cp, ri, lv, db, xb;
    if (!PyArg_ParseTuple(args, "y*y*y*y*y*", &cp, &ri, &lv, &db, &xb)) return nullptr;
    const int64_t n = static_cast<int64_t>(db.len / 8);
    PyObject* out = nullptr;
    std::string err;
    if (cp.len != (n + 1) * 8 || ri.len != lv.len || n == 0 || xb.len % (n * 8) != 0) {
        err = "ldl_solve: buffer sizes do not match";
    } else {
        const int64_t m = static_cast<int64_t>(xb.len / (n * 8));
        out = PyBytes_FromStringAndSize(static_cast<const char*>(xb.buf), xb.len);
        if (out == nullptr) {
            err = "ldl_solve: out of memory";
        } else {
            double* x = reinterpret_cast<double*>(PyBytes_AS_STRING(out));
            Py_BEGIN_ALLOW_THREADS
            ldl_solve_kernel(static_cast<const int64_t*>(cp.buf), static_cast<const int64_t*>(ri.buf),
                             static_cast<const double*>(lv.buf), static_cast<const double*>(db.buf),
                             n, m, x);
            Py_END_ALLOW_THREADS
        }
    }
    for (Py_buffer* b : {&cp, &ri, &lv, &db, &xb}) PyBuffer_Release(b);
    if (!err.empty()) {
        Py_XDECREF(out);
        PyErr_SetString(PyExc_ValueError, err.c_str());
        return nullptr;
    }
    return out;
}

PyMethodDef methods[] = {
    {"colleau_times", py_colleau_times, METH_VARARGS,
     "colleau_times(sire, dam, d, x, k) -> bytes of n*k float64: A x (x row-major n x k) "
     "by the pedigree recursions of A = T D T' (Colleau 2002)."},
    {"inbreeding_depth", py_inbreeding_depth, METH_VARARGS,
     "inbreeding_depth(sire, dam[, mode]) -> (bytes of float64 F, depths, ml_pairs, colleau_columns, "
     "colleau_depths): inbreeding by pedigree depth, per depth the cheaper of Meuwissen & Luo "
     "traces and Colleau columns."},
    {"inbreeding_ml", py_inbreeding_ml, METH_VARARGS,
     "inbreeding_ml(sire, dam) -> bytes of float64 inbreeding coefficients "
     "(Meuwissen & Luo 1992). Parents must precede offspring; -1 = unknown."},
    {"ml_general", py_ml_general, METH_VARARGS,
     "ml_general(sire, dam, c, e, fext) -> bytes of 2n float64 (diag(A), then d); "
     "generalised Meuwissen-Luo trace for metafounders (see abp/core/metafounders.py)."},
    {"symbolic_cholesky", py_symbolic_cholesky, METH_VARARGS,
     "symbolic_cholesky(indptr, indices, n) -> (colptr, rowidx) bytes of int64: strictly lower "
     "Cholesky pattern of a symmetric CSC matrix (see abp/solvers/selinv.py)."},
    {"takahashi", py_takahashi, METH_VARARGS,
     "takahashi(colptr, rowidx, lval, d) -> (zdiag, zval) bytes of float64: selected inverse "
     "of L D L' on the closed pattern (see abp/solvers/selinv.py)."},
    {"mindegree_order", py_mindegree_order, METH_VARARGS,
     "mindegree_order(indptr, indices, n) -> bytes of int64: minimum-degree elimination order "
     "(see abp/solvers/cholesky.py)."},
    {"ldl_numeric", py_ldl_numeric, METH_VARARGS,
     "ldl_numeric(Bp, Bi, Bx, colptr, rowidx) -> (d, lval) bytes: up-looking LDL' on the "
     "symbolic pattern (see abp/solvers/cholesky.py)."},
    {"ldl_solve", py_ldl_solve, METH_VARARGS,
     "ldl_solve(colptr, rowidx, lval, d, x) -> bytes: solves L D L' x = b for the right-hand "
     "sides stored row-wise in x."},
    {"bayes_sweep", py_bayes_sweep, METH_VARARGS,
     "bayes_sweep(Wt, wtw, e, beta, delta, var_j, log_pi, comp_var, sigma_e2, method, z, u): "
     "one in-place Gibbs sweep over markers (see abp/solvers/bayes.py)."},
    {nullptr, nullptr, 0, nullptr}};

PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native",
                      "ABP optional native kernels (see ADR 0002).", -1, methods,
                      nullptr, nullptr, nullptr, nullptr};

}  // namespace

PyMODINIT_FUNC PyInit__native(void) { return PyModule_Create(&module); }
