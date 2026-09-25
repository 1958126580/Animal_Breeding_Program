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
// The GIL is released while computing.  Memory: O(n) doubles/ints plus a
// hash map with one entry per distinct (sire, dam) pair.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
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

PyMethodDef methods[] = {
    {"inbreeding_ml", py_inbreeding_ml, METH_VARARGS,
     "inbreeding_ml(sire, dam) -> bytes of float64 inbreeding coefficients "
     "(Meuwissen & Luo 1992). Parents must precede offspring; -1 = unknown."},
    {nullptr, nullptr, 0, nullptr}};

PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native",
                      "ABP optional native kernels (see ADR 0002).", -1, methods,
                      nullptr, nullptr, nullptr, nullptr};

}  // namespace

PyMODINIT_FUNC PyInit__native(void) { return PyModule_Create(&module); }
