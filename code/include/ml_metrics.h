#include "default_types.h"
#include "log_wrapper.h"

#ifndef INCLUDE_ML_METRICS_H_
#define INCLUDE_ML_METRICS_H_

namespace ml_metrics {

template<typename T>
float ml_accuracy(Vec<T> fst, Vec<T> sec) {
    if (fst.get_size() != sec.get_size()) {
        LOG_(error) << "Cannot compare target vectors: different sizes!";
        return -1.0;
    }

    int sz = fst.get_size();
    float res = 0.0;
    for (int i = 0; i < sz; i++)
        res += fst[i] == sec[i];
    return res/sz;
}

template<typename T>
float ml_mse(Vec<T> fst, Vec<T> sec) {
    int sz = fst.get_size();
    float res = 0.0;
    for (int i = 0; i < sz; i++)
        res += (fst[i] - sec[i])*(fst[i] - sec[i]);
    return res/float(sz);
}

template<typename T>
float ml_mean(Vec<T> vec) {
    T sum = T();
    for (int i = 0, n = vec.get_size(); i < n; i++) {
        sum += vec[i];
    }
    return sum / vec.get_size();
}

template<typename T>
float ml_r2_score(Vec<T> y_true, Vec<T> y_pred) {
    float total_variance = 0.0;
    float explained_variance = 0.0;
    T mean = ml_mean(y_true);
    for (int i = 0, n = y_true.get_size(); i < n; i++) {
        T residual = y_true[i] - y_pred[i];
        T total = y_true[i] - mean;
        explained_variance += residual * residual;
        total_variance += total * total;
    }
    return 1.0 - (explained_variance / total_variance);
}

}

#endif  // INCLUDE_ML_METRICS_H_
