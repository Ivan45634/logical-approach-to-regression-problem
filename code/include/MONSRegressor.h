#include "default_types.h"
#include "MONSClassifier.h"
#include "SampleHandler.h"
#include "clustering_algorithms.h"
#ifndef INCLUDE_MONSREGRESSOR_H_
#define INCLUDE_MONSREGRESSOR_H_

enum RegressionFeatureType {
    kPhenoType  = 0,
    kGenoType   = 1,
    kGenoPhenoType = 2
};
enum RegressionCastType {
    kAverage = 0,
    kMedian = 1,
    kNearestNeighboors = 2
};
enum ClusterAlgoType {
    kMeans = 0,
    kDBSCAN = 1,
    kDMDBSCAN = 2,
    kParzen = 3,
    kVParzen = 4,
    // TODO: Implement more algorithms and find out the best one
};

template<typename S, typename T, typename U>  // S - features, T - answers, U - coverings
class MONSRegressor {
    MONSClassifier<S, U> mons_instance;
    RegressionFeatureType reg_features;
    RegressionCastType reg_cast;
    ClusterAlgoType cluster_algorithm;
    Vec<float> ca_params;

    Mat<S> saved_data;
    Vec<T> saved_target;
    Vec<int> cluster_target;
    Vec<S> delta_data_vec;
    Vec<S> shift_data_vec;

 public:
    MONSRegressor(GeneticDualizer<S, U> init_gen_dual,
                  ClusterAlgoType init_ca, Vec<float> init_ca_params,
                  LBBuilder<S, int>* init_lb_builder = new ComplementLBBuilder<S>(),
                  int init_miter = 1000, float init_eps = 0.0001,
                  RegressionFeatureType init_rf = kGenoType, RegressionCastType init_rc = kAverage);
     ~MONSRegressor() {};

    void fit(const Mat<S>& X, const Vec<T>& y);
    Vec<T> predict(const Mat<S>& X);

 private:
    void fix_dups(Mat<S>* X, Vec<T>* y);
    void get_norm_params();
    Mat<S> get_feature_data(const Mat<S>& X, const Vec<T>& y);
    Vec<T> knn(const Mat<S>& X, const Vec<int>& cl_y, cluster_algos::MetricType metric, int k_param);
};


template<typename S, typename T, typename U>
MONSRegressor<S, T, U>::MONSRegressor(GeneticDualizer<S, U> init_gen_dual,
                                      ClusterAlgoType init_ca, Vec<float> init_ca_params,
                                      LBBuilder<S, int>* init_lb_builder,
                                      int init_miter, float init_eps, RegressionFeatureType init_rf,
                                      RegressionCastType init_rc) :
                    mons_instance(init_gen_dual, init_lb_builder, init_miter, init_eps),
                    reg_features(init_rf), reg_cast(init_rc),
                    cluster_algorithm(init_ca), ca_params(init_ca_params) {
}

template<typename S, typename T, typename U>
void MONSRegressor<S, T, U>::fit(const Mat<S>& X, const Vec<T>& y) {
    // LOG_(info) << "Fit of MONSReg started...";
    Mat<S> goodX = X;
    Vec<T> goody = y;
    fix_dups(&goodX, &goody);
    saved_data = get_feature_data(goodX, goody);
    saved_target = goody;
    // saved_data = get_feature_data(X, y);
    // saved_target = y;
    // if (cluster_algorithm == kDMDBSCAN) {
    //     get_norm_params();
    //     int sx = saved_data.get_sx();
    //     int sy = saved_data.get_sy();
    //     Mat<S> normed_d = saved_data;
    //     for (int i = 0; i < sx; i++)
    //         for (int j = 0; j < sy; j++)
    //             normed_d[i][j] = (saved_data[i][j] - shift_data_vec[j]) / delta_data_vec[j];
    //     saved_data = normed_d;
    // }
    switch (cluster_algorithm) {
        case kDMDBSCAN:
            cluster_target = cluster_algos::dmdbscan<S>(saved_data,
                                                        cluster_algos::MetricType(ca_params[0]),
                                                        int(ca_params[1]) );
            break;

        case kParzen:
            cluster_target = cluster_algos::parzen<S>(Mat<S>(goody), float(ca_params[0]),
                                                    cluster_algos::KernelType(ca_params[1]));
            // cluster_target = cluster_algos::parzen<S>(Mat<S>(y), float(ca_params[0]),
            //                                         cluster_algos::KernelType(ca_params[1]));
            break;

        case kVParzen:
            cluster_target = cluster_algos::vparzen<S>(Mat<S>(goody), int(ca_params[0]),
                                                    cluster_algos::KernelType(ca_params[1]));
            // cluster_target = cluster_algos::parzen<S>(Mat<S>(y), float(ca_params[0]),
            //                                         cluster_algos::KernelType(ca_params[1]));
            break;

        default:
            LOG_(error) << "Unknown Cluster Algorithm Type(code:" << cluster_algorithm << ")";
    }
    LOG_(trace) << "Clusterization worked correctly: " << cluster_target;
    // LOG_(trace) << "y:" << goody;
    mons_instance.fit(goodX, cluster_target);

    // if(cluster_algorithm != kDMDBSCAN)
    get_norm_params();
}

template<typename S, typename T, typename U>
Vec<T> MONSRegressor<S, T, U>::predict(const Mat<S>& X) {
    int target_size = cluster_target.get_size();

    int sx = X.get_sx();
    // int sy = X.get_sy();
    Vec<int> cluster_pred(sx);

    // if (cluster_algorithm == kDMDBSCAN) {
    //     Mat<S> normed_d(sx, sy);
    //     for (int i = 0; i < sx; i++)
    //         for (int j = 0; j < sy; j++)
    //             normed_d[i][j] = (X[i][j] - shift_data_vec[j]) / delta_data_vec[j];
    //     cluster_pred = mons_instance.predict(normed_d);
    // } else {
    cluster_pred = mons_instance.predict(X);
    // }

    Vec<T> pred_vec(sx);
    LOG_(trace) << "pred_vec size:" << pred_vec.get_size();
    LOG_(trace) << "Predictions from MONSClassifier: " << cluster_pred;

    switch (reg_cast) {
        case kAverage: {
            // LOG_(trace) << "kAverage activated";
            // LOG_(trace) << "cluster_target:" << cluster_target;
            // LOG_(trace) << "saved_target:" << saved_target;
            for (int i = 0; i < sx; i++) {
                T group_sum = static_cast<T>(0.0);
                float denominator = 0.0;
                for (int j = 0; j < target_size; j++) {
                    if (cluster_target[j] == cluster_pred[i]) {
                        group_sum += saved_target[j];
                        denominator += 1.0;
                    }
                }
                // LOG_(trace) << "Denominator:" << denominator;
                pred_vec[i] = group_sum / denominator;
            }
            break;
        }
        case kMedian: {
            for (int i = 0; i < sx; i++) {
                Vec<T> class_target(0);
                for (int j = 0; j < target_size; j++) {
                    if (cluster_target[j] == cluster_pred[i]) {
                        class_target.append(cluster_target[j]);
                    }
                }
                int ct_sz = class_target.get_size();
                pred_vec[i] = class_target.sort()[ct_sz/2];
            }
            break;
        }
        case kNearestNeighboors: {
            LOG_(trace) << "kNearestNeighboors activated.";
            pred_vec = knn(X, cluster_pred, cluster_algos::kEuclidean, int(ca_params[2]));
            break;
        }
        default:
            LOG_(error) << "Unknown Regression Cast Type(code:" << reg_cast << ")";
            break;
    }

    return pred_vec;
}

template<typename S, typename T, typename U>
Vec<T> MONSRegressor<S, T, U>::knn(const Mat<S>& X, const Vec<int>& cl_y,
                                   cluster_algos::MetricType metric, int k_param) {
    int obj_num = cl_y.get_size();
    int saved_num = saved_target.get_size();

    int sx = X.get_sx();
    int sy = X.get_sy();

    if (k_param > saved_num)
        k_param = saved_num;

    Vec<T> out_vec(obj_num);
    Vec<int> sort_idx;

    int sdsx = saved_data.get_sx();
    int sdsy = saved_data.get_sy();

    Mat<S> normed_d(sx, sy);
    Mat<S> normed_sd(sdsx, sdsy);


    LOG_(trace) << "Metric:" << metric;
    LOG_(trace) << "k param:" << k_param;
    for (int i = 0; i < sdsx; i++)
        for (int j = 0; j < sdsy; j++)
            normed_sd[i][j] = (saved_data[i][j] - shift_data_vec[j]) / delta_data_vec[j];

    for (int i = 0; i < sx; i++)
        for (int j = 0; j < sdsy; j++)
            normed_d[i][j] = (X[i][j] - shift_data_vec[j]) / delta_data_vec[j];


    LOG_(trace) << "Metric:" << metric;
    LOG_(trace) << "k param" << k_param;
    for (int i = 0; i < obj_num; i++) {
        Mat<S> groupX(0,sy);
        Vec<T> groupy(0);
        for (int j = 0; j < saved_num; j++) {
            if (cluster_target[j] == cl_y[i]) {
                groupX.hadd(normed_sd[j]);
                groupy.append(saved_target[j]);
            }
        }
        int group_sz = groupX.get_sx();

        Vec<float> dist_vec(group_sz);
        for (int j = 0; j < group_sz; j++)
            dist_vec[j] = 1.0/(cluster_algos::find_dist(normed_d[i], groupX[j], metric)+1.);
        // LOG_(trace) << "Dist vec " << i << ":" << dist_vec;
        sort_idx = dist_vec.sort_indices(false);
        // LOG_(trace) << "Indices for sort:" << sort_idx;
        out_vec[i] = static_cast<T>(0.0);
        float total_sum = 0.0;
        for (int j = 0; (j < k_param) && (j < group_sz); j++) {
            // LOG_(trace) << "Summation launched!";
            total_sum += dist_vec[sort_idx[j]];
        }
        for (int j = 0; (j < k_param) && (j < group_sz); j++) {
            out_vec[i] += groupy[sort_idx[j]] * dist_vec[sort_idx[j]]/total_sum;
        }
        // LOG_(trace) << "Out_vec[" << i << "]:" << out_vec[i];
    }
    return out_vec;
}
template<typename S, typename T, typename U>
Mat<S> MONSRegressor<S, T, U>::get_feature_data(const Mat<S>& X, const Vec<T>& y) {
    int sx = X.get_sx();

    switch (reg_features) {
        case kPhenoType: {
            Mat<S> out_mat(sx, 1);
            for (int i = 0; i < sx; i++)
                out_mat[i][0] = static_cast<S>(y[i]);
            return out_mat;
        }

        case kGenoType: {
            return X;
        }

        case kGenoPhenoType: {
            Mat<S> out_mat = X;
            Vec<S> casted_vec(sx);
            for (int i = 0; i < sx; i++)
                casted_vec[i] = static_cast<S>(y[i]);
            return out_mat.vadd(casted_vec);
        }

        default: {
            LOG_(error) << "Unknown Regression Cast Type(code:" << reg_features << ")";
            break;
        }
    }

    return Mat<S>();
}

template<typename S, typename T, typename U>
void MONSRegressor<S, T, U>::get_norm_params() {
    int sx = saved_data.get_sx();
    int sy = saved_data.get_sy();

    Vec<S> maximums = saved_data[0];
    Vec<S> minimums = saved_data[0];

    for (int i = 1; i < sx; i++) {
        for (int j = 0; j < sy; j++) {
            if (saved_data[i][j] < minimums[j])
                minimums[j] = saved_data[i][j];
            else if (saved_data[i][j] > maximums[j])
                maximums[j] = saved_data[i][j];
        }
    }

    delta_data_vec = Vec<S>(sy);
    shift_data_vec = Vec<S>(sy);

    for (int i = 0; i < sy; i++) {
        // delta_data_vec[i] = (maximums[i]-minimums[i])/2.0;
        // shift_data_vec[i] = (maximums[i]+minimums[i])/2.0;
        //
        delta_data_vec[i] = (maximums[i]-minimums[i]);
        shift_data_vec[i] = minimums[i];
    }
    return;
}

template<typename S, typename T, typename U>
void MONSRegressor<S, T, U>::fix_dups(Mat<S>* X, Vec<T>* y) {
    int sx = X->get_sx();
    int sy = X->get_sy();
    Vec<int> correct_mask(sx);
    int total_correct = 0;
    for (int i = 0; i < sx; i++) {
        correct_mask[i] = -1;
    }
    T avg_val = static_cast<T>(0);
    int avg_num = 0;

    for (int i = 0; i < sx; i++) {
        if (correct_mask[i] == 0)
            continue;
        // -1 case
        correct_mask[i] = 1;
        total_correct++;

        avg_val = (*y)[i];
        avg_num = 1;

        for (int j = i+1; j < sx; j++) {
            if ( (*X)[i] == (*X)[j] ) {
                correct_mask[j] = 0;
                avg_val += (*y)[j];
                avg_num++;
            }
        }
        (*y)[i] = avg_val/static_cast<T>(avg_num);
    }
    // LOG_(debug) << "Correctness mask:" << correct_mask;

    Mat<S> old_X = *X;
    Vec<T> old_y = *y;
    (*X) = Mat<S>(total_correct, sy);
    (*y) = Vec<T>(total_correct);
    int curr_counter = 0;
    for (int i = 0; i < sx; i++) {
        if (correct_mask[i] == 1) {
            (*X)[curr_counter] = old_X[i];
            (*y)[curr_counter] = old_y[i];
            curr_counter++;
        }
    }
    return;
}

#endif  // INCLUDE_MONSREGRESSOR_H_
