/* Copyright 2017 Baytekov Nikita */

#include <iostream>
#include <ctime>
#include "log_wrapper.h"
#include "ml_metrics.h"

#include "MONSRegressor.h"
#include "DatasetManager.h"

#include "SampleSet.h"
#include "SampleHandler.h"

using namespace ml_metrics;
using namespace cluster_algos;

int main(int argc, char* argv[]) {
    init_logging(kInfo);
    srand(time(0));

    DatasetManager<float> data_reader;
    char default_path[1000] = "datasets/14_gh_music_lon.csv";
    Mat<float> in_data;
    in_data = (argc > 1) ? data_reader.read_csv(argv[1]) :
                           data_reader.read_csv(default_path);

    Vec<float> in_target = in_data.get_col(-1);
    in_data = in_data.get_rect(0, 0, -1, in_data.get_sy()-1);

    CrossValSH<float, float> cv_sh(10);
    // int data_num = data_mat.get_sx();
    // LOG_(trace) << "Made sample sets successfully!";
    // LOG_(trace) << "Here:" << data_mat;
    Vec<Vec<float>> pgrid(3, {
            Vec<float>(1, {cluster_algos::kEuclidean, cluster_algos::kManhattan}),
            Vec<float>(3, {3, 4, 5, 7, 10}),
            Vec<float>(1, {5, 7, 9, 13}), /* k parameter for kNN */
            /* mutation rate */
    });

    Vec<int> gs_idx(3, {0, 0, 0});
    float best_quality = 20000;
    while(true) {
        LOG_(info) << "Current CV indices:" << gs_idx;
        // Vec<float> ca_params(3, {pgrid[0][gs_idx[0]],
        //                          cluster_algos::kEpanechnikov,
        //                          pgrid[1][gs_idx[1]]});

        Vec<float> ca_params(3, {pgrid[0][gs_idx[0]],
                                 pgrid[1][gs_idx[1]],
                                 pgrid[2][gs_idx[2]]});

        GeneticDualizer<float, bool> gen_dual(5,
                                              0.6,
                                              true, kDataScore, kDelayConverged,
                                              0.005);

        MONSRegressor<float, float, bool> mons(gen_dual, kDMDBSCAN, ca_params,
                                               new ComplementLBBuilder<float>(),
                                               200, 0.007, kGenoType,
                                               kNearestNeighboors);

        Mat<float> dummy_data;
        Vec<float> dummy_target;
        Vec<float> dummy_answer;

        Vec<float> quality(10);

        Mat< SampleSet<float, float> > data_mat = cv_sh.make_samples(in_data, in_target, true, false);
        for (int i = 0; i < 10; i++) {
            data_mat[i][0].get_data(&dummy_data, &dummy_target);
            // LOG_(debug) << "HELLO! Train sx:" << dummy_data.get_sx() << " Train sy:" << dummy_data.get_sy();
            mons.fit(dummy_data, dummy_target);
            data_mat[i][1].get_data(&dummy_data, &dummy_target);
            // LOG_(debug) << "HELLO! Test sx:" << dummy_data.get_sx() << " Test sy:" << dummy_data.get_sy();
            // LOG_(trace) << "Going to predict everything";
            dummy_answer = mons.predict(dummy_data);
            // LOG_(trace) << "Answer:" << dummy_answer;
            // LOG_(trace) << "Real:" << dummy_target;
            quality[i] = ml_mse<float>(dummy_answer, dummy_target);
            LOG_(info) << "MSE " << i << ": " << quality[i];
        }
        float avg_quality = 0.0;
        for (int i = 0; i < 10; i++)
            avg_quality += quality[i];
        avg_quality /= 10.0;
        LOG_(info) << "Quality:" << quality;
        LOG_(info) << "Avg quality:" << avg_quality;

        if (best_quality > avg_quality ) {
            LOG_(info) << "Updated best quality!!";
            best_quality = avg_quality;
        }

        int curr_it = 2;
        while (curr_it >= 0) {
            gs_idx[curr_it]++;
            if (gs_idx[curr_it] >= pgrid[curr_it].get_size()) {
                gs_idx[curr_it] = 0;
                curr_it--;
            } else {
                break;
            }
        }
        if (curr_it < 0)
            break;
    }

    LOG_(info) << "Final quality:" << best_quality;
    return 0;
}
