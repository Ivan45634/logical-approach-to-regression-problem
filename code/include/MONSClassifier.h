#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

#include "default_types.h"
#include "CollFamily.h"
#include "SampleHandler.h"
#include "LBBuilder.h"
#include "GeneticDualizer.h"

#ifndef INCLUDE_MONSCLASSIFIER_H_
#define INCLUDE_MONSCLASSIFIER_H_

template<typename S, typename T>  // S - features, T - coverings
class MONSClassifier {
    GeneticDualizer<S, T> genetic_dualizer;
    LBBuilder<S, int>* lb_builder;
    SplitterSH<S, int> sample_handler;

    Vec< CollFamily<S, int> > coll_sets;
    SampleSet<S, int> train_set;

    int max_iter;
    float eps;

 public:
    explicit MONSClassifier(GeneticDualizer<S, T> init_gen_dual,
                            LBBuilder<S, int>* init_lb_builder = new ComplementLBBuilder<S>(),
                            int init_miter = 1000, float init_eps = 0.001);
    ~MONSClassifier();

    void fit(const Mat<S>& X, const Vec<int>& y);
    Vec<int> predict(const Mat<S>& X);

    bool save_mons_data(const char path[]) const;
    bool load_mons_data(const char path[]);

    void print_colls() {LOG_(debug) << "COLL SETS:" << coll_sets; }

 private:
    bool check_margin(const SampleSet<S, int>& valid, const SampleSet<S, int>& train,
                      Vec<float>* avg_margins, int curr_iter);
    float get_class_estim(const SampleSet<S, int>& train, const Vec<S>& x, int class_tag);
    Vec<float> get_class_estim(const SampleSet<S, int>& train, const Mat<S>& X, int class_tag);

    bool load_coll_sets(std::ifstream* file_ptr);
    bool load_train_set(std::ifstream* file_ptr);
};


template<typename S, typename T>
MONSClassifier<S, T>::MONSClassifier(GeneticDualizer<S, T> init_gen_dual,
                                     LBBuilder<S, int>* init_lb_builder,
                                     int init_miter, float init_eps):
        genetic_dualizer(init_gen_dual),
        lb_builder(init_lb_builder),
        sample_handler(0.9),
        coll_sets(),
        max_iter(init_miter),
        eps(init_eps) {
    srand(time(0));
    LOG_(debug) << "MONSClassifier instance was created (by usual constructor).";
}

template<typename S, typename T>
MONSClassifier<S, T>::~MONSClassifier() {
    delete lb_builder;
    LOG_(debug) << "MONSClassifier was destroyed.";
}

template<typename S, typename T>
void MONSClassifier<S, T>::fit(const Mat<S>& X, const Vec<int>& y) {
    LOG_(trace) << "Process of model fitting has begun...";
    LOG_(trace) << "Initial data set:" << train_set;
    SampleSet<S, int> train, valid;
    Vec< SampleSet<S, int> > data = sample_handler.make_samples(X, y)[0];
    train = data[0];
    valid = data[1];

    // LOG_(trace) << "train:" << train;
    // LOG_(trace) << "valid:" << valid;

    int train_num = train.get_group_num();
    Vec<int> train_tags = train.get_tags();

    train_set = train;
    coll_sets = Vec< CollFamily<S, int> >(train_num);
    for (int i = 0; i < train_num; i++)
        coll_sets[i] = CollFamily<S, int>(0, train_tags[i]);

    Vec<float> avg_margins(valid.get_total_size());
    for (int i = 0; i < valid.get_total_size(); i++)
        avg_margins[i] = 0.0;

    LOG_(trace) << "Starting main fitting loop(class num: " << train_num << ")...";
    for (int i = 1; i <= max_iter; i++) {
        // LOG_(trace) << "Loop iteration: " << i;
        for (int k = 0; k < train_num; k++) {
            // LOG_(trace) << "For class: " << train_tags[k];
            ElColl<S> local_basis = lb_builder->build_lb(train, train_tags[k]);
            // LOG_(trace) << "Local basis: " << local_basis;

            // LOG_(trace) << "Processing Genetic Algorithm...";
            genetic_dualizer.set_init_data(train, local_basis, train_tags[k]);
            Vec< Vec<T> > encoded_colls = genetic_dualizer.execute_ga();
            Vec< ElColl<S> > new_colls = genetic_dualizer.decode_collections(encoded_colls);

            // LOG_(trace) << "Adding new collections...:" << new_colls;
            coll_sets[k].add(new_colls);
            // LOG_(trace) << "New collections were successfully added.";

            // LOG_(trace) << "LOOK HERE! Curr class:" << train_tags[k];
            // LOG_(trace) << "Coll_Families:" << coll_sets;
        }
        if (check_margin(valid, train, &avg_margins, i)) {
            // LOG_(trace) << "Main loop was stopped because of margins' condition";
            break;
        }
    }
    LOG_(trace) << "Successfull end of fit loop for MONSClassifier";
}

template<typename S, typename T>
Vec<int> MONSClassifier<S, T>::predict(const Mat<S> & X) {
    LOG_(trace) << "Predicting stuff! Get prepared!";
    LOG_(trace) << "Collection sets: " << coll_sets;

    int class_num = coll_sets.get_size();
    int obj_num = X.get_sx();
    Vec<int> class_preds(obj_num);

    LOG_(trace) << "Getting predictions for tag 0...";
    Vec<float> estim_vec = get_class_estim(train_set, X, 0);
    int curr_tag = coll_sets[0].get_tag();
    for (int i = 0; i < obj_num; i++)
        class_preds[i] = curr_tag;
    LOG_(trace) << "done!";

    for (int i = 1; i < class_num; i++) {
        curr_tag = coll_sets[i].get_tag();
        LOG_(trace) << "Getting predictions for tag " << curr_tag << "...";
        Vec<float> new_estim_vec = get_class_estim(train_set, X, i);
        LOG_(trace) << "done!";
        for (int j = 0; j < obj_num; j++) {
            LOG_(trace) << " Checking object " << j << " (" << new_estim_vec[j] << " vs. " << estim_vec[j] << ")";
            if (new_estim_vec[j] > estim_vec[j]) {
                LOG_(trace) << "  New estimation is better! Changed!";
                estim_vec[j] = new_estim_vec[j];
                class_preds[j] = curr_tag;
            }
        }
        LOG_(trace) << "After new predictions:" << class_preds;
    }
    return class_preds;
}

template<typename S, typename T>
bool MONSClassifier<S, T>::save_mons_data(const char path[]) const {
    std::ofstream out_file;
    out_file.open(path);
    // LOG_(trace) << "Outputting MONS Classifier data...";

    int total_cs_num = coll_sets.get_size();
    out_file << total_cs_num << std::endl;
    for (int i = 0; i < total_cs_num; i++) {
        out_file << coll_sets[i] << std::endl;
    }

    out_file << train_set;
    out_file.close();
    return 1;
}

template<typename T>
T to_type(const std::string& s) {
    std::istringstream is(s);
    T tmp;
    is >> tmp;
    return tmp;
}

template<typename S, typename T>
bool MONSClassifier<S, T>::load_mons_data(const char path[]) {
    std::ifstream file(path);
    load_coll_sets(&file);
    load_train_set(&file);
    return 1;
}

template<typename S, typename T>
bool MONSClassifier<S, T>::load_coll_sets(std::ifstream* file_ptr) {
    std::string entry_val;
    std::getline(*file_ptr, entry_val);  // get amount of CFamilies
    int fams_num = to_type<int>(entry_val);

    coll_sets = Vec<CollFamily<S, int> >(fams_num);

    for (int i = 0; i < fams_num; i++) {  // loading coll_sets and class_tags
        std::getline(*file_ptr, entry_val, ':');
        std::getline(*file_ptr, entry_val);

        std::getline(*file_ptr, entry_val, '<');
        std::getline(*file_ptr, entry_val, ',');
        int curr_cf_sz = to_type<int>(entry_val);
        std::getline(*file_ptr, entry_val, '>');
        int class_tag = to_type<int>(entry_val);

        Vec<ElColl<S> > cf_colls(curr_cf_sz);

        for (int j = 0; j < curr_cf_sz; j++) {
            std::getline(*file_ptr, entry_val, '<');
            std::getline(*file_ptr, entry_val, '>');

            int curr_elcol_sz = to_type<int>(entry_val);
            ElColl<S> curr_elcol(curr_elcol_sz);

            for (int k = 0; k < curr_elcol_sz; k++) {
                std::getline(*file_ptr, entry_val, '<');
                std::getline(*file_ptr, entry_val, '>');

                int curr_ec_sz = to_type<int>(entry_val);
                Vec<int> ec_cols(curr_ec_sz);
                Vec<S> ec_vals(curr_ec_sz);

                for (int l = 0; l < curr_ec_sz; l++) {
                    std::getline(*file_ptr, entry_val, '(');
                    std::getline(*file_ptr, entry_val, ')');
                    std::replace(entry_val.begin(), entry_val.end(), '-', ' ');
                    std::istringstream is(entry_val);
                    is >> ec_cols[l] >> ec_vals[l];
                }
                curr_elcol[k] = ElClass<S>(curr_ec_sz, ec_cols, ec_vals);
            }
            cf_colls[j] = curr_elcol;
        }
        coll_sets[i] = CollFamily<S, int>(curr_cf_sz, cf_colls, class_tag);
    }
    return 1;
}

template<typename S, typename T>
bool MONSClassifier<S, T>::load_train_set(std::ifstream* file_ptr) {
    std::string entry_val;

    std::getline(*file_ptr, entry_val, ';');
    std::getline(*file_ptr, entry_val, '}');
    int groups_num = to_type<int>(entry_val);
    Vec<GroupSamples<S, int> > gs_vec(groups_num);

    for (int i = 0; i < groups_num; i++) {
        std::getline(*file_ptr, entry_val, '{');
        std::getline(*file_ptr, entry_val, ',');
        std::getline(*file_ptr, entry_val, '}');
        int gs_tag = to_type<int>(entry_val);

        std::getline(*file_ptr, entry_val, '(');
        std::getline(*file_ptr, entry_val, ',');
        int sx = to_type<int>(entry_val);
        std::getline(*file_ptr, entry_val, ')');
        int sy = to_type<int>(entry_val);

        Mat<S> gs_mat(sx, sy);
        for (int j = 0; j < sx; j++) {
            std::getline(*file_ptr, entry_val, '[');
            for (int k = 0; k < sy-1; k++) {
                std::getline(*file_ptr, entry_val, ',');
                gs_mat[j][k] = to_type<int>(entry_val);
            }
            std::getline(*file_ptr, entry_val, ']');
            gs_mat[j][sy-1] = to_type<int>(entry_val);
        }
        gs_vec[i] = GroupSamples<S, int>(gs_mat, gs_tag);
    }
    train_set = SampleSet<S, int>(gs_vec);
    return 1;
}

template<typename S, typename T>
bool MONSClassifier<S, T>::check_margin(const SampleSet<S, int>& valid,
                                        const SampleSet<S, int>& train,
                                        Vec<float>* avg_margins,
                                        int curr_iter) {
    // LOG_(trace) << "Checking margin...";
    // LOG_(trace) << "Prev margins:" << *avg_margins;

    int valid_num = valid.get_group_num();
    Vec<int> valid_tags = valid.get_tags();

    int train_num = train.get_group_num();
    int cs_tags_num = coll_sets.get_size();
    Vec<int> train_tags = train.get_tags();
    Vec<int> tag_matches(train_num);

    for (int i = 0; i < train_num; i++) {
        for (int j = 0; j < cs_tags_num; j++) {
            if(coll_sets[j].get_tag() == train_tags[i]) {
                tag_matches[i] = j;
                break;
            }
        }
    }
    int total_size = valid.get_total_size();
    bool was_satisfied = 1;
    int curr_idx = 0;
    float curr_margin = 0.0;
    Vec<float> new_margins(total_size);
    Vec<float> delta;

    for (int i = 0; i < valid_num; i++) {
        // LOG_(trace) << "Checking class " << valid_tags[i] << "...";
        const GroupSamples<S, int>& valid_samples = valid.get_group(valid_tags[i]);
        // LOG_(trace) << "valid samples:" << valid_samples;
        int valid_samples_size = valid_samples.get_size();
        for (int j = 0; j < valid_samples_size; j++) {
            float my_estim = 0.0, max_estim = -100.0;
            for (int k = 0; k < train_num; k++) {
                float k_estim = get_class_estim(train, valid_samples[j], tag_matches[k]);
                // LOG_(trace) << "Indices: i:" << i << ", k:" << k;
                // LOG_(trace) << "Tag matches:" << tag_matches;
                // LOG_(trace) << "Class tags:" << tag_matches[i] << ", " << tag_matches[k];
                if (tag_matches[k] == tag_matches[i])
                    my_estim = k_estim;
                else if (max_estim < k_estim)
                    max_estim = k_estim;

                // LOG_(trace) << "It's Temmie!";
            }
            curr_margin = my_estim - max_estim;
            new_margins[curr_idx] = ((*avg_margins)[curr_idx]*(curr_iter-1)+curr_margin)/curr_iter;
            delta.append(new_margins[curr_idx] - (*avg_margins)[curr_idx]);
            if (new_margins[curr_idx] - (*avg_margins)[curr_idx] >= eps) {
                LOG_(trace) << "For idx " << curr_idx << " condition is not satisfied!";
                LOG_(trace) << "Delta: " << new_margins[curr_idx] - (*avg_margins)[curr_idx];
                LOG_(trace) << "Eps:" << eps;
                was_satisfied = 0;
            }
            curr_idx++;
        }
    }
    LOG_(trace) << "Delta vec:  " << delta;
    LOG_(trace) << "New margins:" << new_margins;
    *avg_margins = new_margins;

    return was_satisfied;
}

template<typename S, typename T>
float MONSClassifier<S, T>::get_class_estim(const SampleSet<S, int>& train,
                                            const Vec<S>& x, int cf_id) {
    int class_tag = coll_sets[cf_id].get_tag();
    // LOG_(trace) << "Getting class " << class_tag << " estimation...";
    // LOG_(trace) << "train:" << train;
    const GroupSamples<S, int>& needed_class = train.get_group(class_tag);
    // LOG_(trace) << "needed_class:" << needed_class;
    CollFamily<S, int>& needed_family = coll_sets[cf_id];
    // LOG_(trace) << "needed_family:" << needed_family;
    int class_size = needed_class.get_size();
    int coll_num = needed_family.get_size();

    float estim_sum = 0.0;
    for (int i = 0; i < class_size; i++)
        for (int j = 0; j < coll_num; j++)
            estim_sum += needed_family[j].vote_func(x, needed_class[i]);
    // LOG_(trace) << "Estim:" << estim_sum/float(class_size)/float(coll_num);
    return estim_sum/float(class_size)/float(coll_num);
}

template<typename S, typename T>
Vec<float> MONSClassifier<S, T>::get_class_estim(const SampleSet<S, int>& train,
                                                 const Mat<S>& X, int cf_id) {
    int class_tag = coll_sets[cf_id].get_tag();
    // LOG_(trace) << "Launching class estim for tag:" << class_tag;
    // LOG_(trace) << "train:" << train;
    const GroupSamples<S, int>& needed_class = train.get_group(class_tag);
    CollFamily<S, int>& needed_family = coll_sets[cf_id];

    int class_size = needed_class.get_size();
    int coll_num = needed_family.get_size();
    int mat_sx = X.get_sx();

    // LOG_(trace) << "Class size:" << needed_class;
    // LOG_(trace) << "CF size:" << coll_num;
    // LOG_(trace) << "Mat sx:" << mat_sx;

    Vec<float> estim_vec_sum(mat_sx);
    for (int i = 0; i < mat_sx; i++)
        estim_vec_sum[i] = 0;

    for (int i = 0; i < class_size; i++)
        for (int j = 0; j < coll_num; j++)
            for (int k = 0; k < mat_sx; k++)
                estim_vec_sum[k] += needed_family[j].vote_func(X[k], needed_class[i]);

    // LOG_(trace) << "Estim vec sum for tag " << class_tag << ": " << estim_vec_sum;
    float divisor = class_size * coll_num;
    // LOG_(trace) << "Divisor: " << divisor;

    for (int i = 0; i < mat_sx; i++)
        estim_vec_sum[i] /= divisor;
    // LOG_(trace) << "Final estim vec sum:" << estim_vec_sum;
    return estim_vec_sum;
}


#endif  // INCLUDE_MONSCLASSIFIER_H_
