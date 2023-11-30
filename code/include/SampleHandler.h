#include <type_traits>

#include "default_types.h"
#include "CollFamily.h"
#include "SampleSet.h"
#ifndef INCLUDE_SAMPLEHANDLER_H_
#define INCLUDE_SAMPLEHANDLER_H_

//=========================================================================
// SampleHandler returns Mat(X, 2) that consists of X dataset pairs
// Dataset pair consists of train and test dataset as well
//=========================================================================
template<typename S, typename T>  // S - type of features, T - type of answers
class SampleHandler {
 public:
    SampleHandler() {}
    virtual ~SampleHandler() {}

    virtual Mat< SampleSet<S, T> > make_samples(const Mat<S>& X, const Vec<T>& y,
                                                bool mix = true, bool uniform = true) = 0;
};


template<typename S, typename T>
class SplitterSH : public SampleHandler<S, T> {
    float split_frac;
 public:
    SplitterSH(float init_sf) : split_frac(init_sf) {}
    ~SplitterSH() {}

    virtual Mat< SampleSet<S, T> > make_samples(const Mat<S>& X, const Vec<T>& y,
                                                bool mix = true, bool uniform = true);

    float get_split_frac() const { return split_frac; };
    void set_split_frac(float set_sf) { split_frac = set_sf; }
};

template<typename S, typename T>
class CrossValSH : public SampleHandler<S, T> {
    int folds_num;
 public:
    CrossValSH(int init_fn) : folds_num(init_fn) {}
    ~CrossValSH() {}

    virtual Mat< SampleSet<S, T> > make_samples(const Mat<S>& X, const Vec<T>& y,
                                                bool mix = true, bool uniform = true);

    float get_folds_num() const { return folds_num; };
    void set_folds_num(int set_fn) { folds_num = set_fn; }
};

// 'uniform' feature is senseless for LOO, so just ignore it
template<typename S, typename T>
class LeaveOneOutSH : public SampleHandler<S, T> {
 public:
    LeaveOneOutSH() {}
    ~LeaveOneOutSH() {}

    virtual Mat< SampleSet<S, T> > make_samples(const Mat<S>& X, const Vec<T>& y,
                                                bool mix = true, bool uniform = true);
};


template<typename S, typename T>
Mat< SampleSet<S, T> > SplitterSH<S, T>::make_samples(const Mat<S>& X, const Vec<T>& y,
                                                      bool mix, bool uniform) {
    Mat< SampleSet<S, T> > data_mat(1, 2);
    SampleSet<S, T> wrapper_set(X, y);
    if (mix)
        wrapper_set.shuffle();

    if (!uniform) {
        Mat<S> shuffled_X;
        Vec<T> shuffled_y;
        wrapper_set.get_data(&shuffled_X, &shuffled_y);
        Mat<S> train_X = shuffled_X.get_rect(0, 0, shuffled_X.get_sx()*split_frac, -1);
        Vec<T> train_y = shuffled_y.slice(0, shuffled_y.get_size()*split_frac);
        data_mat[0][0] = SampleSet<S, T>(train_X, train_y);
        data_mat[0][0].print_order();
        Mat<S> test_X = shuffled_X.get_rect(shuffled_X.get_sx()*split_frac+1, 0, -1, -1);
        Vec<T> test_y = shuffled_y.slice(shuffled_y.get_size()*split_frac+1, -1);
        data_mat[0][1] = SampleSet<S, T>(test_X, test_y);
        data_mat[0][1].print_order();
        return data_mat;
    }
    Vec<T> tags_vec = wrapper_set.get_tags();
    int group_num = wrapper_set.get_group_num();

    for (int i = 0; i < group_num; i++) {  // class in validation set
        const GroupSamples<S, T>& curr_group = wrapper_set.get_group(tags_vec[i]);
        int curr_size = curr_group.get_size();

        if (curr_size <= 3) {  // extra case
            Vec<T> curr_tags(curr_size);

            for (int j = 0; j < curr_size; j++)  // filling all y vec with
                curr_tags[j] = tags_vec[i];

            data_mat[0][0].append(curr_group.get_objs(), curr_tags);
            data_mat[0][1].append(curr_group.get_objs(), curr_tags);
            continue;
        }
        int train_set_sz = curr_size*split_frac;
        if (train_set_sz > curr_size-1) {
            train_set_sz = curr_size-1;
        } else if (train_set_sz < 1) {
            train_set_sz = 1;
        }

        Mat<S> group_X;
        Vec<T> group_y;
        curr_group.get_data(&group_X, &group_y);

        for (int j = 0; j < curr_size; j++)
            group_y[j] = tags_vec[i];
        Mat<S> train_X = group_X.get_rect(0, 0, train_set_sz, -1);
        Vec<T> train_y = group_y.slice(0, train_set_sz);
        data_mat[0][0].append(train_X, train_y);

        Mat<S> test_X = group_X.get_rect(train_set_sz, 0, -1, -1);
        Vec<T> test_y = group_y.slice(train_set_sz, -1);
        data_mat[0][1].append(test_X, test_y);
    }
    return data_mat;
}

template<typename S, typename T>
Mat< SampleSet<S, T> > LeaveOneOutSH<S, T>::make_samples(const Mat<S>& X, const Vec<T>& y,
                                                         bool mix, bool uniform) {
    int data_sx = X.get_sx();
    Mat< SampleSet<S, T> > data_mat(data_sx, 2);
    SampleSet<S, T> wrapper_set(X, y);

    if (mix)
        wrapper_set.shuffle();
    Mat<S> shuff_X;
    Vec<T> shuff_y;
    wrapper_set.get_data(&shuff_X, &shuff_y);
    Mat<T> trans_y(data_sx, 1);
    for (int i = 0; i < data_sx; i++) {
        trans_y[i][0] = shuff_y[i];
    }
    data_mat[0][0] = SampleSet<S, T>(shuff_X.get_rect(1, 0, -1, -1), shuff_y.slice(1, -1));
    data_mat[0][0].print_order();
    data_mat[0][1] = SampleSet<S, T>(Mat<S>(shuff_X[0]), Vec<T>(trans_y[0]));
    data_mat[0][1].print_order();
    for (int i = 1; i < data_sx; i++) {
        data_mat[i][0] = SampleSet<S, T>(shuff_X.get_rect(0, 0, i, -1), shuff_y.slice(0, i));
        if (i != data_sx-1)
            data_mat[i][0].append(shuff_X.get_rect(i+1, 0, -1, -1), shuff_y.slice(i+1, -1));
        data_mat[i][1] = SampleSet<S, T>(Mat<S>(shuff_X[i]), Vec<T>(trans_y[i]));
    }
    return data_mat;
}


template<typename S, typename T>
Mat< SampleSet<S, T> > CrossValSH<S, T>::make_samples(const Mat<S>& X, const Vec<T>& y,
                                                      bool mix, bool uniform) {
    if (folds_num < 1 || folds_num > X.get_sx()) {
        LOG_(error) << "Incorrect \"folds_num\" parameter's value:" << folds_num;
        return Mat< SampleSet<S, T> >(0, 2);
    }

    Mat< SampleSet<S, T> > data_mat(folds_num, 2);
    SampleSet<S, T> wrapper_set(X, y);
    if (mix)
        wrapper_set.shuffle();

    if (!uniform) {
        LOG_(trace) << "Non-uniform";
        Mat<S> sh_X;
        Vec<T> sh_y;
        wrapper_set.get_data(&sh_X, &sh_y);
        int fold_sz = X.get_sx() / folds_num;
        LOG_(trace) << "fold_sz: " << fold_sz;
        data_mat[0][0] = SampleSet<S, T>(sh_X.get_rect(fold_sz,0,-1,-1), sh_y.slice(fold_sz,-1));
        data_mat[0][1] = SampleSet<S, T>(sh_X.get_rect(0,0,fold_sz,-1), sh_y.slice(0,fold_sz));

        for (int i = 1; i < folds_num; i++) {
            data_mat[i][0] = SampleSet<S, T>(sh_X.get_rect(0,0,i*fold_sz,-1),
                                             sh_y.slice(0,i*fold_sz));
            data_mat[i][0].append(sh_X.get_rect((i+1)*fold_sz,0,-1,-1),
                                  sh_y.slice((i+1)*fold_sz,-1));

            data_mat[i][1] = SampleSet<S, T>(sh_X.get_rect(i*fold_sz,0,(i+1)*fold_sz,-1),
                                             sh_y.slice(i*fold_sz, (i+1)*fold_sz));
        }
        return data_mat;
    }

    Vec<T> tags_vec = wrapper_set.get_tags();
    int group_num = wrapper_set.get_group_num();

    Vec<int> folds_sz(group_num);
    Vec< GroupSamples<S, T> > gs_vec(group_num);
    for (int j = 0; j < group_num; j++) {
        gs_vec[j] = wrapper_set.get_group(tags_vec[j]);
        folds_sz[j] = gs_vec[j].get_size()/folds_num;
    }
    LOG_(trace) << "folds sizes:" << folds_sz;
    LOG_(trace) << "done!";
    LOG_(trace) << "Getting other groups";
    for (int j = 0; j < group_num; j++) {
        int curr_size = gs_vec[j].get_size();
        Mat<S> group_X;
        Vec<T> group_y;
        gs_vec[j].get_data(&group_X, &group_y);

        if (curr_size <= 3 || folds_sz[j] < 1) {
            for (int i = 0; i < folds_num; i++) {
                data_mat[i][0].append(group_X, group_y);
                data_mat[i][1].append(group_X, group_y);
            }
            continue;
        }

        data_mat[0][0].append(group_X.get_rect(folds_sz[j],0,-1,-1), group_y.slice(folds_sz[j],-1));
        data_mat[0][1].append(group_X.get_rect(0,0,folds_sz[j],-1), group_y.slice(0,folds_sz[j]));

        for (int i = 1; i < folds_num; i++) {
            data_mat[i][0].append(group_X.get_rect(0,0,i*folds_sz[j],-1),
                                  group_y.slice(0,i*folds_sz[j]));
            data_mat[i][0].append(group_X.get_rect((i+1)*folds_sz[j],0,-1,-1),
                                  group_y.slice((i+1)*folds_sz[j],-1));
            data_mat[i][1].append(group_X.get_rect(i*folds_sz[j],0,(i+1)*folds_sz[j],-1),
                                  group_y.slice(i*folds_sz[j],(i+1)*folds_sz[j]));
        }
    }
    return data_mat;
}
#endif  // INCLUDE_SAMPLEHANDLER_H_
