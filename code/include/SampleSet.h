#ifndef INCLUDE_SAMPLESET_H_
#define INCLUDE_SAMPLESET_H_

#include <algorithm>
#include <random>
#include <vector>
#include "default_types.h"



template<typename S, typename T>  // S - type of features, T - type of answers
class GroupSamples {
 private:
    Mat<S> objs;
    T group_tag;

 public:
    GroupSamples() : objs(0), group_tag() {}
    GroupSamples(Mat<S> init_objs, T init_tag) :
                objs(init_objs), group_tag(init_tag) {}
    GroupSamples(Vec<S> init_vec, T init_tag) :
                objs(Mat<S>(init_vec)), group_tag(init_tag) {}
    ~GroupSamples() {}



    void shuffle() {std::random_device rd;
    std::mt19937 g(rd()); std::shuffle(&(objs[0]), &(objs[-1])+1, g); }
    void slice_rand(int slice_num, Mat<S>* slice_x, Vec<T>* slice_y) const;
    void append(const Vec<S>& X) { objs.hadd(X); }
    void append(const Mat<S>& X);

    int get_size() const { return objs.get_sx(); }
    int get_sy() const { return objs.get_sy(); }
    T get_tag() const { return group_tag; }
    const Mat<S>& get_objs() const { return objs; }
    void get_data(Mat<S>* data_dummy, Vec<T>* target_dummy) const;

    Vec<S>& operator[](int index) { return objs[index]; }
    const Vec<S>& operator[](int index) const { return objs[index]; }

    template<typename U, typename V>
    friend std::ostream& operator<<(std::ostream& os, const GroupSamples<U, V>& gr_samps);
};

template<typename S, typename T>  // S - type of features, T - type of answers
class SampleSet {
 private:
    Vec< GroupSamples<S, T> > groups;
    Vec<int> out_order;

 public:
    explicit SampleSet(int size = 0) : groups(size) {}  // default constructor
    explicit SampleSet(const Vec<S>& X, const Vec<T>& y);
    explicit SampleSet(const Mat<S>& X, const Vec<T>& y);
    explicit SampleSet(Vec< GroupSamples<S, T> > init_groups);
    SampleSet(const SampleSet<S, T>& copy_obj) :  // copy constructor
                groups(copy_obj.groups),
                out_order(copy_obj.out_order) {}

    explicit SampleSet(SampleSet<S, T>&& move_obj) :  // move constructor
                groups(std::move(move_obj.get_groups())),
                out_order(std::move(move_obj.out_order)) {}
    ~SampleSet() {}

    void append(const Mat<S>& X, const Vec<T>& y);
    void shuffle();
    void slice_rand(int slice_num, Mat<S>* slice_x, Vec<T>* slice_y) const;
    int get_group_num() const { return groups.get_size(); }
    int get_total_size() const;
    Vec< GroupSamples<S, T> >& get_groups() { return groups; }
    Vec<int>& get_out_order() { return out_order; }
    const Vec< GroupSamples<S, T> >& get_groups() const { return groups; }
    const Vec<int>& get_out_order() const { return out_order; }
    const GroupSamples<S, T>& get_group(T index_tag) const;
    SampleSet<S, T> get_antigroup(T index_tag) const;
    void get_data(Mat<S>* data_dummy, Vec<T>* target_dummy) const;
    Vec<T> get_tags() const;
    bool delete_group(T index_tag);
    void print_order() { }//LOG_(trace) << "order:" << out_order; }

    SampleSet<S, T>& operator= (const SampleSet<S, T>& copy_obj);  // copy assignment
    Vec<S>& operator[](int abs_index);  // TODO: Counter-intuitive function -> redefine
    const Vec<S>& operator[](int abs_index) const;  // same here

    template<typename U, typename V>
    friend std::ostream& operator<<(std::ostream& os, const SampleSet<U, V>& sset);
};


template<typename S, typename T>
void GroupSamples<S, T>::slice_rand(int slice_num, Mat<S>* slice_x, Vec<T>* slice_y) const {
    std::vector<int>idx(objs.get_sx());
    std::iota(std::begin(idx), std::end(idx), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(std::begin(idx), std::end(idx), g);

    *slice_x = Mat<S>(slice_num, objs.get_sy());
    *slice_y = Vec<T>(slice_num);

    for (int i = 0; i < slice_num; i++) {
        (*slice_x)[i] = objs[idx[i]];
        (*slice_y)[i] = group_tag;
    }
}

template<typename S, typename T>
void GroupSamples<S, T>::append(const Mat<S>& X) {
    int mat_sx = X.get_sx();

    for (int i = 0; i < mat_sx; i++)
        objs.append(X[i]);
}

template<typename S, typename T>
void GroupSamples<S, T>::get_data(Mat<S>* data_dummy, Vec<T>* target_dummy) const {
    int obj_sx = objs.get_sx(), obj_sy = objs.get_sy();
    *data_dummy = Mat<S>(obj_sx, obj_sy);
    *target_dummy = Vec<T>(obj_sx);
    for (int i = 0; i < obj_sx; i++) {
        (*data_dummy)[i] = objs[i];
        (*target_dummy)[i] = group_tag;
    }
}

template<typename U, typename V>
std::ostream& operator<<(std::ostream& os, const GroupSamples<U, V>& gr_samps) {
    int gsamps_size = gr_samps.objs.get_sx();
    std::stringstream buffer;
    buffer << "GSamps{" << gsamps_size << ", " << gr_samps.group_tag << "}:[" << std::endl;
    buffer << gr_samps.objs;
    os << buffer.str();
    return os;
}

template<typename S, typename T>
SampleSet<S, T>::SampleSet(const Vec<S>& X, const Vec<T>& y) {
    int obj_num = y.get_size();
    Mat<S> dummy_mat(obj_num, 1);
    for (int i = 0; i < obj_num; i++)
        dummy_mat[i][0] = X[i];

    groups.append(GroupSamples<S, T>(dummy_mat[0], y[0]));
    for (int i = 1; i < obj_num; i++) {
        bool was_found = 0;
        for (int j = 0; j < groups.get_size(); j++) {
            if (groups[j].get_tag() == y[i]) {
                groups[j].append(dummy_mat[i]);
                was_found = 1;
                break;
            }
        }
        if (!was_found)
            groups.append(GroupSamples<S, T>(dummy_mat[i], y[i]));
    }
    out_order = Vec<int>(obj_num);
    std::iota(&(out_order[0]), &(out_order[-1])+1, 0);
}

template<typename S, typename T>
SampleSet<S, T>::SampleSet(const Mat<S>& X, const Vec<T>& y) {
    int obj_num = y.get_size();
    groups.append(GroupSamples<S, T>(X[0], y[0]));
    for (int i = 1; i < obj_num; i++) {
        bool was_found = 0;

        for (int j = 0; j < groups.get_size(); j++) {
            if (groups[j].get_tag() == y[i]) {
                groups[j].append(X[i]);
                was_found = 1;
                break;
            }
        }
        if (!was_found)
            groups.append(GroupSamples<S, T>(X[i], y[i]));
    }

    out_order = Vec<int>(obj_num);
    std::iota(&(out_order[0]), &(out_order[-1])+1, 0);
}

template<typename S, typename T>
SampleSet<S, T>::SampleSet(Vec< GroupSamples<S, T> > init_groups) :
        groups(init_groups) {
    int obj_num = get_total_size();
    out_order = Vec<int>(obj_num);
    std::iota(&(out_order[0]), &(out_order[-1])+1, 0);

}

template<typename S, typename T>
void SampleSet<S, T>::append(const Mat<S>& X, const Vec<T>& y) {
    int obj_num = y.get_size();
    int total_num = get_total_size();
    int i = 0;

    if (!total_num) {  // if appending for the first time...
        // LOG_(trace) << "Appending for the first time...";
        groups.append(GroupSamples<S, T>(X[0], y[0]));
        i++;
        out_order.append(0);
    }

    for (; i < obj_num; i++) {
        bool was_found = 0;

        for (int j = 0; j < groups.get_size(); j++) {  // searching within existing groupes
            if (groups[j].get_tag() == y[i]) {
                groups[j].append(X[i]);
                was_found = 1;
                break;
            }
        }
        if (!was_found)  // otherwise adding new group container =)
            groups.append(GroupSamples<S, T>(X[i], y[i]));

        out_order.append(i+total_num);
    }
    // LOG_(trace) << "Matrix of objects was appended: " << (*this);
}

template<typename S, typename T>
void SampleSet<S, T>::shuffle() {
    std::iota(&(out_order[0]), &(out_order[-1])+1, 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(&(out_order[0]), &(out_order[-1])+1, g);
}

template<typename S, typename T>
int SampleSet<S, T>::get_total_size() const {
    int groups_num = groups.get_size();
    int total_size = 0;
    for (int i = 0; i < groups_num; i++)
        total_size += groups[i].get_size();

    return total_size;
}

template<typename S, typename T>
const GroupSamples<S, T>& SampleSet<S, T>::get_group(T index_tag) const {
    int groups_size = groups.get_size();
    for (int i = 0; i < groups_size; i++)
        if (groups[i].get_tag() == index_tag)
            return groups[i];

    LOG_(warning) << "No group with tag " << index_tag << " were found.";
    return groups[0];
}

template<typename S, typename T>
SampleSet<S, T> SampleSet<S, T>::get_antigroup(T index_tag) const {
    SampleSet<S, T> new_sample_set(*this);
    new_sample_set.delete_group(index_tag);
    return new_sample_set;
}

template<typename S, typename T>
void SampleSet<S, T>::get_data(Mat<S>* data_dummy, Vec<T>* target_dummy) const {
    int groups_num = groups.get_size();
    int total_size = get_total_size();
    int features_size = groups[0].get_sy();
    LOG_(trace) << "out_order:" << out_order;
    (*data_dummy) = Mat<S>(total_size, features_size);
    (*target_dummy) = Vec<T>(total_size);
    int curr_idx = 0;
    for (int i = 0; i < groups_num; i++) {
        int group_size = groups[i].get_size();
        T group_tag = groups[i].get_tag();

        for (int j = 0; j < group_size; j++) {
            (*data_dummy)[out_order[curr_idx]] = groups[i][j];
            (*target_dummy)[out_order[curr_idx]] = group_tag;
            curr_idx++;
        }
    }
}

template<typename S, typename T>
void SampleSet<S, T>::slice_rand(int slice_num, Mat<S>* slice_x, Vec<T>* slice_y) const {
    Mat<S> fused_data;
    Vec<T> fused_target;
    get_data(&fused_data, &fused_target);

    std::vector<int>idx(fused_data.get_sx());
    std::iota(std::begin(idx), std::end(idx), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(std::begin(idx), std::end(idx), g);

    *slice_x = Mat<S>(slice_num, fused_data.get_sy());
    *slice_y = Vec<T>(slice_num);

    for (int i = 0; i < slice_num; i++) {
        (*slice_x)[i] = fused_data[idx[i]];
        (*slice_y)[i] = fused_target[idx[i]];
    }
}

template<typename S, typename T>
Vec<T> SampleSet<S, T>::get_tags() const {
    int group_num = groups.get_size();
    Vec<T> tags(group_num);

    for (int i = 0; i < group_num; i++)
        tags[i] = groups[i].get_tag();
    return tags;
}

template<typename S, typename T>
bool SampleSet<S, T>::delete_group(T index_tag) {
    int groups_size = groups.get_size();
    for (int i = 0; i < groups_size; i++) {
        if (groups[i].get_tag() == index_tag) {
            groups.erase(i);
            return 1;
        }
    }
    LOG_(warning) << "No group samples with tag " << index_tag << " were found.";
    return 0;
}

template<typename S, typename T>
Vec<S>& SampleSet<S, T>::operator[](int abs_index) {
    int groups_num = groups.get_size();
    for (int i = 0; i < groups_num; i++) {
        if (abs_index < groups[i].get_size())
            return groups[i][abs_index];

        abs_index -= groups[i].get_size();
    }

    return groups[-1][-1];
}

template<typename S, typename T>
const Vec<S>& SampleSet<S, T>::operator[](int abs_index) const {
    int groups_num = groups.get_size();
    for (int i = 0; i < groups_num; i++) {
        if (abs_index < groups[i].get_size())
            return groups[i][abs_index];

        abs_index -= groups[i].get_size();
    }

    return groups[-1][-1];
}

template<typename S, typename T>
SampleSet<S, T>& SampleSet<S, T>::operator= (const SampleSet<S, T>& copy_obj) {
    groups = copy_obj.get_groups();
    out_order = copy_obj.get_out_order();
    return (*this);
}

template<typename U, typename V>
std::ostream& operator<<(std::ostream& os, const SampleSet<U, V>& sset) {
    std::stringstream buffer;
    int sset_size = sset.groups.get_size();
    int total_size = sset.get_total_size();
    buffer << "SampleSet{" << total_size << "; " << sset_size << "}:[" << std::endl;

    // if (sset_size < 1) {
    //     buffer << "nullptr";
    // } else if (sset_size < 10) {
    //     buffer << sset.groups[0];
    //     for (int i = 1; i < sset_size; i++)
    //         buffer << ", " << sset.groups[i] << std::endl;
    // } else {
    //     buffer << sset.groups[0] << "," << std::endl;
    //     buffer << sset.groups[1] << "," << std::endl;
    //     buffer << sset.groups[2] << "," << std::endl << "..., " << std::endl;
    //     buffer << sset.groups[sset_size-3] << "," << std::endl;
    //     buffer << sset.groups[sset_size-2] << "," << std::endl;
    //     buffer << sset.groups[sset_size-1] << "," << std::endl;
    // }

    if (sset_size < 1) {
        buffer << "nullptr";
    } else {
        buffer << sset.groups[0];
        for (int i = 1; i < sset_size; i++)
            buffer << "," << std::endl << sset.groups[i] << std::endl;
    }
    buffer << "]";
    os << buffer.str();
    return os;
}

#endif  // INCLUDE_SAMPLESET_H_
