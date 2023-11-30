#ifndef INCLUDE_COLLFAMILY_H_
#define INCLUDE_COLLFAMILY_H_

#include <algorithm>
#include "default_types.h"

template<typename T>  // type of features
class ElClass {
    Vec<int> cols;
    Vec<T> vals;

 public:
    explicit ElClass(int rank = 0);

    ElClass(int rank, int init_cols[], T init_vals[]);
    ElClass(int rank, Vec<int> init_cols, Vec<T> init_vals);

    ~ElClass() {}

    bool apply_to_object(const Vec<T>& object) const;
    int get_size() const { return cols.get_size(); }
    const Vec<int>& get_cols() const { return cols; }
    const Vec<T>& get_vals() const { return vals; }

    bool operator==(const ElClass<T>& comp_obj) const;

    template<typename S>
    friend std::ostream& operator<<(std::ostream& os, const ElClass<S>& ec);
};


template<typename T>
class ElColl {
    Vec< ElClass<T> > ecs;

 public:
    explicit ElColl(int size = 0) : ecs(size) {}
    ElColl(int size, ElClass<T> init_ecs[]);
    ~ElColl() {}

    bool add(ElClass<T> ec_obj);  // checks for similar elementary classifiers

    Vec<bool> apply_to_object(const Vec<T>& object) const;
    bool vote_func(const Vec<T>& fst_obj, const Vec<T>& sec_obj) const;

    int get_size() const { return ecs.get_size(); }

    bool operator==(const ElColl<T>& comp_obj) const;
    ElClass<T>& operator[](int idx) { return ecs[idx]; }
    const ElClass<T>& operator[](int idx) const { return ecs[idx]; }

    template<typename S>
    friend std::ostream& operator<<(std::ostream& os, const ElColl<S>& el_coll);
};


template<typename S, typename T>  // S - for features, T - for class tags
class CollFamily {
    Vec< ElColl<S> > colls;
    T class_tag;

 public:
    explicit CollFamily(int size = 0, T init_tag = -1.00) : colls(size), class_tag(init_tag) {}
    CollFamily(int size, ElColl<S> init_colls[], T init_tag);
    CollFamily(int size, Vec< ElColl<S> > init_colls, T init_tag);
    ~CollFamily() {}
    Vec<bool> add(const Vec<ElColl<S> >& new_colls);  // ...

    void set_tag(T init_tag) { class_tag = init_tag; }
    int get_size() const { return colls.get_size(); }
    T get_tag() const { return class_tag; }
    ElColl<S>& operator[](int idx) { return colls[idx]; }
    const ElColl<S>& operator[](int idx) const { return colls[idx]; }

    template<typename U, typename V>
    friend std::ostream& operator<<(std::ostream& os, const CollFamily<U, V>& cfam);
};



template<typename T>
ElClass<T>::ElClass(int rank):
        cols(rank), vals(rank) {
    for (int i = 0; i < rank; i++)
        cols[i] = -1;
}

template<typename T>
ElClass<T>::ElClass(int rank, int init_cols[], T init_vals[]):
        cols(rank), vals(rank) {
    for (int i = 0; i < rank; i++) {
        cols[i] = init_cols[i];
        vals[i] = init_vals[i];
    }
}

template<typename T>
ElClass<T>::ElClass(int rank, Vec<int> init_cols, Vec<T> init_vals):
        cols(init_cols), vals(init_vals) {
}

template<typename T>
bool ElClass<T>::apply_to_object(const Vec<T>& object) const {
    int ec_rank = cols.get_size();
    for (int i = 0; i < ec_rank; i++)
        if (object[cols[i]] != vals[i])
            return 0;
    return 1;
}

template<typename T>
bool ElClass<T>::operator==(const ElClass<T>& comp_obj) const {
    int my_size = cols.get_size();
    if (my_size != comp_obj.get_size())
        return 0;

    const Vec<int>& cols2 = comp_obj.get_cols();
    const Vec<T>& vals2 = comp_obj.get_vals();

    Vec<int> col_inds = cols.sort_indices();
    Vec<int> col_inds2 = comp_obj.get_cols().sort_indices();

    for (int i = 0; i < my_size; i++) {
        if (cols[col_inds[i]] != cols2[col_inds[i]] ||
            vals[col_inds[i]] != vals2[col_inds[i]])
            return 0;
    }
    return 1;
}

template<typename S>
std::ostream& operator<<(std::ostream& os, const ElClass<S>& ec) {
    std::stringstream buffer;
    int ec_rank = ec.cols.get_size();
    buffer << "ElClass<" << ec_rank << ">:[";
    buffer << "(" << ec.cols[0] << "--" << ec.vals[0] << ")";
    for (int i = 1; i < ec_rank; i++)
        buffer << ", (" << ec.cols[i] << "--" << ec.vals[i] << ")";
    buffer << "]";
    os << buffer.str();
    return os;
}

template<typename T>
ElColl<T>::ElColl(int size, ElClass<T> init_ecs[]) : ecs(size) {
    for (int i = 0; i < size; i++)
        ecs[i] = init_ecs[i];
}

template<typename T>
bool ElColl<T>::add(ElClass<T> ec_obj) {
    int ec_num = ecs.get_size();
    for (int i = 0; i < ec_num; i++)  // checking if there any similar elementary classifiers
        if (ecs[i] == ec_obj)
            return 0;
    ecs.append(ec_obj);
    return 1;
}

template<typename T>
Vec<bool> ElColl<T>::apply_to_object(const Vec<T>& object) const {
    int ec_num = ecs.get_size();
    Vec<bool> bin_coll(ec_num);

    for (int i = 0; i < ec_num; i++)
        bin_coll[i] = ecs[i].apply_to_object(object);

    return bin_coll;
}

template<typename T>
bool ElColl<T>::vote_func(const Vec<T>& fst_obj, const Vec<T>& sec_obj) const {
    Vec<bool> fst_coll_vec = apply_to_object(fst_obj);
    Vec<bool> sec_coll_vec = apply_to_object(sec_obj);
    int fst_sz = fst_coll_vec.get_size();
    if (fst_sz != sec_coll_vec.get_size()) {
        LOG_(warning) << "In vote function trying to compare different vectors!";
        return 0;
    }

    for (int i = 0; i < fst_sz; i++)
        if (fst_coll_vec[i] < sec_coll_vec[i])
            return 0;
    return 1;
}

template<typename T>
bool ElColl<T>::operator==(const ElColl<T>& comp_obj) const {
    int ec_num = ecs.get_size();
    if (ec_num != comp_obj.get_size())
        return 0;

    bool was_found = 0;
    for (int i = 0; i < ec_num; i++) {
        for (int j = 0; j < ec_num; j++) {
            if (comp_obj[i] == ecs[j]) {
                was_found = 1;
                break;
            }
        }
        if (!was_found)
            return 0;
        was_found = 1;
    }
    return 1;
}

template<typename S>
std::ostream& operator<<(std::ostream& os, const ElColl<S>& el_coll) {
    std::stringstream buffer;
    int el_coll_size = el_coll.ecs.get_size();
    buffer << "ElColl<" << el_coll_size << ">:[" << std::endl;
    // if (el_coll_size < 1) {
    //     buffer << "nullptr" << std::endl;
    // } else if (el_coll_size < 10) {
    //     for (int i = 0; i < el_coll_size; i++)
    //         buffer << el_coll.ecs[i] << "," << std::endl;
    // } else {
    //     buffer << el_coll.ecs[0] << "," << std::endl;
    //     buffer << el_coll.ecs[1] << "," << std::endl;
    //     buffer << el_coll.ecs[2] << "," << std::endl << "...," << std::endl;
    //     buffer << el_coll.ecs[el_coll_size-3] << "," << std::endl;
    //     buffer << el_coll.ecs[el_coll_size-2] << "," << std::endl;
    //     buffer << el_coll.ecs[el_coll_size-1] << "," << std::endl;
    // }
    if (el_coll_size < 1) {
        buffer << "nullptr" << std::endl;
    } else {
        for (int i = 0; i < el_coll_size; i++)
            buffer << el_coll.ecs[i] << "," << std::endl;
    }
    buffer << "]";
    os << buffer.str();
    return os;
}


template<typename S, typename T>
CollFamily<S, T>::CollFamily(int size, ElColl<S> init_colls[], T init_tag):
        colls(size), class_tag(init_tag) {
    for (int i = 0; i < size; i++)
        colls[i] = init_colls[i];
}

template<typename S, typename T>
CollFamily<S, T>::CollFamily(int size, Vec< ElColl<S> > init_colls, T init_tag):
        colls(init_colls), class_tag(init_tag) {
}

template<typename S, typename T>
Vec<bool> CollFamily<S, T>::add(const Vec<ElColl<S> >& new_colls) {
    int new_coll_num = new_colls.get_size();
    Vec<bool> add_state(new_coll_num);

    for (int i = 0; i < new_coll_num; i++) {
        add_state[i] = 1;
        int coll_num = colls.get_size();  // size of coll vector differs from iter to iter

        for (int j = 0; j < coll_num; j++) {
            if (colls[j] == new_colls[i]) {
                add_state[i] = 0;
                break;
            }
        }

        if (add_state[i])
            colls.append(new_colls[i]);
    }
    return add_state;
}

template<typename U, typename V>
std::ostream& operator<<(std::ostream& os, const CollFamily<U, V>& cfam) {
    std::stringstream buffer;
    int cfam_size = cfam.colls.get_size();
    buffer << "CFamily<" << cfam_size << ", " << cfam.get_tag() << ">:[" << std::endl;

    if (cfam_size <= 0) {
        buffer << "nullptr";
    } else {
        buffer << cfam.colls[0];
        for (int i = 1; i < cfam_size; i++)
            buffer << "," << std::endl << cfam.colls[i];
    }
    buffer << "]";
    os << buffer.str();
    return os;
}

#endif  // INCLUDE_COLLFAMILY_H_
