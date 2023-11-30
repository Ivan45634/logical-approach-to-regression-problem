#ifndef INCLUDE_DEFAULT_TYPES_H_
#define INCLUDE_DEFAULT_TYPES_H_

#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
#include "log_wrapper.h"


template<typename T>
class Vec {
 private:
    T* data;
    int sz;
    int real_sz;

 public:
    explicit Vec(int size = 0);  // default constructor
    Vec(int size, T init_vals[]);  // -//-
    Vec(int size, const std::vector<T>& init_vals);
    Vec(const Vec<T>& vec_obj);  // copy constructor
    explicit Vec(Vec<T>&& vec_obj);  // move constructor
    ~Vec();  // destructor

    Vec<T>& operator=(const Vec<T>& vec_obj);  // copy assignment
    Vec<T>& operator=(Vec<T>&& vec_obj);  // move assignment
    T& operator[](int index);
    const T& operator[](int index) const;
    bool operator==(const Vec<T>& vec_obj) const;

    template<typename S>
    friend std::ostream& operator<<(std::ostream& os, const Vec<S>& vec);

    void append(const T& apnd_obj, int index = -1);
    bool erase(int index = -1);
    int where(const T& obj);
    Vec<T> slice(int begin, int end) const;
    Vec<T> sort() const;
    Vec<int> sort_indices(bool asc = true) const;
    int get_size() const {
        return sz;
    }
};


template<typename T>
class Mat {
 private:
    Vec< Vec<T> > data;
    int sx, sy;

 public:
    explicit Mat(int x = 0, int y = 0);
    Mat(int x, int y, T* init_vals[]);
    Mat(const Mat<T>& mat_obj);
    explicit Mat(const Vec<T>& vec_obj);
    ~Mat() {}

    Mat<T>& hadd(const Vec<T>& h_obj, int index = -1);
    Mat<T>& vadd(const Vec<T>& v_obj, int index = -1);

    Mat<T> get_rect(int x1, int y1, int x2 = -1, int y2 = -1) const;
    Vec<T> get_col(int idx = -1) const;

    Mat<T> Tr() const;

    Mat<T>& operator=(const Mat<T>& mat_obj);
    Vec<T>& operator[](int index);
    const Vec<T>& operator[](int index) const;

    template<typename S>
    friend std::ostream& operator<<(std::ostream& os, const Mat<S>& mat);

    int where(const Vec<T>& obj);
    Vec<int> where(const T& obj);

    int get_sx() const { return sx; }
    int get_sy() const { return sy; }
};


template<typename T>
Vec<T>::Vec(int size) {
    sz = size;
    real_sz = size;

    if (sz > 0)
        data = new T[sz];
    else
        data = nullptr;
}

template<typename T>
Vec<T>::Vec(int size, T init_vals[]) {
    sz = size;
    real_sz = size;

    if (sz > 0) {
        data = new T[sz];
        for (int i = 0; i < sz; i++)
            data[i] = init_vals[i];
    } else {
        data = nullptr;
    }
}

template<typename T>
Vec<T>::Vec(int size, const std::vector<T>& init_vals) {
    sz = size;
    real_sz = size;

    if (sz > 0) {
        data = new T[sz];
        for (int i = 0; i < sz; i++)
            data[i] = init_vals[i];
    } else {
        data = nullptr;
    }
}

template<typename T>
Vec<T>::Vec(const Vec<T>& vec_obj) {
    sz = vec_obj.sz;
    real_sz = vec_obj.real_sz;
    data = new T[real_sz];
    for (int i = 0; i < sz; i++)
        data[i] = vec_obj.data[i];
}

template<typename T>
Vec<T>::Vec(Vec<T>&& vec_obj) {
    sz = vec_obj.sz;
    real_sz = vec_obj.real_sz;
    data = vec_obj.data;
    vec_obj.data = nullptr;
}

template<typename T>
Vec<T>::~Vec() {
    if (data != nullptr)
        delete [] data;
}

template<typename T>
Vec<T>& Vec<T>::operator=(const Vec<T>& vec_obj) {
    if (this == &vec_obj)  // assigning to myself
        return *this;

    if (data != nullptr)
        delete [] data;

    sz = vec_obj.sz;
    real_sz = vec_obj.real_sz;

    data = new T[real_sz];
    for (int i = 0; i < sz; i++)
        data[i] = vec_obj.data[i];

    return *this;
}

template<typename T>
Vec<T>& Vec<T>::operator=(Vec<T>&& vec_obj) {
    T* tmp_data = data;
    sz = vec_obj.sz;
    real_sz = vec_obj.real_sz;
    data = vec_obj.data;

    vec_obj.data = tmp_data;
    tmp_data = nullptr;

    return (*this);
}

template<typename T>
T& Vec<T>::operator[](int index) {
    if (index == -1) {
        return data[sz-1];
    } else if (index > sz) {
        LOG_(error) << "Index " << index << " out of bounds for " << (*this) << "!";
        return data[0];
    }
    return data[index];
}

template<typename T>
const T& Vec<T>::operator[](int index) const {
    if (index == -1) {
        return data[sz-1];
    } else if (index > sz) {
        LOG_(error) << "Index " << index << " out of bounds for " << (*this) << "!";
        return data[0];
    }
    return data[index];
}

template<typename T>
bool Vec<T>::operator==(const Vec<T>& vec_obj) const {
    if (vec_obj.get_size() != sz)
        return 0;

    for (int i = 0; i < sz; i++)
        if (vec_obj[i] != data[i])
            return 0;
    return 1;
}

template<typename S>
std::ostream& operator<<(std::ostream& os, const Vec<S>& vec) {
    std::stringstream buffer;
    buffer << "Vec(" << vec.sz << "/" << vec.real_sz << "):[";

    if (vec.data == nullptr) {
        buffer << "nullptr";
    } else if (vec.sz < 10000) {
        buffer << vec.data[0];
        for (int i = 1; i < vec.sz; i++)
            buffer << ", " << vec.data[i];
    } else {
        buffer << vec.data[0] << ", " << vec.data[1] << ", " << vec.data[2] << ", ..., ";
        buffer << vec.data[vec.sz-3] << ", " << vec.data[vec.sz-2] << ", " << vec.data[vec.sz-1];
    }
    buffer << "]";
    os << buffer.str();
    return os;
}

template<typename T>
void Vec<T>::append(const T& apnd_obj, int index) {
    if (index == -1)
        index = sz;

    // LOG_(trace) << "Appending " << apnd_obj << " on " << index << " position to " << (*this);

    if (data == nullptr || real_sz <= 0) {
        // LOG_(trace) << "Zero-size vector => creating new one";
        data = new T[1];
        data[0] = apnd_obj;
        sz = 1;
        real_sz = 1;
    } else if (sz+1 <= real_sz) {
        // LOG_(trace) << "Virtually extending vec size...";
        for (int i = sz; i > index; i--)
            data[i] = data[i-1];
        data[index] = apnd_obj;
        sz++;
    } else {
        // LOG_(trace) << "Extending vec for real...";
        real_sz *= 2;
        T* old_data = data;
        data = new T[real_sz];

        for (int i = 0; i < index; i++)
            data[i] = old_data[i];
        for (int i = index; i < sz; i++)
            data[i+1] = old_data[i+1];
        data[index] = apnd_obj;

        delete [] old_data;
        old_data = nullptr;
        sz++;
    }
    // LOG_(trace) << "New vec after append: " << (*this);
}

template<typename T>
bool Vec<T>::erase(int index) {
    if (index == -1)
        index = sz-1;

    // LOG_(trace) << "Erasing element on position " << index << "...";
    if (data == nullptr || index < 0 || index >= sz) {
        LOG_(warning) << "Invalid 'erase' operation for " << (*this);
        return 0;
    }

    for (int i = index; i < sz-1; i++)
            data[i] = data[i+1];
    sz--;

    // LOG_(trace) << "New vec after erase: " << (*this);
    return 1;
}

template<typename T>
int Vec<T>::where(const T& obj) {
    for (int i = 0; i < sz; i++)
        if (data[i] == obj)
            return i;
    return -1;
}

template<typename T>
Vec<T> Vec<T>::slice(int begin, int end) const {
    // LOG_(trace) << "Getting slice(" << begin << ", " << end << ") from " << (*this) << "...";

    if (end == -1)
        end = sz;

    if (begin > end) {
        LOG_(error) << " Can't get slice(" << begin << "; " << end << ")";
        return Vec();
    }

    int slice_sz = end-begin;
    Vec<T> slice_vec(slice_sz);

    for (int i = 0; i < slice_sz; i++)
        slice_vec[i] = data[begin+i];

    // LOG_(trace) << "Slice: " << slice_vec;
    return slice_vec;
}

template<typename T>
Vec<T> Vec<T>::sort() const {
    T* new_data = new T[sz];
    for (int i = 0; i < sz; i++)
        new_data[i] = data[i];

    std::sort(new_data, new_data+sz);

    Vec<T> out_vec(sz, new_data);
    // LOG_(trace) << "Sorted vec: " << out_vec;
    return out_vec;
}

template<typename T>
Vec<int> Vec<T>::sort_indices(bool asc) const {
    std::vector<int>idx(sz);
    std::iota(std::begin(idx), std::end(idx), 0);
    if (asc)
        std::sort(std::begin(idx), std::end(idx),
                  [this](int i1, int i2) { return this->data[i1] < this->data[i2]; });
    else
        std::sort(std::begin(idx), std::end(idx),
                  [this](int i1, int i2) { return this->data[i1] > this->data[i2]; });
    Vec<int> out_vec(sz, idx);
    // LOG_(trace) << "Vec of sorted indices: " << out_vec;
    return out_vec;
}


template<typename T>
Mat<T>::Mat(int x, int y) : data(x), sx(x), sy(y) {
    for (int i = 0; i < sx; i++) {
        data[i] = Vec<T>(sy);
    }
}

template<typename T>
Mat<T>::Mat(int x, int y, T *init_vals[]) : data(x), sx(x), sy(y) {
    for (int i = 0; i < sx; i++) {
        data[i] = Vec<T>(sy);
        for (int j = 0; j < sy; j++)
            data[i][j] = init_vals[i][j];
    }
}

template<typename T>
Mat<T>::Mat(const Mat<T>& mat_obj) : data(Vec< Vec<T> >(mat_obj.sx)) {
    sx = mat_obj.sx;
    sy = mat_obj.sy;

    for (int i = 0; i < sx; i++) {
        data[i] = mat_obj[i];
    }
}

template<typename T>
Mat<T>::Mat(const Vec<T> & vec_obj) : data(Vec< Vec<T> >(1)) {
    sx = 1;
    sy = vec_obj.get_size();

    data[0] = vec_obj;
}

template<typename T>
Mat<T>& Mat<T>::hadd(const Vec<T>& h_obj, int index) {
    if (h_obj.get_size() != sy) {
        LOG_(error) << "Can't h-add vector to matrix: " << h_obj.get_size() << " != " << sy;
        return (*this);
    }
    data.append(h_obj, index);
    sx++;
    return (*this);
}

template<typename T>
Mat<T>& Mat<T>::vadd(const Vec<T>& v_obj, int index) {
    if (v_obj.get_size() != sx) {
        LOG_(error) << "Can't v-add vector to matrix: " << v_obj.get_size() << " != " << sx;
        return (*this);
    }

    for (int i = 0; i < sx; i++)
        data[i].append(v_obj[i], index);
    return (*this);
}

template<typename T>
Mat<T> Mat<T>::get_rect(int x1, int y1, int x2, int y2) const {
    if (x2 == -1)
        x2 = sx;
    if (y2 == -1)
        y2 = sy;
    // LOG_(trace) << "Getting rect(" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << ")";
    if (x1 >= x2 || y1 >= y2) {
        LOG_(error) << "Trying to create matrix with zero/negative sides!";
        LOG_(error) << "Bad parameters(" << x1 << "," << y1 << "," << x2 << "," << y2 << ")";
        return Mat<T>();
    }
    Mat<T> out_mat(x2-x1, y2-y1);
    for (int i = x1; i < x2; i++)
        for (int j = y1; j < y2; j++)
            out_mat[i-x1][j-y1] = data[i][j];

    return out_mat;
}

template<typename T>
Vec<T> Mat<T>::get_col(int idx) const {
    if (idx == -1)
        idx = sy-1;
    Vec<T> out_vec(sx);
    for (int i = 0; i < sx; i++)
        out_vec[i] = data[i][idx];
    return out_vec;
}

template<typename T>
Mat<T> Mat<T>::Tr() const {
    Mat<T> out_mat(sy, sx);
    for (int i = 0; i < sx; i++)
        for (int j = 0; j < sy; j++)
            out_mat[j][i] = data[i][j];
    return out_mat;
}

template<typename T>
Mat<T>& Mat<T>::operator=(const Mat<T> & mat_obj) {
    if (this == &mat_obj)  // assigning to myself
        return *this;

    data = Vec< Vec<T> >(mat_obj.sx);

    sx = mat_obj.sx;
    sy = mat_obj.sy;

    for (int i = 0; i < sx; i++) {
        data[i] = mat_obj[i];
    }

    return *this;
}

template<typename T>
Vec<T>& Mat<T>::operator[](int index) {
    if (index == -1) {
        return data[sx-1];
    } else if (index > sx) {
        LOG_(error) << "Index " << index << " out of bounds for " << (*this) << "!";
        return data[0];
    }
    return data[index];
}

template<typename T>
const Vec<T>& Mat<T>::operator[](int index) const {
    if (index == -1) {
        return data[sx-1];
    } else if (index > sx) {
        LOG_(error) << "Index " << index << " out of bounds for " << (*this) << "!";
        return data[0];
    }
    return data[index];
}

template<typename S>
std::ostream& operator<<(std::ostream& os, const Mat<S>& mat) {
    os << "Mat(" << mat.sx << "," << mat.sy << "):|" << std::endl;
    // if (mat.sx < 1) {
    //     os << "nullptr" << std::endl;
    // } else if (mat.sx < 17) {
    //     for (int i = 0; i < mat.sx; i++)
    //         os << mat.data[i] << "," << std::endl;
    // } else {
    //     for (int i = 0; i < 3; i++)
    //         os << mat.data[i] << "," << std::endl;
    //     os << "... ," << std::endl;
    //     for (int i = mat.sx-3; i < mat.sx; i++)
    //         os << mat.data[i] << "," << std::endl;
    // }

    if (mat.sx < 1) {
        os << "nullptr" << std::endl;
    } else{
        for (int i = 0; i < mat.sx; i++)
            os << mat.data[i] << "," << std::endl;
    }
    os << "|";
    return os;
}

template<typename T>
int Mat<T>::where(const Vec<T>& obj) {
    return data.where(obj);
}

template<typename T>
Vec<int> Mat<T>::where(const T& obj) {
    Vec<int> coords(2);
    coords[0] = coords[1] = -1;

    for (int i = 0; i < sx; i++) {
        Vec<T> curr_row = data[i];
        for (int j = 0; j < sy; j++) {
            if (curr_row[j] == obj) {
                coords[0] = i;
                coords[1] = j;
                return coords;
            }
        }
    }

    return coords;
}

#endif  // INCLUDE_DEFAULT_TYPES_H_
