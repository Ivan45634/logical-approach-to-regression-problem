#include "default_types.h"

#ifndef INCLUDE_GENETIC_TYPES_H_
#define INCLUDE_GENETIC_TYPES_H_


template<typename T>
class Chromosome {
    Vec<T> genes;
    float score;

 public:
    Chromosome() : genes(0), score(-1) {}
    explicit Chromosome(const Vec<T>& init_genes, float init_score = 0.0);
    explicit Chromosome(T init_gene, float init_score = 0.0);
    explicit Chromosome(const Chromosome<T>& chromo_obj);
    explicit Chromosome(Chromosome<T>&& chromo_obj);
    ~Chromosome() {}

    int get_size() const { return genes.get_size(); }
    const Vec<T>& get_genes() const { return genes; }
    float get_score() const { return score; }
    void set_score(float score_val) { score = score_val; }

    Chromosome<T>& operator=(const Chromosome<T>& chromo_obj);
    bool operator== (const Chromosome<T>& chromo_obj) const;
    T& operator[] (int idx) { return genes[idx]; }
    const T& operator[] (int idx) const { return genes[idx]; }

    template<typename S>
    friend std::ostream& operator<<(std::ostream& os, const Chromosome<S>& chromo);
};


template<typename T>
class Population {
    Vec< Chromosome<T> > chromo_vec;

 public:
    explicit Population(Vec< Chromosome<T> > init_vec) : chromo_vec(init_vec) {}
    Population() {}
    ~Population() {}

    bool check_chromo(const Chromosome<T>& check_obj) const;
    bool add_chromo(const Chromosome<T>& add_obj, bool allow_dups = false);
    bool del_chromo(int idx);

    const Vec< Chromosome<T> >& get_chromos() const { return chromo_vec; }
    Vec< Vec<T> > get_popul_data() const;
    int get_size() const { return chromo_vec.get_size(); }

    float get_avg_score() const;
    const Chromosome<T>& get_best_ind() const;

    Population<T> slice(int begin, int end) const;
    Chromosome<T>& operator[] (int idx) { return chromo_vec[idx]; }
    const Chromosome<T>& operator[] (int idx) const { return chromo_vec[idx]; }
    Population<T> operator+ (const Population<T>& popul_obj);

    template<typename S>
    friend std::ostream& operator<<(std::ostream& os, const Population<S>& popul);
};


template<typename T>
Chromosome<T>::Chromosome(const Vec<T>& init_genes, float init_score):
        genes(init_genes), score(init_score) {
}

template<typename T>
Chromosome<T>::Chromosome(T init_gene, float init_score):
        genes(1, &init_gene), score(init_score) {
}

template<typename T>
Chromosome<T>::Chromosome(const Chromosome<T>& chromo_obj):
        genes(chromo_obj.get_genes()), score(chromo_obj.get_score()) {
}

template<typename T>
Chromosome<T>::Chromosome(Chromosome<T>&& chromo_obj):
        genes(std::move(chromo_obj.get_genes())), score(chromo_obj.get_score()) {
}

template<typename T>
Chromosome<T>& Chromosome<T>::operator=(const Chromosome<T>& chromo_obj) {
    genes = chromo_obj.get_genes();
    score = chromo_obj.get_score();
    return (*this);
}

template<typename T>
bool Chromosome<T>::operator==(const Chromosome<T>& chromo_obj) const {
    int chromo_size = genes.get_size();
    if (chromo_size != chromo_obj.get_size()) {
        LOG_(warning) << "It is not correct to compare these chromosomes: different sizes!";
        LOG_(warning) << (*this) << " and " << chromo_obj;
        return 0;
    }

    Vec<T> obj_genes = chromo_obj.get_genes();
    for (int i = 0; i < chromo_size; i++)
        if (obj_genes[i] != genes[i]) {
            // LOG_(trace) << "Chromosomes differ at " << i << " position.";
            return 0;
        }
    return 1;
}


template<typename S>
std::ostream& operator<<(std::ostream& os, const Chromosome<S>& chromo) {
    std::stringstream buffer;
    buffer << "Chromo<" << chromo.score << ">(" << chromo.genes << ")";
    os << buffer.str();
    return os;
}

template<typename T>
bool Population<T>::check_chromo(const Chromosome<T>& check_obj) const {
    int chromo_num = chromo_vec.get_size();
    for (int i = 0; i < chromo_num; i++)
        if (check_obj == chromo_vec[i])
            return 0;
    return 1;
}

template<typename T>
bool Population<T>::add_chromo(const Chromosome<T>& add_obj, bool allow_dups) {
    // LOG_(trace) << "Adding " << add_obj << "...";
    if(!allow_dups && !check_chromo(add_obj))
        return 0;
    // LOG_(trace) << "No duplicates were found.";
    chromo_vec.append(add_obj);
    // LOG_(trace) << "New chromosome was added to population: " << (*this);
    return 1;
}

template<typename T>
bool Population<T>::del_chromo(int idx) {
    return chromo_vec.erase(idx);
}

template<typename T>
float Population<T>::get_avg_score() const {
    int chromo_num = chromo_vec.get_size();
    float total_score = 0.0;
    for (int i = 0; i < chromo_num; i++)
        total_score += chromo_vec[i].get_score();
    total_score /= chromo_num;

    // LOG_(trace) << "Average score for " << (*this) << ": " << total_score;
    return total_score;
}

template<typename T>
const Chromosome<T>& Population<T>::get_best_ind() const {
    int chromo_num = chromo_vec.get_size();
    int idx = 0;
    float max_score = chromo_vec[idx].get_score();

    for (int i = 1; i < chromo_num; i++) {
        float curr_score = chromo_vec[i].get_score();
        if (curr_score > max_score) {
            idx = i;
            max_score = curr_score;
        }
    }
    // LOG_(trace) << "Best chromosome for " << (*this) << " is on " << idx << " position";
    return chromo_vec[idx];
}

template<typename T>
Vec< Vec<T> > Population<T>::get_popul_data() const {
    int chromo_len = chromo_vec.get_size();
    Vec< Vec<T> > out_vec(chromo_len);

    for (int i = 0; i < chromo_len; i++)
        out_vec[i] = chromo_vec[i].get_genes();
    return out_vec;
}

template<typename T>
Population<T> Population<T>::slice(int begin, int end) const {
    return Population<T>(chromo_vec.slice(begin, end));
}

template<typename T>
Population<T> Population<T>::operator+(const Population<T>& popul_obj) {
    Population<T> out_popul;
    int my_size = chromo_vec.get_size();
    int obj_size = popul_obj.get_size();

    for (int i = 0; i < my_size; i++)
        out_popul.add_chromo(chromo_vec[i]);

    for (int i = 0; i < obj_size; i++)
        out_popul.add_chromo(chromo_vec[i]);

    return out_popul;
}

template<typename S>
std::ostream& operator<<(std::ostream& os, const Population<S>& popul) {
    std::stringstream buffer;
    int chromo_size = popul.chromo_vec.get_size();
    buffer << "Popul<" << chromo_size << ">:[";
    buffer << popul.chromo_vec[0];
    for (int i = 1; i < chromo_size; i++)
        buffer << ", " << popul.chromo_vec[i];
    buffer << "]";
    os << buffer.str();
    return os;
}

#endif  // INCLUDE_GENETIC_TYPES_H_
