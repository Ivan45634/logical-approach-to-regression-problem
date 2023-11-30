#include "genetic_types.h"

#ifndef INCLUDE_GENETICBREEDER_H_
#define INCLUDE_GENETICBREEDER_H_


enum CrossoverType {
    kOnePoint = 0,
    kMultiPoint = 1,
    kUniform = 2,
    kScoredUniform = 3
};
template<typename T>
class GeneticBreeder {
 protected:
    float popul_frac;
    int popul_lim;
    CrossoverType cross_type;
    float cross_param;

    bool is_max_sf;  // is maximizing or minimizing fitness function

 public:
    explicit GeneticBreeder(float init_pfrac = 0.0, int init_plim = 0, bool init_max_sf = true,
                            CrossoverType init_ctype = kOnePoint, float init_cparam = 1.0);
    GeneticBreeder(const GeneticBreeder<T>& gb_obj);
    virtual ~GeneticBreeder() {}

    virtual Population<T> breed_new_population(const Population<T>& in_popul) = 0;

 protected:
    virtual Vec<Chromosome<T>> apply_crossover(const Chromosome<T>& fst,
                                               const Chromosome<T>& sec);
};

template<typename T>
class PanmixiaBreeder : public GeneticBreeder<T> {
 public:
    explicit PanmixiaBreeder(float init_pfrac = 0.0, int init_plim = 0, bool init_max_sf = true,
                             CrossoverType init_ctype = kOnePoint, float init_cparam = 1.0);
    PanmixiaBreeder(const PanmixiaBreeder<T>& pb_obj);
    ~PanmixiaBreeder() {}

    virtual Population<T> breed_new_population(const Population<T>& in_popul);
};

enum InOutBreederType {
    kInPhenoType  = 0,
    kInGenoType   = 1,
    kOutPhenoType = 2,
    kOutGenoType  = 3
};
template<typename T>
class InOutBreeder : public GeneticBreeder<T> {
    InOutBreederType breeder_type;

 public:
    explicit InOutBreeder(float init_pfrac = 0.0, int init_plim = 0, bool init_max_sf = true,
                          InOutBreederType init_btype = kInPhenoType,
                          CrossoverType init_ctype = kOnePoint, float init_cparam = 1.0);

    InOutBreeder(const InOutBreeder<T>& iob_obj);
    ~InOutBreeder() {}

    virtual Population<T> breed_new_population(const Population<T>& in_popul);

 private:
    virtual const Chromosome<T>& find_match(const Population<T>& in_popul,
                                            const Chromosome<T>& suitor);
    float find_geno_diff(const Chromosome<T>& fst, const Chromosome<T>& sec);
};



template<typename T>
GeneticBreeder<T>::GeneticBreeder(float init_pfrac, int init_plim, bool init_max_sf,
                                  CrossoverType init_ctype, float init_cparam):
        popul_frac(init_pfrac),
        popul_lim(init_plim),
        cross_type(init_ctype),
        cross_param(init_cparam),
        is_max_sf((init_max_sf) ? 1 : 0) {
}

template<typename T>
GeneticBreeder<T>::GeneticBreeder(const GeneticBreeder<T>& gb_obj):
        popul_frac(gb_obj.popul_frac),
        popul_lim(gb_obj.popul_lim),
        cross_type(gb_obj.cross_type),
        cross_param(gb_obj.cross_param),
        is_max_sf((gb_obj.is_max_sf) ? 1 : 0) {
}

template<typename T>
Vec< Chromosome<T> > GeneticBreeder<T>::apply_crossover(const Chromosome<T>& fst,
                                                        const Chromosome<T>& sec) {
    int chromo_len = fst.get_size();
    int rand_val = rand() % chromo_len;
    Vec<T> fst_child_genes(chromo_len);
    Vec<T> sec_child_genes(chromo_len);
    Vec< Chromosome<T> > output_vec;

    int fst_score = fst.get_score();
    int total_score = fst_score + sec.get_score();

    switch (cross_type) {
        case kOnePoint:
        {
            for (int i = 0; i <= rand_val; i++) {
                fst_child_genes[i] = fst[i];
                sec_child_genes[i] = sec[i];
            }
            for (int i = rand_val+1; i < chromo_len; i++) {
                fst_child_genes[i] = sec[i];
                sec_child_genes[i] = fst[i];
            }
            const Chromosome<T> fst_child(fst_child_genes);
            const Chromosome<T> sec_child(sec_child_genes);

            output_vec.append(fst_child);
            output_vec.append(sec_child);
            break;
        }

        case kMultiPoint:
        {
            for (int i = 0; i < cross_param; i++) {
                for (int j = 0; j <= rand_val; j++) {
                    fst_child_genes[j] = fst[j];
                    sec_child_genes[j] = sec[j];
                }
                for (int j = rand_val+1; j < chromo_len; j++) {
                    fst_child_genes[j] = sec[j];
                    sec_child_genes[j] = fst[j];
                }
                const Chromosome<T> fst_child(fst_child_genes);
                const Chromosome<T> sec_child(sec_child_genes);

                output_vec.append(fst_child);
                output_vec.append(sec_child);

                rand_val = rand() % chromo_len;
            }
            break;
        }

        case kUniform:
        {
            for (int i = 0; i < chromo_len; i++)
                fst_child_genes[i] = (rand() % 2) ? fst[i] : sec[i];
            const Chromosome<T> fst_child(fst_child_genes);
            output_vec.append(fst_child);
            break;
        }

        case kScoredUniform:
        {
            for (int i = 0; i < chromo_len; i++) {
                if (rand() % total_score > fst_score)
                    fst_child_genes[i] = fst[i];
                else
                    fst_child_genes[i] = sec[i];
            }
            const Chromosome<T> fst_child(fst_child_genes);
            output_vec.append(fst_child);
            break;
        }
    }

    return output_vec;
}


template<typename T>
PanmixiaBreeder<T>::PanmixiaBreeder(float init_pfrac, int init_plim, bool init_max_sf,
                                    CrossoverType init_ctype, float init_cparam):
        GeneticBreeder<T>(init_pfrac, init_plim, init_max_sf, init_ctype, init_cparam) {
}

template<typename T>
PanmixiaBreeder<T>::PanmixiaBreeder(const PanmixiaBreeder<T>& pb_obj):
        GeneticBreeder<T>(pb_obj) {
}

template<typename T>
Population<T> PanmixiaBreeder<T>::breed_new_population(const Population<T>& in_popul) {
    Population<T> out_popul;
    int in_popul_size = in_popul.get_size();
    int needed_size = (this->popul_lim) ? this->popul_lim : in_popul_size * this->popul_frac;

    if (in_popul_size <= 1) {
        LOG_(error) << "Not enough species to breed new population: " << in_popul_size;
        return in_popul;
    }

    for (int i = 0; i < needed_size;) {
        Chromosome<T> fst_parent(in_popul[rand() % in_popul_size]);
        Chromosome<T> sec_parent(in_popul[rand() % in_popul_size]);
        while (sec_parent == fst_parent)
            sec_parent = in_popul[rand() % in_popul_size];
        Vec< Chromosome<T> > children(this->apply_crossover(fst_parent, sec_parent));
        int children_num = children.get_size();

        for (int j = 0; j < children_num; j++)
            if (out_popul.add_chromo(children[i])) {
                i++;
                if (i >= needed_size)
                    break;
            }
    }
    return out_popul;
}

template<typename T>
InOutBreeder<T>::InOutBreeder(float init_pfrac, int init_plim, bool init_max_sf,
                              InOutBreederType init_btype,
                              CrossoverType init_ctype, float init_cparam):
        GeneticBreeder<T>(init_pfrac, init_plim, init_max_sf, init_ctype, init_cparam),
        breeder_type(init_btype) {
}

template<typename T>
InOutBreeder<T>::InOutBreeder(const InOutBreeder<T>& iob_obj):
        GeneticBreeder<T>(iob_obj),
        breeder_type(iob_obj.breeder_type) {
}

template<typename T>
Population<T> InOutBreeder<T>::breed_new_population(const Population<T>& in_popul) {
    Population<T> out_popul;
    int in_popul_size = in_popul.get_size();
    if (in_popul_size <= 1)
        return in_popul;

    for (int i = 0; i < this->popul_num; i++) {
        Chromosome<T> fst_parent = in_popul[rand() % in_popul_size];
        Chromosome<T> sec_parent = find_match(in_popul, fst_parent);
        Vec< Chromosome<T> > children = apply_crossover(fst_parent, sec_parent);
        int children_num = children.get_size();

        for (int j = 0; j < children_num; j++)
            if (out_popul.add_chromo(children[i])) {
                i++;
                if (i >= this->popul_num)
                    break;
            }
    }
    return out_popul;
}

template<typename T>
const Chromosome<T>& InOutBreeder<T>::find_match(const Population<T>& in_popul,
                                                 const Chromosome<T>& suitor) {
    int in_popul_size = in_popul.get_size();
    int partner_idx = (in_popul[0] == suitor) ? 1 : 0;
    float param_diff = 0.0, curr_param_diff = 0.0;
    float suitor_score = suitor.get_score();

    switch (breeder_type) {
        case kInPhenoType:
            param_diff = fabs(in_popul[partner_idx].get_score() - suitor_score);
            for (int i = (partner_idx+1); i < in_popul_size; i++) {
                curr_param_diff = fabs(in_popul[i].get_score() - suitor_score);
                if (curr_param_diff < param_diff && !(in_popul[i] == suitor)) {
                    partner_idx = i;
                    param_diff = curr_param_diff;
                }
            }
            break;

        case kInGenoType:
            param_diff = find_geno_diff(in_popul[partner_idx], suitor);
            for (int i = (partner_idx+1); i < in_popul_size; i++) {
                curr_param_diff = in_popul[i] % suitor;
                if (curr_param_diff < param_diff && !(in_popul[i] == suitor)) {
                    partner_idx = i;
                    param_diff = curr_param_diff;
                }
            }
            break;

        case kOutPhenoType:
            param_diff = fabs(in_popul[partner_idx].get_score() - suitor_score);
            for (int i = (partner_idx+1); i < in_popul_size; i++) {
                curr_param_diff = fabs(in_popul[i].get_score() - suitor_score);
                if (curr_param_diff > param_diff && !(in_popul[i] == suitor)) {
                    partner_idx = i;
                    param_diff = curr_param_diff;
                }
            }
            break;

        case kOutGenoType:
            param_diff = find_geno_diff(in_popul[partner_idx], suitor);
            for (int i = (partner_idx+1); i < in_popul_size; i++) {
                curr_param_diff = in_popul[i] % suitor;
                if (curr_param_diff > param_diff && !(in_popul[i] == suitor)) {
                    partner_idx = i;
                    param_diff = curr_param_diff;
                }
            }
            break;
    }
    return in_popul[partner_idx];
}

template<typename T>
float InOutBreeder<T>::find_geno_diff(const Chromosome<T>& fst, const Chromosome<T>& sec) {
    int chromo_len = fst.get_size();
    float sum = 0.0;
    for (int i = 0; i < chromo_len; i++)
        sum += (fst[i]-sec[i])*(fst[i]-sec[i]);
    return sqrt(sum);
}

#endif  // INCLUDE_GENETICBREEDER_H_
