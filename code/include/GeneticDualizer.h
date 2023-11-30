#include <type_traits>
#include "default_types.h"
#include "CollFamily.h"
#include "SampleHandler.h"
#include "CoveringHandler.h"
#include "GeneticAlgorithm.h"

#ifndef INCLUDE_GENETICDUALIZER_H_
#define INCLUDE_GENETICDUALIZER_H_

template<typename S, typename T>
class GeneticDualizer;

template<typename S, typename T>
class GenDualInitiator : public GeneticInitiator<S, int, T> {
    GeneticDualizer<S, T>* my_parent;
    CoveringHandler<T> covering_handler;
public:
    explicit GenDualInitiator(GeneticDualizer<S, T>* init_parent_ptr,
                              float init_pfrac = 0.0, int init_plim = 0);
    GenDualInitiator(const GenDualInitiator<S, T>& gi_obj,
                     GeneticDualizer<S, T>* parent = nullptr);
    ~GenDualInitiator() {}

    Population<T> get_init_population(const SampleSet<S, int>& sample_set);
};


template<typename S, typename T>
class GenDualMutator : public GeneticMutator<T> {
    GeneticDualizer<S, T>* my_parent;
    CoveringHandler<T> covering_handler;
    float mutation_rate;
    int64_t curr_iter;
 public:
    explicit GenDualMutator(GeneticDualizer<S, T>* init_parent_ptr,
                            float init_pfrac = 0.0, int init_plim = 0,
                            float init_mrate = 0.2);
    GenDualMutator(const GenDualMutator<S, T>& gm_obj,
                   GeneticDualizer<S, T>* parent = nullptr);
    ~GenDualMutator() {}

    void update_parent(GeneticDualizer<S, T>* new_parent) { my_parent = new_parent; }
    virtual Population<T> mutate_population(const Population<T>& in_popul);
 private:
    Population<T> recover_admissibility(const Population<T>& in_popul);
};


enum ScoreFuncType {
    kDataScore = 0,  // score (quality of whole collection based on how well it separates the classes)
    kWeightedScore = 1  // loss (minimizing weighted columns of coverage)
};

template<typename S, typename T>
class GeneticDualizer : public GeneticAlgorithm<S, int, T> {
    SplitterSH<S, int> sample_handler;
    ElColl<S> local_basis;

    bool is_max_sf;
    ScoreFuncType score_func;
    Vec<float> lb_weights;

    SampleSet<S, int> basic;
    SampleSet<S, int> valid;
    int target_tag;

    Mat<bool> gen_matrix;

 public:
    GeneticDualizer(int popul_size, float mut_frac, bool init_max_sf = true,
                    ScoreFuncType init_sf = kDataScore,
                    TermCritType init_tcrit = kPopulConverged,
                    float init_tcrit_val = 1.0);
    GeneticDualizer(const GeneticDualizer<S, T>& gd_obj);

    ~GeneticDualizer();

    void set_init_data(const SampleSet<S, int>& init_set,
                       const ElColl<S>& init_lb,
                       int init_target_tag);
    void set_matrix(const Mat<bool>& new_matrix);

    Vec<ElColl<S> > decode_collections(Vec< Vec<T> > in_code);
    virtual void update_costs(Population<T>* children_popul,
                              Population<T>* parents_popul);
    virtual void update_costs(Population<T>* in_popul);

    float get_data_quality(const Chromosome<T>& chromo);
    float get_weighted_quality(const Chromosome<T>& chromo);

    ElColl<S> get_chromo_elcoll(const Chromosome<T>& chromo);
    const Mat<bool>& get_gen_matrix() const { return gen_matrix; }
};


template<typename S, typename T>
GenDualInitiator<S, T>::GenDualInitiator(GeneticDualizer<S, T>* init_parent_ptr,
                                         float init_pfrac, int init_plim):
        GeneticInitiator<S, int, T>(init_pfrac, init_plim),
        my_parent(init_parent_ptr),
        covering_handler() {
}

template<typename S, typename T>
GenDualInitiator<S, T>::GenDualInitiator(const GenDualInitiator<S, T>& gi_obj,
                                         GeneticDualizer<S, T>* parent):
        GeneticInitiator<S, int, T>(gi_obj),
        my_parent(parent),
        covering_handler(gi_obj.covering_handler) {
}

template<typename S, typename T>
Population<T> GenDualInitiator<S, T>::get_init_population(const SampleSet<S, int>& sample_set) {
    // LOG_(trace) << "GenDualInitiator is processing initial population...";

    Population<T> out_popul;
    Mat<bool> gen_matrix(my_parent->get_gen_matrix());
    int lim = (this->popul_lim) ? this->popul_lim : sample_set.get_total_size() * this->popul_frac;
    // LOG_(trace) << "Need to get " << lim << " chromosomes.";

    int overheat_cnt = 0;
    for (int i = 0; i < lim; i++) {
        Vec<T> new_genes(covering_handler.build_covering(gen_matrix));
        // LOG_(trace) << "New genes for chromosome were generated as covering: " << new_genes;

        new_genes = covering_handler.make_covering_deadend(gen_matrix, new_genes);
        // LOG_(trace) << "New genes were adapted for deadend-covering: " << new_genes;
        if(!out_popul.add_chromo(Chromosome<T>(new_genes))) {
            overheat_cnt++;
            i--;
        } else {
            overheat_cnt = 0;
        }

        if (overheat_cnt >= lim) {
            break;
        }
    }
    // LOG_(trace) << "Initial population was successfully created.";
    // LOG_(trace) << "Initial population is formed with " << out_popul.get_size() << " chromos.";
    return out_popul;
}


template<typename S, typename T>
GenDualMutator<S, T>::GenDualMutator(GeneticDualizer<S, T>* init_parent_ptr,
                                     float init_pfrac, int init_plim,
                                     float init_mrate):
        GeneticMutator<T>(init_pfrac, init_plim),
        my_parent(init_parent_ptr),
        mutation_rate(init_mrate),
        curr_iter(0) {
}

template<typename S, typename T>
GenDualMutator<S, T>::GenDualMutator(const GenDualMutator<S, T>& gm_obj,
                                     GeneticDualizer<S, T>* parent):
        GeneticMutator<T>(gm_obj),
        my_parent(parent),
        mutation_rate(gm_obj.mutation_rate),
        curr_iter(gm_obj.curr_iter) {
}

template<typename S, typename T>
Population<T> GenDualMutator<S, T>::mutate_population(const Population<T>& in_popul) {
    float mut_frac = 1.0 - 1.0/(mutation_rate * (curr_iter+1) + 1.0);
    // LOG_(trace) << "Mutation fraction: " << mut_frac;
    int popul_size = in_popul.get_size();
    int needed_size = (this->popul_frac) ? popul_size * this->popul_frac : this->popul_lim;
    // LOG_(trace) << "Need to mutate " << needed_size << " out of " << popul_size;
    std::vector<int> idx(popul_size);
    std::iota(idx.begin(), idx.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(idx.begin(), idx.end(), g);
    Population<T> out_popul;
    if (std::is_same<T, bool>::value) {
        // LOG_(trace) << "Using <bool> case...";
        for (int i = 0; i < needed_size; i++) {
            // LOG_(trace) << "Processing " << idx[i] << " chromosome...";
            int chromo_len = in_popul[idx[i]].get_size();
            Vec<T> mut_chromo_vec(chromo_len);
            for (int j = 0; j < chromo_len; j++) {
                if (static_cast<float>(rand()) / (RAND_MAX) > mut_frac) {
                    mut_chromo_vec[j] = in_popul[idx[i]][j];
                } else {
                    // LOG_(trace) << "Mutation occured on " << j << " position";
                    mut_chromo_vec[j] = !in_popul[idx[i]][j];
                }
            }
            Chromosome<T> mut_chromo(mut_chromo_vec);
            // LOG_(trace) << "Mutated chromosome was formed: " << mut_chromo;
            out_popul.add_chromo(mut_chromo);
        }
    } else if (std::is_same<T, int>::value) {
        Mat<bool> gen_matrix(my_parent->get_gen_matrix());

        for (int i = 0; i < needed_size; i++) {
            int chromo_len = in_popul[idx[i]].get_size();
            Vec<T> mut_chromo_vec(chromo_len);
            for (int j = 0; j < chromo_len; j++) {
                if (static_cast<float>(rand()) / (RAND_MAX) > mut_frac) {
                    mut_chromo_vec[j] = in_popul[idx[i]][j];
                } else {
                    Vec<int> ones_vec(covering_handler.get_ones_places(gen_matrix[j]));
                    int ones_size = ones_vec.get_size();
                    int rand_val = ones_vec[rand() % ones_size];
                    while (rand_val == in_popul[idx[i]][j])
                        rand_val = ones_vec[rand() % ones_size];
                    mut_chromo_vec[j] = rand_val;
                }
            }
            Chromosome<T> mut_chromo(mut_chromo_vec);
            out_popul.add_chromo(mut_chromo);
        }
    }
    // LOG_(trace) << "Mutated population was formed:" << out_popul;
    return recover_admissibility(out_popul);
}

template <typename S, typename T>
Population<T> GenDualMutator<S, T>::recover_admissibility(const Population<T>& in_popul) {
    Vec< Vec<T> > in_popul_data = in_popul.get_popul_data();
    Mat<bool> gen_matrix = my_parent->get_gen_matrix();
    Population<T> out_popul;
    Vec<T> new_vec;

    int data_size = in_popul_data.get_size();
    for (int i = 0; i < data_size; i++) {
        new_vec = covering_handler.recover_admissibility(gen_matrix, in_popul_data[i]);
        out_popul.add_chromo(Chromosome<T>(new_vec));
    }

    return out_popul;
}


template<typename S, typename T>
GeneticDualizer<S, T>::GeneticDualizer(int popul_size, float mut_frac, bool init_max_sf,
                                       ScoreFuncType init_sf,
                                       TermCritType init_tcrit, float init_tcrit_val):
            GeneticAlgorithm<S, int, T>(new GenDualInitiator<S, T>(this, 0.0, popul_size),
                                        new RouletteSelector<T>(0.0, 2, init_max_sf),
                                        new PanmixiaBreeder<T>(0.0, 1, init_max_sf, kUniform),
                                        new GenDualMutator<S, T>(this, 0.0, 1, mut_frac),
                                        new SequentialMerger<T>(init_max_sf),
                                        init_tcrit,
                                        init_tcrit_val),
            sample_handler(0.9),
            local_basis(),
            is_max_sf(init_max_sf),
            score_func(init_sf),
            lb_weights(),
            basic(),
            valid(),
            target_tag(-1) {
    LOG_(debug) << "Created GeneticDualizer instance (by usual constructor).";
}

template<typename S, typename T>
GeneticDualizer<S, T>::GeneticDualizer(const GeneticDualizer<S, T>& gd_obj):
            GeneticAlgorithm<S, int, T>(nullptr,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        gd_obj.term_crit,
                                        gd_obj.term_crit_val),
            sample_handler(gd_obj.sample_handler),
            local_basis(gd_obj.local_basis),
            score_func(gd_obj.score_func),
            lb_weights(gd_obj.lb_weights),
            basic(gd_obj.basic),
            valid(gd_obj.valid),
            target_tag(-1) {
    GenDualInitiator<S, T>* initiator_ptr = dynamic_cast<GenDualInitiator<S, T>*>(gd_obj.initiator);
    RouletteSelector<T>* selector_ptr = dynamic_cast<RouletteSelector<T>*>(gd_obj.selector);
    PanmixiaBreeder<T>* breeder_ptr = dynamic_cast<PanmixiaBreeder<T>*>(gd_obj.breeder);
    GenDualMutator<S, T>* mutator_ptr = dynamic_cast<GenDualMutator<S, T>*>(gd_obj.mutator);
    SequentialMerger<T>* merger_ptr = dynamic_cast<SequentialMerger<T>*>(gd_obj.merger);

    this->initiator = new GenDualInitiator<S, T>(*initiator_ptr, this);
    this->selector = new RouletteSelector<T>(*selector_ptr);
    this->breeder = new PanmixiaBreeder<T>(*breeder_ptr);
    this->mutator = new GenDualMutator<S, T>(*mutator_ptr, this);
    this->merger = new SequentialMerger<T>(*merger_ptr);

    LOG_(debug) << "Created GeneticDualizer instance (by copy constructor).";
}

template<typename S, typename T>
GeneticDualizer<S, T>::~GeneticDualizer() {
    // LOG_(trace) << "Deleting pointers to algo instances...";
    delete this->initiator;
    delete this->selector;
    delete this->breeder;
    delete this->mutator;
    delete this->merger;

    LOG_(debug) << "GeneticDualizer instance was destroyed";
}

template<typename S, typename T>
Vec<ElColl<S> > GeneticDualizer<S, T>::decode_collections(Vec< Vec<T> > in_code) {
    int code_len = in_code.get_size();
    int vec_len = in_code[0].get_size();
    Vec<ElColl<S> > answer_colls(code_len);
    if (std::is_same<T, bool>::value) {
        for (int i = 0; i < code_len; i++)
            for (int j = 0; j < vec_len; j++)
                if (in_code[i][j])
                    answer_colls[i].add(local_basis[j]);
    } else if (std::is_same<T, int>::value) {
        for (int i = 0; i < code_len; i++)
            for (int j = 0; j < vec_len; j++)
                answer_colls[i].add(local_basis[in_code[i][j]]);
    }

    return answer_colls;
}

template<typename S, typename T>
void GeneticDualizer<S, T>::update_costs(Population<T>* children_popul,
                                         Population<T>* parents_popul) {
     // LOG_(trace) << "Updating costs of children/parents' population...";
     float min_val = -1.0;

     int children_num = children_popul->get_size();
     float* children_quals = new float[children_num];

     for (int i = 0; i < children_num; i++) {
         switch (score_func) {
             case kDataScore:
             {
                 children_quals[i] = get_data_quality((*children_popul)[i]);
                 break;
             }

             case kWeightedScore:
             {
                 children_quals[i] = get_weighted_quality((*children_popul)[i]);
                 break;
             }
         }

         if (children_quals[i] < min_val || min_val < 0.0)
             min_val = children_quals[i];
     }

     int parents_num = parents_popul->get_size();
     float* parents_quals = new float[parents_num];

     for (int i = 0; i < parents_num; i++) {
         switch (score_func) {
             case kDataScore:
             {
                 parents_quals[i] = get_data_quality((*parents_popul)[i]);
                 break;
             }

             case kWeightedScore:
             {
                 parents_quals[i] = get_weighted_quality((*parents_popul)[i]);
                 break;
             }
         }

         if (parents_quals[i] < min_val || min_val < 0.0)
             min_val = parents_quals[i];
     }
     // LOG_(trace) << "Children' absolute qualities:" << Vec<float>(children_num, children_quals);
     // LOG_(trace) << "Parents' absolute qualities:" << Vec<float>(parents_num, parents_quals);

     for (int i = 0; i < children_num; i++)
         (*children_popul)[i].set_score(children_quals[i]-min_val+1.0);

     for (int i = 0; i < parents_num; i++)
         (*parents_popul)[i].set_score(parents_quals[i]-min_val+1.0);

     delete [] children_quals;
     delete [] parents_quals;
}

template<typename S, typename T>
void GeneticDualizer<S, T>::update_costs(Population<T>* in_popul) {
    // LOG_(trace) << "Updating costs of population " << (*in_popul);

    int chromo_num = in_popul->get_size();
    float* qualities = new float[chromo_num];
    float min_val = -1.0;

    for (int i = 0; i < chromo_num; i++)
        for (int j = i+1; j < chromo_num; j++)
            if ( (*in_popul)[i] == (*in_popul)[j] )
                LOG_(warning) << "Duplicates of chromosomes were found: " << i << ", " << j;
    for (int i = 0; i < chromo_num; i++) {
        switch (score_func) {
            case kDataScore:
            {
                qualities[i] = get_data_quality((*in_popul)[i]);
                break;
            }

            case kWeightedScore:
            {
                qualities[i] = get_weighted_quality((*in_popul)[i]);
                break;
            }
        }

        if (qualities[i] < min_val || min_val < 0.0)
            min_val = qualities[i];
    }

    for (int i = 0; i < chromo_num; i++)
        (*in_popul)[i].set_score(qualities[i]-min_val+1);
    delete [] qualities;
}

template<typename S, typename T>
float GeneticDualizer<S, T>::get_data_quality(const Chromosome<T>& chromo) {
    if (score_func != kDataScore) {
        LOG_(error) << "Using data quality instead of something else!";
        return 0.0;
    }
    // LOG_(trace) << "Getting data quality of " << chromo << " chromosome...";

    const GroupSamples<S, int>& basic_class = basic.get_group(target_tag);
    const GroupSamples<S, int>& valid_class = valid.get_group(target_tag);
    ElColl<S> chromo_coll = get_chromo_elcoll(chromo);

    int basic_class_num = basic_class.get_size();
    int valid_class_num = valid_class.get_size();

    // LOG_(trace) << "basic: " << basic_class;
    // LOG_(trace) << "valid: " << valid_class;

    float quality_sum = 0.0;
    for (int i = 0; i < valid_class_num; i++)
        for (int j = 0; j < basic_class_num; j++) {
            bool vote = chromo_coll.vote_func(valid_class[i], basic_class[j]);
            quality_sum += vote;
        }
    // LOG_(trace) << "Quality sum: " << quality_sum;
    return quality_sum / static_cast<float>(valid_class_num*basic_class_num);
}


template<typename S, typename T>
float GeneticDualizer<S, T>::get_weighted_quality(const Chromosome<T>& chromo) {
    if (score_func != kWeightedScore) {
        LOG_(error) << "Using weighted quality instead of something else!";
        return 0.0;
    }
    // LOG_(trace) << "Getting weighted quality of " << chromo << " chromosome...";

    Vec<T> chromo_genes = chromo.get_genes();
    int vec_len = chromo_genes.get_size();
    int lb_size = local_basis.get_size();
    float weights_sum = 0.0;

    if (std::is_same<T, bool>::value) {
        for (int i = 0; i < vec_len; i++)
            if (chromo_genes[i])
                weights_sum += lb_weights[i];
    } else if (std::is_same<T, int>::value) {
        Vec<bool> bool_form(local_basis.get_size());
        for (int i = 0; i < lb_size; i++)
            bool_form[i] = false;

        for (int i = 0; i < vec_len; i++)
            bool_form[chromo_genes[i]] = true;

        for (int i = 0; i < lb_size; i++)
            if (bool_form[i])
                weights_sum += lb_weights[i];
    }
    return weights_sum;
}

template<typename S, typename T>
ElColl<S> GeneticDualizer<S, T>::get_chromo_elcoll(const Chromosome<T>& chromo) {
    ElColl<S> out_coll;
    Vec<T> chromo_genes = chromo.get_genes();
    int vec_len = chromo_genes.get_size();

    if (std::is_same<T, bool>::value) {
        for (int i = 0; i < vec_len; i++)
            if (chromo_genes[i])
                out_coll.add(local_basis[i]);
    } else if (std::is_same<T, int>::value) {
        for (int i = 0; i < vec_len; i++)
            out_coll.add(local_basis[chromo_genes[i]]);
    }
    return out_coll;
}

template<typename S, typename T>
void GeneticDualizer<S, T>::set_init_data(const SampleSet<S, int>& init_set,
                                          const ElColl<S>& init_lb,
                                          int init_target_tag) {
    // LOG_(trace) << "Setting initial data...";
    Mat<S> init_X;
    Vec<int> init_y;
    init_set.get_data(&init_X, &init_y);
    Vec< SampleSet<S, int> > data = sample_handler.make_samples(init_X, init_y)[0];
    basic = data[0];
    valid = data[1];

    // LOG_(trace) << "Training set was splitted into train/valid sets.";

    local_basis = init_lb;
    target_tag = init_target_tag;
    // LOG_(trace) << "Train set for further computations:" << basic;
    // LOG_(trace) << "Valid set for further computations:" << valid;

    const GroupSamples<S, int>& tag_class = basic.get_group(target_tag);
    SampleSet<S, int> tag_anticlass(basic.get_antigroup(target_tag));

    int tclass_size = tag_class.get_size();
    int tanticlass_size = tag_anticlass.get_total_size();
    int lb_size = local_basis.get_size();

    if (score_func == kWeightedScore) {
        lb_size = local_basis.get_size();
        lb_weights = Vec<float>(lb_size);

        for (int i = 0; i < tclass_size; i++) {
            Vec<bool> class_vec = local_basis.apply_to_object(tag_class[i]);
            for (int j = 0; j < lb_size; j++)
                lb_weights[j] += class_vec[j];
        }

        for (int i = 0; i < tanticlass_size; i++) {
            Vec<bool> class_vec = local_basis.apply_to_object(tag_anticlass[i]);
            for (int j = 0; j < lb_size; j++)
                lb_weights[j] += class_vec[j];
        }

        // LOG_(trace) << "Weights for local basis were created.";
        // LOG_(trace) << local_basis;
        // LOG_(trace) << "Weights:" << lb_weights;
    }

    gen_matrix = Mat<bool>(0, lb_size);
    Vec<bool> bin_vec(lb_size);

    for (int i = 0; i < tclass_size; i++) {
        for (int j = 0; j < tanticlass_size; j++) {
            Vec<bool> class_vec = local_basis.apply_to_object(tag_class[i]);
            Vec<bool> aclass_vec = local_basis.apply_to_object(tag_anticlass[j]);

            for (int k = 0; k < lb_size; k++)
                bin_vec[k] = class_vec[k] && !aclass_vec[k];
            if (gen_matrix.where(bin_vec) == -1)
                gen_matrix.hadd(bin_vec);
        }
    }
    // LOG_(trace) << "Created gen matrix:" << gen_matrix;
}

template<typename S, typename T>
void GeneticDualizer<S, T>::set_matrix(const Mat<bool>& new_matrix) {
    gen_matrix = new_matrix;

    if (score_func == kWeightedScore) {
        int mat_sy = gen_matrix.get_sy();
        lb_weights = Vec<float>(mat_sy);

        for (int i = 0; i < mat_sy; i++)
            lb_weights[i] = 1.0/mat_sy;

        // LOG_(trace) << "Uniform weights were set.";
        // LOG_(trace) << "Weights:" << lb_weights;
    }
}

#endif  // INCLUDE_GENETICDUALIZER_H_
