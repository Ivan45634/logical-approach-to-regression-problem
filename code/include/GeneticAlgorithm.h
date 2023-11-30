#ifndef INCLUDE_GENETICALGORITHM_H_
#define INCLUDE_GENETICALGORITHM_H_

#include "genetic_types.h"
#include "GeneticSelector.h"
#include "GeneticBreeder.h"
#include "GeneticMerger.h"

// TODO(Baytekov): Add weights!

template<typename S, typename T, typename U>  // S - features, T - answers, U - coverings
class GeneticInitiator {
 protected:
    float popul_frac;
    int popul_lim;

 public:
    explicit GeneticInitiator(float init_pfrac = 0.0, int init_plim = 0);
    GeneticInitiator(const GeneticInitiator<S, T, U>& gi_obj);
    virtual ~GeneticInitiator() {}

    virtual Population<U> get_init_population(const SampleSet<S, T>& sample_set) = 0;
};


template<typename T>  // T - coverings
class GeneticMutator {
 protected:
    float popul_frac;
    int popul_lim;

 public:
    explicit GeneticMutator(float init_pfrac = 0.0, int init_plim = 0);
    GeneticMutator(const GeneticMutator<T>& gm_obj);
    virtual ~GeneticMutator() {}

    virtual Population<T> mutate_population(const Population<T>& in_popul) = 0;
};



enum TermCritType {
    kPopulConverged = 0,  // term_crit_val = min delta between two populations
    kBestConverged  = 1,  // term_crit_val = min delta between two fitnesses of best individual
    kDelayConverged = 2,  // term_crit_val = min delta between two populations with delay
    kMaxIterNum    = 3   // term_crit_val = max amount of iterations
};
template<typename S, typename T, typename U>  // S - features, T - answers, U - coverings
class GeneticAlgorithm {
 protected:
    SampleSet<S, T> sample_set;

    GeneticInitiator<S, T, U>* initiator;
    GeneticSelector<U>* selector;
    GeneticBreeder<U>* breeder;
    GeneticMutator<U>* mutator;
    GeneticMerger<U>* merger;

    TermCritType term_crit;
    float term_crit_val;
    int delay_counter;

 public:
    GeneticAlgorithm(GeneticInitiator<S, T, U>* init_initiator, GeneticSelector<U>* init_selector,
                     GeneticBreeder<U>* init_breeder, GeneticMutator<U>* init_mutator,
                     GeneticMerger<U>* init_merger,
                     TermCritType init_tcrit = kPopulConverged, float init_tcrit_val = 1.0);
    virtual ~GeneticAlgorithm() {}

    virtual Vec< Vec<U> > execute_ga();
    virtual bool check_term_crit(Population<U> curr_popul, Population<U> new_popul,
                                 int64_t iter_cnt);
    virtual void set_sample_set(SampleSet<S, T> init_set) { sample_set = init_set; }
    virtual void update_costs(Population<U>* children_popul, Population<U>* parents_popul) = 0;
    virtual void update_costs(Population<U>* in_popul) = 0;
};



template<typename S, typename T, typename U>
GeneticInitiator<S, T, U>::GeneticInitiator(float init_pfrac, int init_plim):
        popul_frac(init_pfrac), popul_lim(init_plim) {
}

template<typename S, typename T, typename U>
GeneticInitiator<S, T, U>::GeneticInitiator(const GeneticInitiator<S, T, U>& gi_obj):
        popul_frac(gi_obj.popul_frac), popul_lim(gi_obj.popul_lim) {
}


template<typename T>
GeneticMutator<T>::GeneticMutator(float init_pfrac, int init_plim):
        popul_frac(init_pfrac), popul_lim(init_plim) {
}

template<typename T>
GeneticMutator<T>::GeneticMutator(const GeneticMutator<T>& gm_obj):
        popul_frac(gm_obj.popul_frac), popul_lim(gm_obj.popul_lim) {
}


template<typename S, typename T, typename U>
GeneticAlgorithm<S, T, U>::GeneticAlgorithm(GeneticInitiator<S, T, U>* init_initiator,
                                            GeneticSelector<U>*        init_selector,
                                            GeneticBreeder<U>*         init_breeder,
                                            GeneticMutator<U>*         init_mutator,
                                            GeneticMerger<U>*          init_merger,
                                            TermCritType init_term_crit, float init_term_crit_val) :
                initiator(init_initiator),
                selector(init_selector),
                breeder(init_breeder),
                mutator(init_mutator),
                merger(init_merger),
                term_crit(init_term_crit),
                term_crit_val(init_term_crit_val),
                delay_counter(7) {
    LOG_(debug) << "Created GeneticAlgorithm instance (by usual constructor).";
}

template<typename S, typename T, typename U>
Vec< Vec<U> > GeneticAlgorithm<S, T, U>::execute_ga() {
    // LOG_(trace) << "Executing Genetic Algorithm...";

    Population<U> curr_population = initiator->get_init_population(sample_set);
    int64_t iter_count = 0L;
    update_costs(&curr_population);

    while (true) {
        // LOG_(trace) << "Current costs: " << curr_population;

        Population<U> breed_population = selector->select_population(curr_population);
        if (breed_population.get_size() <= 1) {
            LOG_(warning) << "Population has degradated.";
            return curr_population.get_popul_data();
        }

        // LOG_(trace) << "Selected population for breeding: " << breed_population;
        Population<U> children_population = breeder->breed_new_population(breed_population);
        // LOG_(trace) << "Children population: " << children_population;

        children_population = mutator->mutate_population(children_population);
        // LOG_(trace) << "Mutated population: " << children_population;

        update_costs(&children_population, &curr_population);

        // LOG_(trace) << "Children population: " << children_population;
        // LOG_(trace) << "Parents population: " << curr_population;

        Population<U> new_population = merger->merge_populations(curr_population,
                                                                 children_population);
        // LOG_(trace) << "New population: " << new_population;

        if (check_term_crit(curr_population, new_population, iter_count)) {
            // LOG_(trace) << "Termination criteria is satisfied.";
            return new_population.get_popul_data();
        }
        curr_population = new_population;
        iter_count++;
    }
}

template<typename S, typename T, typename U>
bool GeneticAlgorithm<S, T, U>::check_term_crit(Population<U> curr_popul, \
                                             Population<U> new_popul, \
                                             int64_t iter_cnt) {
    switch (term_crit) {
        case kPopulConverged:
        {
            if (fabs(new_popul.get_avg_score() - curr_popul.get_avg_score()) < term_crit_val)
                return 1;
            return 0;
        }

        case kBestConverged:
        {
            float new_best_score = new_popul.get_best_ind().get_score();
            float curr_best_score = curr_popul.get_best_ind().get_score();
            if (fabs(new_best_score - curr_best_score) < term_crit_val)
                return 1;
            return 0;
        }

        case kDelayConverged:
        {
            if (fabs(new_popul.get_avg_score() - curr_popul.get_avg_score()) < term_crit_val) {
                if (!delay_counter) {
                    return 1;
                } else {
                    delay_counter--;
                    return 0;
                }
            }
            delay_counter = 3;
            return 0;
        }

        case kMaxIterNum:
        {
            if (iter_cnt >= static_cast<int64_t>(term_crit_val))
                return 1;
            return 0;
        }

        default:
        {
            LOG_(error) << "Unknown termination criteria! (code: " << term_crit << ")";
            return 0;
        }
    }
}

#endif  // INCLUDE_GENETICALGORITHM_H_
