#include <iostream>
#include <algorithm>

#include "genetic_types.h"
#include "SampleSet.h"

#ifndef INCLUDE_GENETICSELECTOR_H_
#define INCLUDE_GENETICSELECTOR_H_

template<typename T>
class GeneticSelector {
 protected:
    float popul_frac;
    int popul_lim;

    bool is_max_sf;  // is maximizing or minimizing score function

 public:
    explicit GeneticSelector(float init_pfrac = 0.0, int init_plim = 0, bool init_max_sf = true);
    GeneticSelector(const GeneticSelector<T>& gs_obj);
    virtual ~GeneticSelector() {}

    virtual Population<T> select_population(const Population<T>& in_popul) = 0;
};


template<typename T>
class TournamentSelector : public GeneticSelector<T> {
 public:
    TournamentSelector(const TournamentSelector<T>& ts_obj);
    explicit TournamentSelector(float init_pfrac = 0.0, int init_plim = 0, bool init_max_sf = true);

    virtual ~TournamentSelector() {}
    virtual Population<T> select_population(const Population<T>& in_popul);
};

template<typename T>
class RouletteSelector : public GeneticSelector<T> {
 public:
    explicit RouletteSelector(float init_pfrac = 0.0, int init_plim = 0, bool init_max_sf = true);
    RouletteSelector(const RouletteSelector<T>& rs_obj);
    virtual ~RouletteSelector() {}

    virtual Population<T> select_population(const Population<T>& in_popul);
};

template<typename T>
class RankingSelector : public GeneticSelector<T> {
    int uniform_thresh;
 public:
    explicit RankingSelector(float init_pfrac = 0.0, int init_plim = 0, bool init_max_sf = true,
                             int init_uniform = 0);
    RankingSelector(const RankingSelector<T>& rs_obj);
    virtual ~RankingSelector() {}

    virtual Population<T> select_population(const Population<T>& in_popul);
};

template<typename T>
class SigmaTruncSelector : public GeneticSelector<T> {
 public:
    explicit SigmaTruncSelector(float init_pfrac = 0.0, int init_plim = 0, bool init_max_sf = true);
    SigmaTruncSelector(const SigmaTruncSelector<T>& sts_obj);
    virtual ~SigmaTruncSelector() {}

    virtual Population<T> select_population(const Population<T>& in_popul);
};



template<typename T>
GeneticSelector<T>::GeneticSelector(float init_pfrac, int init_plim, bool init_max_sf):
        popul_frac(init_pfrac),
        popul_lim(init_plim),
        is_max_sf((init_max_sf) ? 1 : 0) {
    if (popul_frac && popul_lim)
        LOG_(warning) << "Both selector limit options are set. Using popul_frac by default.";
}

template<typename T>
GeneticSelector<T>::GeneticSelector(const GeneticSelector<T>& gs_obj):
        popul_frac(gs_obj.popul_frac),
        popul_lim(gs_obj.popul_lim),
        is_max_sf((gs_obj.is_max_sf) ? 1 : 0) {
}

template<typename T>
TournamentSelector<T>::TournamentSelector(float init_pfrac, int init_plim, bool init_max_sf):
        GeneticSelector<T>(init_pfrac, init_plim, init_max_sf) {
}

template<typename T>
TournamentSelector<T>::TournamentSelector(const TournamentSelector<T>& ts_obj):
        GeneticSelector<T>(ts_obj) {
}

template<typename T>
Population<T> TournamentSelector<T>::select_population(const Population<T>& in_popul) {
    Population<T> out_popul;
    int popul_size = in_popul.get_size();
    int needed_size = (this->popul_frac) ? in_popul * this->popul_frac : this->popul_lim;

    for (int i = 0; i < needed_size; i++) {
        Chromosome<T> fst_chromo = in_popul[rand() % popul_size];
        Chromosome<T> sec_chromo = in_popul[rand() % popul_size];

        if (fst_chromo == sec_chromo) {
            i--;
            continue;
        } else if ((fst_chromo.get_score() >= sec_chromo.get_score()) == this->is_max_sf) {
            if(!out_popul.add_chromo(fst_chromo)) {
                i--;
                continue;
            }
        } else if (!out_popul.add_chromo(sec_chromo)) {
            i--;
            continue;
        }
    }
    return out_popul;
}


template<typename T>
RouletteSelector<T>::RouletteSelector(float init_pfrac, int init_plim, bool is_max_sf):
        GeneticSelector<T>(init_pfrac, init_plim, is_max_sf) {
}

template<typename T>
RouletteSelector<T>::RouletteSelector(const RouletteSelector<T>& rs_obj):
        GeneticSelector<T>(rs_obj) {
}

template<typename T>
Population<T> RouletteSelector<T>::select_population(const Population<T>& in_popul) {
    // LOG_(trace) << "Using Roulette selector...";

    Population<T> out_popul;
    int popul_size = in_popul.get_size();
    int needed_size = (this->popul_frac) ? popul_size * this->popul_frac : this->popul_lim;
    if (needed_size < 1)
        needed_size = 1;

    Vec<float> probs(popul_size);  // probs will be stored here
    float total_score = 0;
    for (int i = 0; i < popul_size; i++) {
        probs[i] = (this->is_max_sf) ? in_popul[i].get_score() : 1.0/in_popul[i].get_score();
        total_score += probs[i];
    }
    for (int i = 0; i < popul_size; i++)
        probs[i] /= total_score;
    // LOG_(trace) << "Probability vector: " << probs;

    int fail_iter = 10;
    for (int i = 0; i < needed_size; i++) {
        float rand_frac = static_cast<float>(rand()) / (RAND_MAX);
        // LOG_(trace) << "Rand fraction: " << rand_frac;

        int chromo_idx = 0;
        for (; chromo_idx < popul_size; chromo_idx++) {
            if (rand_frac < probs[chromo_idx])
                break;
            rand_frac -= probs[chromo_idx];
        }
        // LOG_(trace) << "Matching index: " << chromo_idx;

        if (!out_popul.add_chromo(in_popul[chromo_idx])) {
            // LOG_(trace) << "This chromosome was already chosen...";
            i--;
            if (!(--fail_iter))
                break;
        }
    }
    return out_popul;
}

template<typename T>
RankingSelector<T>::RankingSelector(float init_pfrac, int init_plim, bool init_max_sf,
                                    int init_uniform) :
        GeneticSelector<T>(init_pfrac, init_plim, init_max_sf), uniform_thresh(init_uniform) {
}

template<typename T>
RankingSelector<T>::RankingSelector(const RankingSelector<T>& rs_obj):
        GeneticSelector<T>(rs_obj),
        uniform_thresh(rs_obj.uniform_thresh) {
}

template<typename T>
Population<T> RankingSelector<T>::select_population(const Population<T>& in_popul) {
    Population<T> out_popul;
    int popul_size = in_popul.get_size();
    int needed_size = in_popul * this->popul_frac;
    Vec<float> probs(popul_size);

    for (int i = 0; i < popul_size; i++)
        probs[i] = in_popul.ind_vec[i].get_score();
    Vec<int> indices = probs.sort_indices();

    if (uniform_thresh > 1) {
        float val = 1.0/uniform_thresh;
        for (int i = 0; i < popul_size; i++)
            probs[i] = ((popul_size-indices[i] <= uniform_thresh) == this->is_max_sf) ? val : 0;
    } else {
        float divisor = popul_size*(popul_size+1)/2.0;
        for (int i = 0; i < popul_size; i++)
            probs[i] = (popul_size-indices[i])/divisor;
    }

    for (int i = 0; i < needed_size; i++)
        if (!out_popul.add_chromo(in_popul[rand() % popul_size]))
            i--;
    return out_popul;
}


template<typename T>
SigmaTruncSelector<T>::SigmaTruncSelector(float init_pfrac, int init_plim, bool init_max_sf):
        GeneticSelector<T>(init_pfrac, init_plim, init_max_sf) {
}

template<typename T>
SigmaTruncSelector<T>::SigmaTruncSelector(const SigmaTruncSelector<T>& sts_obj):
        GeneticSelector<T>(sts_obj) {
}

template<typename T>
Population<T> SigmaTruncSelector<T>::select_population(const Population<T>& in_popul) {
    Population<T> out_popul;
    int popul_size = in_popul.get_size();
    int needed_size = (this->popul_frac) ? in_popul * this->popul_frac : this->popul_lim;
    Vec<float> probs(popul_size);

    for (int i = 0; i < popul_size; i++)
        probs[i] = in_popul.ind_vec[i].get_score();
    Vec<int> indices = probs.sort_indices();

    for (int i = 0; i < popul_size; i++)
        if (popul_size-indices[i] <= needed_size)
            out_popul.add_chromo(in_popul[i]);
    return out_popul;
}

#endif  // INCLUDE_GENETICSELECTOR_H_
