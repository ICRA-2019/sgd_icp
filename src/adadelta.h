#ifndef __ADADELTA_H__
#define __ADADELTA_H__


#include "abstract_sgd_optimizer.h"


/**
 * \brief Implementation of the AdaDelta method.
 *
 * Zeiler, Matthew D. "ADADELTA: an adaptive learning rate method."
 * arXiv preprint arXiv:1212.5701 (2012).
 */
class AdaDelta : public AbstractSgdOptimizer
{
    public:
        /**
         * \brief Creates a new optimizer instance.
         *
         * \param initial_values initial values of the parameters
         * \param decay_rate gradient accumulation decay rate, \rho
         * \param preconditioner conditioner for RMS computation, \epsilon
         */
        AdaDelta(
                std::vector<double>     initial_values,
                double                  decay_rate,
                double                  preconditioner
        );


    protected:
        /**
         * \see AbstractSgdOptimizer::do_perform_update
         */
        void do_perform_update(std::vector<double> const& gradients) override;


    private:
        //! Exponential decay rate used for average accumulation
        double                          m_decay_rate;
        //! Preconditioner to ensure numerical stability of RMSE computation
        double                          m_preconditioner;

        //! Accumulated gradient information
        std::vector<double>             m_acc_gradient;
        //! Accumulated update information
        std::vector<double>             m_acc_updates;
};


#endif /* __ADADELTA_H__ */
