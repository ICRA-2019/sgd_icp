#ifndef __ADAM_H__
#define __ADAM_H__


#include "abstract_sgd_optimizer.h"


/**
 * \brief Implementation of the ADAM method.
 *
 * Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization."
 * arXiv preprint arXiv:1412.6980 (2014).
 */
class Adam : public AbstractSgdOptimizer
{
    public:
        /**
         * \brief Creates a new optimizer instance.
         *
         * \param initial_values initial values of the parameters
         * \param step_size base gradient step size, \alpha
         * \param decay_rate_a decay rate for first moment estimates, \beta_1
         * \param decay_rate_b decay rate for second moment estimates, \beta_2
         */
        Adam(
            std::vector<double>         initial_values,
            double                      step_size,
            double                      decay_rate_a,
            double                      decay_rate_b
        );


    protected:
        /**
         * \see AbstractSgdOptimizer::do_perform_update
         */
        void do_perform_update(std::vector<double> const& gradients) override;


    private:
        //! Base step size for updates
        double                          m_step_size;
        //! Decay rate of the first moment estimates
        double                          m_decay_rate_a;
        //! Decay rate of the second moment estimates
        double                          m_decay_rate_b;

        //! Iteration step counter
        int                             m_timestep;
        //! Accumulated first moment decay rate
        double                          m_decay_rate_a_t;
        //! Accumulated second moment decay rate
        double                          m_decay_rate_b_t;

        //! First moment values for each parameter
        std::vector<double>             m_first_moment;
        //! Second moment values for each parameter
        std::vector<double>             m_second_moment;
};


#endif /* __ADAM_H__ */
