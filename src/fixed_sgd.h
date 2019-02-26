#ifndef __FIXED_SGD_H__
#define __FIXED_SGD_H__


#include "abstract_sgd_optimizer.h"


/**
 * \brief Fixed step size SGD implementation.
 */
class FixedSgd : public AbstractSgdOptimizer
{
    public:
        /**
         * \brief Creates a new instance with fixed step size.
         *
         * \param initial_values initial values of the parameters
         * \param step_size fixed step size for updates
         */
        FixedSgd(
                std::vector<double>     initial_values,
                double                  step_size
        );


    protected:
        /**
         * \see AbstractSgdOptimizer::do_perform_update
         */
        void do_perform_update(std::vector<double> const& gradients) override;


    private:
        //! Fixed step size for SGD updates
        double                          m_step_size;
};


#endif /* _FIXED_SGD_H__ */
