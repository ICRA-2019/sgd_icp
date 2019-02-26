#ifndef __ABSTRACT_SGD_OPTIMIZER_H__
#define __ABSTRACT_SGD_OPTIMIZER_H__


#include <vector>


/**
 * \brief Base class for SGD optimization methods.
 */
class AbstractSgdOptimizer
{
    public:
        /**
         * \brief Creates a new instance with initial parameter values.
         *
         * \param initial_values initial values of the parameters
         *      to be optimized
         */
        AbstractSgdOptimizer(std::vector<double> initial_values);

        /**
         * \brief Virtual destructor.
         */
        virtual ~AbstractSgdOptimizer();

        /**
         * \brief Updates the parameters using the provided gradient.
         *
         * \param gradients gradient information for each of the parameters
         * \return new parmeter values after update
         */
        std::vector<double> update_parameters(
                std::vector<double> const& gradients
        );

        /**
         * \brief Returns the current parameter values.
         *
         * \return current values of the parameters
         */
        std::vector<double> get_parameters() const;


    protected:
        /**
         * \brief Virtual function performing the parameter update.
         *
         * \param gradients gradient information for each of the parameters
         */
        virtual
        void do_perform_update(std::vector<double> const& gradients) = 0;


    protected:
        //! Parameters to optimise
        std::vector<double>             m_parameters;
};



#endif /* __ABSTRACT_SGD_OPTIMIZER_H__ */
