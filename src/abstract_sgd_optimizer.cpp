#include <cassert>
#include <iostream>

#include "abstract_sgd_optimizer.h"


AbstractSgdOptimizer::AbstractSgdOptimizer(std::vector<double> initial_values)
    :   m_parameters(initial_values)
{}


AbstractSgdOptimizer::~AbstractSgdOptimizer()
{}


std::vector<double> AbstractSgdOptimizer::update_parameters(
        std::vector<double> const&      gradients
)
{
    assert(gradients.size() == m_parameters.size());

    do_perform_update(gradients);
    return m_parameters;
}

std::vector<double> AbstractSgdOptimizer::get_parameters() const
{
    return m_parameters;
}
