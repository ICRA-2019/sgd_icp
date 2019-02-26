#include <cmath>
#include <stddef.h>

#include "adam.h"


Adam::Adam(
        std::vector<double>         initial_values,
        double                      step_size,
        double                      decay_rate_a,
        double                      decay_rate_b
)   :   AbstractSgdOptimizer(initial_values)
      , m_step_size(step_size)
      , m_decay_rate_a(decay_rate_a)
      , m_decay_rate_b(decay_rate_b)
      , m_timestep(0)
      , m_decay_rate_a_t(1.0)
      , m_decay_rate_b_t(1.0)
      , m_first_moment(initial_values.size(), 0.0)
      , m_second_moment(initial_values.size(), 0.0)
{}

void Adam::do_perform_update(std::vector<double> const& gradients)
{
    // Update time step related variables
    m_timestep++;
    m_decay_rate_a_t *= m_decay_rate_a;
    m_decay_rate_b_t *= m_decay_rate_b;

    // Precompute values
    auto rate_a_inv = 1.0 - m_decay_rate_a;
    auto rate_b_inv = 1.0 - m_decay_rate_b;

    for(size_t i=0; i<gradients.size(); ++i)
    {
        m_first_moment[i] = m_decay_rate_a * m_first_moment[i] +
            rate_a_inv * gradients[i];
        m_second_moment[i] = m_decay_rate_b * m_second_moment[i] +
            rate_b_inv * gradients[i] * gradients[i];

        auto fm_unbiased = m_first_moment[i] / (1.0 - m_decay_rate_a_t);
        auto sm_unbiased = m_second_moment[i] / (1.0 - m_decay_rate_b_t);

        m_parameters[i] -= m_step_size * fm_unbiased / (std::sqrt(sm_unbiased) + 1e-8);
    }
}
