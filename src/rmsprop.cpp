#include <cmath>
#include <stddef.h>
#include <iostream>
#include "rmsprop.h"


Rmsprop::Rmsprop(
           std::vector<double>         initial_values,
           double                      step_size,
           double                      decay_rate
        
           )   :   AbstractSgdOptimizer(initial_values)
, m_step_size(step_size)
, m_decay_rate(decay_rate)
, m_second_moment(initial_values.size(), 0.0)
{}

void Rmsprop::do_perform_update(std::vector<double> const& gradients)
{

    
    // Precompute values
    auto rate_inv = 1.0 - m_decay_rate;
  
    
    for(size_t i=0; i<gradients.size(); ++i)
    {
     
        m_second_moment[i] = m_decay_rate * m_second_moment[i] +
        rate_inv * gradients[i] * gradients[i];
        
        
    m_parameters[i] -=    (m_step_size/ (std::sqrt(m_second_moment[i] + 1e-8) ))*gradients[i] ;
       
    }
}

