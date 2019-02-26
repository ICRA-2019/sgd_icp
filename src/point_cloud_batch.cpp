#include <iostream>

#include "point_cloud_batch.h"


PointCloudBatch::PointCloudBatch(Cloud_t point_cloud, int batch_size)
    :   m_cloud(point_cloud)
      , m_current_offset(0)
      , m_batch_size(batch_size)
{
    // Initialize the random number generator
    std::random_device rand_dev;
    m_generator = std::mt19937(rand_dev());

    shuffle_data();
}

Cloud_t::Ptr PointCloudBatch::next_batch()
{
    Cloud_t::Ptr batch(new Cloud_t);

    auto target_offset = m_current_offset + m_batch_size;

    // If we require more data then left or then present copy the points
    // and then reshuffle (as often as needed) the sequence
    while(target_offset >= m_cloud.points.size())
    {
        while(m_current_offset < m_cloud.points.size())
        {
            batch->points.push_back(m_cloud.points[m_current_offset++]);
        }

        shuffle_data();
        m_current_offset = 0;
        target_offset = target_offset - m_cloud.points.size();
    }

    // Copy the reamining points needed
    while(m_current_offset < target_offset)
    {
        batch->points.push_back(m_cloud.points[m_current_offset++]);
    }

    return batch;
}

void PointCloudBatch::shuffle_data()
{
    std::shuffle(m_cloud.points.begin(), m_cloud.points.end(), m_generator);
}
