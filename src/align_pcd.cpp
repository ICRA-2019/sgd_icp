#include <cmath>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <pcl/common/transforms.h>
#include <pcl/console/time.h>
#include <pcl/filters/filter.h>
#include <pcl/io/pcd_io.h>


#include "fixed_sgd.h"
#include "adadelta.h"
#include "adam.h"
#include "rmsprop.h"
#include "sgdicp.h"
#include "types.h"
#include "utils.h"


namespace po = boost::program_options;
namespace pt = boost::property_tree;


int main(int argc, char * argv[])
{
    // +------------------------------------------------------------------------
    // | Parse command line arguments
    // +------------------------------------------------------------------------

    po::options_description desc("Options");
    desc.add_options()
        ("help", "Show the help message")
        ("source", po::value<std::string>(), "Source point cloud")
        ("target", po::value<std::string>(), "Target point cloud")
        ("config", po::value<std::string>(), "Configuration")
    ;

    // Configure positional arguments
    po::positional_options_description pos_args;
    pos_args.add("source", 1);
    pos_args.add("target", 1);
    pos_args.add("config", 1);

    po::variables_map vm;
    po::store(
            po::command_line_parser(argc, argv)
                .options(desc)
                .positional(pos_args)
                .run(),
            vm
    );
    po::notify(vm);

    if(vm.count("help") || !vm.count("source") || !vm.count("target")
            || !vm.count("config")
    )
    {
        std::cout << desc << std::endl;
        return 1;
    }

     
    // +------------------------------------------------------------------------
    // | Processing of inputs via SGD ICP
    // +------------------------------------------------------------------------
    //pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);

    Cloud_t::Ptr cloud_in(new Cloud_t);
    Cloud_t::Ptr cloud_in2(new Cloud_t);
    Cloud_t::Ptr cloud_out(new Cloud_t);

    Cloud_t::Ptr result(new Cloud_t);

    // Load point clouds and clean them
    if (pcl::io::loadPCDFile<Point_t>(vm["source"].as<std::string>(), *cloud_in)==-1)
    {
        std::cout << "Could not read source file" << std::endl;
        return -1;
    }
    if (pcl::io::loadPCDFile<Point_t>(vm["target"].as<std::string>(), *cloud_out) ==-1)
    {
        std::cout << "Could not read target file" << std::endl;
        return -1;
    }

    pcl::io::loadPCDFile<Point_t>(vm["source"].as<std::string>(), *cloud_in2);
    
    auto indices = std::vector<int>();
    pcl::removeNaNFromPointCloud(*cloud_in, *cloud_in, indices);
    pcl::removeNaNFromPointCloud(*cloud_in2, *cloud_in, indices);
    pcl::removeNaNFromPointCloud(*cloud_out,*cloud_out, indices);
    
     
    // +------------------------------------------------------------------------
    // | Read configuration and setup accordingly
    // +------------------------------------------------------------------------
    auto config = pt::ptree{};
    pt::read_json(vm["config"].as<std::string>(), config);


    auto max_range = get_absolute_max(cloud_in, cloud_out);
    if(config.get<bool>("normalize-cloud"))
    {
        cloud_in = normalise_clouds(cloud_in, max_range);
        cloud_out = normalise_clouds(cloud_out, max_range);
    }

   
    std::vector<double> initial_guess ={config.get<double >("initial-guess.x"),
        config.get<double >("initial-guess.y"),
        config.get<double >("initial-guess.z"),
        config.get<double >("initial-guess.roll"),
        config.get<double >("initial-guess.pitch"),
        config.get<double >("initial-guess.yaw")};
    
    
    std::unique_ptr<SGDICP> sgd_icp;
    if(config.get<std::string>("method") == "fixed")
    {
        sgd_icp.reset(new SGDICP(
                std::unique_ptr<FixedSgd>(
                    new FixedSgd(
                        initial_guess,
                        config.get<double>("fixed.step-size")
                    )
                )
        ));
    }
    else if(config.get<std::string>("method") == "adadelta")
    {
        sgd_icp.reset(new SGDICP(
                std::unique_ptr<AdaDelta>(
                    new AdaDelta(
                        initial_guess,
                        config.get<double>("adadelta.decay-rate"),
                        config.get<double>("adadelta.preconditioner")
                    )
                )
        ));
    }
    else if(config.get<std::string>("method") == "adam")
    {
        sgd_icp.reset(new SGDICP(
                std::unique_ptr<Adam>(
                    new Adam(
                        initial_guess,
                        config.get<double>("adam.step-size"),
                        config.get<double>("adam.decay-rate-a"),
                        config.get<double>("adam.decay-rate-b")
                    )
                )
        ));
    }
    else if(config.get<std::string>("method") == "rmsprop")
    {
        sgd_icp.reset(new SGDICP(
                                 std::unique_ptr<Rmsprop>(
                                                       new Rmsprop(
                                                                initial_guess,
                                                                config.get<double>("rmsprop.step-size"),
                                                                config.get<double>("rmsprop.decay-rate")
                                                                )
                                                       )
                                 ));
    }
    else
    
    {
        std::cout << "Invalid optimizer specified, valid optiosn are: "
                  << "adadelta, adam, fixed, rmsprop" << std::endl;
        return 1;
    }

     
    // +------------------------------------------------------------------------
    // | Perform ICP alignment
    // +------------------------------------------------------------------------
    auto time = pcl::console::TicToc();
    time.tic();
    auto transformation_matrix = sgd_icp->allign_clouds(
            cloud_in,
            cloud_out,
            SGDICP::Parameters(
                config.get<int>("icp.max-iterations"),
                config.get<int>("icp.batch-size"),
                config.get<double>("icp.max-matching-distance"),
                config.get<int>("icp.convergence-steps"),
                config.get<double>("icp.translational-convergence"),
                config.get<double>("icp.rotational-convergence"),
                config.get<bool>("icp.filter")
            )
    );
    std::cout << "ICP Duration: " << time.toc() << " ms" << std::endl;
     auto rmse = compute_rmse(cloud_in, cloud_out, transformation_matrix);
     
    // If the clouds were normalized undo this to obtain the true transformation
    
    if(config.get<bool>("normalize-cloud"))
    {
        rescale_transformation_matrix(transformation_matrix, max_range);
    }
    
    
    std::cout << "RMSE: "<< rmse << "\n";
    std::cout << "Transformation_matrix:\n"
              << transformation_matrix << std::endl;

    auto parameters = get_translation_roll_pitch_yaw(transformation_matrix);

    // Save resulting point cloud
    pcl::transformPointCloud<Point_t>(
            *cloud_in2,
            *result,
            transformation_matrix
    );
    pcl::io::savePCDFileASCII("result_sgdicp.pcd", *result);

    return 0;
}
