/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 1000;
	weights.resize(num_particles); 
	std::random_device rd;
    std::mt19937 gen(rd());
    
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	for (int i=0; i<num_particles; ++i) {
		double sample_x, sample_y, sample_psi;
		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_psi = dist_theta(gen);
		particles.push_back(Particle{i, sample_x, sample_y, sample_psi, 1});
	}
	is_initialized = true;
	cout << "Initialized" << endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	//default_random_engine gen;
	std::random_device rd;
    std::mt19937 gen(rd());
    
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);
	for (int i=0; i<num_particles; ++i) {
		double x, y, theta, v_theta;
		//prediction
		theta = particles[i].theta + yaw_rate*delta_t;
		if(yaw_rate != 0) {
			v_theta = velocity/yaw_rate;
			particles[i].x += v_theta * (sin(theta) - sin(particles[i].theta));
			particles[i].y += v_theta * (cos(particles[i].theta) - cos(theta));
		}
		else {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
		particles[i].theta = theta;
		//add noise update x, y and theta by new prediction for each particle
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(const std::vector<LandmarkObs>& predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (auto& obs : observations) {
		double d;
		for (int i=0; i<predicted.size(); ++i) {		
			if (i == 0) {
				d = dist(obs.x, obs.y, predicted[i].x, predicted[i].y);
				obs.id = i;
			 } else {
			 	double d2 = dist(obs.x, obs.y, predicted[i].x, predicted[i].y);
				if (d2 < d) {
					d = d2;
					obs.id = i;
				}
			}
		}
	}
}

void ParticleFilter::updateWeights(const double& sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs>& observations, const Map& map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	int part_index = 0;
	for (auto& p : particles) {
		p.weight = 1;
		//transform between VEHICLE AND MAP coordinate systems
		std::vector<LandmarkObs> observations_map;
		for (const auto& obs : observations) {
			LandmarkObs obs_map;
			obs_map.x = p.x + obs.x * cos(p.theta) - obs.y * sin(p.theta);
			obs_map.y = p.y + obs.x * sin(p.theta) + obs.y * cos(p.theta);
			observations_map.push_back(obs_map);
		}
		std::vector<LandmarkObs> predicted_map;
		for (const auto& obs : map_landmarks.landmark_list) {
			if (dist(p.x, p.y, obs.x_f, obs.y_f) <= sensor_range) {
				LandmarkObs obs_map;
				obs_map.id = obs.id_i;
				obs_map.x = obs.x_f;
				obs_map.y = obs.y_f;		
				predicted_map.push_back(obs_map);
			}
		}
		if (predicted_map.size() < 1) {
			p.weight = 0;
		} else {
			ParticleFilter::dataAssociation(predicted_map, observations_map);
			for (const auto& obs : observations_map) {
				int index = obs.id;
				p.weight *= normpdf(obs.x, predicted_map[index].x, std_landmark[0],
									obs.y, predicted_map[index].y, std_landmark[1]);
			}
		}
		weights[part_index] = p.weight;
		part_index++;
	}	
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::random_device rd;
    std::mt19937 gen(rd());
    
	std::discrete_distribution<> d(weights.begin(), weights.end());
	std::vector<Particle> new_particles;
	for (int i=0; i<num_particles; ++i){
		int new_p_index = d(gen);
		new_particles.push_back(particles[new_p_index]);
	}
	particles = new_particles;
	
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
