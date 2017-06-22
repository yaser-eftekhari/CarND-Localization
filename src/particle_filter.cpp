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
	// Set the number of particles.
	num_particles = 50;
	is_initialized = false;

	// Create a normal (Gaussian) distribution for x, y, theta.
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	// Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS)
	// Add random Gaussian noise to each particle.
	for(int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;

		particles.push_back(p);

		// Initialize all weights to 1.0
		weights.push_back(1.0);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	double x0, y0, theta0;

	// Create a normal (Gaussian) distribution for x, y, theta.
	// These distributions are set with mean 0 as they will be added to the updated states
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for(int i = 0; i < num_particles; i++) {
		Particle p = particles[i];

		x0 = p.x;
		y0 = p.y;
		theta0 = p.theta;

		//avoid division by zero
		if (fabs(yaw_rate) > 0.001) {
				p.x = x0 + velocity/yaw_rate * ( sin (theta0 + yaw_rate * delta_t) - sin(theta0)) + dist_x(gen);
				p.y = y0 + velocity/yaw_rate * ( cos(theta0) - cos(theta0 + yaw_rate * delta_t) ) + dist_y(gen);
		}
		else {
				p.x = x0 + velocity * delta_t * cos(theta0) + dist_x(gen);
				p.y = y0 + velocity * delta_t * sin(theta0) + dist_y(gen);
		}
		p.theta = fmod(theta0 + yaw_rate * delta_t + dist_theta(gen), 2.0 * M_PI);

		particles[i] = p;
	}
}

// Should preserve the order of predicted IDs to simplify the updateWeights step
vector<int> ParticleFilter::dataAssociation(	const vector<LandmarkObs> predicted,
																			vector<LandmarkObs> &observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	int obs_size = observations.size();
	int pred_size = predicted.size();

	// cout << "dataAssociation: obs_size:" << obs_size << ", pred_size:" << pred_size << endl;

	vector<int> associations;

	for(int obs_count = 0; obs_count < obs_size; obs_count++) {
		LandmarkObs observation = observations[obs_count];
		double min_distance = 50; // Setting to a big value such as sensor range
		int winning_landmark = 0; // we don't have a landmark with id = 0

		for(int pred_count = 0; pred_count < pred_size; pred_count++) {
			LandmarkObs test_landmark = predicted[pred_count];
			double current_dist = dist(observation.x, observation.y, test_landmark.x, test_landmark.y);
			if(current_dist < min_distance) {
				min_distance = current_dist;
				winning_landmark = test_landmark.id;
			}
		}
		observations[obs_count].id = winning_landmark;
		associations.push_back(winning_landmark);
	}

	return associations;
}

void ParticleFilter::updateWeights(	double sensor_range,
																		double std_landmark[],
																		vector<LandmarkObs> observations,
																		Map map_landmarks) {
	// TODO: This should be done once as it only depends on the map data
	vector<LandmarkObs> predicted = find_prediction(map_landmarks);

	for(int p = 0; p < particles.size(); p++) {
		Particle part = particles[p];
		double p_x = part.x;
		double p_y = part.y;
		double p_theta = part.theta;

		vector<LandmarkObs> observations_transformed = transform_observations(observations, p_x, p_y, p_theta);
		vector<int> associations = dataAssociation(predicted, observations_transformed);

		particles[p] = SetAssociations(part, associations, observations);

		// Update the weights of each particle using a multi-variate Gaussian distribution.
		double particle_weight = multivariate_normal_distribution(predicted, observations_transformed, std_landmark[0], std_landmark[1]);

		particles[p].weight = particle_weight;

		weights[p] = particle_weight;

		// cout << "updateWeights - particles[" << p << "].weight = " << particle_weight << endl;
	}
}

void ParticleFilter::resample() {
	random_device rd;
	mt19937 generator(rd());
	discrete_distribution<> distribution (weights.begin(), weights.end());

	// default_random_engine generator;

	vector<Particle> sampled_particles;

	for(int i = 0; i < weights.size(); i++) {
		// cout << "resample - weights[" << i << "] = " << weights[i] << endl;
	}

	// Resample particles with replacement with probability proportional to their weight.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	for(int i = 0; i < num_particles; i++) {
		int sampled_index = distribution(generator);

		// cout << "resample - sampled_index = " << sampled_index << endl;

		sampled_particles.push_back(particles[sampled_index]);
	}

	particles = sampled_particles;
}

Particle ParticleFilter::SetAssociations(	Particle particle,
																					vector<int> associations,
																					vector<LandmarkObs> observations) {
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	vector<double> sense_x, sense_y;

	particle.associations = associations;

	for(int i = 0; i < observations.size(); i++) {
		particle.sense_x.push_back(observations[i].x);
		particle.sense_y.push_back(observations[i].y);
	}

 	return particle;
}

string ParticleFilter::getAssociations(Particle best) {
	vector<int> v = best.associations;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best) {
	vector<double> v = best.sense_x;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best) {
	vector<double> v = best.sense_y;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
