#pragma once

#include <vector>
#include <assert.h>
#include <iostream>

class FullyConnected {
private:
	// инициализируем веса массив массивов
	std::vector < std::vector <double> > weights;
	std::vector <double> out_vals;
	std::vector <double> in_vals;
	int out_dim;
	int in_dim;
	double eta;

public:
	FullyConnected() {}
	// конструктор
	FullyConnected(int input_size, int output_size, double learning_rate)
	{
		assert(input_size > 0);
		assert(output_size > 0);

		in_dim = input_size;
		out_dim = output_size;
		eta = learning_rate;

		//генерируем случайные веса

		for (size_t out = 0; out < out_dim; ++out)
		{
			weights.push_back(std::vector <double>());
			for (size_t in = 0; in < in_dim + 1; ++in)
			{
				weights.back().push_back((double)rand() / RAND_MAX);
			}

		}
	}

	std::vector <double> feedForward(const std::vector<double>& input);
	std::vector <double> backprop(const std::vector<double>& grad);
};

std::vector <double> FullyConnected::feedForward(const std::vector<double>& input)
{
	assert(input.size() == in_dim);

	out_vals = std::vector<double>();
	in_vals = input;

	for (size_t out = 0; out < out_dim; ++out)
	{
		double sum = 0.0;
		for (size_t w = 0; w < in_dim; ++w)
		{
			sum += weights[out][w] * input[w];
		}
		sum += weights[out][in_dim];
		out_vals.push_back(sum);
	}
	return out_vals;
}

std::vector <double> FullyConnected::backprop(const std::vector<double>& grad)
{
	assert(grad.size() == out_dim);

	std::vector<double> prev_layer_grad;

	for (size_t in = 0; in < in_dim; ++in)
	{
		double g = 0.0;
		for (size_t out = 0; out < out_dim; ++out)
		{
			g += (grad[out] * weights[out][in]);
		}
		prev_layer_grad.push_back(g);
	}

	for (size_t out = 0; out < out_dim; ++out)
	{
		for (size_t in = 0; in < in_dim; ++in)
		{
			weights[out][in] -= eta * grad[out] * in_vals[in];
		}
		weights[out][in_dim] -= eta * grad[out];
	}

	return prev_layer_grad;
}