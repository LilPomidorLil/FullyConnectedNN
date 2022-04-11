#pragma once

# include <vector>
# include <assert.h>
# include <math.h>

class Sigmoid
{
private:
	std::vector<double> out_vals;

public:
	Sigmoid() {}
	std::vector<double> feedForward(const std::vector<double>& input);
	std::vector<double> backprop(std::vector<double>& grad);

};

std::vector<double> Sigmoid::feedForward(const std::vector<double>& input)
{
	out_vals = std::vector<double> ();

	for (size_t in = 0; in < input.size(); ++in)
	{
		out_vals.push_back(1.0 / (1.0 + exp(-input[in])));
	}

	return out_vals;
}

std::vector<double> Sigmoid::backprop(std::vector<double>& grad)
{
	assert(grad.size() == out_vals.size());

	for (size_t out = 0; out < out_vals.size(); ++out)
	{
		grad[out] *= out_vals[out] * (1.0 - out_vals[out]);
	}
	return grad;
}