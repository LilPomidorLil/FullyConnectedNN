# include <vector>
# include <iostream>
# include "Model.h"

int main() {
	Model model;

	std::vector< std::vector < double > > train_data;

	for (size_t i = 0; i < 5000; i++)
	{
		double x1 = (double)rand() / RAND_MAX;
		double x2 = (-(5.0 / 7.0) * x1) + ((double)rand() / RAND_MAX) + 0.0001;
		train_data.push_back({ x1, x2, 1.0 });
	}

	for (size_t i = 0; i < 5000; i++)
	{
		double x1 = (double)rand() / RAND_MAX;
		double x2 = (-(5.0 / 7.0) * x1) + ((double)rand() / RAND_MAX) - 0.0001;
		train_data.push_back({ x1, x2, 0.0 });
	}

	//for (size_t i = 0; i < train_data.size(); i++)
	//{
	//	std::cout << train_data[i][0] << ", " << train_data[i][1] << ", " << train_data[i][2] << std::endl;
	//}

	for (size_t i = 0; i < 1000; i++)
	{
		int ind = rand() % train_data.size();
		std::vector<double> output = model.feedForward({ train_data[ind][0], train_data[ind][1] });
		model.backpror({ train_data[ind][2] });
	}

	double acc = 0.0;

	for (size_t i = 0; i < train_data.size(); i++)
	{
		std::vector<double> output = model.feedForward({ train_data[i][0], train_data[i][1] });

		if (output[0] > 0.5 && train_data[i][2] == 0.0) { acc ++; }
		else if (output[0] < 0.5 && train_data[i][2] == 1.0) { acc++; }
	}

	std::cout << "Accuracy: " << acc / train_data.size() << std::endl;
	return 0;
}