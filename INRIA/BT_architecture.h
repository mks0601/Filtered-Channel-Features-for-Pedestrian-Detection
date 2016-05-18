#include <iostream>
#include <vector>
#include <time.h>
#include <Windows.h>
#include <string>
#include <sstream>
#include <fstream>
#include <omp.h>
#include <opencv2/core/core.hpp>
#include <opencv2\opencv.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>


#define NofTree_first 32
#define NofTree_second 128
#define NofTree_third 512
#define NofTree_fourth 2048

#define Nofboost 4
#define maxDepth 2
#define minCount 0.01
#define minDistribution 0.001
#define bin_size 6
#define luv_size 3
#define feature_map_num 10

#define ACF 0
#define CHECKBOARD 1

#define block_size 8
#define block_stride 4
#define cell_size 4
#define padding 16
#define patch_row 128
#define patch_col 64

#define tau_selection_iter 30

#define soft_cascade 20
#define result_unit 100
#define train_negative_num 5000
#define additional_negative 5000

#define scale_per_octave 8
#define pyramid_scale pow(0.5, 1 / (double)scale_per_octave)
#define IoUthr 0.5
#define WINDOW_PER_IMAGE 25

#define ASSERT_WITH_MESSAGE(condition, message) do { \
if (!(condition)) { printf((message)); } \
assert ((condition)); } while(false)

#define positive_train_dir "D:\\Study/Data/INRIA_Pedestrian/train_64x128_H96/pos/"
#define negative_train_dir "D:\\Study/Data/INRIA_Pedestrian/train_64x128_H96/neg/"
#define hard_negative_dir "D:\\Study/Data/INRIA_Pedestrian/Train/neg/"
#define save_dir "D:\\Study/Data/INRIA_Pedestrian/Result/"
#define test_dir "D:\\Study/Data/INRIA_Pedestrian/Test/pos/"
#define result_dir "D:\\Study/Data/INRIA_Pedestrian/Result/"
#define ground_truth_dir "D:\\Study/Data/INRIA_Pedestrian/Test/annotations/"



using namespace std;
using namespace cv;

vector<string> get_all_files_names_within_folder(string folder);

 class ground_truth
{
public:
	int x_min;
	int y_min;
	int x_max;
	int y_max;
	bool is_detected;
};
 class bounding_box
{
public:
	Rect rect;
	bool is_false_positive;
	bool is_suppressed;
	double confidence;
};
class sorted_feature
{
public:
	int sid;
	bool is_pos;
	int filter_index;
	int feature_map_index;
	int row_index;
	int col_index;
	float pos_weight_accul;
	float neg_weight_accul;
	float value;

	bool operator<(const sorted_feature& ref) const { return value < ref.value; }

};
 class feature
{
public:
	vector<Mat> feature_map;
	vector<sorted_feature> extracted_feature;
	bool is_correct;
	double weight;
};
class node
{
public:

	node(){ left_child = NULL; right_child = NULL; };

	int depth;
	int feature_map_index;
	int filter_index;
	int row_index;
	int col_index;
	double tau;
	double alpha;

	double distribution_positive;
	double distribution_negative;

	double count_positive;
	double count_negative;


	vector<feature*> positive_feature;
	vector<feature*> negative_feature;


	node* left_child;
	node* right_child;
};


double get_error(node* tree_ptr)
{
	if (tree_ptr->right_child == NULL && tree_ptr->left_child == NULL)
	{
		double ret = 0;

		if (tree_ptr->distribution_negative > tree_ptr->distribution_positive)
		{
			for (int i = 0; i < tree_ptr->positive_feature.size(); i++)
				ret = ret + tree_ptr->positive_feature.at(i)->weight;


		}
		else
		{
			for (int i = 0; i < tree_ptr->negative_feature.size(); i++)
				ret = ret + tree_ptr->negative_feature.at(i)->weight;

		}

		return ret;
	}
	else
	{
		double ret = 0;

		ret += get_error(tree_ptr->left_child);
		ret += get_error(tree_ptr->right_child);
		return ret;

	}

}
void update_feature(node* tree_ptr)
{
	//decide features are corretly classified
	if (tree_ptr->right_child == NULL && tree_ptr->left_child == NULL)
	{


		if (tree_ptr->distribution_negative > tree_ptr->distribution_positive)
		{
			for (int i = 0; i < tree_ptr->negative_feature.size(); i++)
				tree_ptr->negative_feature.at(i)->is_correct = true;




			for (int i = 0; i < tree_ptr->positive_feature.size(); i++)
				tree_ptr->positive_feature.at(i)->is_correct = false;



		}
		else
		{
			for (int i = 0; i < tree_ptr->negative_feature.size(); i++)
				tree_ptr->negative_feature.at(i)->is_correct = false;




			for (int i = 0; i < tree_ptr->positive_feature.size(); i++)
				tree_ptr->positive_feature.at(i)->is_correct = true;


		}


	}
	else
	{
		update_feature(tree_ptr->left_child);
		update_feature(tree_ptr->right_child);
	}
}
void delete_tree(node* tree_ptr)
{
	if (tree_ptr->left_child == NULL && tree_ptr->right_child == NULL)
		delete tree_ptr;

	else
	{
		delete_tree(tree_ptr->left_child);
		tree_ptr->left_child = NULL;
		delete_tree(tree_ptr->right_child);
		tree_ptr->right_child = NULL;
	}

}
void split_node(node* tree_ptr, int depth, vector<Mat> filter_list, vector<feature*> positive_original, vector<feature*> negative_original)
{

	if (tree_ptr->depth == maxDepth || tree_ptr->count_negative < minCount || tree_ptr->count_positive < minCount || tree_ptr->distribution_negative < minDistribution || tree_ptr->distribution_positive < minDistribution)
		return;
	else
	{

		int save_feature_map_index;
		int save_filter_index;
		int save_row_index;
		int save_col_index;
		double save_tau;
		int save_fid;


		double entropy_original;
		double entropy_left;
		double entropy_right;
		double max_information_gain = -9999.0;
		double information_gain;


		double distribution_positive = tree_ptr->distribution_positive;
		double distribution_negative = tree_ptr->distribution_negative;
		entropy_original = -(distribution_positive * log10(distribution_positive) + distribution_negative * log10(distribution_negative));	//Shannon entropy is used


		double count_left_positive = 0;
		double count_right_positive = 0;
		double count_left_negative = 0;
		double count_right_negative = 0;

		double distribution_left_positive;
		double distribution_right_positive;
		double distribution_left_negative;
		double distribution_right_negative;


		double save_distribution_left_positive;
		double save_distribution_right_positive;
		double save_distribution_left_negative;
		double save_distribution_right_negative;

		double save_count_left_positive;
		double save_count_right_positive;
		double save_count_left_negative;
		double save_count_right_negative;


		tree_ptr->left_child = new node();
		tree_ptr->right_child = new node();

		clock_t begin, end;

		//select feature subset in a greedy way
		#pragma omp parallel for num_threads(omp_get_max_threads())
		for (int fid = 0; fid < positive_original[0]->extracted_feature.size(); fid++)
		{
			sorted_feature* selected_feature = new sorted_feature[tree_ptr->positive_feature.size() + tree_ptr->negative_feature.size()];

			for (int i = 0; i < tree_ptr->positive_feature.size(); i++)
				selected_feature[i] = tree_ptr->positive_feature[i]->extracted_feature[fid];
			for (int i = 0; i < tree_ptr->negative_feature.size(); i++)
				selected_feature[i + tree_ptr->positive_feature.size()] = tree_ptr->negative_feature[i]->extracted_feature[fid];

			sort(selected_feature, selected_feature + tree_ptr->positive_feature.size() + tree_ptr->negative_feature.size());

			for (int i = 0; i < tree_ptr->positive_feature.size() + tree_ptr->negative_feature.size(); i++)
			{
				if (selected_feature[i].is_pos)
				{
					selected_feature[i].pos_weight_accul = positive_original[selected_feature[i].sid]->weight;
					selected_feature[i].neg_weight_accul = 0;
				}
				else
				{
					selected_feature[i].pos_weight_accul = 0;
					selected_feature[i].neg_weight_accul = negative_original[selected_feature[i].sid]->weight;
				}

				if (i > 0)
				{
					selected_feature[i].pos_weight_accul += selected_feature[i - 1].pos_weight_accul;
					selected_feature[i].neg_weight_accul += selected_feature[i - 1].neg_weight_accul;
				}
			}

			//selecting tau
			for (int tau_iter = 1; tau_iter < tau_selection_iter; tau_iter++)
			{

				int tau_index = rand_float(0, tree_ptr->positive_feature.size() + tree_ptr->negative_feature.size() - 1);

				count_left_positive = selected_feature[tau_index].pos_weight_accul;
				count_right_positive = tree_ptr->count_positive - count_left_positive;

				count_left_negative = selected_feature[tau_index].neg_weight_accul;
				count_right_negative = tree_ptr->count_negative - count_left_negative;


				distribution_left_positive = (count_left_positive) / (count_left_positive + count_left_negative);
				distribution_left_negative = (count_left_negative) / (count_left_positive + count_left_negative);
				entropy_left = -(distribution_left_positive * log10(distribution_left_positive) + distribution_left_negative * log10(distribution_left_negative));	//Shannon entropy is used


				distribution_right_positive = (count_right_positive) / (count_right_positive + count_right_negative);
				distribution_right_negative = (count_right_negative) / (count_right_positive + count_right_negative);
				entropy_right = -(distribution_right_positive * log10(distribution_right_positive) + distribution_right_negative * log10(distribution_right_negative));	//Shannon entropy is used


				//calculate information gain
				information_gain = entropy_original - (((count_left_positive + count_left_negative) / (tree_ptr->count_positive + tree_ptr->count_negative))*entropy_left + ((count_right_positive + count_right_negative) / (tree_ptr->count_positive + tree_ptr->count_negative))*entropy_right);


				if (information_gain > max_information_gain)
				{
					max_information_gain = information_gain;

					save_fid = fid;

					save_tau = selected_feature[tau_index].value;
					save_feature_map_index = selected_feature[tau_index].feature_map_index;
					save_filter_index = selected_feature[tau_index].filter_index;
					save_col_index = selected_feature[tau_index].col_index;
					save_row_index = selected_feature[tau_index].row_index;

					save_distribution_left_positive = distribution_left_positive;
					save_distribution_left_negative = distribution_left_negative;
					save_distribution_right_positive = distribution_right_positive;
					save_distribution_right_negative = distribution_right_negative;

					save_count_left_positive = count_left_positive;
					save_count_left_negative = count_left_negative;
					save_count_right_positive = count_right_positive;
					save_count_right_negative = count_right_negative;
				}
			}


			delete[] selected_feature;
		}



		//split complete

		tree_ptr->depth = depth;
		tree_ptr->tau = save_tau;
		tree_ptr->feature_map_index = save_feature_map_index;
		tree_ptr->filter_index = save_filter_index;
		tree_ptr->col_index = save_col_index;
		tree_ptr->row_index = save_row_index;


		tree_ptr->left_child->distribution_positive = save_distribution_left_positive;
		tree_ptr->left_child->distribution_negative = save_distribution_left_negative;
		tree_ptr->left_child->count_positive = save_count_left_positive;
		tree_ptr->left_child->count_negative = save_count_left_negative;
		tree_ptr->left_child->depth = depth + 1;
		


		tree_ptr->right_child->distribution_positive = save_distribution_right_positive;
		tree_ptr->right_child->distribution_negative = save_distribution_right_negative;
		tree_ptr->right_child->count_positive = save_count_right_positive;
		tree_ptr->right_child->count_negative = save_count_right_negative;
		tree_ptr->right_child->depth = depth + 1;
	

		for (int i = 0; i < tree_ptr->positive_feature.size(); i++)
		{
			float filtered_value = tree_ptr->positive_feature[i]->extracted_feature[save_fid].value;

			if (filtered_value <= save_tau)
				tree_ptr->left_child->positive_feature.push_back(tree_ptr->positive_feature[i]);
			else
				tree_ptr->right_child->positive_feature.push_back(tree_ptr->positive_feature[i]);
		}


		for (int i = 0; i < tree_ptr->negative_feature.size(); i++)
		{
			float filtered_value = tree_ptr->negative_feature[i]->extracted_feature[save_fid].value;

			if (filtered_value <= save_tau)
				tree_ptr->left_child->negative_feature.push_back(tree_ptr->negative_feature[i]);
			else
				tree_ptr->right_child->negative_feature.push_back(tree_ptr->negative_feature[i]);
		}


		split_node(tree_ptr->left_child, depth + 1, filter_list, positive_original, negative_original);
		split_node(tree_ptr->right_child, depth + 1, filter_list, positive_original, negative_original);


	}


}


void save_tree_recursion(node* tree, ofstream& of)
{
	stringstream ss;

	if (tree->left_child == NULL && tree->right_child == NULL)
	{
		ss << 0 << " " << tree->depth << " " << tree->feature_map_index << " " << tree->row_index << " " << tree->col_index << " " << tree->tau << " " << tree->alpha << " " << tree->distribution_positive << " " << tree->distribution_negative << " " << tree->count_positive << " " << tree->count_negative << endl;
		of << ss.str();
		return;
	}
	else
	{
		ss << 1 << " " << tree->depth << " " << tree->feature_map_index << " " << tree->row_index << " " << tree->col_index << " " << tree->tau << " " << tree->alpha << " " << tree->distribution_positive << " " << tree->distribution_negative << " " << tree->count_positive << " " << tree->count_negative << endl;
		ss << tree->filter_index << " " << endl;
		of << ss.str();
		save_tree_recursion(tree->left_child, of);
		save_tree_recursion(tree->right_child, of);
	}
}
void save_tree(vector<node*> tree_head)
{
	ofstream of;
	stringstream ss;
	vector<string> file_list;

	ss << save_dir << "tree_save";
	CreateDirectory(ss.str().c_str(), NULL);
	ss.str("");
	ss.clear();

	ss << save_dir << "tree_save/";
	file_list = get_all_files_names_within_folder(ss.str());
	ss.str("");
	ss.clear();

	for (int i = 0; i < file_list.size(); i++)
	{
		ss << save_dir << "tree_save/" << file_list[i];
		remove(ss.str().c_str());
		ss.str("");
		ss.clear();
	}

	for (int i = 0; i < tree_head.size(); i++)
	{
		ss << save_dir << "tree_save/" << i << ".txt";
		of.open(ss.str());
		ss.str("");
		ss.clear();
		save_tree_recursion(tree_head.at(i), of);
		of.close();
	}

}

void load_tree_recursion(node* tree, ifstream& inf)
{
	int stop_criteria;
	stringstream ss;
	string line;
	getline(inf, line);
	ss << line;
	ss >> stop_criteria >> tree->depth >> tree->feature_map_index >> tree->row_index >> tree->col_index >> tree->tau >> tree->alpha >> tree->distribution_positive >> tree->distribution_negative >> tree->count_positive >> tree->count_negative;

	if (stop_criteria == 0)
		return;
	else
	{
		ss.str("");
		ss.clear();
		getline(inf, line);
		ss << line;

		string token;
		getline(ss, token, ' ');
		tree->filter_index = atoi(token.c_str());

		tree->left_child = new node();
		tree->right_child = new node();
		load_tree_recursion(tree->left_child, inf);
		load_tree_recursion(tree->right_child, inf);
	}

}
void load_tree(vector<node*>* tree_head)
{
	ifstream inf;
	stringstream ss;
	node* tree_ptr;

	for (int i = 0;; i++)
	{
		ss << save_dir << "tree_save/" << i << ".txt";

		inf.open(ss.str());
		ss.str("");
		ss.clear();

		if (!inf)
			break;

		tree_ptr = new node();

		load_tree_recursion(tree_ptr, inf);
		tree_head->push_back(tree_ptr);
		inf.close();
	}

}

void save_filter(vector<Mat> filter_list)
{
	ofstream of;
	stringstream ss;

	vector<string> file_list;

	ss << save_dir << "filter_save";
	CreateDirectory(ss.str().c_str(), NULL);
	ss.str("");
	ss.clear();


	ss << save_dir << "filter_save/";
	file_list = get_all_files_names_within_folder(ss.str());
	ss.str("");
	ss.clear();

	for (int i = 0; i < file_list.size(); i++)
	{
		ss << save_dir << "filter_save/" << file_list[i];
		remove(ss.str().c_str());
		ss.str("");
		ss.clear();
	}

	for (int i = 0; i < filter_list.size(); i++)
	{
		ss << save_dir << "filter_save/" << i << ".txt";
		of.open(ss.str());
		ss.str("");
		ss.clear();
		of << filter_list[i].rows << " " << filter_list[i].cols;
		for (int row = 0; row < filter_list[i].rows; row++)
		{
			for (int col = 0; col < filter_list[i].cols; col++)
				of << " " << filter_list[i].at<float>(row, col);
		}
		of.close();
	}
}
void load_filter(vector<Mat>* filter_list)
{
	ifstream inf;
	stringstream ss;

	for (int i = 0;; i++)
	{

		ss << save_dir << "filter_save/" << i << ".txt";
		inf.open(ss.str());
		ss.str("");
		ss.clear();

		if (!inf)
			break;

		string line;
		getline(inf, line);
		ss << line;

		string token;
		getline(ss, token, ' ');
		int filter_row = atoi(token.c_str());
		getline(ss, token, ' ');
		int filter_col = atoi(token.c_str());

		Mat filter_load(Size(filter_col, filter_row), CV_32FC1);


		for (int row = 0; row < filter_row; row++)
		{
			for (int col = 0; col < filter_col; col++)
			{
				getline(ss, token, ' ');
				filter_load.at<float>(row, col) = atoi(token.c_str());
			}
		}

		filter_list->push_back(filter_load);

		inf.close();
		ss.str("");
		ss.clear();
	}
}