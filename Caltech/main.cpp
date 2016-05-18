#include "BT_architecture.h"
#include "Etc.h"


int main()
{
	vector<node*> tree_head;
	node* tree_ptr;

	vector<feature*> positive_feature;
	vector<feature*> negative_feature;
	vector<Mat> filter_list;

	stringstream ss;
	vector<string> folder_list;
	vector<string> sub_folder_list;
	vector<string> file_list;
	int negative_initial_size = 0;

	clock_t begin, end, training_time, testing_time;
	double alpha_sum;
	int* tree_number_list;
	tree_number_list = new int[Nofboost];


	//should be determined before
	tree_number_list[0] = NofTree_first;
	tree_number_list[1] = NofTree_second;
	tree_number_list[2] = NofTree_third;
	tree_number_list[3] = NofTree_fourth;
	tree_number_list[4] = NofTree_fifth;
	//

	srand(time(NULL));

	cout << "Pedestrian detection based on CALTECH dataset" << endl;
	cout << "Using boosted decision tree and feature filtering" << endl;
	cout << "Developed by Gyeongsik Moon" << endl;
	cout << endl << endl;

	string save_load;
	cout << "SAVE or LOAD? ";
	cin >> save_load;

	if (save_load.compare("SAVE") && save_load.compare("LOAD"))
	{
		cout << "Not a proper input..." << endl;
		return 0;
	}


	if (!save_load.compare("SAVE"))
	{

		string which_filter;
		cout << "which filter use?(ACF, CHECKBOARD) ";
		cin >> which_filter;

		if (which_filter.compare("ACF") && which_filter.compare("CHECKBOARD"))
		{
			cout << "Not a proper filter..." << endl;
			return 0;
		}

		////loading dataset
		//loading positive features
		cout << endl << "Extracting feature of training images..." << endl;
		ss << positive_train_dir;
		file_list = get_all_files_names_within_folder(ss.str());
		ss.str("");
		ss.clear();
		for (int positive_image_index = 0; positive_image_index < file_list.size(); positive_image_index++)
		{
			feature* tmp = new feature();

			ss << positive_train_dir;
			ss << file_list.at(positive_image_index);

			Mat positive_image;
			positive_image = imread(ss.str());
			ss.str("");
			ss.clear();

			int width = positive_image.cols;
			int height = positive_image.rows;


			while ((width - block_size) % block_stride != 0)
				width--;
			while ((height - block_size) % block_stride != 0)
				height--;

			vector<Mat> HOG_channel = HOG_extract(positive_image, false, width, height);

			for (int bin = 0; bin < bin_size; bin++)
				tmp->feature_map.push_back(HOG_channel[bin]);



			Rect ROI(padding, padding, patch_col, patch_row);
			Mat positive_image_ROI = positive_image(ROI);

			width = positive_image_ROI.cols;
			height = positive_image_ROI.rows;


			while ((width - block_size) % block_stride != 0)
				width--;
			while ((height - block_size) % block_stride != 0)
				height--;


			vector<Mat>LUV_channel = LUV_extract(positive_image_ROI, width, height);



			for (int luv_index = 0; luv_index < luv_size; luv_index++)
				tmp->feature_map.push_back(LUV_channel[luv_index]);

			Mat gradient_mag = GRADIENT_extract(positive_image_ROI, width, height);
			tmp->feature_map.push_back(gradient_mag);

			positive_image.release();
			positive_image_ROI.release();
			positive_feature.push_back(tmp);
		}
		//
		file_list.clear();


		//loading negative features
		ss << negative_train_dir;
		file_list = get_all_files_names_within_folder(ss.str());
		ss.str("");
		ss.clear();

		for (int negative_image_index = 0; negative_image_index < file_list.size(); negative_image_index++)
		{
			feature* tmp = new feature();

			ss << negative_train_dir;
			ss << file_list.at(negative_image_index);

			Mat negative_image;
			negative_image = imread(ss.str());
			ss.str("");
			ss.clear();

			int width = negative_image.cols;
			int height = negative_image.rows;


			while ((width - block_size) % block_stride != 0)
				width--;
			while ((height - block_size) % block_stride != 0)
				height--;

			vector<Mat> HOG_channel = HOG_extract(negative_image, false, width, height);
			for (int bin = 0; bin < bin_size; bin++)
				tmp->feature_map.push_back(HOG_channel[bin]);


			Rect ROI(padding, padding, patch_col, patch_row);
			Mat negative_image_ROI = negative_image(ROI);

			width = negative_image_ROI.cols;
			height = negative_image_ROI.rows;


			while ((width - block_size) % block_stride != 0)
				width--;
			while ((height - block_size) % block_stride != 0)
				height--;

			vector<Mat>LUV_channel = LUV_extract(negative_image_ROI, width, height);
			for (int luv_index = 0; luv_index < luv_size; luv_index++)
				tmp->feature_map.push_back(LUV_channel[luv_index]);


			Mat gradient_mag = GRADIENT_extract(negative_image_ROI, width, height);
			tmp->feature_map.push_back(gradient_mag);

			negative_image.release();
			negative_image_ROI.release();
			negative_feature.push_back(tmp);
		}
		//
		file_list.clear();
		//
		////

		////filter generation and apply(checkboard filter)
		cout << "Generating filters..." << endl << endl << endl;

		//filter generation
		if (!which_filter.compare("ACF"))
			filter_list = filter_generation(ACF);
		else if (!which_filter.compare("CHECKBOARD"))
			filter_list = filter_generation(CHECKBOARD);
		//
		//
		////
		////TRAINING start!
		cout << "Training start!!" << endl;
		begin = clock();

		for (int boost_number = 0;; boost_number++)
		{

			alpha_sum = 0;

			//weight initialize
			for (int i = 0; i < positive_feature.size(); i++)
				positive_feature.at(i)->weight = (double)1 / (double)(2 * positive_feature.size());

			for (int i = 0; i < negative_feature.size(); i++)
				negative_feature.at(i)->weight = (double)1 / (double)(2 * negative_feature.size());
			//

			cout << "Extracting all feature space..." << endl;


			//Extracting all feature space
			if (boost_number == 0)
			{
				for (int filter_index = 0; filter_index < filter_list.size(); filter_index++)
				{
					for (int feature_map_index = 0; feature_map_index < feature_map_num; feature_map_index++)
					{
						for (int row_index = 0; row_index < (patch_row / cell_size - filter_list[filter_index].rows + 1); row_index++)
						{
							for (int col_index = 0; col_index < (patch_col / cell_size - filter_list[filter_index].cols + 1); col_index++)
							{

								for (int i = 0; i < positive_feature.size(); i++)
								{
									sorted_feature tmp;
									tmp.filter_index = filter_index;
									tmp.feature_map_index = feature_map_index;
									tmp.row_index = row_index;
									tmp.col_index = col_index;
									tmp.sid = i;
									tmp.is_pos = true;
									tmp.pos_weight_accul = 0;
									tmp.neg_weight_accul = 0;

									float filtered_value = 0;
									for (int filter_col = 0; filter_col < filter_list[filter_index].cols; filter_col++)
									{
										for (int filter_row = 0; filter_row < filter_list[filter_index].rows; filter_row++)
											filtered_value += positive_feature[i]->feature_map[feature_map_index].at<float>(row_index + filter_row, col_index + filter_col) * filter_list[filter_index].at<float>(filter_row, filter_col);
									}
									tmp.value = filtered_value;

									positive_feature[i]->extracted_feature.push_back(tmp);

								}

							}
						}
					}
				}

				for (int i = 0; i < positive_feature.size(); i++)
				{
					for (int j = 0; j < feature_map_num; j++)
						positive_feature[i]->feature_map[j].release();
				}
			}


			for (int filter_index = 0; filter_index < filter_list.size(); filter_index++)
			{
				for (int feature_map_index = 0; feature_map_index < feature_map_num; feature_map_index++)
				{
					for (int row_index = 0; row_index < (patch_row / cell_size - filter_list[filter_index].rows + 1); row_index++)
					{
						for (int col_index = 0; col_index < (patch_col / cell_size - filter_list[filter_index].cols + 1); col_index++)
						{
							for (int i = negative_initial_size; i < negative_feature.size(); i++)
							{
								sorted_feature tmp;
								tmp.filter_index = filter_index;
								tmp.feature_map_index = feature_map_index;
								tmp.row_index = row_index;
								tmp.col_index = col_index;
								tmp.sid = i;
								tmp.is_pos = false;
								tmp.pos_weight_accul = 0;
								tmp.neg_weight_accul = 0;

								float filtered_value = 0;
								for (int filter_col = 0; filter_col < filter_list[filter_index].cols; filter_col++)
								{
									for (int filter_row = 0; filter_row < filter_list[filter_index].rows; filter_row++)
										filtered_value += negative_feature[i]->feature_map[feature_map_index].at<float>(row_index + filter_row, col_index + filter_col) * filter_list[filter_index].at<float>(filter_row, filter_col);
								}
								tmp.value = filtered_value;

								negative_feature[i]->extracted_feature.push_back(tmp);

							}

						}
					}
				}
			}

			for (int i = negative_initial_size; i < negative_feature.size(); i++)
			{
				for (int j = 0; j < feature_map_num; j++)
					negative_feature[i]->feature_map[j].release();
			}




			for (int tree_number = 0; tree_number < tree_number_list[boost_number]; tree_number++)
			{

				cout << boost_number + 1 << "th boosting " << tree_number + 1 << "th tree is in training..." << endl;

				double training_error;
				double positive_sum = 0;
				double negative_sum = 0;
				double sum;


				//weight normalization
				for (int i = 0; i < positive_feature.size(); i++)
					positive_sum += positive_feature.at(i)->weight;

				for (int i = 0; i < negative_feature.size(); i++)
					negative_sum += negative_feature.at(i)->weight;

				sum = positive_sum + negative_sum;

				for (int i = 0; i < positive_feature.size(); i++)
					positive_feature.at(i)->weight /= sum;


				for (int i = 0; i < negative_feature.size(); i++)
					negative_feature.at(i)->weight /= sum;
				//


				//tree initialize
				tree_ptr = new node();
				tree_ptr->positive_feature = positive_feature;
				tree_ptr->negative_feature = negative_feature;
				tree_ptr->distribution_positive = positive_sum / (positive_sum + negative_sum);
				tree_ptr->distribution_negative = negative_sum / (positive_sum + negative_sum);
				tree_ptr->count_positive = positive_sum / (positive_sum + negative_sum);
				tree_ptr->count_negative = negative_sum / (positive_sum + negative_sum);
				tree_ptr->depth = 0;
				//


				//grow tree
				split_node(tree_ptr, 0, filter_list, positive_feature, negative_feature);
				//


				training_error = get_error(tree_ptr);


				//if error>=0.5, it is useless!
				if (training_error >= 0.5)
				{
					cout << "training error is larger than 0.5! useless!" << endl;
					delete_tree(tree_ptr);
					break;
				}
				//

				tree_head.push_back(tree_ptr);


				//calculate alpha
				tree_ptr->alpha = 0.5*log((1 - training_error) / training_error);
				alpha_sum += tree_ptr->alpha;
				//


				//Are these features classified correctly or not? and weight update
				update_feature(tree_ptr);
				//

				//weight update
				for (int i = 0; i < positive_feature.size(); i++)
				{
					if (positive_feature.at(i)->is_correct == true)
						positive_feature.at(i)->weight *= exp(-1 * tree_ptr->alpha);
					else
						positive_feature.at(i)->weight *= exp(tree_ptr->alpha);
				}


				for (int i = 0; i < negative_feature.size(); i++)
				{
					if (negative_feature.at(i)->is_correct == true)
						negative_feature.at(i)->weight *= exp(-1 * tree_ptr->alpha);
					else
						negative_feature.at(i)->weight *= exp(tree_ptr->alpha);
				}
				//
			}

			if (boost_number + 1 == Nofboost)
				break;


			//Hard negative mining
			cout << "HARD NEGATIVE collecting..." << endl;

			negative_initial_size = negative_feature.size();
			ss << hard_negative_dir;
			folder_list = get_all_folders_names_within_folder(ss.str());
			ss.str("");
			ss.clear();
			bool finished = false;

			while (!finished)
			{

				int folder_index = rand() % (folder_list.size() - (set05_num + 1)) + 2;


				ss << hard_negative_dir << folder_list[folder_index] << "/";
				sub_folder_list = get_all_folders_names_within_folder(ss.str());
				ss.str("");
				ss.clear();
				int sub_folder_index = rand() % (sub_folder_list.size() - 2) + 2;

				ss << hard_negative_dir << folder_list[folder_index] << "/" << sub_folder_list[sub_folder_index] << "/";
				file_list = get_all_files_names_within_folder(ss.str());
				ss.str("");
				ss.clear();
				int file_index = rand_float(0, file_list.size() - 1);

				vector<ground_truth> ground_truth_bb;
				int dummy = 0;
				ground_truth_bb = ground_truth_parsing(folder_list[folder_index], sub_folder_list[sub_folder_index], file_list[file_index], dummy);

				ss << hard_negative_dir << folder_list[folder_index] << "/" << sub_folder_list[sub_folder_index] << "/" << file_list[file_index];
				Mat boost_image = imread(ss.str());
				ss.str("");
				ss.clear();

				int negative_per_image = 0;
				int original_width, original_height;

				original_width = boost_image.cols;
				original_height = boost_image.rows;


				while ((original_width - block_size) % block_stride != 0)
					original_width--;
				while ((original_height - block_size) % block_stride != 0)
					original_height--;

				vector<feature> down_scaled;
				vector<feature> up_scaled;
				feature no_scaled;

				vector<Mat> HOG_channel = HOG_extract(boost_image, false, original_width, original_height);
				vector<Mat>LUV_channel = LUV_extract(boost_image, original_width, original_height);
				Mat gradient_mag = GRADIENT_extract(boost_image, original_width, original_height);

				for (int i = 0; i < bin_size; i++)
					no_scaled.feature_map.push_back(HOG_channel[i]);
				for (int i = 0; i < luv_size; i++)
					no_scaled.feature_map.push_back(LUV_channel[i]);
				no_scaled.feature_map.push_back(gradient_mag);


				for (int down_scale = 1; down_scale <= down_scale_thr; down_scale++)
				{
					int width, height;
					Mat resized_image;
					if (down_scale == scale_per_octave)	//gaussian pyramid
						pyrDown(boost_image, resized_image, Size(boost_image.cols / 2, boost_image.rows / 2));
					else //just resize
						resize(boost_image, resized_image, Size(floor((double)boost_image.cols * pow(pyramid_scale, down_scale)), floor((double)boost_image.rows * pow(pyramid_scale, down_scale))), 0, 0, INTER_LINEAR);

					width = resized_image.cols;
					height = resized_image.rows;
					while ((width - block_size) % block_stride != 0)
						width--;
					while ((height - block_size) % block_stride != 0)
						height--;

					vector<Mat> HOG_channel = HOG_extract(resized_image, false, width, height);
					vector<Mat>LUV_channel = LUV_extract(resized_image, width, height);
					Mat gradient_mag = GRADIENT_extract(resized_image, width, height);
					feature tmp;

					for (int i = 0; i < bin_size; i++)
						tmp.feature_map.push_back(HOG_channel[i]);
					for (int i = 0; i < luv_size; i++)
						tmp.feature_map.push_back(LUV_channel[i]);
					tmp.feature_map.push_back(gradient_mag);

					resized_image.release();
					down_scaled.push_back(tmp);
				}

				for (int up_scale = 1; up_scale <= up_scale_thr; up_scale++)
				{
					int width, height;
					Mat resized_image;
					if (up_scale == scale_per_octave)	//gaussian pyramid
						pyrUp(boost_image, resized_image, Size(boost_image.cols * 2, boost_image.rows * 2));
					else //just resize
						resize(boost_image, resized_image, Size(floor((double)boost_image.cols / pow(pyramid_scale, up_scale)), floor((double)boost_image.rows / pow(pyramid_scale, up_scale))), 0, 0, INTER_LINEAR);

					width = resized_image.cols;
					height = resized_image.rows;
					while ((width - block_size) % block_stride != 0)
						width--;
					while ((height - block_size) % block_stride != 0)
						height--;

					vector<Mat> HOG_channel = HOG_extract(resized_image, false, width, height);
					vector<Mat>LUV_channel = LUV_extract(resized_image, width, height);
					Mat gradient_mag = GRADIENT_extract(resized_image, width, height);
					feature tmp;

					for (int i = 0; i < bin_size; i++)
						tmp.feature_map.push_back(HOG_channel[i]);
					for (int i = 0; i < luv_size; i++)
						tmp.feature_map.push_back(LUV_channel[i]);
					tmp.feature_map.push_back(gradient_mag);

					resized_image.release();
					up_scaled.push_back(tmp);
				}

				while (!finished && negative_per_image < NEGATIVE_PER_IMAGE)
				{
					int scale = rand() % (up_scale_thr + down_scale_thr + 1);
					int down_scale, up_scale;
					int x, y, width, height;
					vector<Mat> scaled_feature;
					scale++;

					if (scale <= down_scale_thr)	//down_scale
					{
						down_scale = scale;
						down_scale--;
						scaled_feature = down_scaled[down_scale].feature_map;
					}
					else if (scale == down_scale_thr + 1)	//no scaling
						scaled_feature = no_scaled.feature_map;
					else //up scaling
					{
						up_scale = scale - down_scale_thr - 1;
						up_scale--;
						scaled_feature = up_scaled[up_scale].feature_map;
					}

					width = scaled_feature[0].cols;
					height = scaled_feature[0].rows;

					x = rand() % (width - patch_col / cell_size + 1);
					y = rand() % (height - patch_row / cell_size + 1);

				
					double pedestrian_confidence = 0;
					for (int index = 0; index < tree_head.size(); index++)
					{

						tree_ptr = tree_head.at(index);
						double alpha = tree_ptr->alpha;

						for (;;)
						{

							if (tree_ptr->left_child == NULL && tree_ptr->right_child == NULL)
								break;

							else
							{

								float filtered_value = 0;
								for (int col = 0; col < filter_list[tree_ptr->filter_index].cols; col++)
								{
									for (int row = 0; row < filter_list[tree_ptr->filter_index].rows; row++)
											filtered_value += scaled_feature[tree_ptr->feature_map_index].at<float>(y + tree_ptr->row_index + row, x + tree_ptr->col_index + col) * filter_list[tree_ptr->filter_index].at<float>(row, col);	
								}

								if (filtered_value <= tree_ptr->tau)
									tree_ptr = tree_ptr->left_child;
								else
									tree_ptr = tree_ptr->right_child;
							}

						}
						if (tree_ptr->distribution_positive > tree_ptr->distribution_negative)
							pedestrian_confidence += alpha;
						else
							pedestrian_confidence -= alpha;
					}


					pedestrian_confidence /= alpha_sum;


					if (pedestrian_confidence > 0.0)
					{
						//make bounding box
						int scaled_y = (int)((double)original_height * (double)(y) / (double)height);
						int scaled_height = (int)((((double)original_height) * (((double)(y)+(double)(patch_row / cell_size)) / (double)height) - (double)scaled_y));
						int scaled_x = (int)(((double)original_width) * (((double)(x + 1)) / (double)width));
						int scaled_width = (int)(0.41*(double)scaled_height);
						Rect bb_tmp(scaled_x, scaled_y, scaled_width, scaled_height);
						//


						//decide bb_tmp is person or not(if it is person, it is not hard-negative)
						bool is_person = false;
						for (vector<ground_truth> ::iterator it = ground_truth_bb.begin(); it != ground_truth_bb.end(); it++)
						{
							if (it->is_valid)
							{

								int width_intersect = min(it->x_max, bb_tmp.width + bb_tmp.x) - max(bb_tmp.x, it->x_min);
								int height_intersect = min(it->y_max, bb_tmp.height + bb_tmp.y) - max(bb_tmp.y, it->y_min);
								int area_intersect = width_intersect * height_intersect;
								int area_union = bb_tmp.height * bb_tmp.width + (it->x_max - it->x_min)*(it->y_max - it->y_min) - area_intersect;
								double overlap_ratio = (double)(area_intersect) / (double)(area_union);

								if (width_intersect > 0 && height_intersect > 0 && overlap_ratio >= hard_negative_thr)
									is_person = true;
							}
							else
							{
								int width_intersect = min(it->x_max, bb_tmp.width + bb_tmp.x) - max(bb_tmp.x, it->x_min);
								int height_intersect = min(it->y_max, bb_tmp.height + bb_tmp.y) - max(bb_tmp.y, it->y_min);
								int area_intersect = width_intersect * height_intersect;
								int area_union = bb_tmp.height * bb_tmp.width + (it->x_max - it->x_min)*(it->y_max - it->y_min) - area_intersect;
								double overlap_ratio = (double)(area_intersect) / (double)(area_union);

								if (width_intersect > 0 && height_intersect > 0 && (overlap_ratio >= hard_negative_thr || (double)area_intersect / (double)(bb_tmp.width * bb_tmp.height) >= hard_negative_thr))
									is_person = true;
							}
						}
						//


						if (!is_person)
						{
							feature *tmp = new feature();

							for (int feature_id = 0; feature_id < feature_map_num; feature_id++)
							{
								tmp->feature_map.push_back(Mat(Size(patch_col / cell_size, patch_row / cell_size), CV_32FC1));

								for (int row = 0; row < patch_row / cell_size; row++)
								{
									for (int col = 0; col < patch_col / cell_size; col++)
										tmp->feature_map[feature_id].at<float>(row, col) = scaled_feature[feature_id].at<float>(y + row, x + col);
								}
							}

							
							negative_feature.push_back(tmp);
							negative_per_image++;

							if (negative_feature.size() >= negative_initial_size + additional_negative)
								finished = true;

							cout << "HARD NEGATIVE COLLECTING in " << boost_number + 1 << "th boosting... " << negative_initial_size << " -> " << negative_feature.size() << endl;
						}
					}
				}
				for (int i = 0; i < down_scale_thr; i++)
				{
					for (int j = 0; j < feature_map_num; j++)
						down_scaled[i].feature_map[j].release();
				}
				for (int i = 0; i < up_scale_thr; i++)
				{
					for (int j = 0; j < feature_map_num; j++)
						up_scaled[i].feature_map[j].release();
				}
				for(int j = 0; j < feature_map_num; j++)
					no_scaled.feature_map[j].release();
				boost_image.release();
			}


			//remove old trees and grow new trees
			for (int i = 0; i < tree_head.size(); i++)
				delete_tree(tree_head.at(i));

			tree_head.clear();
			//
			//
		}

		for (int i = 0; i < positive_feature.size(); i++)
		{
			for (int j = 0; j < feature_map_num; j++)
				positive_feature[i]->feature_map[j].release();

			delete positive_feature.at(i);
		}
		positive_feature.clear();



		for (int i = 0; i < negative_feature.size(); i++)
		{
			for (int j = 0; j < feature_map_num; j++)
				negative_feature[i]->feature_map[j].release();

			delete negative_feature.at(i);
		}
		negative_feature.clear();


		cout << "Training done!!" << endl << endl;
		end = clock();
		training_time = end - begin;
		cout << (double)training_time / CLOCKS_PER_SEC << "seconds for training..." << endl;

		//save trees and filters
		save_tree(tree_head);
		save_filter(filter_list);
		//
		cout << "SAVING complete..." << endl << endl;
		////
	}
	else if (!save_load.compare("LOAD"))
	{
		load_tree(&tree_head);
		load_filter(&filter_list);

		cout << "LOADING complete..." << endl << endl;
	}



	////TESTING start!!
	cout << "Testing start!!" << endl;
	begin = clock();
	int miss_pedestrian[result_unit];//count missed pedestrians
	int false_positive[result_unit];//count false positive bounding box
	int total_pedestrian = 0;//total number of pedestrian
	int total_image = 0;
	int image_index = 0;
	vector<ground_truth> ground_truth_bb[result_unit];//ground truth
	vector<bounding_box> bb[result_unit];//bounding box

	for (int i = 0; i < result_unit; i++)
	{
		miss_pedestrian[i] = 0;
		false_positive[i] = 0;
	}

	ss << result_dir;
	file_list = get_all_files_names_within_folder(ss.str());
	ss.str("");
	ss.clear();

	for (int file_index = 0; file_index < file_list.size(); file_index++)
	{
		ss << result_dir << file_list[file_index];
		remove(ss.str().c_str());
		ss.str("");
		ss.clear();
	}



	ss << test_dir;
	folder_list = get_all_folders_names_within_folder(ss.str());
	ss.str("");
	ss.clear();

	for (int folder_index = set06_num; folder_index < folder_list.size(); folder_index++)
	{
		ss << test_dir << folder_list[folder_index] << "/";
		sub_folder_list = get_all_folders_names_within_folder(ss.str());
		ss.str("");
		ss.clear();

		for (int sub_folder_index = 2; sub_folder_index < sub_folder_list.size(); sub_folder_index++)
		{
			ss << test_dir << folder_list[folder_index] << "/" << sub_folder_list[sub_folder_index] << "/";

			file_list = get_all_files_names_within_folder(ss.str());
			ss.str("");
			ss.clear();

			total_image += file_list.size();

			for (int file_index = 0; file_index < file_list.size(); file_index++)
			{

				cout << image_index + 1 << "th image is in testing..." << endl;



				for (int i = 0; i < result_unit; i++)
				{
					bb[i].clear();
					ground_truth_bb[i].clear();
				}


				//groundtruth parsing
				vector<ground_truth> ground_truth_tmp = ground_truth_parsing(folder_list[folder_index], sub_folder_list[sub_folder_index], file_list[file_index], total_pedestrian);
				for (int gt_index = 0; gt_index < result_unit; gt_index++)
					ground_truth_bb[gt_index] = ground_truth_tmp;
				//


				ss << test_dir << folder_list[folder_index] << "/" << sub_folder_list[sub_folder_index] << "/" << file_list[file_index];

				Mat test_image;
				test_image = imread(ss.str());
				int down_scale = 0;
				int up_scale = 0;
				Mat down_scale_image = test_image.clone();
				Mat down_gaussian_pyramid = test_image.clone();
				Mat up_gaussian_pyramid = test_image.clone();
				Mat up_scale_image = test_image.clone();
				Mat result_image = test_image.clone();
				test_image.release();
				test_image = up_scale_image;
				ss.str("");
				ss.clear();


				int x;
				int y;
				int width;
				int height;
				int scaled_x;
				int scaled_y;
				int scaled_width;
				int scaled_height;
				double pedestrian_confidence = 0;

				int original_width = test_image.cols;
				int original_height = test_image.rows;


				//modify width and height in order to calculate HOG feature
				while ((original_width - block_size) % block_stride != 0)
					original_width--;
				while ((original_height - block_size) % block_stride != 0)
					original_height--;
				//

				//image pyramid
				while (down_scale < down_scale_thr)
				{

					width = test_image.cols;
					height = test_image.rows;


					while ((width - block_size) % block_stride != 0)
						width--;
					while ((height - block_size) % block_stride != 0)
						height--;

					vector<Mat> HOG_channel = HOG_extract(test_image, false, width, height);
					vector<Mat>LUV_channel = LUV_extract(test_image, width, height);
					Mat gradient_mag = GRADIENT_extract(test_image, width, height);

					x = 0;
					y = 0;


					for (;; x++)
					{

						if (x + patch_col / cell_size > width / cell_size)
						{
							x = 0;
							y++;
						}

						if (y + patch_row / cell_size > height / cell_size)
							break;


						//soft-cascade
						double pedestrian_confidence = 0;
						double sub_alpha_sum = 0;

						ASSERT_WITH_MESSAGE(tree_head.size() > soft_cascade, "\n\nASSERTION FAILED!!\ntree_head size must be larger than soft_cascade\n\n");

						for (int index = 0; index < soft_cascade; index++)
						{

							tree_ptr = tree_head.at(index);
							double alpha = tree_ptr->alpha;
							sub_alpha_sum += alpha;


							for (;;)
							{
								if (tree_ptr->left_child == NULL && tree_ptr->right_child == NULL)
									break;

								else
								{
									double filtered_value = 0;
									for (int col = 0; col < filter_list[tree_ptr->filter_index].cols; col++)
									{
										for (int row = 0; row < filter_list[tree_ptr->filter_index].rows; row++)
										{
											if (tree_ptr->feature_map_index < bin_size)
												filtered_value += HOG_channel[tree_ptr->feature_map_index].at<float>(y + tree_ptr->row_index + row, x + tree_ptr->col_index + col) * filter_list[tree_ptr->filter_index].at<float>(row, col);
											else if (tree_ptr->feature_map_index < bin_size + luv_size)
												filtered_value += LUV_channel[tree_ptr->feature_map_index - bin_size].at<float>(y + tree_ptr->row_index + row, x + tree_ptr->col_index + col) * filter_list[tree_ptr->filter_index].at<float>(row, col);
											else
												filtered_value += gradient_mag.at<float>(y + tree_ptr->row_index + row, x + tree_ptr->col_index + col) * filter_list[tree_ptr->filter_index].at<float>(row, col);
										}

									}

									if (filtered_value <= tree_ptr->tau)
										tree_ptr = tree_ptr->left_child;
									else
										tree_ptr = tree_ptr->right_child;
								}

							}
							if (tree_ptr->distribution_positive > tree_ptr->distribution_negative)
								pedestrian_confidence += alpha;
							else
								pedestrian_confidence -= alpha;
						}

						pedestrian_confidence /= sub_alpha_sum;
						//


						int t = soft_cascade;
						double old_alpha_sum;
						double sub_pedestrian_confidence = 0;

						while (pedestrian_confidence > -0.1 && t < tree_head.size())
						{
							old_alpha_sum = sub_alpha_sum;
							tree_ptr = tree_head.at(t);
							double alpha = tree_ptr->alpha;
							sub_alpha_sum += alpha;

							for (;;)
							{

								if (tree_ptr->left_child == NULL && tree_ptr->right_child == NULL)
									break;

								else
								{
									double filtered_value = 0;
									for (int col = 0; col < filter_list[tree_ptr->filter_index].cols; col++)
									{
										for (int row = 0; row < filter_list[tree_ptr->filter_index].rows; row++)
										{
											if (tree_ptr->feature_map_index < bin_size)
												filtered_value += HOG_channel[tree_ptr->feature_map_index].at<float>(y + tree_ptr->row_index + row, x + tree_ptr->col_index + col) * filter_list[tree_ptr->filter_index].at<float>(row, col);
											else if (tree_ptr->feature_map_index < bin_size + luv_size)
												filtered_value += LUV_channel[tree_ptr->feature_map_index - bin_size].at<float>(y + tree_ptr->row_index + row, x + tree_ptr->col_index + col) * filter_list[tree_ptr->filter_index].at<float>(row, col);
											else
												filtered_value += gradient_mag.at<float>(y + tree_ptr->row_index + row, x + tree_ptr->col_index + col) * filter_list[tree_ptr->filter_index].at<float>(row, col);
										}

									}

									if (filtered_value < tree_ptr->tau)
										tree_ptr = tree_ptr->left_child;
									else
										tree_ptr = tree_ptr->right_child;
								}

							}


							if (tree_ptr->distribution_positive > tree_ptr->distribution_negative)
								sub_pedestrian_confidence = alpha;
							else
								sub_pedestrian_confidence = -alpha;

							pedestrian_confidence = (pedestrian_confidence*old_alpha_sum + sub_pedestrian_confidence) / sub_alpha_sum;
							t++;
						}


						//make bounding box
						int bb_num = 0;
						for (double threshold = 0.0; threshold < 0.3; threshold += 0.003)
						{
							if (pedestrian_confidence > -0.01 + threshold)
							{
								scaled_y = (int)((double)original_height) * (((double)(y)) / ((double)height / cell_size));
								scaled_height = (int)((((double)original_height) * (((double)(y)+(double)(patch_row / cell_size)) / ((double)height / cell_size)) - (double)scaled_y));
								scaled_x = (int)(((double)original_width) * (((double)(x + 1)) / ((double)width / cell_size)));
								scaled_width = (int)(0.41*(double)scaled_height);

								bounding_box tmp;
								tmp.rect = Rect(scaled_x, scaled_y, scaled_width, scaled_height);
								tmp.confidence = pedestrian_confidence;
								tmp.is_false_positive = true;
								tmp.is_suppressed = false;
								tmp.is_valid = true;

								bb[bb_num].push_back(tmp);

							}
							bb_num++;
						}
						//
					}
					if (up_scale != up_scale_thr && up_scale != scale_per_octave)
					{
						Mat resized_image;
						resize(test_image, resized_image, Size(floor((double)test_image.cols / pyramid_scale), floor((double)test_image.rows / pyramid_scale)), 0, 0, INTER_LINEAR);
						up_scale_image.release();
						up_scale_image = resized_image;
						test_image = up_scale_image;
						up_scale++;
					}
					else if (up_scale != up_scale_thr && up_scale == scale_per_octave)
					{
						Mat resized_image;
						pyrUp(up_gaussian_pyramid, resized_image, Size(up_gaussian_pyramid.cols * 2, up_gaussian_pyramid.rows * 2));
						up_gaussian_pyramid.release();
						up_gaussian_pyramid = resized_image.clone();
						up_scale_image = resized_image;
						test_image = up_scale_image;
						up_scale++;
					}
					else if (up_scale == up_scale_thr && down_scale != scale_per_octave)
					{
						Mat resized_image;
						resize(down_scale_image, resized_image, Size(floor((double)down_scale_image.cols * pyramid_scale), floor((double)down_scale_image.rows * pyramid_scale)), 0, 0, INTER_LINEAR);
						down_scale_image.release();
						down_scale_image = resized_image;
						test_image = down_scale_image;
						down_scale++;
					}
					else if (up_scale == up_scale_thr && down_scale == scale_per_octave)
					{
						Mat resized_image;
						pyrDown(down_gaussian_pyramid, resized_image, Size(down_gaussian_pyramid.cols / 2, down_gaussian_pyramid.rows / 2));
						down_gaussian_pyramid.release();
						down_gaussian_pyramid = resized_image.clone();
						down_scale_image = resized_image;
						test_image = down_scale_image;
						down_scale++;
					}

					for (int bin = 0; bin < bin_size; bin++)
						HOG_channel[bin].release();

					for (int luv_index = 0; luv_index < luv_size; luv_index++)
						LUV_channel[luv_index].release();

					gradient_mag.release();

				}
				//
				////


				up_scale_image.release();
				up_gaussian_pyramid.release();
				down_scale_image.release();
				down_gaussian_pyramid.release();

				//Non-Maximum-Suppreesion
				vector<bounding_box> suppressed_bounding_box[result_unit];
				for (int i = 0; i < result_unit; i++)
				{
					int original_size = -1;
					int suppressed_size = 1;


					while (bb[i].size()>1 && suppressed_size != original_size)
					{
						while (1)
						{
							suppressed_size = 0;
							vector<bounding_box>::iterator loc;
							double maxValue = -1;


							for (vector<bounding_box>::iterator it = bb[i].begin(); it != bb[i].end(); it++)
							{
								if (it->confidence > maxValue && it->is_suppressed == false)
								{
									maxValue = it->confidence;
									loc = it;
								}

								if (it->is_suppressed == false)
									suppressed_size++;
							}

							if (suppressed_size == 0)
								break;

							bounding_box NMS_bb;
							NMS_bb.confidence = loc->confidence;
							NMS_bb.is_false_positive = true;
							NMS_bb.is_suppressed = false;
							NMS_bb.is_valid = true;
							NMS_bb.rect = loc->rect;

							NMS(&bb[i], *loc);

							suppressed_bounding_box[i].push_back(NMS_bb);

						}


						original_size = bb[i].size();
						bb[i] = suppressed_bounding_box[i];
						suppressed_bounding_box[i].clear();
						suppressed_size = bb[i].size();

					}
				}
				//





				//decide which pedestrian is detected and which bounding box is false positive
				for (int bb_num = 0; bb_num < result_unit; bb_num++)
				{
					for (vector<ground_truth> ::iterator it = ground_truth_bb[bb_num].begin(); it != ground_truth_bb[bb_num].end(); it++)
					{
						if (it->is_valid)
						{
							for (vector<bounding_box>::iterator it2 = bb[bb_num].begin(); it2 != bb[bb_num].end(); it2++)
							{
								int width_intersect = min(it->x_max, it2->rect.width + it2->rect.x) - max(it2->rect.x, it->x_min);
								int height_intersect = min(it->y_max, it2->rect.height + it2->rect.y) - max(it2->rect.y, it->y_min);
								int area_intersect = width_intersect * height_intersect;
								int area_union = it2->rect.height *it2->rect.width + (it->x_max - it->x_min)*(it->y_max - it->y_min) - area_intersect;
								double overlap_ratio = (double)(area_intersect) / (double)(area_union);

								if (width_intersect > 0 && height_intersect > 0 && overlap_ratio >= IoUthr)
								{
									it->is_detected = true;
									it2->is_false_positive = false;
									it2->is_valid = true;
								}

							}
						}
						else
						{
							for (vector<bounding_box>::iterator it2 = bb[bb_num].begin(); it2 != bb[bb_num].end(); it2++)
							{
								int width_intersect = min(it->x_max, it2->rect.width + it2->rect.x) - max(it2->rect.x, it->x_min);
								int height_intersect = min(it->y_max, it2->rect.height + it2->rect.y) - max(it2->rect.y, it->y_min);
								int area_intersect = width_intersect * height_intersect;
								int area_union = it2->rect.height *it2->rect.width + (it->x_max - it->x_min)*(it->y_max - it->y_min) - area_intersect;
								double overlap_ratio = (double)(area_intersect) / (double)(area_union);

								if (it2->is_false_positive && width_intersect > 0 && height_intersect > 0 && (overlap_ratio >= IoUthr || (double)area_intersect / (double)(it2->rect.width * it2->rect.height) >= 0))
									it2->is_valid = false;

							}
						}


					}
				}
				//


				for (int index = 0; index < result_unit; index++)
				{
					for (vector<bounding_box>::iterator it = bb[index].begin(); it != bb[index].end(); it++)
					{

						if (it->is_valid && it->is_false_positive == true)
							false_positive[index]++;
					}

					for (vector<ground_truth>::iterator it2 = ground_truth_bb[index].begin(); it2 != ground_truth_bb[index].end(); it2++)
					{
						if (it2->is_valid && it2->is_detected == false)
							miss_pedestrian[index]++;
					}

				}

				ss << result_dir << image_index + 1 << ".jpg";
				image_index++;

				//draw bounding box and write image. RED is false positive, BLUE is true positive, GREEN is groundtruth
				for (vector<bounding_box>::iterator it = bb[0].begin(); it != bb[0].end(); it++)
				{
					if (it->is_valid && it->is_false_positive == true)
						rectangle(result_image, it->rect, CV_RGB(255, 0, 0), 2);
					else if (it->is_valid && it->is_false_positive == false)
						rectangle(result_image, it->rect, CV_RGB(0, 0, 255), 2);
				}

				for (vector<ground_truth>::iterator it = ground_truth_bb[0].begin(); it != ground_truth_bb[0].end(); it++)
				{
					if (it->is_valid)
						rectangle(result_image, Rect(it->x_min, it->y_min, it->x_max - it->x_min, it->y_max - it->y_min), CV_RGB(0, 255, 0), 2);
				}

				imwrite(ss.str(), result_image);
				result_image.release();
				ss.str("");
				ss.clear();

			}

		}
	}



	cout << "Testing complete!!" << endl;
	end = clock();
	testing_time = end - begin;

	cout << (double)training_time / CLOCKS_PER_SEC << "seconds for training..." << endl;
	cout << (double)testing_time / CLOCKS_PER_SEC << "seconds for " << total_image << " test images" << endl << endl;


	ofstream of;
	ss << result_dir << "result.txt";
	of.open(ss.str());
	ss.str("");
	ss.clear();


	for (int index = 0; index < result_unit; index++)
		of << (double)miss_pedestrian[index] / (double)total_pedestrian << " " << (double)false_positive[index] / (double)total_image << endl;


	of.close();
	////
	return 0;
}