#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <direct.h>
#include <Windows.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace std;
using namespace cv;

#define partial_occ 0.35
#define patch_row 120
#define patch_col 60
#define crop_per_neg 12
#define iteration_max 30
#define set05_num 7
#define reasonable_height 50
#define image_dir "D:\\Study/Data/Caltech_Pedestrian/data-USA/images/"
#define negative_image_dir "D:\\Study/Data/Caltech_Pedestrian/data-USA/training_images/neg/"
#define positive_image_dir "D:\\Study/Data/Caltech_Pedestrian/data-USA/training_images/pos/"
#define annotation_dir "D:\\Study/Data/Caltech_Pedestrian/data-USA/annotations/"
#define test_dir "D:\\Study/Data/Caltech_Pedestrian/data-USA/test_images/"

vector<string> get_all_files_names_within_folder(string folder)
{
	vector<string> names;
	char search_path[1000];
	sprintf_s(search_path, 1000, "%s*.*", folder.c_str());
	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(search_path, &fd);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			// read all (real) files in current folder
			// , delete '!' read other 2 default folder . and ..
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				names.push_back(fd.cFileName);
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	return names;
}

vector<string> get_all_folders_names_within_folder(string folder)
{
	vector<string> names;
	char search_path[1000];
	sprintf_s(search_path, 1000, "%s*.*", folder.c_str());
	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(search_path, &fd);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			// read all (real) files in current folder
			// , delete '!' read other 2 default folder . and ..
			if ((fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				names.push_back(fd.cFileName);
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	return names;
}

int main()
{
	ifstream inf;
	stringstream ss;
	vector<string> file_list;
	vector<string> folder_list;
	vector<string> sub_folder_list;
	int count = 0;
	string input;

	cout << "POSITIVE or NEGATIVE ";
	cin >> input;

	if (input.compare("POSITIVE") && input.compare("NEGATIVE"))
	{
		cout << "Not a proper input" << endl;
		return 0;
	}

	cout << endl << input << " image cropping is in processing..." << endl;


	if (!input.compare("POSITIVE"))
	{
		ss << positive_image_dir;
		file_list = get_all_files_names_within_folder(ss.str());
		ss.str("");
		ss.clear();

		for (int i = 0; i < file_list.size(); i++)
		{
			ss << positive_image_dir << file_list[i];
			remove(ss.str().c_str());
			ss.str("");
			ss.clear();
		}

		file_list.clear();
		ss.str("");
		ss.clear();
	}
	else
	{
		ss << negative_image_dir;
		file_list = get_all_files_names_within_folder(ss.str());
		ss.str("");
		ss.clear();

		for (int i = 0; i < file_list.size(); i++)
		{
			ss << negative_image_dir << file_list[i];
			remove(ss.str().c_str());
			ss.str("");
			ss.clear();
		}

		file_list.clear();
		ss.str("");
		ss.clear();
	}




	ss << annotation_dir;
	folder_list = get_all_folders_names_within_folder(ss.str());
	ss.str("");
	ss.clear();

	for (int i = 2; i < set05_num; i++)
	{

		ss << annotation_dir << folder_list.at(i) << "/";

		sub_folder_list = get_all_folders_names_within_folder(ss.str());

		ss.str("");
		ss.clear();

		for (int j = 2; j < sub_folder_list.size(); j++)
		{
			ss << annotation_dir << folder_list.at(i) << "/" << sub_folder_list.at(j) << "/";

			file_list = get_all_files_names_within_folder(ss.str());

			ss.str("");
			ss.clear();

			for (int k = 0; k < file_list.size(); k++)
			{
				ss << annotation_dir << folder_list.at(i) << "/" << sub_folder_list.at(j) << "/" << file_list.at(k);

				inf.open(ss.str());

				ss.str("");
				ss.clear();

				string line;
				vector<string> line_save;
				while (getline(inf, line))
					line_save.push_back(line);

				//negative
				if (!input.compare("NEGATIVE"))
				{

					string id;
					vector<int> left_vec;
					vector<int> top_vec;
					vector<int> width_vec;
					vector<int> height_vec;
					int left, top, width, height;

					for (int line_number = 1; line_number < line_save.size(); line_number++)
					{
						ss << line_save.at(line_number);
						ss >> id >> left >> top >> width >> height;
						ss.str("");
						ss.clear();

						left_vec.push_back(left);
						top_vec.push_back(top);
						width_vec.push_back(width);
						height_vec.push_back(height);

					}

					string file_name = file_list.at(k);
					string from = ".txt";
					string to = ".jpg";
					size_t start_pos = file_name.find(from);
					file_name.replace(start_pos, from.length(), to);

					ss << image_dir << folder_list.at(i) << "/" << sub_folder_list.at(j) << "/" << file_name;

					Mat image;
					image = imread(ss.str());

					ss.str("");
					ss.clear();
					bool proper_index = true;
					int iteration = 0;

					for (int negative_number = 0; negative_number < crop_per_neg; negative_number++)
					{
						int col_index = rand() % (image.cols - patch_col);
						int row_index = rand() % (image.rows - patch_row);

						int left_col;
						int right_col;
						int top_row;
						int bottom_row;

						proper_index = true;


						for (int line_number = 0; line_number < line_save.size() - 1; line_number++)
						{
							//if overlapped area is larger tha zero, flag = false and break
							if (col_index < left_vec.at(line_number))
								left_col = left_vec.at(line_number);
							else
								left_col = col_index;


							if (col_index + patch_col < left_vec.at(line_number) + width_vec.at(line_number))
								right_col = col_index + patch_col;
							else
								right_col = left_vec.at(line_number) + width_vec.at(line_number);


							if (row_index < top_vec.at(line_number))
								top_row = top_vec.at(line_number);
							else
								top_row = row_index;


							if (row_index + patch_row < top_vec.at(line_number) + height_vec.at(line_number))
								bottom_row = row_index + patch_row;
							else
								bottom_row = top_vec.at(line_number) + height_vec.at(line_number);

							if ((right_col - left_col) * (bottom_row - top_row) > 0)
							{
								iteration++;
								proper_index = false;
								break;
							}
							//
						}

						if (!proper_index && iteration < iteration_max)
						{
							negative_number--;
							continue;
						}

						Mat crop;
						crop = image(Rect(col_index, row_index, patch_col, patch_row));

						ss << negative_image_dir << count << ".jpg";
						count++;

						imwrite(ss.str(), crop);

						ss.str("");
						ss.clear();
						crop.release();
					}

					image.release();

				}
				//

				//positive
				else if (!input.compare("POSITIVE"))
				{
					for (int line_number = 1; line_number < line_save.size(); line_number++)
					{
						string id;
						int left, top, width, height;
						int v_left, v_top, v_width, v_height;
						int occ, ign, ang;
						ss << line_save.at(line_number);
						ss >> id >> left >> top >> width >> height >> occ >> v_left >> v_top >> v_width >> v_height >> ign >> ang;
						ss.str("");
						ss.clear();

						if (!id.compare("person") && !ign && height >= reasonable_height)
						{
							bool is_valid = false;

							if (occ)
							{
								int width_intersect = min(left + width, v_left + v_width) - max(left, v_left);
								int height_intersect = min(top + height, v_top + v_height) - max(height, v_height);
								int area_intersect = width_intersect * height_intersect;
								int area_union = width * height + v_width * v_height - area_intersect;
								double overlap_ratio = (double)(area_intersect) / (double)(area_union);
								overlap_ratio = 1 - overlap_ratio;
								if (width_intersect > 0 && height_intersect > 0 && overlap_ratio <= partial_occ)
									is_valid = true;
							}

							else
								is_valid = true;

							if (is_valid)
							{
								string file_name = file_list.at(k);
								string from = ".txt";
								string to = ".jpg";
								size_t start_pos = file_name.find(from);
								file_name.replace(start_pos, from.length(), to);

								ss << image_dir << folder_list.at(i) << "/" << sub_folder_list.at(j) << "/" << file_name;


								Mat image;
								image = imread(ss.str());

								ss.str("");
								ss.clear();

								if (left < 0)
									left = 0;
								if (top < 0)
									top = 0;
								if (width < 0)
									width = 0;
								if (height < 0)
									height = 0;

								if (left + width > image.cols)
									width = image.cols - left;

								if (top + height > image.rows)
									height = image.rows - top;

								if (width > 0 && height > 0)
								{
									Mat crop;
									crop = image(Rect(left, top, width, height));

									Mat resized;
									resize(crop, resized, Size(patch_col, patch_row), 0.0, 0.0, INTER_LINEAR);


									ss << positive_image_dir << count << ".jpg";
									count++;

									imwrite(ss.str(), resized);
									ss.str("");
									ss.clear();

									Mat flipped;
									flip(resized, flipped, 1);
									ss << positive_image_dir << count << ".jpg";
									count++;
									imwrite(ss.str(), flipped);

									ss.str("");
									ss.clear();
									crop.release();
									resized.release();
									flipped.release();
								}

								image.release();
							}


						}
					}

				}
				//

				line_save.clear();
				inf.close();


			}


		}

	}



	cout << "Done!" << endl;
	return 0;
}