---
TODO--> replace below path with the actual path of the Demo folder in your computer.
typora-root-url: C:\UserData\data_formatting_demo\Data Formatting Demo\
---

# **Data Formatting Demo**

## Version 1.0

**Author: Shuiliang (Leon) Wu**

## Table of Contents

[toc]

# Introduction

This demo provides a guide to format provided `final_cleaned_diabetes.csv` raw data (cleaned) to meet `Requirements of Data Format` so that date can be recognized by the source code for federated learning. 



# What You Have Before Started

* Data Formatting Demo (this one)
* Source code in `data_formatting_demo` including four files: 
  * `a_encode_data.py`
  * `b_generate_input.py`
  * `c_split_csv.py`
  * `data_prepare_config.json`
* Raw dataset `final_cleaned_diabetes.csv` (in `raw_data` folder)



# Getting Started

This Demo is demonstrated using Windows 11 (operations in macOS are very similar).

1. Open the `source code` folder in `data_formatting_demo`. 

   ![data_1](/data_1.png)

2. Right-click `data_prepare_config.json`, select `Edit in Notepad` then update the parameters accordingly. 

   ![data_3](/data_3.png)

3. Right-click at the blank space in `source code` folder, select `Open in Terminal`. The terminal shall be pop-up as below:

   ![data_2](/data_2.png)

4. Formatting the data by running below in terminal one-by-one:

   ```bash
   python ./a_encode_data.py
   python ./b_generate_input.py
   ```

![data_4](/data_4.png)

5. There shall be `site_name.csv` and `site_name_headers.csv` generated in `input` folder under `dataset` folder.  Replace `site_name` with your actual site name, and they are ready to be used. 

   ![data_5](/data_5.png)

6. Since there will be four sites in total in the demo of `NVIDIA FLARE User Guide for Project Manager` and `NVIDIA FLARE User Guide for Site Admin`, the formatted dataset is split into four datasets with equal amount of data in each datasets by running:

   ```
   python ./c_split_csv.py
   ```

   ![data_6](/data_6.png)

7. There shall be four datasets generated in `output` folder under `dataset` folder. Replace `site#` with the actual site name, and they are ready to be used for demo.

   ![data_7](/data_7.png)