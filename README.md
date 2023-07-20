# Multifunction
This repository is just like a container and there are a lot of python projects which achieve a variety of funtions
* The python file "Ground-air_fire_damage_probability_curve_drawing.py" is a little window program which can help us draw the probability function curve between distance and height,just like the picture below.
  ![image](https://github.com/gh31415/Multifunction/assets/94460269/61209e5d-648d-4e6e-ad1c-8a4db245cace)
* The python file "change color.py" uses k-means algorithm to change the main color of one picture to whatever color you like. There is a for-loop in the code so you can do image batch processing by changing the picture path.
* The python file "cosin.py" shows how to calculate cosin similarity of two different image datasets.
* The python files "cosin_hist_0-255.py","cosin_hist_-1-1_gate.py","cosin_hist_-1-1_scene.py" and "cosin_hist_itself_scene.py" show how to draw subplot.
* The python file "dataset_hist.py" shows how to draw the frequency distribution histogram of a huge image dataset(such as 10k pictures).
  ![image](https://github.com/gh31415/Multifunction/assets/94460269/b192d4fb-d889-41f6-a597-3a7526751d51)
  ![image](https://github.com/gh31415/Multifunction/assets/94460269/2a654125-bfa0-4f5e-881b-45a00a36eeb1)
* The python files "get picture_set pose.py" and "get pictures with poses csv.py" show how to load csv file of drone's poses and generate pictures.The second py file uses 5 cameras.
* The python file "MMD calculation.py" shows how to calculate the MMD(Maximum Mean Discrepancy) distance of two pictures.
* The python file "random pictures get and save pose.py","random pictures get and save pose_relative coordinate.py","random pictures get and save pose_xyz.py" and "random pictures get and save pose_xyzraw.py" are all used to get pictures with airsim API in a paticular scene.You can You can compare these four files and find the differences between them.
* The python file "rename picture.py" can batch the name of numerous pictures. There are two different ways and you can see in the commented out code.
* The python file "resize picture.py" and "resize picure 120.py" can all batch resize pictures.
* The python file "rotate picture.py" can batch rotate images 10 degrees.
* The python file "txt_to_csv.py" can change a .txt file to .csv file(generate a new one).
* The python file "Wasserstein distance.py" can calculate the EMD distance(Wasserstein distance) between two images or distributions.
* The python file "图片名分割.py" can use specific symbol to batch split image names and store them in a csv file.
* The python file "保留相同列.py" can keep identical column of two csv file.
 
