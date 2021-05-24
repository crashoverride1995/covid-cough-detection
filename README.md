# COVID-19 Cough Detection using Real-Time Cough Recordings

## Dataset
[COUGHVID Dataset](https://zenodo.org/record/4048312#.YKv3T6hKjIl)

## Implementation
 * The original dataset is converted to wav format using convert_files.py.
 * The converted files are classified using cough_classify and only those that have cough are used to train the COVID Detection Model. (cough_classify/cough_classify.py)
 * The cough files are then segmented to remove extraneous audio parts using segmentation.py and then each segmented file is concatenated to form a corresponding single wav file. (data_prep.py)
 * The segmented and concatenated wav files are used for MFCC feature extraction. (data_prep.py)
 * The extracted features are then used to train the ANN model. (ANN.ipynb)
 * Model TFlite is generated after training and then added to the Android App to predict on real-time cough recordings. (CovidCough.zip) 
