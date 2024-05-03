# pervasive
During the process of rehabilitation for individuals with leg related injuries, the journey to full recovery involves not only the medical procedures performed but also the physiotherapeutic activities such as RICE (Rest, Ice, Compression, Elevation) to guarantee full recovery. Patients undergoing physiotherapy often face challenges in tracking
their progress especially in terms of restored mobility and walking capabilities, hence we have developed this flask application using python which will receive the user's walking data as a hhtp post request in a json format and then return the recovery rate as response.

### Goals.
To this end, we aim to proffer a solution by creating an application that will track the recovery rate of patients who uses the application for (walking activity).

### Learning Goals.
Our learning goals and objectives involves understanding machine learning techniques and using these techniques to retrieve the walking features of patients, in order to ascertain
their recovery rate.

## Dataset
The dataset used here is gotten from the [MotionSense dataset](https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset) which contains time-series data from accelerometer and gyroscope sensors (attitude, gravity, userAcceleration, and rotation rate). It is gathered using an iPhone6s placed in the participant’s front pocket and with the aid of SensingKit which takes data from the Core Motion framework on the iOS device. A total of 24 volunteers of various genders, ages, weights, and heights completed six activities in 15 trials in the same
setting and conditions: downstairs, upstairs, walking, jogging, sitting, and standing. Saved as a csv file, the DeviceMotion data folder contains time-series collected by both Accelerometer and Gyroscope for all 15 trials. For every trial we have a multivariate timeseries. Thus, we have time-series with 12 features:
attitude.roll, attitude.pitch, attitude.yaw, gravity.x, gravity.y, gravity.z, rotationRate.x, rotationRate.y, rotationRate.z, userAcceleration.x, userAcceleration.y, userAcceleration.z

## METHODOLOGY
A. Preprocessing of the Dataset.
First, the dataset was downloaded and using panda data frame it was cleaned and the empty rows were removed. We selected only the walking samples from the dataset given the scope of our project. We also calculated the magnitude of the following features user acceleration, rotation and attitude in the x, y & z axis.

B. Normalizing the Dataset.
Using the MinMaxScaler method from the sklearn library, we normalized the values of the data frame to a scale of -1 to 1. This was necessary to make it easier to process the large data frame.

C. Cosine Similarity.
The cosine similarity formula is used here to calculate the cosine of the angle between two vectors in a multidimensional space. It's commonly used in various fields, including information retrieval, natural language processing, and recommendation systems. The formula for cosine similarity between two vectors A and B is:

\[
\text{Cosine Similarity}(A, B) = \frac{{A \cdot B}}{{\|A\| \|B\|}} = \frac{{\sum_{i=1}^{n} A_i \times B_i}}{{\sqrt{\sum_{i=1}^{n} (A_i)^2} \times \sqrt{\sum_{i=1}^{n} (B_i)^2}}}
\]

A.B: This represents the dot product of vectors A and B. The dot product is calculated by multiplying corresponding elements of the two vectors and then summing up the results. In other words, it's the sum of the products of the individual components of the vectors. |A| and |B|: These represent the Euclidean norms (also known as magnitudes or lengths) of vectors A and B, respectively. The Euclidean norm of a vector is calculated as the square root of the sum of the squares of its components. The cosine similarity formula yields a value between -1 and
1, where:
• If the cosine similarity is 1, it means the vectors are perfectly aligned (i.e., they point in the same direction).
• If the cosine similarity is 0, it means the vectors are orthogonal (i.e., they are perpendicular to each other).
• If the cosine similarity is -1, it means the vectors are perfectly anti-aligned (i.e., they point in opposite
directions).
In our project, we only focus on values between 0 to 1 as we are searching for the similarities and not the opposite. Hence negative values are considered as zero in our project.
#### NB: This comparison is needed to compare similarity features between the dataset gotten from motionsense (healthy data) and that gotten from the users through the flask end point "/analyze-data" (recovering data) and the similarity score then shows how similar the recovering patients' data is to the motionsense dataset, hence resulting in the recovery score or rate in percentage.

## Chalenges
The major challenge with this approach is, we seem to sometimes get unexpected similarity values when we run the similarity. Especially when we expect the similarity score to be very high (due to the fact that the data is also coming from a healthy person), we get just above 70% similarity. We suspect this behaviour is as a result of the sensors used. In practice, unexpected sensor behaviours can affect the output or outcome of any project.
