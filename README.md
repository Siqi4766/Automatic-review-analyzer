# Automatic review analyzer
This project designs a linear classifier to use for sentiment analysis of product reviews, trained by reviews written by Amazon customers for food products. This is part of the [MITx 6.86x ML course](https://www.edx.org/learn/machine-learning/massachusetts-institute-of-technology-machine-learning-with-python-from-linear-models-to-deep-learning).

## Pipeline
- Implement and compare three types of linear classifiers: the perceptron algorithm, the average perceptron algorithm, and the [Pegasos (Primal Estimated sub-GrAdient SOlver for SVM) algorithm](https://courses.edx.org/assets/courseware/v1/16f13f7ac37ae86ebe0372f2410bcec4/asset-v1:MITx+6.86x+2T2026+type@asset+block/resources_pegasos.pdf).
- Apply the classifiers on the food dataset.
- Experiment with additional features and explore their impact on classifier performance.

## Folder
The labels for the reviews are 1 or -1 (representing positive or negative).
- `project1.py`: functions that design the learning algorithms
- `main.py`: where functions are called and run experiments
- `utils.py`: helper functions provided
- `test.py`: testing/debugging the functions implemented
- `reviews_train.tsv`: training dataset used to train the model
- `reviews_val.tsv`: validation dataset used to tune/check model performance while choosing parameters
- `reviews_test.tsv`: test dataset used to evaluate the final model
- `reviews_submit.tsv`: data for generating predictions to submit
- `toy_data.tsv`: 2D toy dataset used to visualize and test the algorithms
- `200.txt`, `4000.txt`: contain fixed shuffled orders of data indices. They are used to make perceptron and Pegasos training reproducible
- `stopwords.txt`: common words

## Findings & Analysis

### Convergence Testing

I tested convergence by running the three learning algorithms with larger values of `T`, the number of training iterations. If an algorithm converges, its learned parameters `theta` and `theta_0` should become more stable as `T` increases.

| Algorithm | T = 10 | T = 500 | T = 1000 | Converged? |
|---|---|---|---|---|
| Perceptron | theta = [3.9174, 4.1640], theta_0 = -8.0 | theta = [3.3189, 5.5648], theta_0 = -8.0 | theta = [2.5128, 4.1252], theta_0 = -7.0 | No |
| Average Perceptron | theta = [3.4783, 3.6111], theta_0 = -6.373 | theta = [3.8688, 3.8903], theta_0 = -7.0581 | theta = [3.8761, 3.9073], theta_0 = -7.0819 | Yes |
| Pegasos | theta = [0.7346, 0.6300], theta_0 = -1.2195 | theta = [0.6708, 0.5850], theta_0 = -1.2309 | theta = [0.6614, 0.5950], theta_0 = -1.2311 | Yes |

The regular perceptron did not converge on this dataset. Its parameters continued to change noticeably between larger values of `T`. 

The average perceptron and Pegasos showed more stable behavior. Their final `theta`s and `theta_0`s changed much less as `T` increased.

One possible explanation is that perceptron algorithm won't converge if the data is not linearly separable, but the average perceptron algorithm converges because its averaging the results repeatly. 

The Pegasos algorithm converges because it performs SGD on the convex SVM objective function, which has an global optimum (minimum). Its solving the SVM optimization problem by minimizing the objective function: maximizing the margin of boundaries while making sure that the hinge loss is small (correct classification). Its learning rate also decreases over time to make sure the update/step size becomes smaller and more stable.

### Stopwords & Feature representation
Stopwords are very common words such as "the", "is", "and", and "to". These words appear frequently in many reviews but often do not carry strong sentiment by themselves. Removing stopwords can reduce noise and make the model focus more on words that may be more meaningful for sentiment classification, such as "great", "bad", "boring", or "excellent".

Feature representation controls how each review is converted into a numeric vector.

With **binary features**, each word is represented by whether it appears in the review or not. If a word appears at least once, its value is 1; if it does not appear, its value is 0. For example, if the dictionary contains "good" and "really", the review "really really good" would become [1, 1]. Binary features ignore repeated words.

With **count features**, each word is represented by how many times it appears in the review. Using the same example, "really really good" would become [1, 2] if the dictionary order is [good, really]. Count features keep frequency information, so repeated words have more influence on the model.

### Tuning hyperparameters
Conditions: used binary features, kept stopwords.

Used the validation dataset to tune the hyperparameters. Experimented with `T`: [1, 5, 10, 15, 25, 50] and `L`: [0.001, 0.01, 0.1, 1, 10]. For pegasos algorithm, first fixed `L` = 0.01 to tune `T`, then use the best `T` to tune `L`.

| Algorithm | Hyperparameter tested | Best value | Best validation accuracy |
|---|---:|---:|---:|
| Perceptron | `T` | 25 | 0.7940 |
| Average Perceptron | `T` | 25 | 0.8000 |
| Pegasos | `T` | 25 | 0.8060 |
| Pegasos | `L` | 0.01 | 0.8060 |

We can conclude that the **Pegasos algorithm** performed the best with hyperparameters `T` = 25 and `L` = 0.01, resulted in validation accuracy of 0.8060. This setting was used to evaluate the test dataset.

### Accuracy on the test dataset
- Test accuracy for Pegasos algorithm with stopwords kept, used binary features: 0.8020
    - Most explanatory positive word features:
['delicious', 'great', '!', 'best', 'perfect', 'loves', 'wonderful', 'glad', 'love', 'quickly']
    - Most explanatory negative word features:
['disappointed', 'bad', 'not', 'however', 'but', 'unfortunately', 'awful', 'money', 'ok', '$']

- Test accuracy for Pegasos algorithm with stopwords removed, used binary features: 0.8080
    - Most Explanatory Positive Word Features: ['delicious', 'great', 'loves', '!', 'best', 'perfect', 'excellent', 'wonderful', 'favorite', 'tasty']
    - Most Explanatory Negative Word Features: ['disappointed', 'bad', 'however', 'ok', 'disappointment', 'unfortunately', '?', 'thought', 'stale', 'money']

- Test accuracy for Pegasos algorithm with stopwords removed, used count features: 0.7700
    - Most Explanatory Positive Word Features: ['delicious', 'loves', 'great', 'perfect', 'best', 'favorite', 'wonderful', 'excellent', 'tasty', 'thank']
    - Most Explanatory Negative Word Features: ['disappointed', 'bad', 'however', 'money', 'ok', 'unfortunately', 'stale', 'fine', 'disappointment', 'awful']

- Test accuracy for Pegasos algorithm with stopwords kept, used count features: 0.7880
    - Most Explanatory Positive Word Features: ['delicious', 'great', 'perfect', 'loves', 'best', 'find', 'excellent', 'favorite', 'without', 'quickly']
    - Most Explanatory Negative Word Features: ['disappointed', 'bad', 'not', 'however', 'money', 'unfortunately', 'chai', 'waste', 'worst', 'rather']

Overall, by using the Pegasos algorithm with `T` = 25, `L` = 0.01, removing the stopwords and using the binary feature representation, the model performed the best on the test dataset with accuracy of 0.8080.