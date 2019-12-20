# project-bbecker-hbeilinson

Hannah, 12/4/19, 2ish hours:
- Cloned the repo locally and added the dataset into the folder.
- Put the dataset into the gitignore since it's so large.
- Set up main.py and imported SKlearn RandomForests.
- Chose to partition dataset into 80% training data and 20% testing data. Wrote a function to make this partition.


Brian, 12/4/19, Also 2ish hours:
- Cloned the repo
- Pushed the proposal pdf
- Wrote util.py, with function read_csv, which converts data to a label vector and examples matrix

Hannah and Brian, in the lab, advised by Sara, 12/5/19:
- DONE: Shuffle data, but use seed so it shuffles the same way each time.
- Can use FC in tensorflow and Random Forest in sklearn, but maybe can check if both exists in one library.
- If accuracy is low, try binning to have fewer labels.
- Could try RandomForestRegressor, might be weird.

Hannah, 12/12/19, in lab:
- Shuffled order of reading in lines in util.py so that songs are not organized by playlist when partitioning train and test data. Used seed = 42 so that it will shuffle the same way each time. Chose to shuffle in util rather than when partitioning train and test data so that the X and y can stay together.
- Tested training and testing of RandomForest in main.

Brian, 12/12/19, in lab:
- Wrote train/test Random Forest functions in main.py using sklearn.ensemble Random Forest models.
- These functions take a boolean parameter called `regressor` which determines whether RFClassifier or RFRegressor is used.
- Wrote accuracy and MSE functions in main.py

Hannah, 12/15/19, many hours:
- Added confusion matrix code. Tried to use print_confusion_matrix(), but had the wrong version of sklearn. Eventually decided to use seaborn for heatmap visualization.
- Added code for binning in util
- Tested random forest. Right now accuracy is very low. Regardless of how much we bin, accuracy is usually about twice as good as random guessing (without binning accuracy is around 2%, with bin_step=25 accuracy is around 40%). Tried increasing T to 1000, but accuracy did not really increase.
- Worked on trying to get tensorflow to work to see if an FC neural network will perform better.

Brian 12/15/19:
- Added feature normalization in util.py. Makes each feature out of 1 and mean centers at 0.
- Added files fc_nn.py and run_nn_tf.py to implement fully connected neural nets using tensorflow. Unfortunately, we ran into a strange error during training so decided to switch to sklearn.

Hannah, 12/16/19:
- Tried again to improve random forest through changing hyperparameters. Changed max_depth to 2, but this caused it to almost always predict the same popularity score (roughly 54). Then changed min_samples_leaf to 10, but saw very few changes in results. I think that this happens because most of the examples are clustered in a few popularity scores.
- Added sklearn FC architecture. Initial results have similar accuracy to random forest (0.020435967302452316), but it didn't converge. Increased max_iter to 1000, but accuracy hardly improved.
- Observation: It's kind of interesting that accuracy is almost exactly twice as good as random regardless of binning or model. Says something inherent about dataset maybe?
- Above is probably because it can eliminate all the low scores and is otherwise random.
- Plotted accuracy and proportional accuracy vs. different binning size, and across both models.
- Made plot of correlations (after Brian calculated correlations)

Brian, 12/16/19:
- Computed correlation between label and features in `testCor` found in `main.py`. Found that it is very near 0. Holy mackerel. We think this may relate to our low accuracies.
- Wrote `Perturber` class. Takes a vector of feature importances and perturbs the most important ones in test data by some specified amount. We hope to see how this will actually affect classification. I.e, how important these features really are to a song's popularity.

To do list, 12/15:
- Run accuracies/MSEs on random forests without binning
- Run accuracies/MSEs on FC without binning
- Set up binning
- Rerun the above with binning
- Write a feature perturber
- Run tests to collect mean errors post perturbation

Visualizations we want for presentation:
- DONE: Confusion matrix (heat map style, if possible)
- DONE: Feature importance in random forest
- NULL: Feature importance in FC
- Some graph of mean errors (not squared) after perturbation
- DONE: Accuracy vs. bin steps
- DONE: Accuracy proportional to random guessing vs. bin steps
- DONE: correlations between features and label
- DONE: Frequency of labels

References:
https://www.kaggle.com/edalrami/19000-spotify-songs/data
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
https://getaravind.com/blog/confusion-matrix-seaborn-heatmap/
