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

Brian, 12/12/19, in lab:
- Wrote train/test Random Forest functions in main.py using sklearn.ensemble Random Forest models.
- These functions take a boolean parameter called `regressor` which determines whether RFClassifier or RFRegressor is used.
- Wrote accuracy and MSE functions in main.py

References:
https://www.kaggle.com/edalrami/19000-spotify-songs/data

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
