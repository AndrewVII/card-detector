# obs-card-detector
A script to detect MTG cards on the screen

# Information


- `scripts/helpers/generate_source_images.py` gets random frames from an array of youtube URLs, saves them in the data folder
- These can then be used for annotation purposes. Boxes can be drawn around them in labelling software, such as Label Studio (what I used)
- `scripts/helpers/split_yolo_training_data.py` then splits the training data into training, validation, and test datasets, with a ratio of 0.7, 0.2, and 0.1, respectively. It expects the training data to be in `yolo/data`
- `scripts/model_training/train_yolo.py` will then train the model, and output it in `yolo/models`
- this will open up a webcam, you can put a card in the screen and it will draw a box around it
- `scripts/card_finder.py` works by taking an image of a card, and using a model to match the image to its closest match in the directory data/cards. These cards can be generated by running `scripts/helpers/get_all_cards.py`. NOTE! this functionality does not yet work and will very likely not match

