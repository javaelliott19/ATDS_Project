# ATDS_Project
Forest Fire Identification and Localization Through Tensorflow using AutoKeras and tfHParams

Using AutoKeras for NAS on tf_flowers dataset
Using best model to transfer learn onto fire dataset
Using Tensorboard and HPParams for Hyperparameter optimization during feature extraction

96% model accuracy for fire detection test dataset

def train_test_model(hparams): <br>
  model = tf.keras.Sequential()	<br>
  for layer in autokeras_model.layers[:-2]: # Skip first and last layer	<br>
    model.add(layer)	<br>
  model.add(tf.keras.layers.Dropout(hparams[HP_DROPOUT]))	<br>
  model.add(tf.keras.layers.Dense(fire_num_classes, activation='softmax'))<br>	
	<br>
  model.compile(	<br>
      optimizer = tf.keras.optimizers.Adam(learning_rate=hparams[HP_LR]),	<br>
      loss = tf.keras.losses.SparseCategoricalCrossentropy(),	<br>
      metrics=['accuracy'])	<br>
	
  history = model.fit(train_fire,	<br>
                      epochs=init_epochs)	<br>
  _,accuracy = model.evaluate(valid_fire)	<br>
  return accuracy	<br>
	<br>
def run(run_dir, hparams):	<br>
  with tf.summary.create_file_writer(run_dir).as_default():<br>	
    hp.hparams(hparams)  # record the values used in this trial	<br>
    accuracy = train_test_model(hparams)	<br>
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)	<br>
