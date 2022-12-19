# ATDS_Project
Forest Fire Identification and Localization Through Tensorflow using AutoKeras and tfHParams

Using AutoKeras for NAS on tf_flowers dataset
Using best model to transfer learn onto fire dataset
Using Tensorboard and HPParams for Hyperparameter optimization during feature extraction

96% model accuracy for fire detection test dataset

def train_test_model(hparams): <br>
  model = tf.keras.Sequential()	
  for layer in autokeras_model.layers[:-2]: # Skip first and last layer	
    model.add(layer)	
  model.add(tf.keras.layers.Dropout(hparams[HP_DROPOUT]))	
  model.add(tf.keras.layers.Dense(fire_num_classes, activation='softmax'))	
	
  model.compile(	
      optimizer = tf.keras.optimizers.Adam(learning_rate=hparams[HP_LR]),	
      loss = tf.keras.losses.SparseCategoricalCrossentropy(),	
      metrics=['accuracy'])	
	
  history = model.fit(train_fire,	
                      epochs=init_epochs)	
  _,accuracy = model.evaluate(valid_fire)	
  return accuracy	
	
def run(run_dir, hparams):	
  with tf.summary.create_file_writer(run_dir).as_default():	
    hp.hparams(hparams)  # record the values used in this trial	
    accuracy = train_test_model(hparams)	
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)	
