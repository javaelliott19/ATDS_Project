# ATDS_Project
Forest Fire Identification and Localization Through Tensorflow using AutoKeras and tfHParams
![image](https://user-images.githubusercontent.com/43414937/208487090-29ccca45-cdee-46e7-896a-614b3bc143de.png)

## Using AutoKeras for NAS on tf_flowers dataset
https://www.tensorflow.org/datasets/catalog/tf_flowers
## Using best model to transfer learn onto fire detection dataset
https://data.mendeley.com/datasets/gjmr63rz2r/1

Using Tensorboard and HPParams for Hyperparameter optimization during feature extraction (transfer learning)
![image](https://user-images.githubusercontent.com/43414937/208486470-b3d645d6-723f-40a0-bb57-43770830ef63.png)

Model accuracy jump from base model 48$ accuracy
![image](https://user-images.githubusercontent.com/43414937/208486640-82d3c55f-5581-49b8-b019-345e54a4eee3.png)

...to model accuracy of 96%
![image](https://user-images.githubusercontent.com/43414937/208486735-3e6ffe93-46db-4292-8969-916417e1247e.png)

Function to run HParams Hyperparameter Optimization during feature extraction
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
    
    
