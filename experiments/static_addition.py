from keras.models import Model
from keras.layers import Input
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

import os
import sys
module_dir = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, module_dir)
from nac import NAC
from experiments.data_utils import generate_static_dataset

# Check if weights folder exists
# This folder store the trained model weights
if not os.path.exists('weights'):
    os.makedirs('weights')


# Hyper parameters used for training
units = 2
num_samples = 1000

# Create a task function for addition
task_name = 'addition'
task_fn = lambda x, y: x + y

# Generate the model with an input layer and two NAC layers
ip = Input(shape=(100,))
x = NAC(units)(ip)
x = NAC(1)(x)

model = Model(ip, x)
model.summary()

# Compile the model
# Use RMSprop as the optimizer and mean squared error as the loss function
optimizer = RMSprop(0.1)
model.compile(optimizer, 'mse')

# Generate training and testing datasets
X_train, y_train = generate_static_dataset(task_fn, num_samples, mode='interpolation')
X_test, y_test = generate_static_dataset(task_fn, num_samples, mode='extrapolation')

# Use 'ModelCheckpoint' callback to save the model weights during training, specifically saving only the best-performing weights based on validation loss
weights_path = 'weights/%s_weights.h5' % (task_name)
checkpoint = ModelCheckpoint(weights_path, monitor='val_loss',
                             verbose=1, save_weights_only=True, save_best_only=True)

callbacks = [checkpoint]

# Train model and includes validation using the test dataset
model.fit(X_train, y_train, batch_size=64, epochs=500,
          verbose=2, callbacks=callbacks, validation_data=(X_test, y_test))

# Evaluate the model on the test data set and the mean squared error of the model on the test dataset is printed
model.load_weights(weights_path)

scores = model.evaluate(X_test, y_test, batch_size=128)

print("Mean Squared error : ", scores)