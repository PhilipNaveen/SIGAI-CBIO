import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid


# (X, y), (X_test, y_test) = our data
# X = X.reshape(-1, 28, 28, 1).astype("float32") / 255.0
# X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def build_resnet(input_shape, num_classes, dropout_rate):
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv2D(64, kernel_size=7, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.AveragePooling2D(pool_size=3, strides=2, padding="same")(x)

    x = residual_block(x, filters=64, dropout_rate=dropout_rate)
    x = residual_block(x, filters=64, dropout_rate=dropout_rate)
    x = residual_block(x, filters=64, dropout_rate=dropout_rate)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)

def residual_block(x, filters, dropout_rate):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv2D(filters, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def build_transformer(input_shape, num_classes, dropout_rate, num_patches=196, projection_dim=64, num_heads=4, transformer_units=[128, 64], transformer_layers=8):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Creating patches
    patch_size = input_shape[0] // int(np.sqrt(num_patches))
    patches = tf.keras.layers.Conv2D(projection_dim, kernel_size=patch_size, strides=patch_size, padding="valid")(inputs)
    patches = tf.keras.layers.Reshape((num_patches, projection_dim))(patches)
    
    # Add positional encoding
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embedding = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)
    encoded_patches = patches + pos_embedding
    
    # Create multiple Transformer layers
    for _ in range(transformer_layers):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=dropout_rate)(x1, x1)
        # Skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=dropout_rate)
        # Skip connection 2
        encoded_patches = layers.Add()([x3, x2])
    
    # Apply global average pooling to generate a [batch_size, projection_dim] representation
    representation = layers.GlobalAveragePooling1D()(encoded_patches)
    
    # Add MLP
    representation = layers.Dropout(dropout_rate)(representation)
    features = layers.Dense(128, activation="relu")(representation)
    features = layers.Dropout(dropout_rate)(features)
    
    # Classify outputs.
    outputs = layers.Dense(num_classes, activation="softmax")(features)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def build_combined_model(input_shape, num_classes, dropout_rate):
    resnet_model = build_resnet(input_shape, num_classes, dropout_rate)
    transformer_model = build_transformer(input_shape, num_classes, dropout_rate)
    
    inputs = tf.keras.Input(shape=input_shape)
    resnet_outputs = resnet_model(inputs)
    transformer_outputs = transformer_model(inputs)
    
    combined_outputs = layers.Concatenate()([resnet_outputs, transformer_outputs])
    
    outputs = layers.Dense(num_classes, activation="softmax")(combined_outputs)
    
    return tf.keras.Model(inputs, outputs)

input_shape = X_train.shape[1:]
num_classes = 10  # Number of classes in Fashion MNIST

# Define the parameter grid
param_grid = {
    'dropout_rate': [0.3, 0.5],
    'learning_rate': [1e-3, 1e-4]
}

# Training settings
epochs = 10
batch_size = 32

# Grid search
best_accuracy = 0
best_params = {}

for params in ParameterGrid(param_grid):
    dropout_rate = params['dropout_rate']
    learning_rate = params['learning_rate']
    
    model = build_combined_model(input_shape, num_classes, dropout_rate)
    
    # Use Adam optimizer with AMSGrad
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)
    
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    print(f"Training with params: {params}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )
    
    val_accuracy = history.history['val_accuracy'][-1]
    
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_params = params
        best_model = model

# Evaluate the best model on the test set
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print(f"Best params: {best_params}")
print(f"Test accuracy: {test_accuracy:.4f}")
