import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class KLDivergenceLayer(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return inputs

def build_vae(input_dim, latent_dim):
    # Encoder
    encoder_inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(encoder_inputs)
    x = layers.Dense(32, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    kl_loss = KLDivergenceLayer()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(32, activation="relu")(latent_inputs)
    x = layers.Dense(64, activation="relu")(x)
    decoder_outputs = layers.Dense(input_dim, activation="sigmoid")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    # VAE model
    vae_inputs = keras.Input(shape=(input_dim,))
    z_mean, z_log_var, z = encoder(vae_inputs)
    reconstructions = decoder(z)
    vae = keras.Model(vae_inputs, reconstructions, name="vae")

    return vae, encoder, decoder

def train_vae(data):
    logging.info("Starting VAE training")
    logging.info(f"Initial data shape: {data.shape}")
    
    try:
        # Preprocess data
        numeric_cols = data.select_dtypes(include=['number']).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        logging.info(f"Numeric columns: {numeric_cols.tolist()}")
        logging.info(f"Categorical columns: {categorical_cols.tolist()}")

        # One-hot encode categorical variables
        data_encoded = pd.get_dummies(data, columns=categorical_cols)
        
        # Convert all columns to numeric, replacing non-numeric values with NaN
        for col in data_encoded.columns:
            data_encoded[col] = pd.to_numeric(data_encoded[col], errors='coerce')
        
        # Drop columns with all NaN values
        data_encoded = data_encoded.dropna(axis=1, how='all')
        
        # Fill remaining NaN values with the mean of each column
        data_encoded = data_encoded.fillna(data_encoded.mean())
        
        logging.info(f"Encoded data shape: {data_encoded.shape}")
        
        # Normalize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_encoded)
        
        logging.info(f"Scaled data shape: {scaled_data.shape}")
        
        input_dim = scaled_data.shape[1]
        latent_dim = min(10, input_dim // 2)  # Adjust latent_dim based on input_dim
        
        logging.info(f"Input dimension: {input_dim}, Latent dimension: {latent_dim}")

        vae, encoder, decoder = build_vae(input_dim, latent_dim)
        
        vae.compile(optimizer=keras.optimizers.Adam(), loss='mse')
        
        history = vae.fit(scaled_data, scaled_data,
                          epochs=50,
                          batch_size=32,
                          validation_split=0.2,
                          verbose=0)
        
        logging.info("VAE training completed")
        return vae, encoder, decoder, scaler, data_encoded.columns
    except Exception as e:
        logging.error(f"Error in VAE training: {str(e)}")
        logging.error(f"Data types: {data.dtypes}")
        logging.error(f"Data head: {data.head()}")
        raise

def generate_synthetic_data(decoder, num_samples, latent_dim, scaler, columns):
    try:
        random_latent_points = np.random.normal(size=(num_samples, latent_dim))
        synthetic_data = decoder.predict(random_latent_points)
        synthetic_data = scaler.inverse_transform(synthetic_data)
        synthetic_df = pd.DataFrame(synthetic_data, columns=columns)
        return synthetic_df
    except Exception as e:
        logging.error(f"Error in generating synthetic data: {str(e)}")
        return None