from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, Model
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    LATENT_DIM, GP_LAMBDA, N_CRITIC, WGAN_LR, WGAN_BETA1, WGAN_BETA2,
    TRIAL_SHAPE,
)

# ─── Generator ────────────────────────────────────────────────────────────────
def build_generator(latent_dim: int = LATENT_DIM) -> Model:
    """
    Maps noise z ∈ R^{latent_dim} to a synthetic CWT image (50, 375, 5).
    Batch normalisation after every transposed-conv block stabilises training
    and acts as regularisation.
    """
    z = layers.Input(shape=(latent_dim,), name="z")

    # Dense projection + reshape to spatial seed for upsampling
    x = layers.Dense(8192, use_bias=False)(z)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Reshape((4, 4, 512))(x)            # (4, 4, 512)

    # Upsample block 1:  (4,4,512) → (11,11,256)  [valid padding, stride 2]
    x = layers.Conv2DTranspose(256, (5, 5), strides=(2, 2),
                               padding="valid", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)                           # (11, 11, 256)

    # Upsample block 2:  (11,11,256) → (25,25,128)
    x = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2),
                               padding="valid", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)                           # (25, 25, 128)

    # Upsample block 3:  (25,25,128) → (50,125,64)  [same padding, stride (2,5)]
    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 5),
                               padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)                           # (50, 125, 64)

    # Output block:  (50,125,64) → (50,375,5)  [same padding, stride (1,3)]
    out = layers.Conv2DTranspose(
        TRIAL_SHAPE[-1], (5, 5), strides=(1, 3),
        padding="same", activation="tanh", name="cwt_out"
    )(x)                                           # (50, 375, 5)

    return Model(z, out, name="generator")


# ─── Critic ───────────────────────────────────────────────────────────────────
def build_critic(input_shape: tuple = TRIAL_SHAPE) -> Model:
    """
    Scores the realism of CWT images with a scalar output (no sigmoid).

    Using LeakyReLU in the critic and Dropout(0.2) for regularisation.
    Linear final layer is required for Wasserstein distance estimation
    (Section 2.2.5, point 1).
    """
    x_in = layers.Input(shape=input_shape, name="cwt_image")

    # Conv block 1:  (50,375,5) → (25,125,64)
    x = layers.Conv2D(64, (5, 5), strides=(2, 3), padding="same")(x_in)
    x = layers.LeakyReLU(0.2)(x)                  # (25, 125, 64)

    # Conv block 2:  (25,125,64) → (25,25,128)
    x = layers.Conv2D(128, (5, 5), strides=(1, 5), padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.2)(x)                    # (25,  25, 128)

    # Output head
    x = layers.Flatten()(x)                       # (80000,)
    x = layers.Dropout(0.2)(x)
    score = layers.Dense(1, name="wasserstein_score")(x)

    return Model(x_in, score, name="critic")


# ─── WGAN-GP Training Model ───────────────────────────────────────────────────
class WGANGP(tf.keras.Model):
    """
    Critic is updated N_CRITIC times per generator step.
    Gradient penalty enforces the 1-Lipschitz constraint.
    """

    def __init__(
        self,
        generator: Model,
        critic:    Model,
        latent_dim: int   = LATENT_DIM,
        n_critic:   int   = N_CRITIC,
        gp_lambda:  float = GP_LAMBDA,
    ):
        super().__init__()
        self.generator  = generator
        self.critic     = critic
        self.latent_dim = latent_dim
        self.n_critic   = n_critic
        self.gp_lambda  = gp_lambda

        # Keras metrics tracked across batches
        self._c_loss_metric = tf.keras.metrics.Mean(name="critic_loss")
        self._g_loss_metric = tf.keras.metrics.Mean(name="generator_loss")

    def compile(self, g_optimizer, c_optimizer, **kwargs):
        super().compile(**kwargs)
        self.g_optimizer = g_optimizer
        self.c_optimizer = c_optimizer

    @property
    def metrics(self):
        return [self._c_loss_metric, self._g_loss_metric]

    def _gradient_penalty(
        self, real: tf.Tensor, fake: tf.Tensor
    ) -> tf.Tensor:
        """
        x̂ = ε·real + (1-ε)·fake,   ε ~ U(0,1)
        GP = E[(‖∇_{x̂} C(x̂)‖₂ − 1)²]
        """
        batch = tf.shape(real)[0]
        eps   = tf.random.uniform([batch, 1, 1, 1], 0.0, 1.0)
        real_f = tf.cast(real, tf.float32)
        fake_f = tf.cast(fake, tf.float32)
        eps_f = tf.cast(eps, tf.float32)
        x_hat = real_f + eps_f * (fake_f - real_f)

        with tf.GradientTape() as tape:
            tape.watch(x_hat)
            score = tf.cast(self.critic(x_hat, training=True), tf.float32)

        grads = tape.gradient(score, x_hat)
        norms = tf.sqrt(
            tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-10
        )
        return tf.reduce_mean((norms - 1.0) ** 2)

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        batch = tf.shape(real_images)[0]

        for _ in range(self.n_critic):
            z = tf.random.normal([batch, self.latent_dim])
            with tf.GradientTape() as tape:
                fake        = self.generator(z, training=True)
                real_score  = self.critic(real_images, training=True)
                fake_score  = self.critic(fake,        training=True)
                gp          = self._gradient_penalty(real_images, fake)
                # L_critic = E[fake] − E[real] + λ·GP
                c_loss = (
                    tf.reduce_mean(tf.cast(fake_score, tf.float32))
                    - tf.reduce_mean(tf.cast(real_score, tf.float32))
                    + self.gp_lambda * gp
                )
            c_grads = tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(
                zip(c_grads, self.critic.trainable_variables)
            )

        z = tf.random.normal([batch, self.latent_dim])
        with tf.GradientTape() as tape:
            fake       = self.generator(z, training=True)
            fake_score = self.critic(fake, training=True)
            g_loss = -tf.reduce_mean(tf.cast(fake_score, tf.float32))
        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_grads, self.generator.trainable_variables)
        )

        self._c_loss_metric.update_state(c_loss)
        self._g_loss_metric.update_state(g_loss)
        return {m.name: m.result() for m in self.metrics}
