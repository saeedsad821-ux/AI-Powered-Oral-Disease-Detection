"""M2 perf benchmark: EfficientNetB0 forward+backward on CPU (1 core)."""

import time

import tensorflow as tf

tf.keras.utils.set_random_seed(42)

model = tf.keras.applications.EfficientNetB0(weights=None, include_top=False)
model = tf.keras.Sequential([model, tf.keras.layers.GlobalAveragePooling2D(),
                             tf.keras.layers.Dense(6, activation="softmax")])
model.compile(optimizer="adam", loss="categorical_crossentropy")

x = tf.random.uniform((32, 224, 224, 3), 0, 255)   # float32 [0,255] batch
y = tf.keras.utils.to_categorical([i % 6 for i in range(32)], 6).astype("float32")

for _ in range(2):  # warmup / JIT
    model.train_on_batch(x, y)

times = []
for _ in range(8):
    t0 = time.perf_counter()
    model.train_on_batch(x, y)
    times.append(time.perf_counter() - t0)

dt = min(times) / 32  # best per-image step time (train_on_batch = 1 fwd+bwd step)
print(f"train step (batch 32, fwd+bwd): best {min(times)*1000:.0f} ms, "
      f"median {sorted(times)[len(times)//2]*1000:.0f} ms")
print(f"per-image (fwd+bwd): {dt*1000:.1f} ms")
print(f"epoch of 8,624 train images @bs32: {min(times)/32*8624/60:.1f} min "
      f"(pipe-limited), model-only: {min(times)*270/60:.1f} min")
