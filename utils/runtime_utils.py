import tensorflow as tf

def configure_runtime():
    physical_gpus = tf.config.list_physical_devices("GPU")
    print("Available GPUs:", physical_gpus)

    if physical_gpus:
        try:
            for gpu in physical_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print("Could not set memory growth:", e)
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
        print("No GPU detected. Using CPU with float32.")