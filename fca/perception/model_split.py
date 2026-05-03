"""
fca/perception/model_split.py — Stage B preparation. RUN ON YOUR LAPTOP.

PLACEHOLDER — TO BE IMPLEMENTED WHEN YOU'RE READY FOR STAGE B.

═══════════════════════════════════════════════════════════════════════════
REVISED PLAN — NO SPLITTING, JUST FEATURE EXTRACTOR EXPORT
═══════════════════════════════════════════════════════════════════════════
We're NOT splitting the existing INT8 model. The full model keeps driving the
car. We export a SEPARATE TFLite graph that goes from input → 1280-d feature
vector. Both graphs run in parallel on the Pi.

This avoids the quantization-mismatch risk of splitting a quantized model.

═══════════════════════════════════════════════════════════════════════════
WHAT THIS SCRIPT NEEDS TO DO (PSEUDO-CODE)
═══════════════════════════════════════════════════════════════════════════
This script lives on your laptop because:
  - It needs the original Keras .keras file (not just the .tflite)
  - It needs INT8 calibration with a representative dataset
  - These dependencies aren't on the Pi

Implementation outline:

    import tensorflow as tf
    import numpy as np
    import os

    # 1. Load the original full Keras model
    full_model = tf.keras.models.load_model(
        'best_model_finetuned.keras',
        custom_objects={
            'se_block': se_block_fn,         # your custom SE block
            'weighted_bce': weighted_bce_fn, # your custom loss
        },
        compile=False,
    )

    # 2. Identify the layer to extract features from.
    #    For your architecture, the candidates are:
    #      - 'global_average_pooling2d' (1280-d, before Dense 512) ← RECOMMENDED
    #      - 'dense_128_post_bn'        (128-d, mid-trunk)         ← compact alt
    #
    #    Inspect the layer names: [l.name for l in full_model.layers]
    FEATURE_LAYER_NAME = 'global_average_pooling2d'

    # 3. Build a NEW model that outputs the feature vector.
    #    This model SHARES weights with full_model — we're not modifying it.
    feature_extractor = tf.keras.Model(
        inputs=full_model.input,
        outputs=full_model.get_layer(FEATURE_LAYER_NAME).output,
    )

    # 4. Calibration dataset — a generator yielding ~100 representative images
    def representative_dataset():
        # Load images from your training set; preprocess EXACTLY as the model
        # was trained (320×240 → crop → 224×224 → /127.5 − 1)
        for img_path in sorted(os.listdir('calibration_images'))[:100]:
            img = preprocess_for_calibration(img_path)
            yield [img.astype(np.float32)]

    # 5. Convert to INT8 TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(feature_extractor)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    with open('feature_extractor_int8.tflite', 'wb') as f:
        f.write(tflite_model)

    # 6. Sanity check: run the feature extractor on a few images and verify
    #    you get reasonable 1280-d vectors (not all zeros, not all saturated).

    # 7. Optionally also export a 128-d version from the mid-trunk layer.
    #    This becomes a useful ablation for the paper:
    #      "1280-d visual feature vs 128-d task-compressed feature"

═══════════════════════════════════════════════════════════════════════════
DEPLOY TO PI
═══════════════════════════════════════════════════════════════════════════
    scp feature_extractor_int8.tflite pi@<ip>:~/fca/tflite_models/

═══════════════════════════════════════════════════════════════════════════
NEXT STEPS (PI-SIDE)
═══════════════════════════════════════════════════════════════════════════
    1. Implement FeatureExtractor in feature_extractor.py
    2. Implement DeepAdapter in learning/adapter_deep.py
    3. Modify AdaptiveController.predict() to call both:
         (a) self.base_model.predict_raw(image) — existing
         (b) self.feature_extractor.predict_features(image) — new
       Then pass (b)'s output to the adapter.
    4. Run: python3 run.py --mode drive --adapter deep
"""

import sys


def main():
    print(__doc__)
    print()
    print("This is a placeholder. Implement when ready for Stage B.")
    print("Stage A works without this — use --adapter scalar.")
    sys.exit(1)


if __name__ == "__main__":
    main()
