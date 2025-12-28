import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tf_slim as slim
import numpy as np
import os
import sys
import re
from PIL import Image
import imp
import pandas as pd

# --- CONFIGURATION ---

DOCFACE_PROJECT_DIR = '/home/tasneem/repos/bassil_face/RD/DocFace'
# FINETUNED_MODEL_DIR = '/home/tasneem/repos/bassil_face/RD/DocFace/log/faceres_finetuned/20251111-113103'
FINETUNED_MODEL_DIR = '/home/tasneem/repos/bassil_face/RD/DocFace/log/faceres_finetuned_frontalized_ids/20251208-213904'  # <-- path to your fine-tuned model folder
CSV_INPUT_PATH = '/home/tasneem/repos/bassil_face/disjoint_no_shuffle/docface_data_set_final/train.csv'   # <-- path to your CSV
CSV_OUTPUT_PATH = '/home/tasneem/repos/bassil_face/disjoint_no_shuffle/docface_data_set_final/train__out.csv'
THRESHOLD = 0.5

# --- IMPORT NETWORK ---
SRC_PATH = os.path.join(DOCFACE_PROJECT_DIR, 'src')
sys.path.insert(0, SRC_PATH)

try:
    import sibling_net_V1 as sibling_net
    print("Loaded 'sibling_net_V1.py'")
except ImportError:
    import sibling_net
    print("Loaded 'sibling_net.py'")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def remove_black_border(image):
    img_array_orig = np.array(image)
    gray_array = np.array(image.convert('L'))
    non_black_mask = gray_array > 10
    rows_with_content = np.any(non_black_mask, axis=1)
    cols_with_content = np.any(non_black_mask, axis=0)
    if not np.any(rows_with_content) or not np.any(cols_with_content):
        return image.convert('RGB')
    top = np.argmax(rows_with_content)
    bottom = len(rows_with_content) - np.argmax(rows_with_content[::-1])
    left = np.argmax(cols_with_content)
    right = len(cols_with_content) - np.argmax(cols_with_content[::-1])
    cropped_array = img_array_orig[top:bottom, left:right]
    return Image.fromarray(cropped_array).convert('RGB')


def prewhiten(x):
    mean = 127.5
    std = 128.0
    return (x.astype(np.float32) - mean) / std


def preprocess_image(image_path, target_size_hw, is_selfie):
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"ERROR: Could not open image {image_path}: {e}")
        return None
    if is_selfie:
        img = remove_black_border(img)
    pil_size = (target_size_hw[0], target_size_hw[1])
    img = img.resize(pil_size, Image.Resampling.BILINEAR).convert('RGB')
    img_array = np.array(img, dtype=np.float32)
    return prewhiten(img_array)


def compute_cosine_similarity(embedding_A, embedding_B):
    emb_A_norm = embedding_A / np.linalg.norm(embedding_A)
    emb_B_norm = embedding_B / np.linalg.norm(embedding_B)
    return float(np.dot(emb_A_norm, emb_B_norm.T))


def main():
    print("\n--- DocFace Batch Cosine Similarity ---")

    CONFIG_PATH = os.path.join(FINETUNED_MODEL_DIR, 'config.py')
    config = imp.load_source('config', CONFIG_PATH)

    num_classes = getattr(config, 'num_classes', 307096)
    abs_network_path = os.path.join(DOCFACE_PROJECT_DIR, config.network)
    config.network = abs_network_path

    network = sibling_net.SiblingNetwork()
    target_shape_hw = config.image_size
    network.image_batch_placeholder = tf.placeholder(tf.float32, shape=[None, target_shape_hw[0], target_shape_hw[1], 3], name='image_batch')
    network.initialize(config, num_classes)
    graph = network.graph

    image_batch_in = graph.get_tensor_by_name("image_batch:0")
    switch_all_in = graph.get_tensor_by_name("switch_all:0")
    embeddings_A_out = graph.get_tensor_by_name("embeddings_A:0")
    embeddings_B_out = graph.get_tensor_by_name("embeddings_B:0")
    phase_train_in = graph.get_tensor_by_name("phase_train:0")
    keep_prob_in = graph.get_tensor_by_name("keep_prob:0")
    sess = network.sess

    latest_checkpoint = tf.train.latest_checkpoint(FINETUNED_MODEL_DIR)
    restore_vars = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    saver = tf.train.Saver(restore_vars)
    saver.restore(sess, latest_checkpoint)
    print(f"Model loaded from {latest_checkpoint}")

    # --- READ CSV ---
    df = pd.read_csv(CSV_INPUT_PATH)
    similarities = []

    for idx, row in df.iterrows():
        id_path = row['id']
        selfie_path = row['selfie']
        print(f"\n[{idx+1}/{len(df)}] {os.path.basename(selfie_path)} vs {os.path.basename(id_path)}")

        id_array = preprocess_image(id_path, target_shape_hw, is_selfie=False)
        selfie_array = preprocess_image(selfie_path, target_shape_hw, is_selfie=True)
        if id_array is None or selfie_array is None:
            similarities.append(np.nan)
            continue

        batch_images = np.stack([id_array, selfie_array], axis=0)
        batch_switch = np.array([False, True])

        feed_dict = {
            image_batch_in: batch_images,
            switch_all_in: batch_switch,
            phase_train_in: False,
            keep_prob_in: 1.0
        }

        embedding_A_batch, embedding_B_batch = sess.run([embeddings_A_out, embeddings_B_out], feed_dict=feed_dict)
        similarity = compute_cosine_similarity(embedding_A_batch[0], embedding_B_batch[0])
        similarities.append(similarity)
        print(f"  → Cosine Similarity = {similarity:.4f}")

    df['Cosine_Similarity'] = similarities
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"\n✅ Done. Results saved to {CSV_OUTPUT_PATH}")

    sess.close()


if __name__ == "__main__":
    main()
