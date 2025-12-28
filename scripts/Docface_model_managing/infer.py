import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() # Must be first, after tf import
import tf_slim as slim # Must import this
import numpy as np
import os
import sys
import re
from PIL import Image
import imp

# --- 1. CONFIGURATION: YOU MUST EDIT THIS ---

# 1. Set the path to your main 'DocFace' project folder
#    (This is the folder that contains the 'src' and 'nets' directories)
DOCFACE_PROJECT_DIR = '/home/tasneem/repos/bassil_face/RD/DocFace'

# 2. Set the path to your FINE-TUNED model *FOLDER*
#    (This is the folder from your screenshot, e.g., log/faceres_finetuned/...)
FINETUNED_MODEL_DIR = '/home/tasneem/repos/bassil_face/RD/DocFace/log/faceres_finetuned/20251111-113103'

# 3. Set the paths to the two images you want to compare
ID_IMAGE_PATH = "/home/tasneem/repos/bassil_face/docface_data_sett/test_set/person_29/A_000.jpg"
SELFIE_IMAGE_PATH = "/home/tasneem/repos/bassil_face/docface_data_sett/test_set/person_29/B_000.jpg"

# 4. Set your threshold (start with 0.5 or use one from your logs)
THRESHOLD = 0.5 

# --- END OF CONFIGURATION ---


# --- CORRECTLY IMPORT THE SIBLING NETWORK CLASS ---
# We add the 'src' directory from your project path
SRC_PATH = os.path.join(DOCFACE_PROJECT_DIR, 'src')
sys.path.insert(0, SRC_PATH)

try:
    # Try to import the renamed file first
    import sibling_net_V1 as sibling_net
    print("Loaded 'sibling_net_V1.py'")
except ImportError:
    try:
        # Fallback to the original name
        import sibling_net
        print("Loaded 'sibling_net.py'")
    except ImportError:
        print(f"FATAL ERROR: Could not find 'sibling_net.py' or 'sibling_net_V1.py' in {SRC_PATH}/")
        sys.exit(1)
# --- END IMPORT FIX ---


# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def remove_black_border(image):
    """
    Remove black border from a PIL image containing a face.
    Handles both transparent (PNG) and pixel (JPG) borders.
    """
    img_array_orig = np.array(image)
    
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        alpha_image = image.convert('RGBA')
        img_array = np.array(alpha_image)
        alpha = img_array[:, :, 3]
        if np.any(alpha < 255): 
            non_black_mask = alpha > 0
        else:
            gray_array = np.array(image.convert('L'))
            non_black_mask = gray_array > 10
    else:
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
    
    if bottom <= top: bottom = top + 1
    if right <= left: right = left + 1

    try:
        cropped_array = img_array_orig[top:bottom, left:right]
    except IndexError:
        return image.convert('RGB')
    
    result_image = Image.fromarray(cropped_array)
    return result_image.convert('RGB')

def prewhiten(x):
    """Applies the same pre-whitening as the training script."""
    mean = 127.5
    std = 128.0
    return (x.astype(np.float32) - mean) / std

def preprocess_image(image_path, target_size_hw, is_selfie):
    """
    Loads and preprocesses a single image for the model.
    target_size_hw is [Height, Width]
    """
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"ERROR: Could not open image {image_path}: {e}")
        return None

    # 1. Clean the selfie border if necessary
    if is_selfie:
        img = remove_black_border(img)

    # 2. Resize to model's expected input size
    # PIL.Image.resize expects (Width, Height)
    pil_size = (target_size_hw[0], target_size_hw[1]) 
    img = img.resize(pil_size, Image.Resampling.BILINEAR)
    img = img.convert('RGB')
    
    # 3. Convert to Numpy array and prewhiten (standardize)
    img_array = np.array(img, dtype=np.float32)
    img_array = prewhiten(img_array)
    
    # DO NOT add batch dimension, we will stack them
    return img_array

def main():
    print("--- DocFace Fine-Tuned Model Inference Script ---")
    
    # 1. Load the configuration file
    # The config.py is inside your model log folder
    CONFIG_PATH = os.path.join(FINETUNED_MODEL_DIR, 'config.py')
    print(f"Loading config from: {CONFIG_PATH}")
    try:
        config = imp.load_source('config', CONFIG_PATH)
    except FileNotFoundError:
        print(f"FATAL ERROR: Config file not found at {CONFIG_PATH}")
        print("Please check your FINETUNED_MODEL_DIR path.")
        sys.exit(1)
    
    # Get the number of classes from the config file
    try:
        num_classes = config.num_classes
        print(f"Found {num_classes} classes in config.")
    except AttributeError:
        # This is the fix for your previous error
        print(f"INFO: 'num_classes' not in config.py. Using hardcoded 307096.")
        num_classes = 307096 # Hardcoded from your training log

    # 2. Build the model graph
    
    # --- FIX for FileNotFoundError ---
    # Convert the relative network path from the config into an absolute path
    abs_network_path = os.path.join(DOCFACE_PROJECT_DIR, config.network)
    if not os.path.exists(abs_network_path):
        print(f"FATAL ERROR: The network file '{config.network}' was not found at:")
        print(f"{abs_network_path}")
        print("Please check the 'network' path in your config.py OR")
        print(f"Make sure the file exists at: {abs_network_path}")
        sys.exit(1)
    
    print(f"Loading network architecture: {abs_network_path}")
    # Overwrite the config value with the absolute path
    config.network = abs_network_path
    # --- END FIX ---
    
    # Instantiate the SiblingNetwork class from the imported file
    network = sibling_net.SiblingNetwork()
    
    # Get the image size from the config file
    try:
        target_shape_hw = config.image_size # This should be [96, 112]
        print(f"Using image size from config: {target_shape_hw} (Height, Width)")
    except Exception as e:
        print(f"FATAL ERROR: Could not read 'image_size' from config. {e}")
        sys.exit(1)
    
    # Override placeholders to use the correct, single shape from your config
    network.image_batch_placeholder = tf.placeholder(tf.float32, shape=[None, target_shape_hw[0], target_shape_hw[1], 3], name='image_batch')
    
    # Now initialize the network
    # This is the line that was failing
    network.initialize(config, num_classes)

    # Get the graph and the tensor "plugs"
    
    # --- THIS IS THE FIX ---
    # We must use the graph *from the network object*, not the default graph
    graph = network.graph
    # --- END FIX ---
    
    try:
        # --- These are the REAL tensor names from sibling_net.py ---
        image_batch_in = graph.get_tensor_by_name("image_batch:0")
        switch_all_in = graph.get_tensor_by_name("switch_all:0")
        embeddings_A_out = graph.get_tensor_by_name("embeddings_A:0")
        embeddings_B_out = graph.get_tensor_by_name("embeddings_B:0")
        phase_train_in = graph.get_tensor_by_name("phase_train:0")
        keep_prob_in = graph.get_tensor_by_name("keep_prob:0")
    except KeyError as e:
        print(f"ERROR: Could not find tensor in graph: {e}")
        print("This means the model's tensor names are not as expected.")
        return

    # 3. Start TensorFlow session
    # We must use the session *from the network object*
    sess = network.sess
        
    # 4. Load your *FINE-TUNED* weights
    print(f"Finding latest checkpoint in: {FINETUNED_MODEL_DIR}")
    try:
        latest_checkpoint = tf.train.latest_checkpoint(FINETUNED_MODEL_DIR)
        if latest_checkpoint is None:
            print(f"FATAL ERROR: No checkpoint found in {FINETUNED_MODEL_DIR}")
            print("Your log folder may be one level deeper. Check the path.")
            sys.exit(1)
            
        print(f"Loading weights from: {latest_checkpoint}")
        
        # --- THIS IS THE FIX FOR "AM-Softmax/Momentum" ERROR ---
        # We only restore the TRAINABLE_VARIABLES, which excludes
        # the 'Momentum' and other optimizer variables.
        # restore_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        restore_vars = network.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        saver = tf.train.Saver(restore_vars)
        # --- END FIX ---
        
        saver.restore(sess, latest_checkpoint)
        
    except Exception as e:
        print(f"\nFATAL ERROR: Could not load checkpoint.")
        print(f"Please check your FINETUNED_MODEL_DIR variable.")
        print(f"Error details: {e}")
        sess.close()
        sys.exit(1)
        
    print("Model loaded successfully.")

    # 5. Preprocess both images
    print(f"Processing ID photo: {ID_IMAGE_PATH}")
    id_array = preprocess_image(ID_IMAGE_PATH, target_shape_hw, is_selfie=False)
    
    print(f"Processing Selfie: {SELFIE_IMAGE_PATH}")
    selfie_array = preprocess_image(SELFIE_IMAGE_PATH, target_shape_hw, is_selfie=True)

    if id_array is None or selfie_array is None:
        print("Could not process images. Exiting.")
        sess.close()
        return
        
    # 6. Create the Batch and Switch
    # We are feeding a batch of 2 images
    batch_images = np.stack([id_array, selfie_array], axis=0)
    
    # The switch: [False, True]
    # False = image 0 is an ID (it's an 'A' file)
    # True = image 1 IS a selfie (it's a 'B' file)
    batch_switch = np.array([False, True])

    # 7. Run inference ONCE
    feed_dict = {
        image_batch_in: batch_images,
        switch_all_in: batch_switch,
        phase_train_in: False,
        keep_prob_in: 1.0
    }
    
    # We ask for both embedding outputs.
    # The 'A' output will contain the ID embedding.
    # The 'B' output will contain the Selfie embedding.
    embedding_A_batch, embedding_B_batch = sess.run([embeddings_A_out, embeddings_B_out], feed_dict=feed_dict)

    # embedding_A_batch will be a [1, 512] array
    # embedding_B_batch will be a [1, 512] array
    embedding_A = embedding_A_batch[0]
    embedding_B = embedding_B_batch[0]

    # 8. Calculate Cosine Similarity
    # Normalize the embeddings (L2 normalization)
    emb_A_norm = embedding_A / np.linalg.norm(embedding_A)
    emb_B_norm = embedding_B / np.linalg.norm(embedding_B)
    
    # Calculate dot product
    cosine_similarity = np.dot(emb_A_norm, emb_B_norm.T)

    print("\n--- Inference Complete ---")
    print(f"Cosine Similarity: {cosine_similarity:.4f}")
    
    if cosine_similarity > THRESHOLD:
        print(f"Result: MATCH (Similarity > {THRESHOLD})")
    else:
        print(f"Result: NO MATCH (Similarity <= {THRESHOLD})")
        
    # 9. Close the session
    sess.close()

if __name__ == "__main__":
    main()