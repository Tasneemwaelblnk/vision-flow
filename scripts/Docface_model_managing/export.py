import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() # Must be first, after tf import
import tf_slim as slim # Must import this
import numpy as np
import os
import sys
import imp

# --- Check for tf2onnx at the start ---
try:
    import tf2onnx
except ImportError:
    print("\nFATAL ERROR: 'tf2onnx' is not installed.")
    print("Please run: pip install tf2onnx")
    sys.exit(1)
# --- END CHECK ---

# --- 1. CONFIGURATION: YOU MUST EDIT THIS ---

# 1. Set the path to your main 'DocFace' project folder
#    (This is the folder that contains the 'src' and 'nets' directories)
DOCFACE_PROJECT_DIR = '/home/tasneem/repos/bassil_face/RD/DocFace'

# 2. Set the path to your FINE-TUNED model *FOLDER*
#    (The folder containing your ckpt-4000 files)
FINETUNED_MODEL_DIR = '/home/tasneem/repos/bassil_face/RD/DocFace/log/faceres_finetuned_frontalized_ids/20251208-213904'

# 3. Set the desired output path for your new ONNX model
OUTPUT_ONNX_PATH = 'docface_model_frontalized_ids128epoch.onnx'

# --- END OF CONFIGURATION ---

# --- Correctly import the SiblingNetwork class ---
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

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    print("--- DocFace ONNX Exporter ---")
    
    # 1. Load the configuration file from your model's log
    CONFIG_PATH = os.path.join(FINETUNED_MODEL_DIR, 'config.py')
    print(f"Loading config from: {CONFIG_PATH}")
    try:
        config = imp.load_source('config', CONFIG_PATH)
    except FileNotFoundError:
        print(f"FATAL ERROR: Config file not found at {CONFIG_PATH}")
        sys.exit(1)

    try:
        num_classes = config.num_classes
        print(f"Found {num_classes} classes in config.")
    except AttributeError:
        print(f"INFO: 'num_classes' not in config.py. Using hardcoded 307096.")
        num_classes = 307096

    # 2. Build the model graph
    print(f"Loading network architecture: {config.network}")
    abs_network_path = os.path.join(DOCFACE_PROJECT_DIR, config.network)
    if not os.path.exists(abs_network_path):
        print(f"FATAL ERROR: The network file '{config.network}' was not found at: {abs_network_path}")
        sys.exit(1)
    
    config.network = abs_network_path
    
    network = sibling_net.SiblingNetwork()
    target_shape_hw = config.image_size[::-1]
    network.image_batch_placeholder = tf.placeholder(tf.float32, shape=[None, target_shape_hw[0], target_shape_hw[1], 3], name='image_batch')
    network.initialize(config, num_classes)
    
    # --- 3. Define the Input and Output Tensors ---
    
    # --- THIS IS THE FIX ---
    # The frozen graph no longer has phase_train or keep_prob as inputs.
    # The only inputs are the image batch and the domain switch.
    input_tensor_names = [
        "image_batch:0",
        "switch_all:0"
    ]
    # --- END FIX ---
    
    # We only want the final embeddings
    output_tensor_names = [
        "embeddings_A:0",
        "embeddings_B:0"
    ]
    # We need the node names (without the ':0') for the freeze function
    output_node_names = [name.split(':')[0] for name in output_tensor_names]

    # 4. Start TensorFlow session
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=network.graph) as sess:
        
        # 5. Load your *FINE-TUNED* weights
        print(f"Finding latest checkpoint in: {FINETUNED_MODEL_DIR}")
        latest_checkpoint = tf.train.latest_checkpoint(FINETUNED_MODEL_DIR)
        if latest_checkpoint is None:
            print(f"FATAL ERROR: No checkpoint found in {FINETUNED_MODEL_DIR}")
            sys.exit(1)
            
        print(f"Loading weights from: {latest_checkpoint}")
        # We only restore trainable variables (to avoid optimizer state errors)
        restore_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        saver = tf.train.Saver(restore_vars)
        saver.restore(sess, latest_checkpoint)
        print("Model loaded successfully.")
        
        # 6. Freeze the graph
        # This converts all the learned weights (Variables) into constants
        print("Freezing graph (this may take a moment)...")
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            output_node_names # A list of the output node *names*
        )
        print("Graph frozen.")

    # 7. Convert the Frozen Graph to ONNX
    print("Converting frozen graph to ONNX...")
    try:
        # We re-create a simple graph just for the converter
        g = tf.Graph()
        with g.as_default():
            tf.import_graph_def(frozen_graph_def, name="")
        
        # Convert using tf2onnx
        # We specify the inputs and outputs by their full tensor names
        with tf.Session(graph=g):
            onnx_graph = tf2onnx.tfonnx.process_tf_graph(
                g,
                input_names=input_tensor_names,
                output_names=output_tensor_names,
                opset=11 # A good default
            )
            model_proto = onnx_graph.make_model("DocFaceModel")
            
            # Save the ONNX file
            with open(OUTPUT_ONNX_PATH, "wb") as f:
                f.write(model_proto.SerializeToString())

        print("\n--- Success! ---")
        print(f"Model saved to: {OUTPUT_ONNX_PATH}")
        
    except Exception as e:
        print(f"\nAn error occurred during ONNX conversion: {e}")
        print("This often happens if an operation is not supported by ONNX (opset 11).")

if __name__ == "__main__":
    main()