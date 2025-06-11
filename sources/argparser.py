
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="SAUTE.")
    subparsers = parser.add_subparsers(required=True, dest="action")

    # Training parameters
    train_parser = subparsers.add_parser('train', help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=1,
                              help="Number of epochs")
    train_parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'gelu'],
                              help='Activation function')
    train_parser.add_argument('--layers', type=int, default=1,
                              help='Number of layers')

    # Inference
    inference_parser = subparsers.add_parser('inference', help="Inference the model")
    inference_parser.add_argument("--filepath", type=str, help="Input Filepath dialog", required=True)
    inference_parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for random sampling")
    inference_parser.add_argument("--random-sampling", required=False, default=False, action="store_true")

    parser.add_argument('--datasets', type=str, default="allenai/soda",
                        help='Dataset Reposity HugginFace')
    
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                        help='Device to train on')

    return parser.parse_args()