# main.py (continued)
 # main.py
import os
import argparse
import yaml

def load_config(path='config/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Multi-Perspective Health Summarization")
    
    # Main arguments
    parser.add_argument("--mode", choices=["train", "inference"], required=True,   help="Mode of operation")
    
    # Training arguments
    train_group = parser.add_argument_group('Training arguments')
    train_group.add_argument("--train-classifier", action="store_true", 
                             help="Train the perspective classifier")
    train_group.add_argument("--train-llm", action="store_true", 
                             help="Fine-tune the LLM for summarization")
    
    # Inference arguments
    inference_group = parser.add_argument_group('Inference arguments')
    inference_group.add_argument("--input", 
                                help="Input file path for inference")
    inference_group.add_argument("--output", 
                                help="Output file path for inference results")
    inference_group.add_argument("--model-path", 
                                help="Path to fine-tuned model for inference")
    
    # Config arguments
    parser.add_argument("--config", default="config/config.yaml", 
                        help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Execute the appropriate mode
    if args.mode == "train":
        if args.train_classifier:
            from training.train_classifier import train_classifier
            train_classifier()
            
        if args.train_llm:
            from training.train_llm import train_llm
            train_llm()
            
    elif args.mode == "inference":
        if not args.input or not args.output:
            parser.error("--input and --output required for inference mode")
            
        from inference.generate_summary import generate_summaries
        generate_summaries(
            input_file=args.input,
            output_file=args.output,
            model_path=args.model_path
        )

if __name__ == "__main__":
    main()