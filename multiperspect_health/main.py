# main.py

from modules.perspective_pipeline import train_or_load_classifier, predict_perspectives
from modules.llm_pipeline import train_or_load_summariser, generate_summaries
from data.data_utils import load_dataset, load_config, save_predictions_to_json

def main():
    config = load_config()

    print("\n===== STEP 1: TRAINING/LOADING PERSPECTIVE CLASSIFIER =====")
    classifier_model, classifier_tokenizer = train_or_load_classifier(config)

    print("\n===== STEP 2: PREDICTING PERSPECTIVES ON TEST SET =====")
    test_data = load_dataset(config["data"]["test_path"])
    predicted_test_data = predict_perspectives(classifier_model, classifier_tokenizer, test_data, config)  
    save_predictions_to_json(predicted_test_data)

    print("\n===== STEP 3: TRAINING/LOADING LLM FOR SUMMARISATION =====")
    summariser_model, summariser_tokenizer = train_or_load_summariser(config)

    print("\n===== STEP 4: GENERATING PERSPECTIVE-WISE SUMMARIES =====")
    generate_summaries(summariser_model, summariser_tokenizer, predicted_test_data, config)

if __name__ == "__main__":
    main()
