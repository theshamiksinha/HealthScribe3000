from data.data_utils import load_dataset, load_config, save_predictions_to_json
from pipelines.llm_pipeline import train_or_load_summariser, generate_summaries
from pipelines.perspective_pipeline import train_or_load_classifier, predict_perspectives


def main(testing_size: int = None) -> None:
    config = load_config()

    print("\n===== STEP 1: TRAINING/LOADING PERSPECTIVE CLASSIFIER =====")
    classifier_model, classifier_tokenizer = train_or_load_classifier(config)

    print("\n===== STEP 2: PREDICTING PERSPECTIVES ON TEST SET =====")
    test_data = load_dataset(config["data"]["test_path"])
    if testing_size is not None:
        test_data = test_data[:testing_size]

    predicted_test_data = predict_perspectives(classifier_model, test_data, config)
    save_predictions_to_json(predicted_test_data)

    if testing_size is not None:
        print("Predicted Perspectives:")
        for result in predicted_test_data:
            print(f"Question: {result['question']} \n\n Perspectives: {result["predicted_perspectives"]}")

    print("\n===== STEP 3: TRAINING/LOADING LLM FOR SUMMARIZATION =====")
    summariser_model, summariser_tokenizer = train_or_load_summariser(config)

    print("\n===== STEP 4: GENERATING PERSPECTIVE-WISE SUMMARIES =====")
    generate_summaries(summariser_model, summariser_tokenizer, predicted_test_data, config)


if __name__ == "__main__":
    main(testing_size=1)
