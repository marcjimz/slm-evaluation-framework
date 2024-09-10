import os
import pickle
from typing import Any, Union
import json
import pandas as pd
import random
from collections import Counter

from my_utils.ml_logging import get_logger

# Set up logging
logger = get_logger()


def save_dataframe(
    df: pd.DataFrame, path: Union[str, pd._typing.PathLike], file_format: str = "csv"
) -> None:
    """
    Save the given dataframe to the specified path in the desired format.

    :param df: Input DataFrame.
    :param path: The path where the dataframe should be saved.
    :param file_format: The format in which the dataframe should be saved. Default is 'csv'.
    :return: None
    :raises ValueError: If the specified file format is unsupported.
    """
    try:
        if file_format == "csv":
            df.to_csv(path, index=False)
            logger.info(f"DataFrame saved successfully at {path} in CSV format.")
        elif file_format == "excel":
            df.to_excel(path, index=False)
            logger.info(f"DataFrame saved successfully at {path} in Excel format.")
        elif file_format == "parquet":
            df.to_parquet(path, index=False)
            logger.info(f"DataFrame saved successfully at {path} in Parquet format.")
        elif file_format == "feather":
            df.to_feather(path)
            logger.info(f"DataFrame saved successfully at {path} in Feather format.")
        else:
            raise ValueError(
                f"Unsupported file format: {file_format}. Supported formats are: ['csv', 'excel', 'parquet', 'feather']."
            )
    except Exception as e:
        logger.error(f"Error while saving DataFrame: {e}")
        raise


def save_model_to_pickle(estimator: Any, file_path: str) -> None:  # nosec
    """
    Save a trained model to the specified file using Pickle.

    :param estimator: The trained model to save.
    :param file_path: The path where the model will be saved.
    :return: None.
    :raises Exception: If an error occurs while saving the model.
    """
    try:
        # Ensure the directory exists; if not, create it.
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        logger.info(f"Saving model to {file_path}.")
        with open(file_path, "wb") as file:
            pickle.dump(estimator, file)  # nosec
        logger.info(f"Model saved successfully to {file_path}.")
    except Exception as e:
        logger.error(f"Error occurred while saving model to {file_path}: {e}")
        raise e


def load_model_from_pickle(file_path: str) -> Any:
    """
    Load a trained model from the specified file using Pickle.

    :param file_path: The path from where the model is loaded.
    :return: Loaded model.
    :raises Exception: If an error occurs while loading the model.
    """
    try:
        logger.info(f"Loading model from {file_path}.")
        with open(file_path, "rb") as file:
            estimator = pickle.load(file)  # nosec
        logger.info(f"Model loaded successfully from {file_path}.")
        return estimator
    except Exception as e:
        logger.error(f"Error occurred while loading model from {file_path}: {e}")
        raise e


def load_dataframe_from_path(path: str) -> pd.DataFrame:
    """
    Load a dataframe from the specified path based on the file's extension.

    :param path: The path from where the dataframe should be loaded.
    :return: Loaded DataFrame.
    :raises ValueError: If the file format (determined by its extension) is unsupported.
    """
    _, file_extension = os.path.splitext(path)
    try:
        logger.info(f"Loading DataFrame from {path}.")
        if file_extension == ".csv":
            df = pd.read_csv(path)
        elif file_extension == ".excel" or file_extension == ".xlsx":
            df = pd.read_excel(path)
        elif file_extension == ".parquet":
            df = pd.read_parquet(path)
        elif file_extension == ".feather":
            df = pd.read_feather(path)
        else:
            raise ValueError(
                f"""Unsupported file format: {file_extension}. Supported formats are:
                ['.csv', '.excel', '.parquet', '.feather']."""
            )
        logger.info(f"DataFrame loaded successfully from {path}.")
        return df
    except Exception as e:
        logger.error(f"Error occurred while loading DataFrame from {path}: {e}")
        raise e


def resolve_python_object(path_str):
    module_str, obj_str = path_str.rsplit(".", 1)
    module = __import__(module_str, fromlist=[obj_str])
    return getattr(module, obj_str)

def csv_to_jsonl_with_mapping(source_csv, output_jsonl, key_mapping):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(source_csv)
    
    # Open the output file in write mode
    with open(output_jsonl, 'w') as jsonl_file:
        # Iterate through each row in the DataFrame
        for _, row in df.iterrows():
            # Create a new dictionary based on the key mapping
            mapped_row = {output_key: row[input_key] for input_key, output_key in key_mapping.items() if input_key in row}
            
            # Convert the mapped dictionary to a JSON string
            json_str = json.dumps(mapped_row)
            # Write the JSON string to the file, followed by a newline
            jsonl_file.write(json_str + '\n')

def split_dataset(file_path, field_to_optimize, training_split=0.7, incremental_split=0.2, random_seed=None):
    # Load data from the file using built-in open function
    data = []
    with open(file_path, 'r') as reader:
        for line in reader:
            obj = json.loads(line.strip())  # Parse each line as a JSON object
            data.append(obj)

    # Set random seed for reproducibility if provided
    if random_seed is not None:
        random.seed(random_seed)

    # Sort or shuffle based on the field to optimize
    field_counts = Counter([item[field_to_optimize] for item in data])
    data_sorted = sorted(data, key=lambda x: field_counts[x[field_to_optimize]], reverse=True)

    # Shuffle the sorted data to randomize within the optimized field
    random.shuffle(data_sorted)

    # Calculate the split indices
    total_size = len(data_sorted)
    training_size = int(total_size * training_split)
    incremental_size = int(total_size * incremental_split)

    # Split data into training, incremental training, and validation
    training_dataset = data_sorted[:training_size]
    incremental_dataset = data_sorted[training_size:training_size + incremental_size]
    validation_dataset = data_sorted[training_size + incremental_size:]

    # Function to print stats for each dataset
    def print_dataset_stats(dataset, dataset_name):
        field_count = Counter([item[field_to_optimize] for item in dataset])
        print(f"--- {dataset_name} Stats ---")
        for field, count in field_count.items():
            print(f"{field}: {count} instances")
        print(f"Total: {len(dataset)}\n")

    # Print stats for each dataset
    print_dataset_stats(training_dataset, "Training Dataset")
    print_dataset_stats(incremental_dataset, "Incremental Training Dataset")
    print_dataset_stats(validation_dataset, "Validation Dataset")

    return training_dataset, incremental_dataset, validation_dataset

def generate_nshot_prompt(prompt, examples, new_example, n=0):
    prompt += "\n\n"
    counter = 0
    for i, example in enumerate(examples, start=1):
        if n != 0 and counter < n:
            # print("Adding example to prompt.")
            prompt += f"Example {i}:\n"
            prompt += f"Question: {example['question']}\n"
            prompt += f"Answer: {example['answer']}\n"
            prompt += f"General Context: {example['general_context']}\n"
            prompt += f"Specialized Context: {example['specialized_context']}\n"
            prompt += f"Evaluation: {example.get('evaluation', 'Provide a human-like evaluation for this pair.')}\n\n"
            # print("Skipping as reached max n-shot examples.")
        counter+=1
        
    prompt += "Now, evaluate the following new question-answer-context pair:\n"
    prompt += f"Question: {new_example['question']}\n"
    prompt += f"Answer: {new_example['answer']}\n"
    prompt += f"General Context: {new_example['general_context']}\n"
    prompt += f"Specialized Context: {new_example['specialized_context']}\n"
    prompt += "Evaluation:"

    return prompt

def generate_prompt(new_example):
    prompt = f"Question: {new_example['question']}\n"
    prompt += f"Answer: {new_example['answer']}\n"
    prompt += f"General Context: {new_example['general_context']}\n"
    prompt += f"Specialized Context: {new_example['specialized_context']}\n"
    prompt += "Evaluation:"

    return prompt

def convert_to_jsonl_in_memory(data):
    jsonl_data = ""
    for item in data:
        jsonl_data += json.dumps(item) + "\n"
    return jsonl_data