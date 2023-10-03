import json
import logging
import os.path
import pprint
from typing import Dict, List, Any, Tuple

import pandas as pd


def read_jsonl_input(opened_file) -> pd.DataFrame:
    """
    Read a json lines input and map it to a pandas dataframe
    :param opened_file: A python read object

    :return: pandas dataframe containing the loaded data
    """
    # temporary output, the format is similar to how a dataframe represents
    # data
    df_dict: Dict[str, List[Any]] = {}
    # read the file contents line by line
    for line in opened_file.readlines():
        # convert it to a python dictionary
        json_load = json.loads(line)
        for (key, value) in json_load.items():
            if key not in df_dict:
                df_dict[key] = []
            df_dict[key].append(value)
    df = pd.DataFrame(df_dict)
    return df


def process_two_datasets(ref_df: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame | None:
    """
    Process two datasets, combining them to one dataset with combined rows from
    datasets
    :param ref_df: Reference dataframe, this is the en-EN dataframe
    :param other: Other dataframe, can be any type of en-XX dataframe
    :return: A combined dataframe with the following columns [id,partition,intent,en_utt,en_annot_utt,xx_utt,xx_annot_utt]
    'xx' is the locale of the other dataframe
    """
    if len(other) == 0:
        # we need at least one row to extract locale
        # so if we have less just quit
        return None

    # format is language-COUNTRY
    other_locale: str = other.iloc[0]["locale"]
    locale = other_locale.split("-")[0]  # extracts 'language'
    # rename other columns
    other.rename(columns={"utt": locale + "_utt", "annot_utt": locale + "_annot_utt"}, inplace=True)
    # select interesting columns from each dataframe
    other_interesting = other[[locale + "_utt", locale + "_annot_utt"]]
    ref_interesting = ref_df[["id", "partition", "intent", "utt", "annot_utt"]]
    # for reference dataframe use `en` as prefix for the utterances columns
    ref_interesting = ref_interesting.rename(columns={"utt": "en_utt", "annot_utt": "en_annot_utt"})
    # the index is messed when we concat the two dataframes creating Nan values, this fixes it
    ref_interesting.reset_index(inplace=True, drop=True)
    other_interesting.reset_index(inplace=True, drop=True)
    # create the new df with the two combined rows
    new_df = pd.concat([ref_interesting, other_interesting], axis=1)
    new_df.reset_index(inplace=True, drop=True)
    return new_df


def generate_pivot_file(input_directory: str, output_dir: str):
    en_file = "en-US.jsonl"
    en_path = os.path.join(input_directory, en_file)
    if not os.path.exists(os.path.join(input_directory, en_file)):
        logging.error(f"The path {en_path} doesn't exist, cannot continue anymore")
        return
    if not os.path.isdir(output_dir):
        logging.error(f"Path {output_dir} doesn't exist")
        return
    files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if os.path.isfile(f)]

    with open(en_path) as op:
        # read the reference file here
        en_pd = read_jsonl_input(opened_file=op)
        logging.info(f"Loaded the reference EN file at {en_path}\n")
        for ref_file in files:
            if os.path.samefile(en_path, ref_file):
                # don't generate en-en mappings
                continue
            with open(ref_file) as opened_ref:
                # read other mappings
                mapping_df = read_jsonl_input(opened_ref)
                logging.info(f"Loaded file {ref_file} as dataframe")
                if len(mapping_df) == 0:
                    raise Exception("The dataframe doesn't contain any rows")
                processed_df: pd.DataFrame | None = process_two_datasets(en_pd, mapping_df)
                if processed_df is None:
                    raise Exception()
                # extract locale from the first df row
                other_locale: str = mapping_df.iloc[0]["locale"]
                # ensure the datatype is string
                assert isinstance(other_locale, str)
                locale = other_locale.split("-")[0]
                # finally write the data
                # create a new directory for the output
                os.makedirs("./outputs/task1", exist_ok=True)
                name = "./outputs/task1/en-" + locale + ".xlsx"
                logging.info(f"Saving to name {name}\n")
                processed_df.to_excel(name, engine="openpyxl")


def walk_directory(ref_file: str, in_dir: str):
    """
    Walk a directory containing massive dataset and generate a ref-xx pivot sheet

    param ref_file: Reference file
    param in_dir: A directory containing the massive dataset
    """
    # iterate to get all files into a list
    files = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f))]
    generate_pivot_file(ref_file, files)


def separate_jsonl(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dev: pd.DataFrame = df[df["partition"] == "dev"]
    test: pd.DataFrame = df[df["partition"] == "test"]
    train: pd.DataFrame = df[df["partition"] == "train"]

    return train, test, dev


def partition_and_save(df: pd.DataFrame, out_file_name: str):
    (train, test, dev) = separate_jsonl(df)
    # extract locale from the first df row
    other_locale: str = df.iloc[0]["locale"]
    # ensure the datatype is string
    assert isinstance(other_locale, str)
    locale = other_locale.split("-")[0]
    train.to_json(out_file_name + f"/{locale}_train.jsonl", orient="records", lines=True)
    test.to_json(out_file_name + f"/{locale}_test.jsonl", orient="records", lines=True)
    dev.to_json(out_file_name + f"/{locale}_dev.jsonl", orient="records", lines=True)

    # we return train test because it's used for the next shit
    return train


def start_w3(en, sw, dw):
    path_name = "./outputs/task2/"
    en_df = read_jsonl_input(en)
    sw_df = read_jsonl_input(sw)
    de_df = read_jsonl_input(dw)
    os.makedirs(path_name, exist_ok=True)
    train_en = partition_and_save(en_df, path_name)
    train_sw = partition_and_save(sw_df, path_name)
    train_df = partition_and_save(de_df, path_name)

    # drop indices
    train_en.reset_index(drop=True, inplace=True)
    train_sw.reset_index(drop=True, inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    # combine
    train_data = pd.concat([train_en, train_sw, train_df])
    # drop all un-needed columns
    train_data = train_data[["id", "utt"]]
    # reset indices since the concatenated dataframes have duplicate indices
    train_data.reset_index(drop=True, inplace=True)
    train_json = []
    for each_line in train_data.itertuples():
        record = {"id": each_line[1], "utt": each_line[2]}
        train_json.append(record)
    string_json = json.dumps(train_json, indent=4, ensure_ascii=False)
    print(string_json)
