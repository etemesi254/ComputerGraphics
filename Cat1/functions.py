import json
import logging
import os.path
from typing import Dict, List, Any

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


def generate_file(en_file: str, ref_files: List[str]):
    with open(en_file) as op:
        # read the reference file here
        en_pd = read_jsonl_input(opened_file=op)
        logging.info(f"Loaded the reference EN file at {en_file}")
        for ref_file in ref_files:
            if os.path.samefile(en_file, ref_file):
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
                name = "./en-" + locale + ".xlsx"
                logging.info(f"Saving to name {name}\n")
                processed_df.to_excel(name, engine="openpyxl")


def walk_directory(en_file: str, in_dir: str):
    files = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f))]
    generate_file(en_file, files)
