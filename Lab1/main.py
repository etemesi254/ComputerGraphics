import logging

import pandas as pd
from functions import *

pd.set_option('display.width', 1000)


def execute_single_df_operations(df: pd.DataFrame, file_name: str):
    emails = generate_emails(df)
    save_to_file(emails, file_name)
    (females, males) = get_genders(emails)
    logging.info(females)
    logging.info(males)
    get_special_characters(emails)
    #cluster_embeddings(females, males)
    return emails


if __name__ == "__main__":
    class_3b = load_sheet("./Test Files.xlsx", "3B")
    class_3c = load_sheet("./Test Files.xlsx", "3C")

    class_3b_refined = execute_single_df_operations(class_3b, "./outputs/3B_data")
    class_3c_refined = execute_single_df_operations(class_3c, "./outputs/3C_data")

    # concat both classes
    all_classes = pd.concat([class_3b_refined, class_3c_refined])
    # shuffle both classes
    shuffled_class = all_classes.sample(frac=1)
    save_files(shuffled_class, "./outputs/all_classes")
