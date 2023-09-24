import string
from typing import Tuple
import numpy as np
import pandas as pd
import re
import json
#from sklearn.metrics.pairwise import cosine_similarity
import pprint

# Model will be loaded here
model = None


def load_sheet(file_name: str, sheet_name: str) -> pd.DataFrame:
    """
    Load a excel sheet
    :param file_name:
    :param sheet_name:
    :return:
    """
    xls = pd.ExcelFile(file_name)
    df = pd.read_excel(xls, sheet_name)
    xls.close()
    return df


def generate_emails(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate emails from student names
    :param df: A pandas dataframe containing a "Student Name" column
    :return: A new dataframe with emails filled
    """
    names = df["Student Name"]
    email_list = []
    for name in names:
        full_name = name.split(" ")
        first_name = full_name[1].lower()
        last_name = full_name[0].lower()
        clean_last_name = re.sub("[^a-z]", "", last_name)
        email = first_name[0] + clean_last_name + "@gmail.com"
        # check for uniqueness
        if email in email_list:
            email = clean_last_name + first_name[0] + "@gmail.com"
        email_list.append(email)

    df["Email Address"] = email_list
    return df


def save_to_file(df: pd.DataFrame, path: str) -> None:
    """
    Save a file as csv and tsv format
    :param df: The dataframe for which we are saving data
    :param path: The path to file, should be the absolute path.
    :return: None
    """
    df.to_csv(f"{path}.csv")
    df.to_csv(f"{path}.tsv", sep="\t")


def get_genders(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns a filtered list of genders of a student class
    :param df:  A dataframe containing class details
    :return:
    """
    females = df[df["Gender"] == "F"]
    males = df[df["Gender"] == "M"]

    return females, males


def get_special_characters(df: pd.DataFrame):
    """
    Returns a list of people with special characters in their names
    :param df:
    :return:
    """
    # Use a regular expression to take the names, ignoring comas and whitespace
    suffix = re.compile("[^A-Za-z0-9,\\s].*")
    names = df["Student Name"]
    # We create a boolean with each containing fields that indicate whether
    # a row name contains special characters or not
    bool_condition = []
    for name in names:
        if suffix.search(name):
            bool_condition.append(True)
        else:
            bool_condition.append(False)

    df["special_chars"] = bool_condition


def save_files(df: pd.DataFrame, path_name: str):
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True)
    df.to_json(f"{path_name}.json", lines=True, orient='records')

    # write in jsonl format
    data = []
    for d in df.itertuples():
        first_scope = {"id": d[1], "student_number": d[2]}
        inner_scope = {"dob": d[4], "gender": d[7], "name_similar": "[no]",
                       "special_character": "[yes]" if d[8] is True else "[no]"}
        first_scope["additional_details"] = inner_scope
        data.append(first_scope)
    jsonl = json.dumps(data, indent=4)

    file = open(f"./{path_name}.jsonl", mode="w")
    file.write(jsonl)
    file.close()


def load_model():
    global model
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/LaBSE')


# Next extract embeddings for every name using a function, we return a list containing an array of all embeddings
def create_embeddings(df: pd.DataFrame):
    """
  Create embeddings based on student names
  :param df: Dataframe  containing student names

  :returns  A 2-D array of embeddings
  """
    names = df["Student Name"].to_list()
    name_embeddings = model.encode(names)
    return name_embeddings


def create_male_and_female_embeddings(all_classes: pd.DataFrame) -> np.ndarray:
    """
    Creates embeddings for males and female names extracted from a dataframe

    :params male_df: Male dataframe names
    :returns  embeddingd
    """
    # concatenate both classes
    # then create embeddings
    return create_embeddings(all_classes)


def create_similarity_matrix(embeddings: np.ndarray):
    """
  Create a similarity matrix that maps the similarity between i and j as
  similarity[i][j]

  :param embeddings: Embeddings for the class, should be a 2D array

  :returns: Similarity matrix, matrix[i][j] relates similarity between i and j
  """
    # desugar the dimensions, extracting the outer dimension,
    # the inner dimension is the same for all embeddings
    (outer_dim, inner_dim) = embeddings.shape
    # output array which will contain the mapping similarity[i][j]
    similarity_matrix = np.zeros((outer_dim, outer_dim))
    for i in range(outer_dim):
        for j in range(outer_dim):
            # callculate the cosine similarity
            # this is O(n^2) but can be simplified to run in half that time but mahn
            # I got a life (the trick is to notice that similarity[i][j] == similarity[j][i])
            cosine_sim = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))
            similarity_matrix[i][j] = cosine_sim
            if i == j:
                similarity_matrix[i][j] = 0.0

    return similarity_matrix


def slink(similarity_matrix: np.ndarray, df: pd.DataFrame):
    labels = [-1] * len(df)
    cluster_num = 0
    cutoff = 0.5
    for (i, names) in enumerate(df.iterrows()):
        max_value = similarity_matrix[i].argmax()
        # print(similarity_matrix[i][max])
        e = similarity_matrix[i][max_value]
        similarity_matrix[i][max_value] = 0.0

        if e > cutoff:
            # print("\n")
            # expand cluster
            lowest_disim = e
            row = similarity_matrix[i]
            # whatever value we were in update it to 10 to prevent it from recurring
            similarity_matrix[i][i] = 0.0
            labels[i] = int(cluster_num)
            labels[max_value] = int(cluster_num)
            while lowest_disim > cutoff:
                lowest_dissim = -1
                # we have a row of dissimilarity matrices, let's find the next
                # cluster with the minimum dissimilarity
                pos = similarity_matrix[i].argmax()
                x = row[pos]
                if x < cutoff:
                    break
                # before we merge this, we need to check if it has a nearer row
                # cluster than this so fetch this row again
                m = similarity_matrix[pos]
                min_row = m.argmax()

                n = m[min_row]
                # check if this row is the most similar to the top row
                if min_row == i and m[min_row] > cutoff:
                    labels[pos] = int(cluster_num)
                    # update similarity matrix at i
                    similarity_matrix[pos][min_row] = 0.0
                    lowest_disim = n
                # indicate we saved this
                row[pos] = 0.0
            row[i] = 0.0
            cluster_num += 1
    # We finally have labels.
    # so create clusters
    m = {}
    max_value = -1
    negative_counters = max(labels) + 1
    for (j, i) in enumerate(labels):
        if i == -1:
            m[negative_counters] = [df.iloc[j]["Student Name"]]
            negative_counters += 1
        else:
            if i not in m:
                m[i] = []
            m[i].append(df.iloc[j]["Student Name"])
    pprint.pprint(m)


import pprint


def slink(similarity_matrix: np.ndarray, df: pd.DataFrame):
    labels = [-1] * len(df)
    cluster_num = 0
    cutoff = 0.5
    for (i, names) in enumerate(df.iterrows()):
        max_value = similarity_matrix[i].argmax()
        # print(similarity_matrix[i][max])
        e = similarity_matrix[i][max_value]
        similarity_matrix[i][max_value] = 0.0

        if e > cutoff:
            print(df.iloc[i]["Student Name"], df.iloc[max_value]["Student Name"], e)
            # print("\n")
            # expand cluster
            lowest_disim = e
            row = similarity_matrix[i]
            # whatever value we were in update it to 10 to prevent it from recurring
            similarity_matrix[i][i] = 0.0
            labels[i] = int(cluster_num)
            labels[max_value] = int(cluster_num)
            while lowest_disim > cutoff:
                lowest_dissim = -1
                # we have a row of dissimilarity matrices, let's find the next
                # cluster with the minimum dissimilarity
                pos = similarity_matrix[i].argmax()
                x = row[pos]
                if x < cutoff:
                    break
                # before we merge this, we need to check if it has a nearer row
                # cluster than this so fetch this row again
                m = similarity_matrix[pos]
                min_row = m.argmax()

                n = m[min_row]
                # check if this row is the most similar to the top row
                if min_row == i and m[min_row] > cutoff:
                    labels[pos] = int(cluster_num)
                    # update similarity matrix at i
                    similarity_matrix[pos][min_row] = 0.0
                    lowest_disim = n
                # indicate we saved this
                row[pos] = 0.0
            row[i] = 0.0
            cluster_num += 1
    # We finally have labels.
    # so create clusters
    m = {}
    print(labels)
    max_value = -1
    negative_counters = max(labels) + 1
    print(negative_counters)
    for (j, i) in enumerate(labels):
        if i == -1:
            m[negative_counters] = [df.iloc[j]["Student Name"]]
            negative_counters += 1
        else:
            if i not in m:
                m[i] = []
            m[i].append(df.iloc[j]["Student Name"])
    pprint.pprint(m)


def cluster_embeddings(male_df: pd.DataFrame, female_df: pd.DataFrame):
    # get embeddings
    all_classes = pd.concat([male_df, female_df])
    embeddings = create_male_and_female_embeddings(all_classes)
    matrix = create_similarity_matrix(embeddings)
    # next is to threshold names based on the matrix
    slink(matrix, all_classes)
