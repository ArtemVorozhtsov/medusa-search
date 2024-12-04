"""
Script is used to rename spectra filenames in reverse indexes based on hrms database path
For example, if you moved hrms_database_path to the new place and don't want to perform new time consuming indexation,
you can just run this script. The values, which contain mass spectra paths will be changed automatically, based on new hrms database path and the previous one.
"""

import os
import pickle as pkl
from argparse import ArgumentParser

from tqdm import tqdm


def rename_values(v, previous_db_path, new_db_path):
    """rename dict_values based on previous_db_path and new_db_path"""

    new_v = []
    for filepath in v:
        new_v.append(new_db_path + filepath.split(previous_db_path)[1])

    return new_v


def process_batch_dict(batch_dict, previous_db_path, new_db_path):
    """changes previous_db_path into new_db_path in batch_dict values"""

    for k, v in batch_dict.items():
        new_v = rename_values(v, previous_db_path, new_db_path)
        batch_dict[k] = new_v

    return batch_dict


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--index_dictionary_path", type=str, help="Path to the index dictionary"
    )
    parser.add_argument(
        "--new_index_dictionary_path",
        type=str,
        help="Path to the new index dictionary. If new_index_dictionary_path is 0, reindexation is made inplace.",
    )
    parser.add_argument(
        "--previous_db_path", type=str, help="Path to the previous database path"
    )
    parser.add_argument("--new_db_path", type=str, help="Path to the new database path")
    args = parser.parse_args()

    path_to_index_dict = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), args.index_dictionary_path
    )

    if args.new_index_dictionary_path != "0":
        path_to_new_index_dict = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), args.new_index_dictionary_path
        )
    batch_names = os.listdir(path_to_index_dict)

    for batch_name in tqdm(batch_names):
        with open(os.path.join(path_to_index_dict, batch_name), "rb") as f:
            batch_dict = pkl.load(f)

        new_batch_dict = process_batch_dict(
            batch_dict, args.previous_db_path, args.new_db_path
        )

        if args.new_index_dictionary_path == "0":
            with open(os.path.join(path_to_index_dict, batch_name), "wb") as f:
                pkl.dump(new_batch_dict, f)

        else:
            with open(os.path.join(path_to_new_index_dict, batch_name), "wb") as f:
                pkl.dump(new_batch_dict, f)
