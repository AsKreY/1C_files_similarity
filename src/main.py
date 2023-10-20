#!/usr/bin/python
import argparse
import multiprocessing as mp
import numpy as np
from pathlib import Path


def damerau_levenshtein_distance(s1, s2):
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    d = np.zeros((lenstr1 + 1, lenstr2 + 1))

    for i in range(lenstr1 + 1):
        for j in range(lenstr2 + 1):
            if i > 1 and j > 1 and\
               s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                              d[i - 1][j - 1] +
                              (1 if s1[i - 1] != s2[j - 1] else 0),
                              d[i - 2][j - 2] + 1)
            if i > 0 and j > 0:
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                              d[i - 1][j - 1] +
                              (1 if s1[i - 1] != s2[j - 1] else 0))
            elif i > 0:
                d[i][j] = d[i - 1][j] + 1
            elif j > 0:
                d[i][j] = d[i][j - 1] + 1
            else:
                d[i][j] = 0
    return d[lenstr1][lenstr2] / max(lenstr1, lenstr2)


class BinaryComparator:
    danger_score = 0.33  # Danger percent of similarity

    def __init__(self, metric):
        self.__metric = metric

    def __load_files(self, first_filename, second_filename):
        """Load files from filenames and parse them"""
        with open(first_filename, "rb") as first_file, \
                open(second_filename, "rb") as second_file:
            self.first_byte_code = first_file.read()
            self.second_byte_code = second_file.read()

    def Compare(self, first_filename, second_filename):
        first_file_size = first_filename.stat().st_size
        second_file_size = second_filename.stat().st_size
        if first_file_size / second_file_size < self.danger_score or \
           second_file_size / first_file_size < self.danger_score:
            return 0
        self.__load_files(first_filename, second_filename)
        return (first_filename, second_filename,
                round(1 - self.__metric(self.first_byte_code,
                                        self.second_byte_code), 2))


def worker(line):
    """Compare two filenames from one line"""
    comparator = BinaryComparator(damerau_levenshtein_distance)
    return comparator.Compare(*line)


def writing_results(out_filename, results, first_path, second_path):
    score = BinaryComparator.danger_score
    with open(out_filename, "w") as w:
        w.write("Идентичные файлы:\n")
        for result in results:
            if result[2] == 1:
                w.write("{}, {}\n".format(result[0].parts[-1],
                                          result[1].parts[-1]))

        w.write("Схожие файлы:\n")
        for result in results:
            if result[2] > score and result[2] != 1:
                w.write("{}, {}, {}%\n".format(result[0].parts[-1],
                                               result[1].parts[-1],
                                               result[2] * 100))

        w.write("Отсутсвующие файлы:\n")
        for first_dir_file in first_path.rglob("*"):
            has_similar = False
            for result in results:
                if result[0] == first_dir_file and result[2] > score:
                    has_similar = True
                    break
            if not has_similar:
                w.write("First directory: {}\n".format(first_dir_file))

        for second_dir_file in second_path.rglob("*"):
            has_similar = False
            for result in results:
                if result[1] == second_dir_file and result[2] > score:
                    has_similar = True
                    break
            if not has_similar:
                w.write("Second directory: {}\n".format(second_dir_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compares python files and produces similarity")
    parser.add_argument("first_folder")
    parser.add_argument("second_folder")
    parser.add_argument("score")
    parser.add_argument("results")
    args = parser.parse_args()

    BinaryComparator.danger_score = float(args.score)
    filename_pairs = []

    first_path = Path(args.first_folder)
    second_path = Path(args.second_folder)
    for first_dir_file in first_path.rglob("*"):
        for second_dir_file in second_path.rglob("*"):
            filename_pairs.append((first_dir_file, second_dir_file))

    with mp.Pool(mp.cpu_count()) as p:
        results = p.map(worker, filename_pairs)

    writing_results(args.results, results, first_path, second_path)
