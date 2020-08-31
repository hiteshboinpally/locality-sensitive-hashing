import pandas as pd
import random
import matplotlib.pyplot as plt
import time


class LSH:
    def __init__(self, data, shingle_length, permutations, num_rows_per_band, num_buckets, calc_jaccard=False):
        """
        An Locality Sensitive Hashing program that places similar documents in the same hash bucket with high
        probability and dissimilar documents in different buckets with high probability.

        :param data: a list of documents or genomes
        :param shingle_length: the length of each gram/shingle of the document/data point
        :param permutations: the number of permutations that will be used in the min-Hash Function
        :param num_rows_per_band: the number of rows in the band where we will hash column values into
        :param num_buckets: the number of buckets in each band
        """
        self.shingle_length = shingle_length
        self.num_permutations = permutations
        self.num_rows_per_band = num_rows_per_band
        self.num_buckets = num_buckets
        self.num_documents = len(data)

        start_time = time.time()
        shingles_document = self.shingling(data)

        if calc_jaccard:
            self.jaccard_similarity = self.jaccard(shingles_document)
            self.jaccard_time = time.time()

        sig_matrix = self.min_hash(shingles_document)
        similar_documents = self.lsh(sig_matrix)
        self.sim_docs_set = set()
        self.get_set_of_sim_docs(similar_documents)

        self.time_taken = time.time() - start_time


    def get_runtime(self):
        """
        Calculates runtime of program

        :return: runtime of program in seconds
        """
        return self.time_taken


    def get_set_of_sim_docs(self, similar_documents):
        """
        Finds the similar documents within a the original list of documents

        :param similar_documents: list of similar documents or genomes
        """
        for set_docs in similar_documents:
            if len(set_docs) > 1:
                self.sim_docs_set.add(frozenset(set_docs))


    def shingling(self, data):
        """
        Creates k length shingles for all the documents in the original data file

        :param data: list of documents or genomes
        :return: a list of all the shingles in a given document
        """
        shingles = set()
        data_as_shingles = []
        for data_file in data:
            for i in range(len(data_file) - self.shingle_length):
                shingles.add(data_file[i:i+self.shingle_length])
        shingles_document = []
        for shingle in shingles:
            document_in_current_shingle = []
            for data_file in data:
                if shingle in data_file:
                    document_in_current_shingle.append(1)
                else:
                    document_in_current_shingle.append(0)
            shingles_document.append(document_in_current_shingle)
        return shingles_document


    def jaccard(self, shingles_document):
        """
        Calculates the Jaccard Index in a given document, or the intersection of all shingles divided by the
        union of all shingles in a document or genome.

        :param shingles_document: a matrix where the rows are the shingles and the columns are the different files
        :return: a float: that represents the Jaccard Index of a given document or genome
        """
        intersection_count = 0
        union_count = len(shingles_document)
        for i in range(len(shingles_document)):
            if shingles_document[i][0] == 1 and shingles_document[i][1] == 1:
                intersection_count += 1
        return float(intersection_count) / float(union_count)


    def min_hash(self, shingles_document):
        """
        Calculates and returns the min hash signature matrix

        :param shingles_document: a matrix where the rows are the shingles and the columns are the different files
        :return signature matrix: a matrix where the rows represent different files and if the columns are similar
                                    there is a high probability that the documents are similar too
        """
        num_shingles = len(shingles_document)
        perm_indices = list(range(num_shingles))
        signature_matrix = []
        for i in range(self.num_permutations):
            random.shuffle(perm_indices)
            signature_matrix_row = [-1] * self.num_documents
            for j in range(len(perm_indices)):
                index = perm_indices.index(j)
                row = shingles_document[index]
                k = 0
                for document in row:
                    if signature_matrix_row[k] == -1 and document == 1:
                        signature_matrix_row[k] = index
                    k += 1
            signature_matrix.append(signature_matrix_row)
        return signature_matrix


    def lsh(self, sig_matrix):
        """
        Returns a list of sets of similar documents or genomes from the given data set

        :param sig_matrix: a matrix where the rows represent different files and if the columns are similar
                                    there is a high probability that the documents are similar too
        :return: a list: that contains sets of similar documents
        """
        buckets = {}
        similar_documents = []
        for i in range(self.num_buckets):
            similar_documents.append(set())
        for i in range(0,self.num_permutations,self.num_rows_per_band):
            buckets.clear()
            current_rows = sig_matrix[i:i+self.num_rows_per_band]
            for j in range(self.num_documents):
                values_in_band = tuple([row[j] for row in current_rows])
                bucket_in = hash(values_in_band) % self.num_buckets
                if bucket_in not in buckets:
                    buckets[bucket_in] = set()
                buckets[bucket_in].add(j)
            for index,docs in buckets.items():
                if len(docs) > 1:
                    docs = frozenset(docs)
                    similar_documents[index].update(docs)
        return similar_documents


    def is_similar(self, file1, file2):
        """
        Determines if two documents are similar
        :param file1: Document number 1 that will be compared
        :param file2: Document number 2 that will be compared
        :return: a boolean: true if the file 1 and file 2 are similar, otherwise false
        """
        for sim_docs in self.sim_docs_set:
            if file1 in sim_docs and file2 in sim_docs:
                return True
        return False


def permutations_vs_jaccard(data, shingle_length, rows_per_band, buckets, max_perms=15, num_trials=10):
    """
    plots a graph where the number of permutation is x-axis and the jaccard index is the y-axis

    :param data: a list of documents or genomes
    :param shingle_length: the length of each gram/shingle of the document/data point
    :param max_perms: the maximum number of permutations that will be used in the min-Hash Function
    :param rows_per_band: the number of rows in the band where we will hash column values into
    :param num_trials: the trials that will be conducted
    :return: a plot: where the number of permutation is x-axis and the jaccard index is the y-axis
    """
    file_one_idx = random.randint(0,len(data))
    file_two_idx = file_one_idx
    while file_two_idx == file_one_idx:
        file_two_idx = random.randint(0,len(data))
    file_one = data[file_one_idx]
    file_two = data[file_two_idx]
    test_files = [file_one, file_two]
    jaccard_lsh = LSH(test_files, shingle_length, 1, rows_per_band, buckets, True)
    jaccard_similarity = jaccard_lsh.jaccard_similarity
    min_hash_similarities = []
    for i in range(0, max_perms, rows_per_band):
        print("calculating permuation", i)
        similarity_ct = 0
        for j in range(num_trials):
            curr_lsh = LSH(data, shingle_length, i, rows_per_band, buckets)
            if curr_lsh.is_similar(file_one_idx, file_two_idx):
                similarity_ct += 1
        min_hash_similarities.append(similarity_ct / num_trials)

    permutations = list(range(0, max_perms, rows_per_band))
    jaccard_sims = [jaccard_similarity] * len(permutations)

    plt.plot(permutations, min_hash_similarities, "ro", label='min hash')
    plt.plot(permutations, jaccard_sims, "b--", label='jaccard')
    plt.legend(loc='lower right')
    plt.xlabel('Number of Bands')
    plt.ylabel('Percentage of Similarity')
    plt.title('Number of Bands vs Similarity Percentage')
    plt.savefig('ra-plots/perms_vs_jaccard.png')


def rows_vs_jaccard(data, shingle_length, max_rows_per_band, buckets, num_trials=10):
    """
    plots a graph where the number of permutation is x-axis and the jaccard index is the y-axis

    :param data: a list of documents or genomes
    :param shingle_length: the length of each gram/shingle of the document/data point
    :param buckets: the number of buckets in each band
    :param max_rows_per_band: the maximum number of rows in the band where we will hash column values into
    :param num_trials: the trials that will be conducted
    :return: a plot: where the number of permutation is x-axis and the jaccard index is the y-axis
    """
    file_one_idx = random.randint(0,len(data))
    file_two_idx = file_one_idx
    while file_two_idx == file_one_idx:
        file_two_idx = random.randint(0,len(data))
    file_one = data[file_one_idx]
    file_two = data[file_two_idx]
    test_files = [file_one, file_two]
    jaccard_lsh = LSH(test_files, shingle_length, 1, 1, 1, True)
    jaccard_similarity = jaccard_lsh.jaccard_similarity
    min_hash_similarities = []
    for i in range(1, max_rows_per_band):
        print("calculating rows per band", i)
        similarity_ct = 0
        for j in range(num_trials):
            curr_lsh = LSH(data, shingle_length, i * 5, i, buckets)
            if curr_lsh.is_similar(file_one_idx, file_two_idx):
                similarity_ct += 1
        min_hash_similarities.append(similarity_ct / num_trials)

    permutations = list(range(1, max_rows_per_band))
    jaccard_sims = [jaccard_similarity] * len(permutations)

    plt.plot(permutations, min_hash_similarities, "ro", label='min hash')
    plt.plot(permutations, jaccard_sims, "b--", label='jaccard')
    plt.legend(loc='lower right')
    plt.xlabel('Number of Rows per Band')
    plt.ylabel('Percentage of Similarity')
    plt.title('Rows per Band vs Similarity Percentage')
    plt.savefig('ra-plots/rows_vs_jaccard.png')


"""
def document_ct_vs_runtime(data, shingle_length, permutations, rows_per_band, buckets, num_trials=10):

    lsh_times = []
    for i in range(1, len(data)):
        print('document numbers', i)
        lsh_time = 0
        for j in range(num_trials):
            curr_lsh = LSH(data, shingle_length, permutations, rows_per_band, buckets)
            curr_lsh_time = curr_lsh.get_runtime()
            lsh_time += curr_lsh_time
        lsh_times.append(lsh_time / num_trials)

    # print('jaccard times', jaccard_times)
    # print('min hash times', lsh_times)

    document_cts = range(1, len(data))

    # plt.plot(document_cts, jaccard_times, "-r", label='Jaccard Runtime')
    plt.plot(document_cts, lsh_times, "-b", label='LSH Runtime')
    plt.legend(loc='lower right')
    plt.xlabel('Number of Documents')
    plt.ylabel('Runtime (s)')
    plt.title('Number of Documents vs Runtime in Seconds')
    plt.savefig('ra-plots/doc_cts_vs_runtime.png')
"""


def main():
    data = []
    for i in range(1, 23):
        bad_chars = ['\n', 'W']
        file_string = open("ra-data/strain" + str(i) + ".txt","r").read()
        file_string = file_string.replace('\n', '')
        file_string = file_string.replace('W', '')
        data.append(file_string)
    lsh = LSH(data, 5, 100, 10, 50)
    file_one_idx = random.randint(0, len(data))
    file_two_idx = file_one_idx
    while file_two_idx == file_one_idx:
        file_two_idx = random.randint(0, len(data))
    similar = lsh.is_similar(file_one_idx, file_two_idx)
    if similar:
        print("Strand number", file_one_idx + 1, " and strand number", file_two_idx + 1, "are similar")
    else:
        print("Strand number", file_one_idx + 1, " and strand number", file_two_idx + 1, "are not similar")
    # permutations_vs_jaccard(data, 5, 10, 50, 100, 2)
    # rows_vs_jaccard(data, 5, 20, 50, 2)


if __name__ == "__main__":
    main()