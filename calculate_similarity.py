import sqlite3
import json
import tfidf
import sys

SQLITE_DB_PATH = './corpus/corpus.db'

def float_to_fixed_length(f, width=7, precision=7):
    formatted_str = '{:{}.{}f}'.format(f, width, precision)
    return formatted_str

if __name__ == "__main__":

    print("loading all entries in memory")
    # title: "test", author: "test", content: json({word1: 29, word2: 14,....})
    conn = None
    cursor = None
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        query = "SELECT * FROM corpus"
        cursor.execute(query)
        entries = cursor.fetchall()
    except sqlite3.Error as e:
        print(e)
        sys.exit()

    print("fill corpus")
    preprocessed_corpus = []
    for entry in entries:
        content = {}
        try:
            content = json.loads(entry[4])
            preprocessed_corpus.append(content)
        except Exception as e:
            print(e)
            continue

    print("generate base vector")
    basic_tfidf_vector = tfidf.get_basic_vector(preprocessed_corpus)

    print("creating tfidf matrix")
    tfidf_matrix = []
    for entry in entries:
        id = entry[0]
        content = json.loads(entry[4])
        tfidf_vector, words = tfidf.tfidf(content, preprocessed_corpus, basic_tfidf_vector)
        custom_vector = (id, tfidf_vector)
        tfidf_matrix.append(custom_vector)

    print("calculate similarites")
    for vector in tfidf_matrix:
        id = vector[0]
        top1 = (0, "")
        top2 = (0, "")
        top3 = (0, "")
        top4 = (0, "")
        top5 = (0, "")
        for inner_vector in tfidf_matrix:
            inner_id = inner_vector[0]
            if id == inner_id:
                continue
            cs = tfidf.cosine_similarity(vector[1], inner_vector[1])
            if cs > top1[0]:
                top1 = (cs, inner_id)
            if cs > top2[0] and cs < top1[0]:
                top2 = (cs, inner_id)
            if cs > top3[0] and cs < top2[0]:
                top3 = (cs, inner_id)
            if cs > top4[0] and cs < top3[0]:
                top4 = (cs, inner_id)
            if cs > top5[0] and cs < top4[0]:
                top5 = (cs, inner_id)

        top1val, top1id = float_to_fixed_length(top1[0]),  top1[1]
        top2val, top2id = float_to_fixed_length(top2[0]),  top2[1]
        top3val, top3id = float_to_fixed_length(top3[0]),  top3[1]
        top4val, top4id = float_to_fixed_length(top4[0]),  top4[1]
        top5val, top5id = float_to_fixed_length(top5[0]),  top5[1]
        final_string = top1id + ";" + top1val + ";"
        final_string += top2id + ";" + top2val + ";"
        final_string += top3id + ";" + top3val + ";"
        final_string += top4id + ";" + top4val + ";"
        final_string += top5id + ";" + top5val 
        update_query = '''
                        UPDATE corpus
                        SET sim = (?)
                        WHERE id = (?)'''
        try:
            if cursor is not None and conn is not None:
                cursor.execute(update_query, (final_string, id))
                print("{} transformed".format(id))
                conn.commit()
        except sqlite3.Error as e:
            print(e)
            continue

    if conn is not None:
        conn.close()
    
