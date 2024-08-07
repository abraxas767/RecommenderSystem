import sqlite3
import json
import tfidf
import sys
import gc

SQLITE_DB_PATH = './corpus/corpus.db'

def float_to_fixed_length(f, width=7, precision=7):
    formatted_str = '{:{}.{}f}'.format(f, width, precision)
    return formatted_str

def load_json_in_batches(entries, batch_size=1000):
    for i in range(0, len(entries), batch_size):
        batch = entries[i:i+batch_size] 
        for entry in batch:
            tmp = json.loads(entry[0])
            del entries[i]
            gc.collect()
            yield tmp

if __name__ == "__main__":

    # LOADS 4GB into memory
    print("loading all entries in memory")
    # title: "test", author: "test", content: json({word1: 29, word2: 14,....})
    conn = None
    cursor = None
    entries = []
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        query = "SELECT id FROM corpus"
        cursor.execute(query)
        ids = cursor.fetchall()

        query = "SELECT content FROM corpus"
        cursor.execute(query)
        entries = cursor.fetchall()
    except sqlite3.Error as e:
        print(e)
        sys.exit()

    # LOADS ANOTHER 4GB into MEMORY
    print("fill corpus")
    preprocessed_corpus = []
    for batch in load_json_in_batches(entries):
        preprocessed_corpus.append(batch)
    #for i in range(len(entries) - 1, -1, -1):
    #    #preprocessed_corpus.insert(0, json.loads(entries[i][0]))
    #    preprocessed_corpus.append(json.loads(entries[i][0]))
    #    del entries[i]

    preprocessed_corpus.reverse()

    print("generate base vector")
    basic_tfidf_vector = tfidf.get_basic_vector(preprocessed_corpus)

    print("creating tfidf matrix")
    tfidf_matrix = []
    for idx, entry in enumerate(entries):
        id = ids[idx]
        content = preprocessed_corpus[idx]
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
    
