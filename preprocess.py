from tfidf import *
import pyzim
import zipfile as z
import io
from ebooklib import epub
import ebooklib as ebook
from bs4 import BeautifulSoup
import json
import uuid
import sqlite3
import sys

TMP_EPUB_PATH = './epub/'
SQLITE_DB_PATH = './corpus/corpus.db'
ZIM_PATH = 'corpus/output.zim'


def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text()

def get_stream_length(stream):
    stream.seek(0, 2)
    length = stream.tell()
    stream.seek(0)
    return length

def unzip_epub(epub, extract_to):
    with z.ZipFile(epub) as zipref:
        zipref.extractall(extract_to)

def get_sqlite_connection():
    # SQLITE CONNECTION
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS corpus (
                            id TEXT PRIMARY KEY,
                            title TEXT,
                            author TEXT,
                            year DATE,
                            content TEXT,
                            sim TEXT
                            )
                       ''') 
    return conn, cursor


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Counter):
            return dict(obj)
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":


    # OPEN SQLITE CONNECTION
    conn, cursor = None, None
    try:
        conn, cursor = get_sqlite_connection()
    except Exception as e:
        print(e)
        sys.exit()

    # PREPROCESS CORPUS
    corpus = []
    count_found = 0
    count_saved = 0
    with pyzim.Zim.open(ZIM_PATH) as zim:

        for entry in zim.iter_entries():

            all_text = ''
            if entry.mimetype == "application/epub+zip":
                count_found += 1 
                try:
                    ebup_bytes = entry.read()
                    epub_stream = io.BytesIO(ebup_bytes)
                    unzip_epub(epub_stream, TMP_EPUB_PATH + 'tmp_epub.epub')
                    book = epub.read_epub(TMP_EPUB_PATH + 'tmp_epub.epub/')
                    title = book.get_metadata('DC', 'title')[0][0]
                    for item in corpus:
                        if item['title'] == title:
                            print("ITEM ALREADY IN CORPUS, SKIPPING")
                            continue
                    year = book.get_metadata('DC', 'date')[0][0]
                    author = book.get_metadata('DC', 'creator')[0][0] 
                    content = ''
                    for i in book.get_items_of_type(ebook.ITEM_DOCUMENT):
                        content += extract_text_from_html(i.get_content())
                    content = dict(preprocess(content))
                    content_json = json.dumps(content, ensure_ascii=True)
                    id = uuid.uuid4().hex
                    cursor.execute('''
                                INSERT INTO corpus (id, title, author, year, content)
                                VALUES (?, ?, ?, ?, ?)
                                   ''', (id, title, author, year, content_json))
                    print("{} saved to db".format(id))
                    conn.commit()
                    count_saved += 1

                except Exception  as e:
                    print(e)
                    continue

        zim.close()

    conn.close()

    print('found {} ebooks, added {} to database'.format(count_found, count_saved))
