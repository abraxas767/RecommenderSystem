from ctypes import sizeof
import requests as r
import json


if __name__ == "__main__":

    page_index = 1
    total_book_count = 73923

    with open("./gutenberg_full.json", "w") as json_file:
        json_file.write('[\n')
        try:
            while True:
                url = "https://gutendex.com/books/?languages=en&page={}".format(page_index)
                response = r.get(url)

                if response.status_code == 200:
                    page_content = response.text 
                    if response.text == '{"detail":"Invalid page."}':
                        json_file.close()
                        raise Exception
                    try:
                        json_obj = json.loads(page_content)
                        books = json_obj['results']
                        for book in books:
                            json_file.write(',\n')
                            json.dump(book, json_file, indent=4)
                        total_book_count -= len(books)
                        print("items left: ", total_book_count)

                    except Exception as a:
                        json_file.write('\n]')
                        json_file.close()
                        print(a)

                else:
                    json_file.write('\n]')
                    json_file.close()
                    print("failure")

                page_index += 1
            
        except KeyboardInterrupt:
            json_file.write('\n]')
            json_file.close()

        except Exception as e:
            print(e)
            json_file.write('\n]')
            json_file.close()









