import argparse
from tqdm import tqdm
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default=None, type=str, required=True,
                        help=".tsv file with MSMarco documents to be translated.")
    parser.add_argument("--output_file", default=None, type=str, required=True,
                        help=".tsv file with MSMarco documents to be translated.")

    args = parser.parse_args()
    counter = 0
    doc_number = 0
    with open(args.input_file, 'r') as f_in, open(args.output_file, 'w') as f_out:
        current_id = None
        current_document = ""
        for line in tqdm(f_in):
            doc_id, document = line.strip().split('\t', 1)
            if current_id is None:
                current_id = doc_id

            if doc_id != current_id:
                f_out.write(current_id + '\t' + current_document.strip() + '\n')
                doc_number += 1
                current_id = doc_id
                current_document = document
            else:
                if current_document == "":
                    current_document = document
                else:
                    counter += 1
                    current_document = current_document + ' ' + document

        # Write the last document
        if current_document:
            doc_number += 1
            f_out.write(current_id + '\t' + current_document.strip() + '\n')
    print(f"Number of documents merged: {counter}")
    print(f"Number of documents written: {doc_number}")
    print("Done!")


if __name__ == '__main__':
    main()