import sys
import argparse
import os
import json
import re
import spacy
import html.parser


indir = '/u/cs401/A1/data/'

nlp = spacy.load("en", disable=["parser", "ner"])

def preproc1(comment, steps=range(1, 11)):
    ''' This function pre-processes a single comment

    Parameters:
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step

    Returns:
        modComm : string, the modified comment
    '''

    modComm = ''
    if 1 in steps:
        comment = comment.replace('\n', '')
    if 2 in steps:
        comment = html.unescape(comment)
    if 3 in steps:
        if 'www' in comment or 'http' in comment:
            comment = re.sub(r"http\S+", "", comment)
            comment = re.sub(r"www\S+", "", comment)

    if 4 in steps:
        p = '!\"#$%&()*\+,\-/:;<=>?@\[\]\^\_\`\{\|\}\~+'

        comment = re.sub(' {2,}', ' ', comment)
        comment = re.sub(u'([{}])'.format(p), r' \1 ', comment)
        comment = re.sub(' {2,}', '', comment)
        comment = re.sub(r'([{}])(\d)'.format(p), r'\1 \2', comment)
        comment = re.sub(r'(\d)([{}])'.format(p), r'\1 \2', comment)
        comment = re.sub(r'([{}])([a-zA-Z])'.format(p), r'\1 \2', comment)
        comment = re.sub(r'([a-zA-Z])([{}])'.format(p), r'\1 \2', comment)

    if 5 in steps:
        comment = re.sub(r"(n't|')", r" \1", comment)
    if 6 in steps:
        new_comment = []
        utt = nlp(u"" + comment)
        for token in utt:
            new_comment.append(str(token) + '/' + str(token.tag_))
        comment = ' '.join(new_comment)

    if 7 in steps:
        stopword_file = open('/u/cs401/Wordlists/StopWords', 'r')
        for stopword in stopword_file:
            stopword = stopword.strip('\n')
            pattern = u'((^' + stopword + ')/[A-Z]+|\s(' + stopword + ')/[A-Z]+)'
            comment = re.sub(pattern, "", comment)
        stopword_file.close()

    if 8 in steps:
        new_comment = ''
        comment_list = comment.split(' ')
        for comb in comment_list:
            if '/' in comb:
                word = comb.split('/')[0]
                tag = comb.split('/')[1]
                if len(word) > 0:
                    utt = nlp(u"" + word)
                    lemma = utt[0].lemma_
                    if '-' != lemma[0]:
                        comb = lemma + '/' + tag + ' '
                    new_comment += comb
        comment = new_comment[:-1]

    if 9 in steps:
        comment = re.sub(u'(\?\/\.\s+| $\?\/\.)', r'\1\n', comment)
        comment = re.sub(u'(\!\/\.\s+| $\!\/\.)', r'\1\n', comment)

        abbrev_file = open('/u/cs401/Wordlists/abbrev.english', 'r')
        for abbrev in abbrev_file:
            abbrev = abbrev.replace('\n', '')
            comment = comment.replace(abbrev, abbrev[:-1] + '&')
        abbrev_file.close()
        comment = re.sub(u'(\.\/\.\s+| $\.\/\.)', r'\1\n ', comment)

        abbrev_file = open('/u/cs401/Wordlists/abbrev.english', 'r')
        for abbrev in abbrev_file:
            abbrev = abbrev.replace('\n', '')
            comment = comment.replace(abbrev[:-1] + '&', abbrev)
        abbrev_file.close()

    if 10 in steps:
        comment = comment.lower()
    return comment

def main( args ):

    allOutput = []

    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)

            data = json.load(open(fullFile))

            for i in range(args.max):
                line = data[i]
                line = json.loads(line)

                to_drop = ['archived', 'edited', 'author_flair_css_class',
                           'link_id', 'retrieved_on']

                for key in to_drop:
                    if key in line:
                        del line[key]

                line['cat'] = file

                line['body'] = preproc1(line['body'])

                allOutput.append(line)

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    if (args.max > 200272):
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)

    main(args)
