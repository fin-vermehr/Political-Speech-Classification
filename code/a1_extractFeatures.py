import numpy as np
import sys
import argparse
import os
import json
import re
import csv

def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''

    feats = np.zeros(174)
    # punctuation_tags = ('(\/\.|\/:|\/nfp|\/\'\'|\/\`\`|\/sym|\/,|\/-lrb-|\/-rrb-|\/hyph|\/\$)')

    # Number of first-person pronouns:
    feats[0] = len(re.findall('''(\s+I\/|\s+me\/|\s+my\/|\s+mine\/|\s+we\/|
    \s+us\/|\s+our\/|\s+ours\/)''', comment))

    # Number of second-person pronouns
    feats[1] = len(re.findall('(\s+you\/|\s+your\/|\s+yours\/|\s+u\/|\s+ur\/|\s+urs\/)', comment))

    # Number of third-person pronouns
    feats[2] = len(re.findall('''(\s+he\/|\s+him\/|\s+his\/|\s+she\/|\s+her\/|
    \s+hers\/|\s+it\/|\s+its\/|\s+they\/|\s+them\/|\s+their\/|
    \s+theirs\/)''', comment))

    # Number of Coordinatin conjunction
    feats[3] = len(re.findall('(\/cc)', comment))

    # Number of past-tense verbs
    feats[4] = len(re.findall('(\/vbd)', comment))

    # Number of future-tense verbs
    feats[5] = len(re.findall('(\'ll|gonna|going\/[a-z]+\sto\/[a-z]+\s[a-z]+\/vb|will|shall|)', comment))

    # Number of commas
    feats[6] = len(re.findall('(/,)', comment))

    # Number of multi-character punctuation tokens
    feats[7] = len(re.findall('''.((\/\.|\/:|\/nfp|\/\'\'|\/\`\`|\/sym|\/,|
                \/-lrb-|\/-rrb-|\/hyph|\/\$)\s[^a-zA-Z\d]){2,}''', comment))

    # Number of common nouns
    feats[8] = len(re.findall('(\/nns|\/nn)', comment))

    # Number of proper nouns
    feats[9] = len(re.findall('(\/nnps|\/nnp)', comment))

    # Number of adverbs
    feats[10] = len(re.findall('(\/rbs|\/rbr|\/rb)', comment))

    # Number of wh-words
    feats[11] = len(re.findall('(\/wdt|\/wp|\/wp\$|wrb)', comment))

    # Number of slang acronyms
    feats[12] = len(re.findall('''(\s+smh\/|\s+fwb\/|\s+lmfao\/|\s+lmao\/|
                                    \s+lms\/|\s+tbh\/|\s+rofl\/|\s+wtf\/|
                                    \s+bff\/|\s+wyd\/|\s+lylc\/|\s+brb\/|
                                    \s+atm\/|\s+imao\/|\s+sml\/|\s+btw\/|
                                    \s+bw\/|\s+imho\/|\s+fyi\/|\s+ppl\/|
                                    \s+sob\/|\s+ttyl\/|\s+imo\/|\s+ltr\/|
                                    \s+thx\/|\s+kk\/|\s+omg\/|\s+omfg\/|
                                    \s+ttys\/|\s+afn\/|\s+bbs\/|\s+cya\/|
                                    \s+ez\/|\s+f2f\/|\s+gtr\/|\s+ic\/|\s+jk\/|
                                    \s+k\/|\s+ly\/|\s+ya\/|\s+nm\/|\s+np\/|
                                    \s+plz\/|\s+ru\/|\s+tc\/|\s+tmi\/|
                                    \s+ym\/|\s+ur\/|\s+u\/|\s+sol\/|
                                    \s+fml\/)''', comment))

    # Number of uppercase words:
    feats[13] = 0

    # Average length of sentences in tokens
    iter_n = re.finditer('\\n', comment)
    indices_n = len([m.start(0) for m in iter_n])
    iter_s = re.finditer('\s+', comment)
    indices_s = len([m.start(0) for m in iter_s])

    if indices_n == 0:
        indices_n = 1

    feats[14] = indices_s / indices_n

    # Average length of tokens
    cleaned_comment = re.sub(u'\/[a-z]+', r'', comment)
    letters = len(re.findall('[a-z\']', cleaned_comment))
    words = len(re.findall('[a-z\']+', cleaned_comment))
    if words == 0:
        words = 1
    feats[15] = letters / words

    # Number of sentences
    feats[16] = indices_n

    # Lexical Norms

    word_comment = re.sub(u'[^\s^a-z]+', r'', cleaned_comment).split(' ')

    AoA_list = []
    IMG_list = []
    FAM_list = []

    V_Warr = []
    A_Warr = []
    D_Warr = []

    with open('/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv', 'r') as BNG_f:
        with open('/u/cs401/Wordlists/Ratings_Warriner_et_al.csv', 'r') as RW_f:

            for word in word_comment:
                if word != '' and word != '\n':

                    BNG_reader = csv.reader(BNG_f)
                    RW_reader = csv.reader(RW_f)

                    for row in BNG_reader:
                        if row[1] == word:
                            AoA_list.append(int(row[3]))
                            IMG_list.append(int(row[4]))
                            FAM_list.append(int(row[5]))
                            continue

                    for row in RW_reader:
                        if row[1] == word:
                            V_Warr.append(float(row[2]))
                            A_Warr.append(float(row[5]))
                            D_Warr.append(float(row[8]))
                            continue
                BNG_f.seek(0)
                RW_f.seek(0)
    BNG_f.close()
    RW_f.close()


    AoA Average
    feats[17] = np.mean(AoA_list)
    feats[18] = np.mean(IMG_list)
    feats[19] = np.mean(FAM_list)

    feats[20] = np.std(AoA_list)
    feats[21] = np.std(IMG_list)
    feats[22] = np.std(FAM_list)

    feats[23] = np.mean(V_Warr)
    feats[24] = np.mean(A_Warr)
    feats[25] = np.mean(D_Warr)

    feats[26] = np.std(V_Warr)
    feats[27] = np.std(A_Warr)
    feats[28] = np.std(D_Warr)

    return feats


def main( args ):

    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173 + 1))

    Center_file = open('/u/cs401/A1/feats/Center_IDs.txt', 'r')
    Left_file = open('/u/cs401/A1/feats/Left_IDs.txt', 'r')
    Right_file = open('/u/cs401/feats/Right_IDs.txt', 'r')
    Alt_file = open('/u/cs401/feats/Alt_IDs.txt', 'r')

    Center_feats = np.load('/u/cs401/feats/Center_feats.dat.npy')
    Left_feats = np.load('/u/cs401/feats/Left_feats.dat.npy')
    Right_feats = np.load('/u/cs401/feats/Right_feats.dat.npy')
    Alt_feats = np.load('/u/cs401/feats/Alt_feats.dat.npy')

    for i in range(len(data)):

        if data[i]['cat'] == 'Center':
            y = 1
        elif data[i]['cat'] == 'Left':
            y = 0
        elif data[i]['cat'] == 'Right':
            y = 2
        else:
            y = 3

        comment = data[i]

        feats[i] = extract1(comment['body'])

        if comment['cat'] == 'Center':
            file = Center_file
            pol_feats = Center_feats
        elif comment['cat'] == 'Left':
            file = Left_file
            pol_feats = Left_feats
        elif comment['cat'] == 'Right':
            file = Right_file
            pol_feats = Right_feats
        else:
            file = Alt_file
            pol_feats = Alt_feats

        row_index = 0
        for line in file:
            if line[:-1] == comment['id']:
                for j in range(144):
                    feats[i, j + 29] = pol_feats[row_index, j]
                continue
                feats[i, 173] = y
            row_index += 1
        file.seek(0)

    Center_file.close()
    Left_file.close()
    Right_file.close()
    Alt_file.close()

    np.savez_compressed( args.output, feats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()

    main(args)
