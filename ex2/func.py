def split_text(text, split_type):
    if split_type[1]=='char':
        return list(text)
    elif split_type[1]=='word':
        return text.split()
    elif split_type[1]=='n-gram':
        return mk_ngram(int(split_type[2]),list(text))

def mk_ngram(n, char_list):
    ngram_list = []
    for i in range(0,len(char_list)-n):
        ngram_list.append(''.join(char_list[i:i+n]))
    return ngram_list
