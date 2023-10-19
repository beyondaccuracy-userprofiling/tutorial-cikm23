import gzip
import hickle
import _pickle as cPickle
import itertools
import time

def get_num_neighbor(G,etype):
    print(G.edges(etype=etype))
    for i in G.edges(etype=etype):
        print(i)
    #     exit()


def neighbormap(df,dic,user_dic,new_item_dic,col_user='user_id',col_item='item_id'):
    t=time.time()
    print('Start time')
    for i in range(len(df)):
        user=df.at[i,col_user]
        item=df.at[i,col_item]
        if item in new_item_dic:
            dic[user_dic[user]].append(new_item_dic[item])

    print('End time',time.time()-t)
    return dic

def split_char(str):
    english = 'abcdefghijklmnopqrstuvwxyz0123456789'
    output = []
    buffer = ''
    try:
        for s in str:
            if s in english or s in english.upper(): # English or numeric
                buffer += s
            elif s in ' （）*()【】/-.': # If it is a special symbol such as a space, skip it
                continue
            else: # Chinese
                if buffer:
                    output.append(buffer)
                buffer = ''
                output.append(s)
        if buffer:
            output.append(buffer)
    except:
        print(str)
    return output



def filter_sample(threshold,dic):

    del_index = []
    out = []
    for key,value in dic.items():
        if len(set(value)) < threshold:
            del_index.append(key)
        else:
            neirghbor = value
            out.append(neirghbor[:threshold])
    return out,del_index

def combination(df,users,col_user='user_id',col_item='item_id'):


    df = df[df[col_user].isin(users)]   # Filtering, the user must be a user who meets the conditions
    df.reset_index(drop=True, inplace=True)
    df_item=df[col_item].value_counts()
    items = df_item[df_item >= 10].to_dict().keys()  # Filtered, the number of users clicked on the item should be greater than a certain value
    df = df[df[col_item].isin(items)]
    df.reset_index(drop=True, inplace=True)
    print(df.shape,len(list(df.groupby([col_item]))))
    out = []
    for iter in df.groupby([col_item]):
        l = iter[1][col_user].tolist()
        l = [x for x in l if x in set(users)]
        pairs = list(itertools.combinations(l, 2))[:10 if 10>len(l) else len(l)]
        out.extend(pairs)

    out = list(zip(*set(out)))
    print('Number of sides after de-duplication:', len(out[0]))
    return out
