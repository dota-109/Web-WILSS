from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from scipy import spatial
import numpy as np
import nltk
from nltk.tag import pos_tag
import os
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
import shutil

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

psc_classes = ["airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dog",
            "table", "horse", "motorbike", "person", "plant", "sheep", "sofa", "train", "monitor"]

def get_synsets(word):
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(word)
    return wn.synsets(lemma, pos=wn.NOUN)

def cos(word1, word2):
    synset1 = get_synsets(word1)[:1]
    synset2 = get_synsets(word2)[:1]
    max_length = max(max(len(hyp_path) for hyp_path in ss.hypernym_paths()) for ss in wn.all_synsets())
    vec1 = np.zeros(max_length)
    vec2 = np.zeros(max_length)
    i=0
    for s1 in synset1:
        for synset in s1.closure(lambda s: s.hypernyms()):
            vec1[synset.max_depth()] += 1*0.05*synset.max_depth()
    for s2 in synset2:
        for synset in s2.closure(lambda s: s.hypernyms()):
            vec2[synset.max_depth()] += 1*0.05*synset.max_depth()

    return 1 - spatial.distance.cosine(vec1/np.max(vec1), vec2/np.max(vec2))



def extract_subject_verb(sentence):
    tokens = word_tokenize(sentence) 
    tags = pos_tag(tokens) 
    subj = []
    for i in range(len(tags)):
        word, tag = tags[i]
        if tag=="NN":
            subj.append(word)
    
    return subj


def generate_NN(source_dir_path):
    dst_path = source_dir_path+"_NN"
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    files = sorted(os.listdir(source_dir_path))
    for file in tqdm(files[:]):
        with open(os.path.join(source_dir_path,file), 'r') as f_src:
            with open(os.path.join(dst_path, file), 'w') as f_dst:
                while True:
                    cnt = f_src.readline()
                    if cnt:
                        cnt = cnt.rstrip("\n").split(" ")
                        img_name = cnt[0]
                        subj = extract_subject_verb(" ".join(cnt[1:]))
                        f_dst.write(img_name)
                        f_dst.write(" ")
                        if len(subj)==0:
                            f_dst.write("bg")
                            
                        else:
                            for i in range(len(subj)):
                                # print(subj[i])
                                f_dst.write(str(subj[i]))
                                f_dst.write(" ")
                        f_dst.write("\n")

                    else:
                        break


def filter_replay_with_NN():
    file_path = r"../web_data_cap_file_NN"
    psc_cls_file = r"./pascal_classes.txt"
    dst_path = file_path+"-filtered"
    finished_length = 0
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    else:
        finished_length = len(os.listdir(dst_path))-1
        print(finished_length)
    files = sorted(os.listdir(file_path))[finished_length:]
    with open(psc_cls_file,'r') as f_psc:
        if finished_length>0:
                for i in range(finished_length):
                    _ = f_psc.readline()
        for file in (files[:]):
            psc_cls_cnt = f_psc.readline()
            psc_cls = psc_cls_cnt.rstrip("\n").split(" ")
            psc_cls = [_ for _ in psc_cls[1:-1]]
            with open(os.path.join(file_path,file), 'r') as f_src:
                num = 0
                with open(os.path.join(dst_path, file), 'w') as f_dst:
                    while True:
                        if num>1:
                            break
                        cnt = f_src.readline()                        
                        if cnt:
                            cnt = cnt.rstrip("\n").split(" ")
                            img_name = cnt[0]
                            
                            # print(cnt[1:])
                            flag_founded = 0
                            for nn in cnt[1:min(len(cnt)-1,2)]:
                                for psc in psc_cls:
                                    score = cos(nn, psc)
                                    print("check")
                                    print(file+" "+nn+" "+psc+" "+str(score))
                                    if score>0.6:
                                        f_dst.write(img_name)
                                        f_dst.write("\n")
                                        num+=1
                                        flag_founded = 1
                                        break
                                if flag_founded == 1:
                                    break

                        else:
                            break


def filter_webnew_with_cls():
    file_path = r"./data/voc/web_data_cls_cap_file_NN"
    dst_path = file_path+"-filtered"
    finished_length = 0
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    else:
        finished_length = len(os.listdir(dst_path))-1
        print(finished_length)
    files = sorted(os.listdir(file_path))[finished_length:]
    for file in (files[:]):
        print(file)
        with open(os.path.join(file_path,file), 'r') as f_src:
            with open(os.path.join(dst_path, file), 'w') as f_dst:
                i = 0
                while True:
                    if i>499:
                        break
                    cnt = f_src.readline()                        
                    if cnt:
                        cnt = cnt.rstrip("\n").split(" ")
                        img_name = cnt[0]
                        
                        # print(cnt[1:])
                        flag_founded = 0
                        for nn in cnt[1:min(len(cnt)-1,3)]:
                            score = cos(nn, file[:-4])
                            print("check")
                            print(file+" "+nn+" "+file[:-4]+" "+str(score))
                            if score>0.6:
                                f_dst.write(img_name)
                                f_dst.write("\n")
                                flag_founded = 1
                                i+=1
                                break
                            if flag_founded == 1:
                                break

                    else:
                        break

def move_with_filtered_caption():
    web_src_path = r"your_pascal_web_data_path"
    web_dst_path = r"your_dest_path"
    web_file_path = r"your_selected_caption_folder_path"
    cls_files = os.listdir(web_src_path)
    for cls in tqdm(sorted(cls_files)):
        if not os.path.exists(os.path.join(web_dst_path, cls)):
            os.makedirs(os.path.join(web_dst_path, cls))
        with open( os.path.join(web_file_path, cls+".txt"), "r" ) as f:
            while True:
                cnt = f.readline()
                if not cnt:
                    break
                else:
                    img_name = cnt.rstrip("\n")
                    img_src_path = os.path.join(web_src_path, cls+"/"+img_name)
                    img_dst_path = os.path.join(web_dst_path, cls+"/"+img_name)
                    shutil.copy(img_src_path, img_dst_path)



if __name__ == "__main__":
    # 1.
    filter_replay_with_NN()
    # 2.
    # move_with_filtered_caption()
