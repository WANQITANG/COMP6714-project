import torch
from config import config
from torch.nn import functional as F
import math
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
_config = config()


def evaluate(golden_list, predict_list):
    bias=math.exp(-700)
    #处理格式 glist=[T,H]
    #print(len(golden_list))
    
    new_glists=[]### 装处理好的g list
    for item in golden_list:
        tmp_glist=[]
        newtar=''
        newhyp=''
        for i in range(len(item)):
            if item[i]=='B-TAR' or item[i]=='I-TAR':
                newtar+=item[i]
            if item[i]=='B-HYP' or item[i]=='I-HYP':
                newhyp+=item[i]
        if newtar is '':
            newtar ='0'
        if newhyp is '':
            newhyp='0'
        tmp_glist.append(newtar)
        tmp_glist.append(newhyp)
        new_glists.append(tmp_glist)
    #print(new_glists)
    #print(len(new_glists))
    #print(predict_list[0])

    new_plists=[]  ### 装处理好的p list
    for item in predict_list:
        tmp_plist=[]
        newtar=''
        newhyp=''
        for i in range(len(item)):
            if item[i]=='B-TAR' or item[i]=='I-TAR':
                newtar+=item[i]
            if item[i]=='B-HYP' or item[i]=='I-HYP':
                newhyp+=item[i]
        if newtar is '':
            newtar ='0'
        if newhyp is '':
            newhyp='0'
        tmp_plist.append(newtar)
        tmp_plist.append(newhyp)
        #print(tmp_plist)
        new_plists.append(tmp_plist)
    #print(new_plists)

    #calculate FN 在Glist 里面出现的， 在Plist 里面没有出现的次数
    FN=0
    FP=0
    TP=0
    for i in range(len(new_glists)): # glist he plist 长度一杨
        tmp_g=new_glists[i]
        tmp_p=new_plists[i]          
        for j in range(len(tmp_g)):#item 长度一样
            if tmp_g[j]!='0' and tmp_g[j]!=tmp_p[j]: # 如果在g中出现了（不是0）在p中没有出现 FN+=1
                   FN+=1
            if tmp_g[j]!='0' and tmp_g[j]==tmp_p[j]: # 如果在g中出现了（不是0）也在p中出现了 TP+=1
                   TP+=1
            if tmp_p[j]!='0' and tmp_p[j]!=tmp_g[j]: # 如果在p中出现了（不是0），但在g中没有出现FP
                   FP+=1

    print("FN:",FN,"FP:",FP,"TP",TP) # should be  correct
    # calculate f1
    if TP+FP==0:
        return 0
    else:
        P=TP/(TP+FP) #precision
        R=TP/(TP+FN) #Recall
        if (P+R)!=0:
            f1=2*P*R/(P+R)
        else:
            print('this is 00000000')
            f1=2*P*R/((P+R)+bias)
        print(f1)
        return f1
    
        

    
    


def new_LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    if input.is_cuda:
        igates = F.linear(input, w_ih)
        hgates = F.linear(hidden[0], w_hh)
        state = fusedBackend.LSTMFused.apply
        return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)

    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + ((1-forgetgate) * cellgate) # 按照要求修改c=f1*c+（1-f1)*C_new
    hy = outgate * torch.tanh(cy)

    return hy, cy


def get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists):
    ##char_embed
    input_char_embeds = model.char_embeds(batch_char_index_matrices)
    ##reshape to (70,11,50) to sort
    char_embeds=torch.reshape(input_char_embeds,[-1,input_char_embeds.shape[2],input_char_embeds.shape[3]])
    # char embeding变成（70）
    batch_word_len_lists=torch.reshape(batch_word_len_lists,[-1])
    ##sort
    perm_idx, sorted_batch_word_len_lists = model.sort_input(batch_word_len_lists)
    sorted_input_embeds = char_embeds[perm_idx]
    _, desorted_indices = torch.sort(perm_idx, descending=False)
    # padding
    output_sequence = pack_padded_sequence(sorted_input_embeds, lengths=sorted_batch_word_len_lists.data.tolist(),
                                           batch_first=True)
    #into lstm
    output_sequence, state = model.char_lstm(output_sequence)
    output_sequence, _ = pad_packed_sequence(output_sequence, batch_first=True)
    output_sequence = output_sequence[desorted_indices]
    # get forwar and backword
    forward_back=output_sequence.view(output_sequence.shape[0],output_sequence.shape[1],2,50)
    # print(forward_back.shape)
    ##define word len

    index_0 = torch.LongTensor([0])### first index
    index_1= torch.LongTensor([1])##second index
    # index__1=torch.LongTensor([batch_word_len_lists[0]-torch.tensor([1])])##last index
    forward=torch.squeeze(torch.index_select(forward_back,2,index_0))##forward(70,11,50)
    backward=torch.squeeze(torch.index_select(forward_back,2,index_1))##backword(70,11,50)
    ##get last char in each word
    # print(forward.shape)
    new_forward=torch.ones(forward.shape[0],1,50)
    # print('fot',new_forward.shape)##70,1,50
    # print(sorted_batch_word_len_lists.shape[0])#70
    for i in range(batch_word_len_lists.shape[0]):##each word  0_69
        # print('changdu',sorted_batch_word_len_lists[-1])
        select_row=torch.LongTensor([batch_word_len_lists[i]-torch.LongTensor([1])])
        # print('forwardzuih',forward[-1])
        # print('dasdasd',torch.index_select(forward[-1],0,torch.LongTensor(select_row)),'row',torch.LongTensor([sorted_batch_word_len_lists[-1]-torch.LongTensor([1])]))
        new_forward[i]=torch.index_select(forward[i],0,select_row)
        # print(new_forward[i])
    # forward=torch.index_select(forward,1,index__1-1)##last_in forward
    # print(backward.shape)
    backward = torch.index_select(backward, 1, index_0)#first_in backword
    # print('f20',forward[-1])
    # print('n20',new_forward.shape)
    # print(backward.shape)
    #(70,1,1,50)to(70,50)
    forward=torch.squeeze(new_forward)
    ##(reshape to (10,7,50)
    forward=forward.view(input_char_embeds.shape[0],input_char_embeds.shape[1],-1)
    ##same method for backwords
    backward = torch.squeeze(backward)
    backward = backward.view(input_char_embeds.shape[0], input_char_embeds.shape[1], -1)
    ##concat
    out=torch.cat([forward, backward], dim=-1)
    return out

