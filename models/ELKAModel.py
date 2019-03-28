# AdaAttMo with Memory

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttModel(nn.Module):
    def __init__(self, opts):
        super(AttModel, self).__init__()
        self.vocab_size = opts.vocabulary_size
        self.input_encoding_size = opts.input_encoding_size
        # self.rnn_type = opts.rnn_type
        self.rnn_size = opts.rnn_size
        self.num_layers = opts.num_layers
        self.drop_prob_lm = opts.dropout_prob
        self.seq_length = opts.max_caption_length
        self.fc_feat_size = opts.fc_feat_size
        self.att_feat_size = opts.att_feat_size
        self.att_hid_size = opts.att_hid_size

        self.ss_prob = 0.0  # Schedule sampling probability

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                   nn.ReLU(),
                                   nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                      nn.ReLU(),
                                      nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.rnn_size),
                                       nn.ReLU(),
                                       nn.Dropout(self.drop_prob_lm))
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(self.num_layers, bsz, self.rnn_size).zero_(),
                weight.new(self.num_layers, bsz, self.rnn_size).zero_())

    def forward(self, fc_feats, att_feats, seq):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        outputs = list()

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        _att_feats = self.att_embed(att_feats.view(-1, self.att_feat_size))
        att_feats = _att_feats.view(*(att_feats.size()[:-1] + (self.rnn_size,)))

        # Project the attention feats first to reduce memory and computation consumptions. 进入
        p_att_feats = self.ctx2att(att_feats.view(-1, self.rnn_size))
        p_att_feats = p_att_feats.view(*(att_feats.size()[:-1] + (self.att_hid_size,)))

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0:  # otherwise no need to sample
                sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()

                    prob_prev = torch.exp(outputs[-1].data)  # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    it.requires_grad = False
            else:
                it = seq[:, i].clone()
                # break if all the sequences end
            if i >= 1 and seq[:, i].data.sum() == 0:
                break

            xt = self.embed(it)

            output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state)
            output = F.log_softmax(self.logit(output))
            outputs.append(output)

        return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def get_logprobs_state(self, it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, state):
        # 'it' is Variable contraining a word index
        xt = self.embed(it)

        output, state = self.core(xt, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, state)
        logprobs = F.log_softmax(self.logit(output))

        return logprobs, state

    def sample_beam(self, fc_feats, att_feats, opts={}):
        beam_size = opts.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        _att_feats = self.att_embed(att_feats.view(-1, self.att_feat_size))
        att_feats = _att_feats.view(*(att_feats.size()[:-1] + (self.rnn_size,)))

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats.view(-1, self.rnn_size))
        p_att_feats = p_att_feats.view(*(att_feats.size()[:-1] + (self.att_hid_size,)))

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = fc_feats[k:k + 1].expand(beam_size, fc_feats.size(1))
            tmp_att_feats = att_feats[k:k + 1].expand(*((beam_size,) + att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = p_att_feats[k:k + 1].expand(*((beam_size,) + p_att_feats.size()[1:])).contiguous()

            for t in range(1):
                if t == 0:  # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
                    it.requires_grad = False
                    xt = self.embed(it)

                output, state = self.core(xt, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, state)
                logprobs = F.log_softmax(self.logit(output))

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats,
                                                  opts=opts)
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def sample(self, fc_feats, att_feats, opts={}):
        sample_max = opts.get('sample_max', 1)
        beam_size = opts.get('beam_size', 1)
        temperature = opts.get('temperature', 1.0)
        if beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, opts)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        _att_feats = self.att_embed(att_feats.view(-1, self.att_feat_size))
        att_feats = _att_feats.view(*(att_feats.size()[:-1] + (self.rnn_size,)))

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats.view(-1, self.rnn_size))
        p_att_feats = p_att_feats.view(*(att_feats.size()[:-1] + (self.att_hid_size,)))

        seq = []
        seqLogprobs = []
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.data.new(batch_size).long().zero_()
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu()  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1).cuda()
                it.requires_grad = False
                sampleLogprobs = logprobs.gather(1, it)  # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing

            it.requires_grad = False
            xt = self.embed(it)

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it)  # seq[t] the input of t+2 time step

                seqLogprobs.append(sampleLogprobs.view(-1))

            output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state)
            logprobs = F.log_softmax(self.logit(output))

        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)

    def beam_search(self, state, logprobs, *args, **kwargs):
        # args are the miscelleous inputs to the core in addition to embedded word and state
        # kwargs only accept opts

        def beam_step(logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            # INPUTS:
            # logprobsf: probabilities augmented after diversity
            # beam_size: obvious
            # t        : time instant
            # beam_seq : tensor contanining the beams
            # beam_seq_logprobs: tensor contanining the beam logprobs
            # beam_logprobs_sum: tensor contanining joint logprobs
            # OUPUTS:
            # beam_seq : tensor containing the word indices of the decoded captions
            # beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            # beam_logprobs_sum : joint log-probability of each beam

            ys, ix = torch.sort(logprobsf, 1, True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols):  # for each column (word, essentially)
                for q in range(rows):  # for each beam expansion
                    # compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q, c]
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob.cpu()
                    candidates.append(dict(c=ix[q, c], q=q,
                                           p=candidate_logprob,
                                           r=local_logprob))
            candidates = sorted(candidates, key=lambda x: -x['p'])

            new_state = [_.clone() for _ in state]
            # beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1:
                # we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                # fork beam index q into index vix
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                # rearrange recurrent states
                for state_ix in range(len(new_state)):
                    #  copy over state in previous beam q to new beam at vix
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']]  # dimension one is time step
                # append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c']  # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r']  # the raw logprob here
                beam_logprobs_sum[vix] = v['p']  # the new (sum) logprob along this beam
            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates

        # start beam search
        opts = kwargs['opts']
        beam_size = opts.get('beam_size', 10)

        beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
        beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
        # running sum of logprobs for each beam
        beam_logprobs_sum = torch.zeros(beam_size)
        done_beams = []

        for t in range(self.seq_length):
            """pem a beam merge. that is,
            for every previous beam we now many new possibilities to branch out
            we need to resort our beams to maintain the loop invariant of keeping
            the top beam_size most likely sequences."""
            logprobsf = logprobs.data.float()  # lets go to CPU for more efficiency in indexing operations
            # suppress UNK tokens in the decoding
            logprobsf[:, logprobsf.size(1) - 1] = logprobsf[:, logprobsf.size(1) - 1] - 1000

            beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates_divm = beam_step(
                logprobsf,
                beam_size,
                t,
                beam_seq,
                beam_seq_logprobs,
                beam_logprobs_sum,
                state)

            for vix in range(beam_size):
                # if time's up... or if end token is reached then copy beams
                if beam_seq[t, vix] == 0 or t == self.seq_length - 1:
                    final_beam = {
                        'seq': beam_seq[:, vix].clone(),
                        'logps': beam_seq_logprobs[:, vix].clone(),
                        'p': beam_logprobs_sum[vix]
                    }
                    done_beams.append(final_beam)
                    # don't continue beams from finished sequences
                    beam_logprobs_sum[vix] = -1000

            # encode as vectors
            it = beam_seq[t]
            logprobs, state = self.get_logprobs_state(it.cuda(), *(args + (state,)))

        done_beams = sorted(done_beams, key=lambda x: -x['p'])[:beam_size]
        return done_beams


class ELKALSTM(nn.Module):
    def __init__(self, opts):
        super(ELKALSTM, self).__init__()
        self.input_encoding_size = opts.input_encoding_size
        # self.rnn_type = opts.rnn_type
        self.rnn_size = opts.rnn_size
        self.num_layers = opts.num_layers
        self.drop_prob_lm = opts.drop_prob_lm
        self.fc_feat_size = opts.fc_feat_size
        self.att_feat_size = opts.att_feat_size
        self.att_hid_size = opts.att_hid_size

        # Build a LSTM
        # 单词w到隐藏h_t，全局视觉v_g到隐藏h_t，即输入x到隐藏h_t
        self.w2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.v2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        # 这里只考虑一层，所以i2h为空，h2h只有一层，即隐藏h_t-1到隐藏h_t
        self.i2h = nn.ModuleList(
            [nn.Linear(self.rnn_size, 5 * self.rnn_size) for _ in range(self.num_layers - 1)])
        self.h2h = nn.ModuleList(
            [nn.Linear(self.rnn_size, 5 * self.rnn_size) for _ in range(self.num_layers)])

        # Layers for getting the fake region 这里只考虑一层
        if self.num_layers == 1:
            # sentinel需要的单词和全局视觉，即输入x到g_t
            self.r_w2h = nn.Linear(self.input_encoding_size, self.rnn_size)
            self.r_v2h = nn.Linear(self.rnn_size, self.rnn_size)
        else:
            # 暂时不管了
            self.r_i2h = nn.Linear(self.rnn_size, self.rnn_size)
        # sentinel需要的来自LSTM的隐藏h_t-1，即h_t-1到g_t
        self.r_h2h = nn.Linear(self.rnn_size, self.rnn_size)

    def forward(self, xt, img_fc, state):
        # 用于保存所有的隐藏状态和记忆单元，暂时不管了
        hs = list()
        cs = list()
        # 因为只有一层，因此这里只执行一次
        for L in range(self.num_layers):
            # c,h from previous timesteps 取上一个状态中拿到的h_t-1和c_t-1
            prev_h = state[0][L]
            prev_c = state[1][L]
            # the input to this layer
            if L == 0:
                # 输入，由当前单词w_t和全局视觉特征v_g，也就是x_t到隐藏状态h_t
                x = xt
                i2h = self.w2h(x) + self.v2h(img_fc)
            else:
                # 暂时不管了
                x = hs[-1]
                x = F.dropout(x, self.drop_prob_lm, self.training)
                i2h = self.i2h[L - 1](x)

            # 隐藏到隐藏，h_t-1到h_t，这里直接加和准备后面的处理
            all_input_sums = i2h + self.h2h[L](prev_h)

            # 前三行的三个门
            sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
            sigmoid_chunk = F.sigmoid(sigmoid_chunk)
            # decode the gates
            in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
            forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
            out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

            # decode the write inputs 后两行隐藏状态，做maxout
            in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size)
            in_transform = torch.max(
                in_transform.narrow(1, 0, self.rnn_size),
                in_transform.narrow(1, self.rnn_size, self.rnn_size))

            # perform the LSTM update c的更新
            next_c = forget_gate * prev_c + in_gate * in_transform
            # gated cells form the output
            tanh_nex_c = F.tanh(next_c)

            # h的更新
            next_h = out_gate * tanh_nex_c

            # 只有一层，一次执行
            if L == self.num_layers - 1:
                if L == 0:
                    # 单词和全局视觉到g_t，即x_td到g_t
                    i2h = self.r_w2h(x) + self.r_v2h(img_fc)
                else:
                    # 暂时不管了
                    i2h = self.r_i2h(x)

                # h_t-1 到 g_t，最终的g_t
                n5 = i2h + self.r_h2h(prev_h)

                # 最终的s_t
                fake_region = F.sigmoid(n5) * tanh_nex_c

            # 这里用于保存两层LSTM，暂时不管了
            cs.append(next_c)
            hs.append(next_h)

        # set up the decoder 多层RNN需要取顶层，这里只有一层，也就是底层LSTM结果h_t，加一层dropout，准备送Att
        top_h = hs[-1]
        top_h = F.dropout(top_h, self.drop_prob_lm, self.training)

        # 这里s_t也做一次dropout处理？？？
        fake_region = F.dropout(fake_region, self.drop_prob_lm, self.training)

        # 传回状态，只考虑一层，那么也就是一个h和一个c
        state = (torch.cat([_.unsqueeze(0) for _ in hs], 0),
                 torch.cat([_.unsqueeze(0) for _ in cs], 0))

        # 前两个都用于Att
        return top_h, fake_region, state


class ELKAAttention(nn.Module):
    def __init__(self, opts):
        super(ELKAAttention, self).__init__()
        self.input_encoding_size = opts.input_encoding_size
        # self.rnn_type = opts.rnn_type
        self.rnn_size = opts.rnn_size
        self.drop_prob_lm = opts.drop_prob_lm
        self.att_hid_size = opts.att_hid_size

        # fake region embed s_t的嵌入，先转回图像编码宽度，再嵌到Att宽度
        self.fr_linear = nn.Sequential(
            nn.Linear(self.rnn_size, self.input_encoding_size),
            nn.ReLU(),
            nn.Dropout(self.drop_prob_lm))

        self.fr_embed = nn.Linear(self.input_encoding_size, self.att_hid_size)

        # h out embed h_t的嵌入，同样的操作
        self.ho_linear = nn.Sequential(
            nn.Linear(self.rnn_size, self.input_encoding_size),
            nn.Tanh(),
            nn.Dropout(self.drop_prob_lm))

        self.ho_embed = nn.Linear(self.input_encoding_size, self.att_hid_size)

        self.alpha_net = nn.Linear(self.att_hid_size, 1)

        self.att2h = nn.Linear(self.input_encoding_size, self.rnn_size)

    def forward(self, h_out, fake_region, conv_feat, conv_feat_embed):
        # View into three dimensions
        # 计算空间区域个数，也就是att * att
        att_size = conv_feat.numel() // conv_feat.size(0) // self.rnn_size
        conv_feat = conv_feat.view(-1, att_size, self.rnn_size) # 处理前的视觉特征 batch * attsize * rnnsize
        conv_feat_embed = conv_feat_embed.view(-1, att_size, self.att_hid_size) # 处理好的视觉特征 batch * attsize * atthid

        # view neighbor from bach_size * neighbor_num x rnn_size to bach_size x rnn_size * neighbor_num
        fake_region = self.fr_linear(fake_region)
        fake_region_embed = self.fr_embed(fake_region)

        h_out_linear = self.ho_linear(h_out)
        h_out_embed = self.ho_embed(h_out_linear)

        # 扩展第二维
        txt_replicate = h_out_embed.unsqueeze(1).expand(h_out_embed.size(0), att_size + 1, h_out_embed.size(1))

        # 拼在一起
        img_all = torch.cat((fake_region.view(-1, 1, self.input_encoding_size), conv_feat), 1)
        img_all_embed = torch.cat((fake_region_embed.view(-1, 1, self.input_encoding_size), conv_feat_embed), 1)

        # z_t
        hA = F.tanh(img_all_embed + txt_replicate)
        hA = F.dropout(hA, self.drop_prob_lm, self.training)

        # 转换维度，最后一维变成1
        hAflat = self.alpha_net(hA.view(-1, self.att_hid_size))

        # 算注意力
        PI = F.softmax(hAflat.view(-1, att_size + 1)) # 原来是 batch * (attsize + 1) * 1 去掉最后一维算softmax

        # 最终注意力结果
        visAtt = torch.bmm(PI.unsqueeze(1), img_all) # batch * 1 * （attsize + 1） **** batch * (attsize + 1) * incodingsize
        visAttdim = visAtt.squeeze(1) # batch * incodingsize 注意力结果

        # 传到MLP的内容
        atten_out = visAttdim + h_out_linear # batch * incodingsize

        # MLP
        h = F.tanh(self.att2h(atten_out))
        h = F.dropout(h, self.drop_prob_lm, self.training)
        return h


class ELKACore(nn.Module):
    def __init__(self, opts):
        super(ELKACore, self).__init__()
        self.lstm = ELKALSTM(opts)
        self.attention = ELKAAttention(opts)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state):
        # LSTM 产生一个h，分别送到Att和MLP，这里只考虑到送到Att的部分
        h_out, p_out, state = self.lstm(xt, fc_feats, state)
        atten_out = self.attention(h_out, p_out, att_feats, p_att_feats)
        return atten_out, state


class ELKAModel(AttModel):
    def __init__(self, opts):
        super(ELKAModel, self).__init__(opts)
        self.core = ELKACore(opts)
