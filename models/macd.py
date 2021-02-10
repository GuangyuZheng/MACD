import torch
import torch.nn as nn
from utils.pretrain_config import cfg
from utils.GlobalAttention import func_attention, func_attention_textual


# ##################Loss for matching text-image###################
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


class MACDUnidirectional(nn.Module):
    def __init__(self, project_size):
        self.project_size = project_size
        super(MACDUnidirectional, self).__init__()

    @staticmethod
    def global_loss(cnn_code, rnn_code, labels, eps=1e-8):
        # --> seq_len x batch_size x nef
        if cnn_code.dim() == 2:
            cnn_code = cnn_code.unsqueeze(0)
            rnn_code = rnn_code.unsqueeze(0)

        # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
        cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
        rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
        # scores* / norm*: seq_len x batch_size x batch_size
        scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
        norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
        scores0 = scores0 / norm0.clamp(min=eps) * cfg.SMOOTH.GAMMA3

        # --> batch_size x batch_size
        scores0 = scores0.squeeze()
        scores1 = scores0.transpose(0, 1)
        if labels is not None:
            loss0 = nn.CrossEntropyLoss()(scores0, labels)
            loss1 = nn.CrossEntropyLoss()(scores1, labels)
        else:
            loss0, loss1 = None, None
        return loss0, loss1

    @staticmethod
    def local_loss(img_features, words_emb, labels,
                   sentence_mask, batch_size):
        """
            words_emb(query): batch x nef x seq_len
            img_features(context): batch x nef x 17 x 17
        """
        att_maps = []
        similarities = []
        words_emb = words_emb.permute(0, 2, 1)
        for i in range(batch_size):
            # Get the i-th text description
            words_num = int(torch.sum(sentence_mask[i]))
            # -> 1 x nef x words_num
            word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
            # -> batch_size x nef x words_num
            word = word.repeat(batch_size, 1, 1)
            # batch x nef x 17*17
            context = img_features
            """
                word(query): batch x nef x words_num
                context: batch x nef x 17 x 17
                weiContext: batch x nef x words_num
                attn: batch x words_num x 17 x 17
            """
            weiContext, attn = func_attention(word, context, cfg.SMOOTH.GAMMA1)
            att_maps.append(attn[i].unsqueeze(0).contiguous())
            # --> batch_size x words_num x nef
            word = word.transpose(1, 2).contiguous()
            weiContext = weiContext.transpose(1, 2).contiguous()
            # --> batch_size*words_num x nef
            word = word.view(batch_size * words_num, -1)
            weiContext = weiContext.view(batch_size * words_num, -1)
            #
            # -->batch_size*words_num
            row_sim = cosine_similarity(word, weiContext)
            # --> batch_size x words_num
            row_sim = row_sim.view(batch_size, words_num)

            row_sim.mul_(cfg.SMOOTH.GAMMA2).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)

            # --> 1 x batch_size
            # similarities(i, j): the similarity between the i-th image and the j-th text description
            similarities.append(row_sim)

        # batch_size x batch_size
        similarities = torch.cat(similarities, 1)
        similarities = similarities * cfg.SMOOTH.GAMMA3
        similarities1 = similarities.transpose(0, 1)
        if labels is not None:
            loss0 = nn.CrossEntropyLoss()(similarities, labels)
            loss1 = nn.CrossEntropyLoss()(similarities1, labels)
        else:
            loss0, loss1 = None, None
        return loss0, loss1, att_maps

    def forward(self, cnn_code, rnn_code, img_features, words_emb, sentence_mask, labels):
        batch_size = words_emb.size(0)
        w_loss0, w_loss1, attn_maps = self.local_loss(img_features, words_emb, labels,
                                                      sentence_mask=sentence_mask, batch_size=batch_size)
        s_loss0, s_loss1 = self.global_loss(cnn_code, rnn_code, labels)
        return w_loss0, w_loss1, s_loss0, s_loss1


class MACDBidirectional(nn.Module):
    def __init__(self, project_size):
        self.project_size = project_size
        super(MACDBidirectional, self).__init__()

    @staticmethod
    def nce(positive, noise):
        # positive: batch,  noise: batch, T
        positive = torch.exp(positive * cfg.SMOOTH.GAMMA3)
        noise = torch.exp(noise * cfg.SMOOTH.GAMMA3)
        ret = torch.log(positive) - torch.log(torch.sum(noise, dim=-1))
        return -1 * ret  # batch,

    def global_loss(self, cnn_code, rnn_code, labels, eps=1e-8):
        # --> seq_len x batch_size x nef
        if cnn_code.dim() == 2:
            cnn_code = cnn_code.unsqueeze(0)
            rnn_code = rnn_code.unsqueeze(0)

        # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
        cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
        rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
        # scores* / norm*: seq_len x batch_size x batch_size
        scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
        norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
        scores0 = scores0 / norm0.clamp(min=eps) * cfg.SMOOTH.GAMMA3

        # --> batch_size x batch_size
        scores0 = scores0.squeeze()
        scores1 = scores0.transpose(0, 1)
        if labels is not None:
            loss0 = nn.CrossEntropyLoss()(scores0, labels)
            loss1 = nn.CrossEntropyLoss()(scores1, labels)
        else:
            loss0, loss1 = None, None
        return loss0, loss1

    @staticmethod
    def local_loss_visual(img_features, words_emb, labels, sentence_mask, batch_size):
        """
            words_emb(query): batch x seq_len x nef
            img_features(context): batch x nef x 17 x 17
            sentence_mask: batch x seq_len
        """

        words_emb = words_emb.permute(0, 2, 1)  # batch, nef, seq_len

        similarities = []
        for i in range(batch_size):
            # Get the i-th text description
            words_num = int(torch.sum(sentence_mask[i]))
            # -> 1 x nef x words_num
            word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
            # -> batch_size x nef x words_num
            word = word.repeat(batch_size, 1, 1)

            # batch_size x nef x 17 x 17
            context = img_features
            weiContext, attn = func_attention(word, context, cfg.SMOOTH.GAMMA1)

            # --> batch_size x words_num x nef
            word = word.transpose(1, 2).contiguous()
            weiContext = weiContext.transpose(1, 2).contiguous()
            # --> batch_size*words_num x nef
            word = word.view(batch_size * words_num, -1)
            weiContext = weiContext.view(batch_size * words_num, -1)

            # -->batch_size*words_num
            row_sim = cosine_similarity(word, weiContext)
            # --> batch_size x words_num
            row_sim = row_sim.view(batch_size, words_num)

            row_sim.mul_(cfg.SMOOTH.GAMMA2).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)

            # --> 1 x batch_size
            # similarities(i, j): the similarity between the i-th image and the j-th text description
            similarities.append(row_sim)

        # batch_size x batch_size
        similarities = torch.cat(similarities, 1)
        similarities = similarities * cfg.SMOOTH.GAMMA3
        similarities1 = similarities.transpose(0, 1)
        if labels is not None:
            loss0 = nn.CrossEntropyLoss()(similarities, labels)
            loss1 = nn.CrossEntropyLoss()(similarities1, labels)
        else:
            loss0, loss1 = None, None

        return loss0 + loss1

    @staticmethod
    def local_loss_textual(img_features, words_emb, labels, sentence_mask, batch_size):
        """
            words_emb(context): batch x seq_len x nef
            img_features(query): batch x nef x 17 x 17
            sentence_mask: batch x seq_len
        """

        img_patch_num = img_features.size(2) * img_features.size(3)
        img_features = img_features.reshape(batch_size, -1, img_patch_num)  # batch, nef, 289
        # img_features = img_features.permute(0, 2, 1)  # batch, 289, nef

        words_emb = words_emb.permute(0, 2, 1)  # batch, nef, seq_len

        similarities = []
        for i in range(batch_size):
            # Get the i-th img description
            img = img_features[i, :, :].unsqueeze(0).contiguous()
            # -> batch_size x nef x 289
            img = img.repeat(batch_size, 1, 1)

            # batch_size x nef x seq_len
            context = words_emb
            weiContext, attn = func_attention_textual(img, context, sentence_mask, cfg.SMOOTH.GAMMA1)

            # --> batch_size x 289 x nef
            img = img.transpose(1, 2).contiguous()
            weiContext = weiContext.transpose(1, 2).contiguous()
            # --> batch_size*words_num x nef
            img = img.view(batch_size * img_patch_num, -1)
            weiContext = weiContext.view(batch_size * img_patch_num, -1)

            # -->batch_size*289
            row_sim = cosine_similarity(img, weiContext)
            # --> batch_size x 289
            row_sim = row_sim.view(batch_size, img_patch_num)

            row_sim.mul_(cfg.SMOOTH.GAMMA2).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)

            # --> 1 x batch_size
            # similarities(i, j): the similarity between the i-th image and the j-th text description
            similarities.append(row_sim)

        # batch_size x batch_size
        similarities = torch.cat(similarities, 1)
        similarities = similarities * cfg.SMOOTH.GAMMA3
        similarities1 = similarities.transpose(0, 1)
        if labels is not None:
            loss0 = nn.CrossEntropyLoss()(similarities, labels)
            loss1 = nn.CrossEntropyLoss()(similarities1, labels)
        else:
            loss0, loss1 = None, None

        return loss0 + loss1

    # def local_loss(self, img_features, words_emb, sentence_mask, batch_size):
    #     """
    #         cnn_code: batch x nef
    #         rnn_code: batch x nef
    #         words_emb(query): batch x seq_len x nef
    #         img_features(context): batch x nef x 17 x 17
    #         sentence_mask: batch x seq_len
    #     """
    #     dim = words_emb.size(-1)
    #     seq_len = words_emb.size(1)
    #     img_patch_num = img_features.size(2) * img_features.size(3)
    #     img_features = img_features.reshape(batch_size, -1, img_patch_num)  # batch, nef, 289
    #     img_features = img_features.permute(0, 2, 1)  # batch, 289, nef
    #
    #     positive_words_cos = []
    #     noise_words_cos = []
    #     for i in range(batch_size):
    #         # Get the i-th text description
    #         # -> 1 x words_num x nef
    #         word = words_emb[i, :, :].unsqueeze(0)  # 1 x seq_len x nef
    #         # -> batch_size x seq_len x nef
    #         word_expand = word.expand(batch_size, seq_len, dim)
    #         context = img_features
    #         weiContext = self.attentive_img_features(word_expand, context, gamma1=cfg.SMOOTH.GAMMA1)  # batch, seq_len, nef
    #
    #         positive_sample_c = weiContext[i, :, :].unsqueeze(0)  # 1 x seq_len x nef
    #         positive_word_cos = F.cosine_similarity(word, positive_sample_c, dim=-1) * cfg.SMOOTH.GAMMA2  # 1, seq_len
    #         positive_words_cos.append(positive_word_cos)
    #
    #         noise_word_cos = F.cosine_similarity(word_expand, weiContext, dim=-1).unsqueeze(0) * cfg.SMOOTH.GAMMA2  # 1, batch, seq_len
    #         noise_word_cos = noise_word_cos.permute(0, 2, 1)  # 1, seq_len, batch
    #         noise_words_cos.append(noise_word_cos)
    #
    #     positive_words_cos = torch.cat(positive_words_cos, dim=0).view(-1,)  # batch * seq_len,
    #     noise_words_cos = torch.cat(noise_words_cos, dim=0).view(-1, batch_size)  # batch * seq_len, batch
    #     loss_x_i = self.nce(positive_words_cos, noise_words_cos).view(batch_size, -1)  # batch, seq_len
    #     loss_x = loss_x_i.sum(-1)  # batch,
    #
    #     positive_imgs_cos = []
    #     noise_imgs_cos = []
    #     for i in range(batch_size):
    #         # Get the i-th text description
    #         # -> 1 x patch_num x nef
    #         img = img_features[i, :, :].unsqueeze(0)  # 1 x patch_num x nef
    #         # -> batch_size x patch_num x nef
    #         img_expand = img.expand(batch_size, img_patch_num, -1)
    #         context = words_emb
    #         weiContext = self.attentive_sentence_features(img_expand, context, sentence_mask=sentence_mask, gamma1=cfg.SMOOTH.GAMMA1)  # batch, pathc_num, nef
    #
    #         positive_sample_c = weiContext[i, :, :].unsqueeze(0)  # 1 x patch_num x nef
    #         positive_img_cos = F.cosine_similarity(img, positive_sample_c, dim=-1) * cfg.SMOOTH.GAMMA2  # 1, patch_num
    #         positive_imgs_cos.append(positive_img_cos)
    #
    #         noise_img_cos = F.cosine_similarity(img_expand, weiContext, dim=-1).unsqueeze(0) * cfg.SMOOTH.GAMMA2  # 1, batch, patch_num
    #         noise_img_cos = noise_img_cos.permute(0, 2, 1)  # 1, patch_num, batch
    #         noise_imgs_cos.append(noise_img_cos)
    #
    #     positive_imgs_cos = torch.cat(positive_imgs_cos, dim=0).view(-1, )  # batch * patch_num,
    #     noise_imgs_cos = torch.cat(noise_imgs_cos, dim=0).view(-1, batch_size)  # batch * patch_num, batch
    #     loss_y_i = self.nce(positive_imgs_cos, noise_imgs_cos).view(batch_size, -1)  # batch, patch_num
    #
    #     loss_y = loss_y_i.sum(dim=-1)  # batch,
    #
    #     return loss_x, loss_y

    def forward(self, cnn_code, rnn_code, img_features, words_emb, sentence_mask, labels, class_ids=None):
        batch_size = words_emb.size(0)
        w_loss0 = self.local_loss_visual(img_features, words_emb, labels, sentence_mask=sentence_mask, batch_size=batch_size)
        w_loss1 = self.local_loss_textual(img_features, words_emb, labels, sentence_mask=sentence_mask, batch_size=batch_size)
        s_loss0, s_loss1 = self.global_loss(cnn_code, rnn_code, labels)
        return w_loss0.mean(), w_loss1.mean(), s_loss0, s_loss1
