import torch
from torch.nn import functional as F
from transformers import Trainer, TrainerCallback


class KLRegTrainer(Trainer):
    def __init__(self, kl_weight, orig_model, *args, **kwargs):
        self.kl_weight = kl_weight
        self.orig_model = orig_model
        super(KLRegTrainer, self).__init__(*args, **kwargs)

    def _calc_kl_div(self, x1, x2, label):
        log_dis1 = F.log_softmax(x1, dim=-1)
        dis2 = F.softmax(x2, dim=-1)
        kl_divs = F.kl_div(log_dis1, dis2, reduction='none')
        mask = (label != -100).unsqueeze(-1)
        return torch.sum(kl_divs * mask.float())

    def compute_loss(self, model, inputs, return_outputs=False):

        def calc_kl_reg_loss(data_batch):
            data_outputs = model(input_ids=data_batch["input_ids"], labels=data_batch["labels"])
            if self.kl_weight > 0:
                with torch.no_grad():
                    orig_logit = self.orig_model(input_ids=data_batch["input_ids"], labels=data_batch["labels"])\
                        .logits.detach()
                kl_penal = self.kl_weight * self._calc_kl_div(data_outputs.logits[:, :-1, :],
                                                         orig_logit[:, :-1, :], data_outputs['labels'][:, 1:])
                return data_outputs.loss + kl_penal, data_outputs
            else:
                return data_outputs.loss, data_outputs

        loss_dict, output_dict = {}, {}
        total_loss = 0
        for key in inputs.keys():
            loss, output = calc_kl_reg_loss(inputs[key])
            loss_dict[key + '_loss'] = loss.item()
            output_dict[key] = output
            total_loss += loss
        self.log(loss_dict)
        if return_outputs:
            return total_loss, output_dict
        else:
            return total_loss


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels
