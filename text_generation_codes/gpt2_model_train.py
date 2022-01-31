from text_generation_codes.gpt2_model_envs import MODEL_CLASSES
from text_generation_codes.lambada_dataloader import LambadaHelper
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from text_generation_codes.gpt2_generation import NextWordPrediction
from torch.nn import CrossEntropyLoss
from torch import Tensor
from time import time
import torch
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def nll_loss_computation(logits: Tensor, labels: Tensor):
    assert labels.dim() == 2
    loss_func = CrossEntropyLoss()
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


def GPT2_Training(args):
    accumulating_batch_count = 0
    model_class, tokenizer_class, config_class = MODEL_CLASSES[args.model_name_or_path]
    configuration = config_class.from_pretrained(args.model_name_or_path, output_hidden_states=False)
    model = model_class.from_pretrained(args.model_name_or_path, config=configuration)
    model.to(args.device)
    logger.info('Model Parameter Configuration:')
    for name, param in model.named_parameters():
        logging.info('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
    logger.info('*' * 75)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=-1
    )
    datahelper = LambadaHelper(config=args)
    train_dataloader = datahelper.train_data_loader()
    best_dev_acc = 0.0
    step = 0
    total_train_loss = 0
    for epoch_idx in range(args.train_epochs):
        logger.info('Epoch {} starting'.format(epoch_idx + 1))
        start_time = time()
        for batch_idx, batch in tqdm(enumerate(train_dataloader)):
            model.train()
            input_tensor = batch['input_ids']
            attn_mask = batch['attn_mask']
            target_tensor = batch['labels']
            input_tensor = input_tensor.to(args.device)
            attn_mask = attn_mask.to(args.device)
            target_tensor = target_tensor.to(args.device)
            outputs = model(input_tensor, labels=input_tensor, attention_mask=attn_mask)
            loss = outputs[0]
            # outputs = model(input_tensor)
            # loss = nll_loss_computation(logits=outputs[0], labels=target_tensor)
            # print(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if (accumulating_batch_count % args.gradient_accumulation_steps) == 0:
                step = step + 1
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None
            if accumulating_batch_count % args.train_log_step == 0:
                logger.info('Epoch={}, Batch={}, Step={}, Loss={:.4f} Slap = {:.4f}'.format(epoch_idx + 1,
                                                                                            batch_idx + 1,
                                                                                            step,
                                                                                            loss.data.item(),
                                                                                            time() - start_time))

        if epoch_idx <= 20:
            continue
        if (epoch_idx + 1) % 10 == 0:
            val_acc = model_evaluation(model=model, args=args, data_loader=datahelper.valid_data_loader())
            logger.info('Validation accuracy = {:.5f} at epoch = {}'.format(val_acc, epoch_idx + 1))
            if best_dev_acc < val_acc:
                best_dev_acc = val_acc
                test_acc = model_evaluation(model=model, args=args, data_loader=datahelper.test_data_loader())
                logger.info('Test accuracy = {:.5f} at Epoch {}'.format(test_acc, epoch_idx + 1))

    val_acc = model_evaluation(model=model, args=args, data_loader=datahelper.valid_data_loader())
    logger.info('Validation accuracy = {:.5f}'.format(val_acc))
    test_acc = model_evaluation(model=model, args=args, data_loader=datahelper.test_data_loader())
    logger.info('Test accuracy = {:.5f}'.format(test_acc))


def model_evaluation(model, data_loader, args):
    word_predictor = NextWordPrediction(config=args, model=model)
    res = []
    total_count = 0
    match_count = 0
    start_time = time()
    word_predictor.model.eval()
    for batch_idx, batch in tqdm(enumerate(data_loader)):
        next_words = word_predictor.batch_predict_next_word(context_list=batch['prompt_text'])
        target_words = batch['target_word']
        total_count = total_count + len(target_words)
        for _ in range(len(next_words)):
            pred_word = next_words[_]
            targ_word = target_words[_]
            match = 1 if pred_word.lower() == targ_word else 0
            res.append((pred_word, targ_word, match))
            match_count = match_count + match
        if (batch_idx + 1) % args.valid_log_step == 0:
            logger.info('--Match count = {}, runtime = {:.4f}'.format(match_count, time() - start_time))
    logger.info('Runtime = {:.4f}'.format(time() - start_time))
    accuracy = match_count * 1.0 / total_count
    logger.info('Accuracy = {:.4f}'.format(accuracy))
    return accuracy
