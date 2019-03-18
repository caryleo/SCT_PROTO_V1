# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
FILENAME:       train.py
BY:             Gary 2019.3.12
LAST MODIFIED:  2019.3.12
DESCRIPTION:    train core file
"""

import logging
import os
from six.moves import cPickle

from tool.dataloader import DataLoader
import models
import misc.utils as utils

try:
    import tensorflow as tf
except:
    logging.warning("Tensorflow is not installed, no Tensorboard logging.")
    tf = None


def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)


def train(opts, device):
    # load data
    logging.info("Loading data")
    loader = DataLoader(opts)
    logging.info("Load data complete")
    opts.vocabulary_size = loader.vocabulary_size
    opts.max_caption_length = loader.max_caption_length
    logging.info("The vocabulary size: %d" % opts.vocabulary_size)
    logging.info("Max caption length: %d" % opts.max_caption_length)

    # tensorflow summary
    tf_summary_writer = tf and tf.summary.FileWriter(opts.checkpoint_path)

    infos = dict()
    histories = dict()
    if opts.start_from is not None:
        logging.info("Starting from checkpoint")
        # open old infos and check if models are compatible
        with open(os.path.join(opts.start_from, 'infos_' + opts.train_id + '.pkl')) as infofile:
            infos = cPickle.load(infofile)
            saved_model_opts = infos['opts']
            entries = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for entry in entries:
                assert vars(saved_model_opts)[entry] == vars(opts)[
                    entry], "Command line argument and saved model disagree on '%s' " % entry

        if os.path.isfile(os.path.join(opts.start_from, 'histories_' + opts.train_id + '.pkl')):
            with open(os.path.join(opts.start_from, 'histories_' + opts.train_id + '.pkl')) as historiesfile:
                histories = cPickle.load(historiesfile)

        logging.info("Load checkpoint complete")

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', dict())
    loss_history = histories.get('loss_history', dict())
    lr_history = histories.get('lr_history', dict())
    ss_prob_history = histories.get('ss_prob_history', dict())

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_index = infos.get('split_ix', loader.split_index)
    if opts.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    model = models.setup(opts)
    model.cuda(device=device)

    update_lr_flag = True
    # Assure in training mode
    model.train()

    crit = utils.LanguageModelCriterion()

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)

    # Load the optimizer
    if vars(opt).get('start_from', None) is not None:
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    while True:
        if update_lr_flag:
            # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate ** frac
                opt.current_lr = opt.learning_rate * decay_factor
                utils.set_lr(optimizer, opt.current_lr)  # set the decayed rate
            else:
                opt.current_lr = opt.learning_rate
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob
            update_lr_flag = False

        start = time.time()
        # Load data from train split (0)
        data = loader.get_batch('train')
        print('Read data:', time.time() - start)

        torch.cuda.synchronize()
        start = time.time()

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]
        tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
        fc_feats, att_feats, labels, masks = tmp

        optimizer.zero_grad()
        loss = crit(model(fc_feats, att_feats, labels), labels[:, 1:], masks[:, 1:])
        loss.backward()
        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        train_loss = loss.item()
        torch.cuda.synchronize()
        end = time.time()
        print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
              .format(iteration, epoch, train_loss, end - start))

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            if tf is not None:
                add_summary_value(tf_summary_writer, 'train_loss', train_loss, iteration)
                add_summary_value(tf_summary_writer, 'learning_rate', opt.current_lr, iteration)
                add_summary_value(tf_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
                tf_summary_writer.flush()

            loss_history[iteration] = train_loss
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            # eval model
            eval_kwargs = {'split': 'val',
                           'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.eval_split(model, crit, loader, eval_kwargs)

            # Write validation result into summary
            if tf is not None:
                add_summary_value(tf_summary_writer, 'validation loss', val_loss, iteration)
                for k, v in lang_stats.items():
                    add_summary_value(tf_summary_writer, k, v, iteration)
                tf_summary_writer.flush()
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            best_flag = False
            if True:  # if true
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)

                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['split_ix'] = loader.split_ix
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader.get_vocab()

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history
                with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '.pkl'), 'wb') as infofile:
                    cPickle.dump(infos, infofile)
                with open(os.path.join(opt.checkpoint_path, 'histories_' + opt.id + '.pkl'), 'wb') as infofile:
                    cPickle.dump(histories, infofile)

                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '-best.pkl'), 'wb') as infofile:
                        cPickle.dump(infos, infofile)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break


