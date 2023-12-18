import os
import re
from typing import Dict, Optional, List, Union, Callable, Any, Tuple
import time, datetime
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import jax, optax
from jax import random, numpy as jnp
import jax.lax as lax
import numpy as np

from flax import linen as nn
from flax import jax_utils
from flax.training import train_state, checkpoints
from flax.training import common_utils

import torch
try:
  from torchtext.datasets import Multi30k
except OSError:
  from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import spacy.cli 
spacy.cli.download('en_core_web_sm')
spacy.cli.download('de_core_news_sm')


def padding(
    array: jnp.array,
    config: Config
    ) -> jnp.array:
  """Takes a 2D array and adds pads to max length on the last dimension.
   
  Args:
    array: 2d array of shape [batch_size, length] whose length <= max length.
    config: config class containing hyperparameters.

  Returns:
    A padded array of shape [batch_size, max_length].
  """

  assert array.ndim == 2                                                        # [Batch, SeqLen]
  batch_size, seqlen = array.shape[0], array.shape[1]
  assert seqlen <= config.max_len

  if seqlen < config.max_len:
    pads = jnp.ones((batch_size, config.max_len - seqlen), 
                   dtype=config.id_dtype) * config.pad_idx                      # [Batch, MaxLen-SeqLen]
    padded_array = jnp.concatenate((array, pads), axis=-1)                      # [Batch, MaxLen]
  else:
    padded_array = array
  
  return padded_array

@partial(jax.jit, static_argnums=(3,))
def compute_weighted_cross_entropy(
    logits: jnp.array,
    trg: jnp.array,
    weight: jnp.array,
    label_smoothing: float =0.0
    ) -> jnp.array:
  """Calculate weighted cross entropy.
   
  Args:
    logits: output from Transformer of shape [batch_size, length, target_vocab_size].
    trg: targets of shape [batch_size, length].
    weight: boolean array of shape [batch_size, length]. the pads positions in targets is 0. otherwise, 1.
    label_smoothing: label smoothing constant.

  Returns:
    Scalar loss.
  """

  assert logits.ndim == 3                                                       # [Batch, SeqLen, VocabSize]
  assert trg.ndim == 2                                                          # [Batch, SeqLen]
  assert weight.ndim == 2                                                       # [Batch, SeqLen]
  
  batch_size, vocab_size = logits.shape[0], logits.shape[2]
  confidence = 1.0 - label_smoothing
  low_confidence = (1.0 - confidence) / (vocab_size - 1)
  normalizing_constant = -(
      confidence * jnp.log(confidence) +
      (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20))
  soft_target = common_utils.onehot(
      trg, vocab_size, on_value=confidence, off_value=low_confidence)
  
  loss = -jnp.sum(soft_target * nn.log_softmax(logits), axis=-1)
  loss = loss - normalizing_constant

  normalizing_factor = batch_size                                               # normalize by batch_size
  loss = loss * weight
  loss = loss.sum() / normalizing_factor
  
  return loss

@partial(jax.jit, static_argnums=(1,5))
def train_step(
    state: train_state.TrainState,
    model: Transformer,
    src: jnp.array,
    trg: jnp.array,
    dropout_rng: jax.random.PRNGKey,
    config: Config
    ) -> Tuple[train_state.TrainState, jnp.array]:
  """Runs a training step.
  In order to minimize the number of jit compile and accelerate, 
  this step takes padded src and trg that has always same shapes. 

  Args:
    state: training state.
    model: Transformer model.
    src: padded sources of shape [batch_size, max_length].
    trg: padded targets of shape [batch_size, max_length].
    dropout_rng: PRNGKey for dropout.
    config: config class containing hyperparameters.
  
  Returns:
    new_state: updated training state.
    loss: scalar loss. 
  """

  trg_for_input = jnp.where(trg == config.eos_idx, config.pad_idx, trg)[:, :-1]
  trg_for_loss = trg[:, 1:]
  weight = jnp.where(trg_for_input == config.pad_idx, 0, 1).astype(config.id_dtype)

  def loss_fn(params):
    logits = model.apply(
      {"params": params},
      src,
      trg_for_input,
      train = True,
      rngs={"dropout": dropout_rng})
    loss = compute_weighted_cross_entropy(logits, trg_for_loss, weight)
    return loss
    
  loss, grads = jax.value_and_grad(loss_fn)(state.params)
  new_state = state.apply_gradients(grads=grads)
  return new_state, loss

def train(
    state: train_state.TrainState,
    model: Transformer,
    train_iterator,
    config,
    dropout_rng
    ) -> Tuple[train_state.TrainState, float]:
  """Runs a training loop.

  Args:
    state: training state.
    model: transformer model.
    train_iterator: iterator for training.
    config: config class containing hyperparameters.
    dropout_rng: PRNGKey for dropout.
  
  Returns:
    state: updated training state.
    loss: average loss of 1 epoch. 
  """

  loss_history = []
  for i, batch in enumerate(train_iterator):
    src, trg = padding(jnp.asarray(batch.src), config), padding(jnp.asarray(batch.trg), config)
    dropout_rng = jax.random.fold_in(dropout_rng, state.step)
    state, loss = train_step(state, model, src, trg, dropout_rng, config)
    loss_history.append(loss)
  
  train_loss = sum(loss_history) / len(loss_history)
  return state, float(train_loss)

@partial(jax.jit, static_argnums=(1,4))
def valid_step(
    state: train_state.TrainState,
    model: Transformer,
    src: jnp.array,
    trg: jnp.array,
    config: Config
    )-> jnp.array:
  """Runs a validation step.
  In order to minimize the number of jit compile and accelerate, 
  this step takes padded src and trg that has always same shapes. 

  Args:
    state: training state.
    model: Transformer model.
    src: padded sources of shape [batch_size, max_length].
    trg: padded targets of shape [batch_size, max_length].
    config: config class containing hyperparameters.
  
  Returns:
    loss: scalar loss. 
  """

  trg_for_input = jnp.where(trg == config.eos_idx, config.pad_idx, trg)[:, :-1]
  trg_for_loss = trg[:, 1:]
  weight = jnp.where(trg_for_input != config.pad_idx, 1, 0).astype(config.id_dtype)

  logits = model.apply(
                      {"params": state.params}, 
                      src, 
                      trg_for_input, 
                      train = False)
  loss = compute_weighted_cross_entropy(logits, trg_for_loss, weight)
    
  return loss

def valid(
    state: train_state.TrainState,
    model: Transformer,
    valid_iterator,
    config: Config
    ) -> float:
  """Runs a validation loop.

  Args:
    state: training state.
    model: transformer model.
    valid_iterator: iterator for validation.
    config: config class containing hyperparameters.
  
  Returns:
    loss: average loss of 1 epoch. 
  """

  loss_history = []
  for _, batch in enumerate(valid_iterator):
    src, trg = padding(jnp.asarray(batch.src), config), padding(jnp.asarray(batch.trg), config)
    loss = valid_step(state, model, src, trg, config)
    loss_history.append(loss)
  
  valid_loss = sum(loss_history) / len(loss_history)

  return float(valid_loss)

def translator(
    src: jnp.array,
    state: train_state.TrainState,
    model: Transformer,
    config: Config,
    return_attn: bool = False
    ) -> Union[List[List[str]], Tuple[List[List[str]], List[jnp.array], List[List[jnp.array]], List[List[jnp.array]]]]:
  """Translate batch sources by greedy search.

  Args:
    src: sources of shape [Batch, length]
    state: training state.
    model: transformer model.
    config: config class containing hyperparameters.
    return_attn: if true, returns Self-Attention and Source-Target-Attention matrixes both from encoder and decoder.
  
    Returns: 
      If return_attn is True, 
        the list of translatoin list of english tokens of shape [batch_size, translation length (<=max length)],
        list of attention matrix in encoder Self-Attention for the number of encoder layers,
        list of list of attention matrix in decoder Self-Attention for the number of decoder layers for the number of translation length,
        and list of list of attention matrix in decoder Source-Target-Attention for the number of decoder layers for the number of translation length.
      else,
        the logits.
  """
    
  assert src.ndim == 2                                                        # [Batch, SeqLen]

  if return_attn:
    memory, encoder_attention_list = model.apply(
        {"params": state.params}, 
        src,
        train = False,
        return_attn = True,
        method=model.encode)
  else:
    memory = model.apply(
        {"params": state.params}, 
        src,
        train = False,
        return_attn = False,
        method=model.encode)

  batch_size = src.shape[0]
  translation_id = jnp.ones((batch_size, 1), 
                            dtype=config.id_dtype) * config.bos_idx           # [Batch, 1]
  translation_done = jnp.zeros((batch_size, 1), dtype=bool)                   # [Batch, 1]

  decoder_attention_his, src_trg_attention_his = [], []
  for i in range(config.max_len):
      if return_attn:
        logits, decoder_attention_list, src_trg_attention_list = model.apply(
            {"params": state.params}, 
            translation_id, 
            src, 
            memory,
            train = False,
            return_attn = True,
            method=model.decode)
          
        decoder_attention_his.append(decoder_attention_list)
        src_trg_attention_his.append(src_trg_attention_list)
      else:
        logits = model.apply(
            {"params": state.params}, 
            translation_id, 
            src, 
            memory,
            train = False,
            return_attn = False,
            method=model.decode)                                              # [Batch, SeqLen, VocabSize]
        
      pred_id = jnp.argmax(logits, axis=-1)[:, -1][:, jnp.newaxis]            # [Batch, 1]
      translation_id = jnp.concatenate((translation_id, pred_id), axis=-1)    # [Batch, TranslationLen]
      translation_done = jnp.where(pred_id == config.eos_idx, True, translation_done)
      if jnp.all(translation_done):
        break
    
  translation = []
  for sent_id in translation_id:
    sent = itos_en(list(sent_id))
    translation.append(sent)

  if return_attn:
    return translation, encoder_attention_list, decoder_attention_his, src_trg_attention_his
  else:
    return translation

if __name__ == "__main__":
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

    def tokenize_de(text: str):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text: str):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    SRC = Field(tokenize = tokenize_de,
                tokenizer_language="de",
                init_token = '<bos>',
                eos_token = '<eos>',
                pad_token='<pad>',
                lower = True,
                batch_first = True)

    TRG = Field(tokenize = tokenize_en,
                tokenizer_language="en",
                init_token = '<bos>',
                eos_token = '<eos>',
                pad_token='<pad>',
                lower = True,
                batch_first = True)

    train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                        fields = (SRC, TRG))
    SRC.build_vocab(train_data, min_freq = 2)
    TRG.build_vocab(train_data, min_freq = 2)
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size = 128)

    def stoi_de(tokens: List[str]) -> List[int]:
        return [SRC.vocab.stoi[token] for token in tokens]

    def stoi_en(tokens: List[str]) -> List[int]:
        return [TRG.vocab.stoi[token] for token in tokens]

    def itos_de(ids: List[int]) -> List[str]:
        return [SRC.vocab.itos[id] for id in ids]

    def itos_en(ids: List[int]) -> List[str]:
        return [TRG.vocab.itos[id] for id in ids]

    max_input_len = 0
    max_target_len = 0

    for iterator in [train_iterator, valid_iterator, test_iterator]:
    for batch in iterator:
        src = jnp.asarray(batch.src)
        trg = jnp.asarray(batch.trg)
        if src.shape[1] >= max_input_len:
            max_input_len = src.shape[1]
        if trg.shape[1] >= max_target_len:
            max_target_len = trg.shape[1]

    print('max_input_len', max_input_len)
    print('max_target_len', max_target_len)

    # from jax.config import config
    # config.update("jax_debug_nans", True) # For Debug
    # from jax.config import config
    # config.update("jax_debug_nans", False) # For Usual

    config = Config()
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)
    src_shape = (config.batch_size, config.max_len)
    trg_shape = (config.batch_size, config.max_len)

    model = Transformer(config)

    print(model.tabulate(init_rng,
            jnp.ones(src_shape, config.id_dtype),
            jnp.ones(trg_shape, config.id_dtype)))
    initial_variables = jax.jit(model.init)(init_rng,
                                        jnp.ones(src_shape, dtype=config.id_dtype),
                                        jnp.ones(trg_shape, dtype=config.id_dtype)
                                        )

    def rsqrt_schedule(
        init_value: float,
        shift: int = 0,
    ):
        def schedule(count):
            return init_value * (count + shift)**-.5 * shift**.5
        return schedule

    def create_learning_rate_schedule(learning_rate: float, warmup_steps: int):
        return optax.join_schedules([
            optax.linear_schedule(
                init_value=0, end_value=learning_rate, transition_steps=warmup_steps),
            rsqrt_schedule(init_value=learning_rate, shift=warmup_steps),
        ],
                                    boundaries=[warmup_steps])

    learning_rate_fn = create_learning_rate_schedule(
        learning_rate=config.learning_rate, warmup_steps=config.warmup_steps)

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=initial_variables["params"],
        tx=optax.adam(
            learning_rate=learning_rate_fn,
            b1=0.9,
            b2=0.98,
            eps=1e-9
        )
    )
    del initial_variables

    if config.restore_checkpoints:
        latest_state_path = checkpoints.latest_checkpoint(config.ckpt_dir, config.ckpt_prefix)
        
        if latest_state_path is not None:
        state = checkpoints.restore_checkpoint(latest_state_path, state)
        last_epoch = int(re.findall(r"\d+", latest_state_path)[-1])
        print(f'Restore {latest_state_path} and restart training from epoch {last_epoch + 1}')
        else: 
        last_epoch = 0
        print('No checkpoints found. Start training from epoch 1')

    else:
        last_epoch = 0
        print('Start training from epoch 1')

    def get_h_m_s(seconds: int):
        min, sec = divmod(seconds, 60)
        hour, min = divmod(min, 60)
        return hour, min, sec

    print("================START TRAINING================")
    training_start_time = time.time()
    training_start_date = datetime.datetime.now()
    training_history = []
    validation_history = []
    sample_id = stoi_de(tokenize_de('Ich bin Student bei NAIST.'))
    sample = jnp.asarray(sample_id)[jnp.newaxis, :]

    for epoch in range(last_epoch + 1, last_epoch + 1 + config.num_epochs):
    epoch_start_time = time.time()
    is_last_epoch = epoch == last_epoch + config.num_epochs
    train_metrics = {}
    valid_metrics = {}
    print(f"Epoch_{epoch}")

    #TRAIN
    state, loss = train(state, model, train_iterator, config, dropout_rng=rng)
    print(f'Train      : loss {loss:.5f}')
    train_metrics["epoch"] = epoch
    train_metrics["loss"] = loss
    hour, min, sec = get_h_m_s(time.time() - epoch_start_time)
    print(f'Epoch Time : {hour:.0f}h {min:.0f}m {sec:.0f}s')
    train_metrics["hour"] = hour
    train_metrics["min"] = min
    train_metrics["sec"] = sec
    training_history.append(train_metrics)

    translation = translator(sample, state, model, config)[0]
    print(f'Translation: {translation}')

    #VALIDATE
    if epoch % config.valid_every_epochs == 0 or is_last_epoch:
        loss = valid(state, model, valid_iterator, config)
        print(f'Validate   : loss {loss:.5f}')
        valid_metrics["epoch"] = epoch
        valid_metrics["loss"] = loss
        validation_history.append(valid_metrics)

    #SAVE CHECKPOINTS
    if epoch % config.save_ckpt_every_epochs == 0 or is_last_epoch:
        checkpoints.save_checkpoint(
                ckpt_dir=config.ckpt_dir, prefix=config.ckpt_prefix,
                target=state, step=epoch, overwrite=True, keep=10)

    hour, min, sec = get_h_m_s(time.time() - training_start_time)
    print(f"-------------{hour:.0f}h {min:.0f}m {sec:.0f}s------------")

    if is_last_epoch:
        train_hour, train_min, train_sec = hour, min, sec

    print("================FINISH TRAINING================")

    #MAKE TRAINING LOG FILE
    with open(config.ckpt_dir + f'/train_log_from_epoch{last_epoch+1}.txt', 'w') as f:
    text = f'Training Date: {training_start_date}\n'
    text += '===================Config===================\n'
    members = [attr for attr in dir(config) if not callable(getattr(config, attr)) and not attr.startswith("__")]
    for m in members:
        text += f'{m} : {getattr(config, m)}\n'
    text += '===================Training===================\n'
    for metrics in training_history:
        text += f'epoch_{metrics["epoch"]}: loss {metrics["loss"]:.5f} Epoch Time: {metrics["hour"]:.0f}h {metrics["min"]:.0f}m {metrics["sec"]:.0f}s\n'
    text += f'Whole Training took {train_hour:.0f}h {train_min:.0f}m {train_sec:.0f}s\n'
    text += '===================Validation===================\n'
    for metrics in validation_history:
        text += f'epoch_{metrics["epoch"]}: loss {metrics["loss"]:.5f}\n'
    f.write(text)