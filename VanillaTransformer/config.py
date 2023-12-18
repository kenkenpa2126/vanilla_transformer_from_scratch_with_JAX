class Config:
  # Architecture Config
  src_vocab_size: int = 7853
  trg_vocab_size: int = 5893
  pad_idx: int = 1
  bos_idx: int = 2
  eos_idx: int = 3
  max_len: int = 46
  embed_dim: int = 256
  id_dtype: Any = jnp.int16
  dtype: Any = jnp.float32
  num_heads: int = 8
  num_layers: int = 3
  q_dim: int = embed_dim // num_heads
  v_dim: int = embed_dim // num_heads
  ff_dim: int = 512
  dropout_rate: float = 0.1
  # Training Config
  special_idxes: List[int] = [1,2,3]
  special_tokens: List[str] = ['<bos>', '<eos>', '<pad>']
  seed: int = 0
  batch_size: int = 128
  learning_rate: float = 0.0005
  warmup_steps: int = 100
  num_epochs: int = 150
  valid_every_epochs: int = 2
  save_ckpt_every_epochs: int = 1
  restore_checkpoints: bool = True
  ckpt_prefix: str = 'translation_ckpt_'
  ckpt_dir: str = '/content/drive/My Drive/checkpoints/translation'