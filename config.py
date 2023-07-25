from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Config:
    model_name_or_path: Optional[str] = field(
        default="roberta-base",
        metadata={"help": "Model name or path"},
    )
    
    hidden_dim: Optional[int] = field(
        default=256,
        metadata={"help": "Hidden dimension"},
    )

    num_train_epochs: Optional[int] = field(
        default=4,
        metadata={"help": "Number of training epochs"},
    )

    learning_rate: Optional[float] = field(
        default=2e-5,
        metadata={"help": "Learning rate"},
    )

    batch_size: Optional[int] = field(
        default=32,
        metadata={"help": "Batch size for training and evaluation"},
    )

    data_dir1: Optional[str] = field(
        default="./data/csv/prompts_train.csv",
        metadata={"help": "Data directory 1"},
    )

    data_dir2: Optional[str] = field(
        default="./data/csv/summaries_train.csv",
        metadata={"help": "Data directory 2"},
    )

    k_folds_dir: Optional[str] = field(
        default="./data/k_folds",
        metadata={"help": " K_folds directory"},
    )

    k_folds_pt_dir: Optional[str] = field(
        default="./data/k_folds_pt",
        metadata={"help": " K_folds_pt directory"},
    )

    checkpoints_dir: Optional[str] = field(
        default="./checkpoints",
        metadata={"help": " Checkpoints directory"},
    )

    output_dir: Optional[str] = field(
        default="./data/output",
        metadata={"help": "Output directory"},
    )

    max_seq_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max sequence length"},
    )
    
    folds: Optional[int] = field(
        default=4,
        metadata={"help": "Fold number"},
    )

    num_proc: Optional[int] = field(
        default=4,
        metadata={"help": "Number of processes"},
    )
        
    dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "Amount of dropout to apply"},
    )

    