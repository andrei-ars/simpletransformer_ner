import argparse
import logging
from simpletransformers.language_modeling import LanguageModelingModel
from simpletransformers.language_modeling import LanguageModelingArgs
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

this_folder = "lm_train"

output_dir_name = "lm_outputs_test"


# Variant 1

train_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "num_train_epochs": 3,
    "save_eval_checkpoints": False,
    "block_size": 509,
    "max_seq_length": 509,
    # "save_model_every_epoch": False,
    "learning_rate": 1e-4,
    "train_batch_size": 8, #16,
    "gradient_accumulation_steps": 4,
    "mlm": False,
    "dataset_type": "simple",
    "logging_steps": 100,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 3000,
    "evaluate_during_training_verbose": True,
    "use_cached_eval_features": True,
    "sliding_window": True,
    "use_multiprocessing": False,
    "vocab_size": 10000,
    "output_dir": "{}/from_scratch_".format(output_dir_name),
    "best_model_dir": "{}/from_scratch/best_model".format(output_dir_name),
    #"output_dir": "../{}/from_scratch_".format(output_dir_name),
    #"best_model_dir": "../{}/from_scratch/best_model".format(output_dir_name),
    "fp16": False,
    "local_rank": -1,
}
parser = argparse.ArgumentParser()
parser.add_argument(
    "--local_rank", type=int, default=-1, help="Local rank. Necessary for using the torch.distributed.launch utility."
)
args = parser.parse_args()
train_args["local_rank"] = args.local_rank
train_file = "{}/data/train.txt".format(this_folder)
test_file = "{}/data/test.txt".format(this_folder)
#model = LanguageModelingModel("gpt2", None, args=train_args, train_files=train_file,)
#model = LanguageModelingModel("bert", None, args=train_args, train_files=train_file, use_cuda=False)


# Variant 2

model_args = LanguageModelingArgs()
model_args.config = {
    "hidden_size": 384, # 768
    "num_hidden_layers": 3,  # 12
    "num_attention_heads": 3,
}
model_args.vocab_size = 20000
model_args.output_dir = "{}/from_scratch_".format(output_dir_name)
model_args.best_model_dir = "{}/from_scratch/best_model".format(output_dir_name)
model_args.num_train_epochs = 3
model_args.save_eval_checkpoints = True
model_args.overwrite_output_dir = True

model_args.evaluate_during_training = True, # needed to save the best model!
model_args.evaluate_during_training_verbose = True,
#model_args.evaluate_during_training_steps = 3000,

model = LanguageModelingModel("bert", None, args=model_args, train_files=train_file, use_cuda=False)

logging.info("Train model")

model.train_model(
    train_file, eval_file=test_file,
)

logging.info("Eval model")

model.eval_model(test_file)

logging.info("Finish")